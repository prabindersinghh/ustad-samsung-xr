"""Track 4 entry point: one first-person video in, one <video_id>.json out.

    python -m ustad_t4.run <video.mp4> <out.json>

Stages (each degrades gracefully when its model is not available):
  1. duration           - ffprobe
  2. motion curve       - model-free importance
  3. actions            - LaViLa TimeSformer-B (EPIC-100 verb/noun heads)
  4. highlights         - coverage-greedy over confident verb-noun pairs
  5. label              - zero-shot video classifier over the 16 kit labels

Nothing here looks at the video ID beyond echoing it into the output: no
per-video branches, no ground truth, as the reproducibility rules require.

Set USTAD_SAVE_PREDS=<dir> to also keep the raw per-window predictions, so
selection can be re-tuned without re-running the model.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import actions as action_model
from . import importance, label as label_model, postprocess, schema, select, video

TEAM_ID = os.environ.get("TEAM_ID", "INCOGNITO")
KITCHEN_GATE = 0.15  # CLIP probability of "cooking" above which the action model runs
HERE = Path(__file__).resolve().parent
VOCAB = json.loads((HERE / "supported_classes.json").read_text())  # copy of the kit file

POLICY = Path(os.environ.get("USTAD_POLICY", HERE.parents[1] / "weights" / "policy.json"))


def load_policy():
    """RL highlight policy (scripts/train_policy.py), only if it beat the greedy rule held-out."""
    if not POLICY.exists():
        return None
    p = json.loads(POLICY.read_text())
    if p.get("heldout_coverage_policy", 0) <= p.get("heldout_coverage_greedy_rule", 1):
        return None
    return p["weights"]


def summarize(path: str, gate: bool = True) -> dict:
    """gate=False always runs the action model (the Ustad app: hands-on work
    footage that is not a kitchen, where the action timeline is the point)."""
    t0 = time.time()
    dur = video.duration(path)
    motion = importance.motion_curve(path, dur)

    # Kitchen footage gets the EPIC action model; other activities (FPVSum's
    # sports) are scored on highlight F1 only, so the kitchen model would only
    # add noise there. The gate is CLIP's view of the content, never the ID.
    kitchen = (not gate) or label_model.cooking_prob(path, dur) >= KITCHEN_GATE

    acts: list[dict] = []
    preds = []
    if kitchen and action_model.available():
        preds = action_model.predict(path)
        acts = postprocess.build_actions(preds, VOCAB)
        save = os.environ.get("USTAD_SAVE_PREDS")
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            (Path(save) / f"{Path(path).stem}.preds.json").write_text(
                json.dumps({"duration": dur, "motion": motion, "windows": [asdict(w) for w in preds]}))

    lab = label_model.classify(path, dur, preds, VOCAB)

    cands = postprocess.highlight_candidates(preds, dur, motion)
    if cands and any(c.pairs for c in cands):
        policy = load_policy()
        chosen = select.by_policy(cands, dur, policy) if policy else select.by_coverage(cands, dur)
    else:
        excite = label_model.excitement_curve(path, dur, lab["name"])
        curve = motion if excite is None else [0.4 * m + 0.6 * e for m, e in zip(motion, excite)]
        chosen = select.by_importance(select.candidate_windows(curve, dur, seg=postprocess.HL_LEN), dur)

    doc = {
        "team_id": TEAM_ID,
        "video_id": Path(path).stem,
        "duration_sec": round(dur, 2),
        "label": lab,
        "actions": acts,
        "highlights": [{"segment": [round(w.start, 2), round(w.end, 2)]} for w in chosen],
    }
    print(f"{doc['video_id']}: {dur:.1f}s, {len(chosen)} highlights, {len(acts)} actions, "
          f"label={lab['name']}, {time.time() - t0:.1f}s", file=sys.stderr)
    return doc


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m ustad_t4.run <video.mp4> <out.json>", file=sys.stderr)
        return 2
    doc = summarize(argv[1])
    errors, warnings = schema.check(doc, vocab=VOCAB)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    if errors:
        for e in errors:
            print("error:", e, file=sys.stderr)
        return 1
    schema.write(Path(argv[2]), doc)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
