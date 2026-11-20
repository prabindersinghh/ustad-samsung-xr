"""Rebuild EPIC output JSONs from saved per-window predictions with the current code.

The cloud batch saved raw model outputs (preds/<id>.preds.json) next to each
JSON. Selection and action scoring are cheap, so after any change to them
(fine-tuned scoring head, RL policy) outputs are regenerated here instead of
re-running the backbone. The label comes from the cloud JSON (it needs pixels).
The result is identical to what run.sh produces for the same video.

    python scripts/assemble_epic.py outputs_cloud/preds outputs_cloud/outputs outputs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ustad_t4 import postprocess, run, schema, select  # noqa: E402
from ustad_t4.actions import WindowPred  # noqa: E402


def main(preds_dir: str, cloud_dir: str, out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = run.load_policy()
    n = bad = 0
    for pf in sorted(Path(preds_dir).glob("*.preds.json")):
        vid = pf.name[: -len(".preds.json")]
        cloud = Path(cloud_dir) / f"{vid}.json"
        if not cloud.exists():
            print(f"{vid}: no cloud json, skipped")
            continue
        p = json.loads(pf.read_text())
        preds = [WindowPred(**w) for w in p["windows"]]
        dur, motion = p["duration"], p["motion"]
        acts = postprocess.build_actions(preds, run.VOCAB)
        cands = postprocess.highlight_candidates(preds, dur, motion)
        if cands and any(c.pairs for c in cands):
            chosen = select.by_policy(cands, dur, policy) if policy else select.by_coverage(cands, dur)
            hl = [{"segment": [round(w.start, 2), round(w.end, 2)]} for w in chosen]
        else:
            hl = json.loads(cloud.read_text())["highlights"]
        doc = {"team_id": run.TEAM_ID, "video_id": vid, "duration_sec": round(dur, 2),
               "label": json.loads(cloud.read_text())["label"], "actions": acts, "highlights": hl}
        errors, _ = schema.check(doc, vocab=run.VOCAB)
        if errors:
            bad += 1
            print(vid, errors)
            continue
        schema.write(out / f"{vid}.json", doc)
        n += 1
    print(f"assembled {n}, invalid {bad}, policy={'on' if policy else 'off'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
