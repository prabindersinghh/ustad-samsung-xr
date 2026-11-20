"""Score the current selection code on EPIC-100 *train* episodes.

    python scripts/eval_train.py [preds_dir]

Each episode = window predictions saved by cloud/train_run.py + the public train
annotations for that video. Re-runs build_actions / highlight selection with the
current code and reports coverage and action mAP. Never touches the scored
manifest videos.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ustad_t4 import postprocess, scoring, select  # noqa: E402
from ustad_t4.actions import WindowPred  # noqa: E402

VOCAB = json.loads((ROOT / "kit" / "supported_classes.json").read_text())


def secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_gt() -> dict:
    gt = defaultdict(list)
    for r in csv.DictReader((ROOT / "data" / "epic_ann" / "EPIC_100_train.csv").open()):
        gt[r["video_id"]].append({
            "verb": VOCAB["verbs"][int(r["verb_class"])],
            "noun": VOCAB["nouns"][int(r["noun_class"])],
            "segment": [secs(r["start_timestamp"]), secs(r["stop_timestamp"])],
        })
    return gt


def load_episodes(preds_dir: Path) -> list[dict]:
    eps = []
    for f in sorted(preds_dir.glob("*.preds.json")):
        d = json.loads(f.read_text())
        wins = [WindowPred(w["start"], w["end"], [tuple(x) for x in w["verbs"]], [tuple(x) for x in w["nouns"]])
                for w in d["windows"]]
        eps.append({"video_id": f.name.split(".")[0], "duration": d["duration"], "motion": d["motion"], "windows": wins})
    return eps


def evaluate(eps: list[dict], gt: dict, params: dict | None = None) -> dict:
    params = params or {}
    saved = {k: getattr(postprocess, k) for k in params}
    for k, v in params.items():
        setattr(postprocess, k, v)
    try:
        cov, mp = [], []
        for e in eps:
            g = gt.get(e["video_id"], [])
            acts = postprocess.build_actions(e["windows"], VOCAB)
            cands = postprocess.highlight_candidates(e["windows"], e["duration"], e["motion"])
            hl = select.by_coverage(cands, e["duration"])
            cov.append(scoring.coverage(g, [[w.start, w.end] for w in hl], e["duration"]))
            mp.append(scoring.action_map(g, acts))
        return {"coverage": sum(cov) / len(cov), "action_map": sum(mp) / len(mp), "n": len(eps)}
    finally:
        for k, v in saved.items():
            setattr(postprocess, k, v)


if __name__ == "__main__":
    preds_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "train_preds"
    print(evaluate(load_episodes(preds_dir), load_gt()))
