"""Fine-tune the action scoring head on EPIC-100 train episodes.

Each proposed run (one verb-noun pair over consecutive windows) is labelled 1 if
it matches a train ground-truth action of the same pair at tIoU >= 0.3, else 0.
A logistic head over postprocess.run_features is fit with gradient descent.
Held-out train videos measure Action mAP before and after.

    python scripts/train_refine.py      # writes weights/refine.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from ustad_t4 import postprocess, refine, scoring  # noqa: E402
from eval_train import VOCAB, load_episodes, load_gt  # noqa: E402

SEED = 7


def dataset(eps, gt):
    X, y = [], []
    for e in eps:
        g = gt.get(e["video_id"], [])
        for r in postprocess.action_runs(e["windows"]):
            v, n = VOCAB["verbs"][r["v"]], VOCAB["nouns"][r["n"]]
            seg = [r["start"], r["end"]]
            hit = any(a["verb"] == v and a["noun"] == n and scoring.tiou(seg, a["segment"]) >= 0.3 for a in g)
            X.append(postprocess.run_features(r))
            y.append(1.0 if hit else 0.0)
    return np.array(X), np.array(y)


def mean_map(eps, gt):
    return float(np.mean([scoring.action_map(gt.get(e["video_id"], []), postprocess.build_actions(e["windows"], VOCAB))
                          for e in eps]))


def main() -> int:
    eps = load_episodes(ROOT / "data" / "train_preds")
    gt = load_gt()
    random.Random(SEED).shuffle(eps)
    n_test = max(3, len(eps) // 5)
    test, train = eps[:n_test], eps[n_test:]

    if refine.PATH.exists():
        refine.PATH.unlink()
    refine._w = None
    before = mean_map(test, gt)

    X, y = dataset(train, gt)
    pos = y.mean()
    w = np.zeros(X.shape[1])
    # class-balanced logistic regression
    sw = np.where(y > 0, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ ((p - y) * sw) / len(y) + 1e-3 * w
        w -= 0.5 * grad

    refine.PATH.write_text(json.dumps({
        "weights": w.tolist(),
        "features": ["best", "mean", "log1p_count", "top1", "best_x_top1", "bias"],
        "positives": float(pos), "examples": int(len(y)),
        "train_videos": [e["video_id"] for e in train], "heldout_videos": [e["video_id"] for e in test],
    }, indent=2))
    refine._w = None
    after = mean_map(test, gt)
    meta = json.loads(refine.PATH.read_text())
    meta.update({"heldout_map_before": before, "heldout_map_after": after})
    refine.PATH.write_text(json.dumps(meta, indent=2))
    print(f"{len(y)} runs ({pos:.1%} positive); held-out action mAP {before:.4f} -> {after:.4f}")
    if after < before:
        refine.PATH.unlink()
        print("fine-tuned head did not help on held-out videos; removed, hand-written score stays")
    return 0


if __name__ == "__main__":
    sys.exit(main())
