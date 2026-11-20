"""Action scoring head, fine-tuned on EPIC-100 train (scripts/train_refine.py).

A logistic head over run features from postprocess.run_features. It learns, from
train-split ground truth, how likely a proposed (verb, noun) run is to be a
real action - which is exactly the ranking Action mAP rewards. When
weights/refine.json is absent the hand-written score is used instead.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

PATH = Path(os.environ.get("USTAD_REFINE", Path(__file__).resolve().parents[2] / "weights" / "refine.json"))
_w = None


def available() -> bool:
    return PATH.exists()


def score(feats: list[float]) -> float:
    global _w
    if _w is None:
        _w = json.loads(PATH.read_text())["weights"]
    z = sum(a * b for a, b in zip(_w, feats))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
