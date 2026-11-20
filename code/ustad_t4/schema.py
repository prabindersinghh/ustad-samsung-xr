"""Track 4 output schema: build and check one <video_id>.json.

The checks mirror the student kit README (section 3 and 4). validate.py was
listed in the kit but not shipped, so this is our own implementation of the
same rules; replace with the official one when it arrives.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

BUDGET = 0.15
MIN_SEG = 4.0
KEYS = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check(doc: dict, manifest_duration: float | None = None, vocab: dict | None = None) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for k in KEYS:
        if k not in doc:
            errors.append(f"missing key {k}")
    if errors:
        return errors, warnings
    if not isinstance(doc["team_id"], str) or not doc["team_id"]:
        errors.append("team_id must be a non-empty string")
    if not isinstance(doc["video_id"], str) or not doc["video_id"]:
        errors.append("video_id must be a non-empty string")
    dur = doc["duration_sec"]
    if not _num(dur) or dur <= 0:
        errors.append("duration_sec must be a positive number")
        return errors, warnings
    if manifest_duration is not None and abs(dur - manifest_duration) > 0.5:
        errors.append(f"duration_sec {dur} differs from manifest {manifest_duration} by more than 0.5 s")

    lab = doc["label"]
    if not isinstance(lab, dict) or not isinstance(lab.get("name"), str) or not _num(lab.get("score")) or not 0 <= lab["score"] <= 1:
        errors.append("label must be {name: str, score: number in [0,1]}")
    elif vocab and lab["name"] not in vocab["labels"]:
        errors.append(f"label {lab['name']!r} not in supported labels")

    acts = doc["actions"]
    if not isinstance(acts, list) or len(acts) > 2000:
        errors.append("actions must be a list of 0-2000 items")
    else:
        for i, a in enumerate(acts):
            seg = a.get("segment") if isinstance(a, dict) else None
            if not (isinstance(seg, list) and len(seg) == 2 and all(_num(v) for v in seg) and 0 <= seg[0] < seg[1] <= dur + 0.5):
                errors.append(f"actions[{i}].segment invalid")
                continue
            if not isinstance(a.get("verb"), str) or not isinstance(a.get("noun"), str):
                errors.append(f"actions[{i}] verb/noun must be strings")
            elif vocab and (a["verb"] not in vocab["verbs"] or a["noun"] not in vocab["nouns"]):
                errors.append(f"actions[{i}] verb/noun outside the supported vocabulary")
            if not _num(a.get("score")) or not 0 <= a["score"] <= 1:
                errors.append(f"actions[{i}].score must be in [0,1]")

    hls = doc["highlights"]
    if not isinstance(hls, list) or not hls:
        errors.append("highlights must be a non-empty list")
        return errors, warnings
    prev_end, total, long_one = -1.0, 0.0, False
    for i, h in enumerate(hls):
        seg = h.get("segment") if isinstance(h, dict) else None
        if not (isinstance(seg, list) and len(seg) == 2 and all(_num(v) for v in seg) and 0 <= seg[0] < seg[1] <= dur + 0.5):
            errors.append(f"highlights[{i}].segment invalid")
            continue
        if seg[0] < prev_end:
            errors.append(f"highlights[{i}] overlaps or is out of order")
        prev_end = seg[1]
        length = seg[1] - seg[0]
        total += length
        if length >= MIN_SEG:
            long_one = True
        else:
            warnings.append(f"highlights[{i}] is {length:.2f} s (< 4 s, ignored in scoring)")
    if not long_one:
        errors.append("at least one highlight must be >= 4.0 s")
    if total > BUDGET * dur + 1e-6:
        warnings.append(f"highlights total {total:.1f} s exceeds 15% budget ({BUDGET * dur:.1f} s)")
    return errors, warnings


def write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
