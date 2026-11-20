"""Our implementation of the Track 4 metrics, from the kit's one-line definitions.

The official scorer is not public, so these follow the README as closely as its
wording allows. They are used only on EPIC-100 *train* videos (never the scored
manifest videos), to tune and to train the selection policy.

coverage   share of distinct ground-truth (verb, noun) pairs whose segment
           overlaps a selected highlight, times a soft penalty when highlights
           exceed 15% of the duration.
action_map mean over tIoU 0.1..0.5 of per-pair average precision for the
           predicted action segments (ActivityNet-style detection AP).
"""
from __future__ import annotations

from collections import defaultdict

TIOUS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def tiou(a, b) -> float:
    inter = _overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def coverage(gt: list[dict], highlights: list[list[float]], duration: float, budget: float = 0.15) -> float:
    pairs = {(g["verb"], g["noun"]) for g in gt}
    if not pairs:
        return 0.0
    hl = [h for h in highlights if h[1] - h[0] >= 4.0]
    hit = {(g["verb"], g["noun"]) for g in gt if any(_overlap(g["segment"], h) > 0 for h in hl)}
    total = sum(h[1] - h[0] for h in hl)
    penalty = min(1.0, budget * duration / total) if total > 0 else 0.0
    return len(hit) / len(pairs) * penalty


def _ap(scores_hits: list[tuple[float, bool]], n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    scores_hits.sort(key=lambda x: -x[0])
    tp = fp = 0
    ap, prev_recall = 0.0, 0.0
    for _, hit in scores_hits:
        if hit:
            tp += 1
        else:
            fp += 1
        recall = tp / n_gt
        precision = tp / (tp + fp)
        if hit:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
    return ap


def action_map(gt: list[dict], preds: list[dict]) -> float:
    gt_by = defaultdict(list)
    for g in gt:
        gt_by[(g["verb"], g["noun"])].append(g["segment"])
    pred_by = defaultdict(list)
    for p in preds:
        pred_by[(p["verb"], p["noun"])].append((p["score"], p["segment"]))
    maps = []
    for th in TIOUS:
        aps = []
        for key, gts in gt_by.items():
            used = [False] * len(gts)
            hits = []
            for score, seg in sorted(pred_by.get(key, []), key=lambda x: -x[0]):
                best, bi = 0.0, -1
                for i, g in enumerate(gts):
                    if not used[i]:
                        t = tiou(seg, g)
                        if t > best:
                            best, bi = t, i
                if best >= th:
                    used[bi] = True
                    hits.append((score, True))
                else:
                    hits.append((score, False))
            aps.append(_ap(hits, len(gts)))
        maps.append(sum(aps) / len(aps) if aps else 0.0)
    return sum(maps) / len(maps)
