"""Window predictions -> scored outputs.

actions (Action mAP, 25%)
    Every window proposes its top verb x top noun pairs, scored p(verb) * p(noun).
    Consecutive windows proposing the same pair are merged into one segment.
    Proposing several pairs per moment raises recall at tIoU 0.1-0.5; the
    ranking by score keeps precision, which is what mAP rewards.

highlights (Action coverage 40%, Highlight F1 25%)
    Candidate segments of HL_LEN seconds slide over the video. Each carries the
    set of verb-noun pairs the model is confident about inside it, and an
    importance score (action confidence blended with visual motion). The
    coverage selector then buys distinct pairs per second of budget.
"""
from __future__ import annotations

from collections import defaultdict

from .select import Window

PAIR_TOPK = 3          # verbs x nouns considered per window
MIN_PAIR_SCORE = 0.02  # ignore pairs below this joint probability
MAX_ACTIONS = 2000     # kit limit
CONF_PAIR = 0.25       # joint prob at which a pair counts as "seen" for coverage
RUN_MAXLEN = 6.0       # longest run-fitted highlight candidate (s)
HL_LEN = 4.2           # highlight segment length (kit ignores segments < 4.0 s)
HL_STRIDE = 1.0


def window_pairs(w, k: int | None = None) -> list[tuple[int, int, float]]:
    k = PAIR_TOPK if k is None else k
    out = []
    for vi, pv in w.verbs[:k]:
        for ni, pn in w.nouns[:k]:
            s = pv * pn
            if s >= MIN_PAIR_SCORE:
                out.append((vi, ni, s))
    return out


def action_runs(preds) -> list[dict]:
    """Merge per-window pair proposals into runs, with features for scoring.

    A run is one (verb, noun) pair proposed by consecutive windows. Its features
    (all from model outputs) feed either the hand-written score or the scoring
    head fine-tuned on EPIC-100 train (see refine.py).
    """
    open_runs: dict[tuple[int, int], dict] = {}
    done: list[dict] = []
    for w in preds:
        pairs = window_pairs(w)
        top = max(pairs, key=lambda x: x[2])[:2] if pairs else None
        present = {(v, n): s for v, n, s in pairs}
        for key in list(open_runs):
            if key not in present:
                done.append(open_runs.pop(key))
        for key, s in present.items():
            r = open_runs.get(key)
            if r:
                r["end"] = w.end
                r["best"] = max(r["best"], s)
                r["total"] += s
                r["count"] += 1
                r["top1"] = r["top1"] or key == top
            else:
                open_runs[key] = {"v": key[0], "n": key[1], "start": w.start, "end": w.end,
                                  "best": s, "total": s, "count": 1, "top1": key == top}
    done.extend(open_runs.values())
    return done


def run_features(r: dict) -> list[float]:
    import math
    mean = r["total"] / r["count"]
    return [r["best"], mean, math.log1p(r["count"]), float(r["top1"]), r["best"] * float(r["top1"]), 1.0]


def build_actions(preds, vocab: dict) -> list[dict]:
    """Merge per-window pair proposals into scored action segments."""
    from . import refine

    actions = []
    for r in action_runs(preds):
        if refine.available():
            score = refine.score(run_features(r))
        else:
            score = 0.7 * r["best"] + 0.3 * (r["total"] / r["count"])
        actions.append({
            "segment": [round(r["start"], 2), round(r["end"], 2)],
            "verb": vocab["verbs"][r["v"]],
            "noun": vocab["nouns"][r["n"]],
            "score": round(min(1.0, max(0.0, score)), 4),
        })
    actions.sort(key=lambda a: a["score"], reverse=True)
    return actions[:MAX_ACTIONS]


def highlight_candidates(preds, duration: float, motion: list[float]) -> list[Window]:
    """Highlight candidates and the confident pairs each one captures.

    The organisers count a ground-truth action as captured when at least 50% of
    its duration lies inside the highlights, so a pair belongs to a candidate
    only when at least half of its predicted run is inside it. Candidates are
    sliding HL_LEN-second windows plus one window fitted to each confident run
    (widened to HL_LEN when shorter, skipped when longer than RUN_MAXLEN).
    """
    if not preds:
        return []
    per_sec_conf = defaultdict(float)
    for w in preds:
        best = max((s for _, _, s in window_pairs(w, 1)), default=0.0)
        for t in range(int(w.start), int(w.end) + 1):
            per_sec_conf[t] = max(per_sec_conf[t], best)
    runs = [r for r in action_runs(preds) if r["best"] >= CONF_PAIR]

    def captured(lo: float, hi: float) -> frozenset:
        out = set()
        for r in runs:
            length = r["end"] - r["start"]
            inside = max(0.0, min(hi, r["end"]) - max(lo, r["start"]))
            if length <= 0 or inside / length >= 0.5:
                out.add((r["v"], r["n"]))
        return frozenset(out)

    def stats(lo: float, hi: float):
        secs = range(int(lo), int(hi) + 1)
        conf = sum(per_sec_conf[s] for s in secs) / len(secs)
        mot = sum(motion[s] for s in secs if s < len(motion)) / max(1, len(secs))
        return conf, mot

    cands = []
    score_at = {}
    t = 0.0
    while t + HL_LEN <= duration + 1e-6:
        lo, hi = t, t + HL_LEN
        peak = max((s for w in preds if lo <= (w.start + w.end) / 2 <= hi for _, _, s in window_pairs(w)), default=0.0)
        conf, mot = stats(lo, hi)
        score = 0.7 * conf + 0.3 * mot
        score_at[round(lo)] = score
        cands.append(Window(round(lo, 2), round(min(hi, duration), 2), score, captured(lo, hi),
                            conf=conf, motion=mot, peak=peak))
        t += HL_STRIDE
    for r in runs:
        lo, hi = r["start"], r["end"]
        if hi - lo < HL_LEN:
            mid = (lo + hi) / 2
            lo, hi = mid - HL_LEN / 2, mid + HL_LEN / 2
        if hi - lo > RUN_MAXLEN:
            continue
        lo, hi = max(0.0, lo), min(duration, hi)
        if hi - lo < 4.0:
            continue
        conf, mot = stats(lo, hi)
        cands.append(Window(round(lo, 2), round(hi, 2), score_at.get(round(lo), 0.0) + 0.01 * r["best"],
                            captured(lo, hi), conf=conf, motion=mot, peak=r["best"]))
    return cands
