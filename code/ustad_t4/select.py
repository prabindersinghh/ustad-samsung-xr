"""Highlight selection under the Track 4 rules.

Rules from the student kit: highlights are sorted, non-overlapping, each segment
should be >= 4.0 s (shorter ones are ignored by the scorer), at least one must be
>= 4.0 s, and the total should stay within 15% of the duration.

Two selectors share those rules:

* `by_importance` - pick the highest-scoring windows (used when we only have a
  per-second importance curve, e.g. FPVSum, scored by frame-level F1).
* `by_coverage` - greedily pick the window that adds the most *new* verb-noun
  pairs per second, then fill leftover budget by importance. Action coverage is
  40% of the score and counts distinct pairs, so five clips of the same action
  are worth one.
"""
from __future__ import annotations

from dataclasses import dataclass

BUDGET = 0.15
MIN_SEG = 4.0


@dataclass
class Window:
    start: float
    end: float
    score: float
    pairs: frozenset = frozenset()
    conf: float = 0.0      # mean top-pair probability inside the window
    motion: float = 0.0    # mean motion inside the window
    peak: float = 0.0      # highest single pair probability inside the window

    @property
    def length(self) -> float:
        return self.end - self.start


def policy_features(w: Window, covered: set, used_frac: float) -> list[float]:
    """State-action features for the learned selection policy.

    Everything here is computable at inference time from model outputs alone;
    ground truth is only ever used to compute the *reward* during training.
    """
    new = len(w.pairs - covered)
    return [
        new / max(w.length, 1e-6),        # new confident pairs per second
        float(new > 0),                   # adds anything new at all
        w.conf,                           # how sure the action model is here
        w.peak,                           # sharpest single detection
        w.motion,                         # visual activity
        len(w.pairs) / max(w.length, 1e-6),
        used_frac,                        # how much budget is already spent
        1.0,                              # bias
    ]


def by_policy(windows: list[Window], duration: float, weights: list[float], budget: float = BUDGET) -> list[Window]:
    """Pick segments with the RL-trained linear policy (greedy at inference)."""
    cap = budget * duration
    pool = [w for w in windows if w.length >= MIN_SEG]
    chosen: list[Window] = []
    covered: set = set()
    used = 0.0
    while True:
        feasible = [w for w in pool if used + w.length <= cap + 1e-6 and not _overlaps(w, chosen)]
        if not feasible:
            break
        frac = used / cap if cap > 0 else 1.0
        best = max(feasible, key=lambda w: sum(a * b for a, b in zip(weights, policy_features(w, covered, frac))))
        chosen.append(best)
        covered |= best.pairs
        used += best.length
    return finalize(chosen, duration, pool)


def _overlaps(w: Window, chosen: list[Window]) -> bool:
    return any(w.start < c.end and c.start < w.end for c in chosen)


def candidate_windows(importance: list[float], duration: float, seg: float = 5.0, stride: float = 1.0) -> list[Window]:
    """Sliding windows over a per-second importance curve."""
    out = []
    t = 0.0
    while t + seg <= duration + 1e-6:
        lo, hi = int(t), max(int(t) + 1, int(t + seg))
        vals = importance[lo:hi] or [0.0]
        out.append(Window(round(t, 2), round(min(t + seg, duration), 2), sum(vals) / len(vals)))
        t += stride
    if not out:  # video shorter than one window
        out.append(Window(0.0, round(duration, 2), 1.0))
    return out


def by_importance(windows: list[Window], duration: float, budget: float = BUDGET) -> list[Window]:
    cap = budget * duration
    chosen: list[Window] = []
    used = 0.0
    for w in sorted(windows, key=lambda w: w.score, reverse=True):
        if used + w.length > cap + 1e-6:
            continue
        if _overlaps(w, chosen):
            continue
        chosen.append(w)
        used += w.length
    return finalize(chosen, duration, windows)


def by_coverage(windows: list[Window], duration: float, budget: float = BUDGET) -> list[Window]:
    cap = budget * duration
    chosen: list[Window] = []
    covered: set = set()
    used = 0.0
    pool = [w for w in windows if w.length >= MIN_SEG]
    while True:
        best, best_gain = None, 0.0
        for w in pool:
            if used + w.length > cap + 1e-6 or _overlaps(w, chosen):
                continue
            new = len(w.pairs - covered)
            if new == 0:
                continue
            gain = new / w.length + 1e-3 * w.score
            if gain > best_gain:
                best, best_gain = w, gain
        if best is None:
            break
        chosen.append(best)
        covered |= best.pairs
        used += best.length
    # Spend what is left on the most important footage.
    for w in sorted(pool, key=lambda w: w.score, reverse=True):
        if used + w.length > cap + 1e-6 or _overlaps(w, chosen):
            continue
        chosen.append(w)
        used += w.length
    return finalize(chosen, duration, pool)


def finalize(chosen: list[Window], duration: float, pool: list[Window] | None = None) -> list[Window]:
    """Sort, merge touching segments, and guarantee one valid >= 4 s segment.

    When the 15% budget is shorter than 4 s (videos under ~27 s), nothing fits,
    yet the kit requires one segment >= 4 s. We then take the single best
    candidate - most distinct pairs, then highest score - trimmed to exactly 4 s,
    which keeps the overshoot as small as the rules allow.
    """
    chosen = sorted(chosen, key=lambda w: w.start)
    merged: list[Window] = []
    for w in chosen:
        if merged and w.start <= merged[-1].end + 1e-6:
            last = merged[-1]
            merged[-1] = Window(last.start, max(last.end, w.end), max(last.score, w.score), last.pairs | w.pairs)
        else:
            merged.append(w)
    merged = [w for w in merged if w.length >= MIN_SEG]
    if not merged:
        if pool:
            best = max(pool, key=lambda w: (len(w.pairs), w.score))
            start = min(best.start, max(0.0, duration - MIN_SEG))
        else:
            start = max(0.0, duration / 2 - MIN_SEG / 2)
        merged = [Window(round(start, 2), round(min(duration, start + MIN_SEG), 2), 0.0)]
    return merged
