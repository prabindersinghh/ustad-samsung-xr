"""Experiment: highlight candidates whose pairs must be >=50% inside the window
(the organisers' capture rule), plus candidates fitted to predicted action runs.
EPIC-100 train episodes only."""
import random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "scripts"))
from eval_train import load_episodes, load_gt, VOCAB
from ustad_t4 import postprocess as pp, scoring, select
from ustad_t4.select import Window

def cands_runs(preds, dur, motion, conf, maxlen, pad, sliding):
    runs = [r for r in pp.action_runs(preds) if r["best"] >= conf]
    def inside_pairs(lo, hi):
        out = set()
        for r in runs:
            L = r["end"] - r["start"]
            ov = max(0.0, min(hi, r["end"]) - max(lo, r["start"]))
            if L <= 0 or ov / L >= 0.5:
                out.add((r["v"], r["n"]))
        return frozenset(out)
    base = pp.highlight_candidates(preds, dur, motion)
    score = {round(w.start): w.score for w in base}
    out = []
    if sliding:
        for w in base:
            out.append(Window(w.start, w.end, w.score, inside_pairs(w.start, w.end), conf=w.conf, motion=w.motion, peak=w.peak))
    for r in runs:
        lo, hi = r["start"] - pad, r["end"] + pad
        if hi - lo < pp.HL_LEN:
            c = (lo + hi) / 2; lo, hi = c - pp.HL_LEN / 2, c + pp.HL_LEN / 2
        if hi - lo > maxlen:
            continue
        lo = max(0.0, lo); hi = min(dur, hi)
        if hi - lo < 4.0:
            continue
        out.append(Window(round(lo, 2), round(hi, 2), score.get(round(lo), 0.0) + 0.01 * r["best"], inside_pairs(lo, hi)))
    return out

def evaluate(eps, gt, **kw):
    cov = []
    for e in eps:
        if kw.get("baseline"):
            c = pp.highlight_candidates(e["windows"], e["duration"], e["motion"])
        else:
            c = cands_runs(e["windows"], e["duration"], e["motion"], **kw)
        hl = select.by_coverage(c, e["duration"])
        cov.append(scoring.coverage(gt.get(e["video_id"], []), [[w.start, w.end] for w in hl], e["duration"]))
    return sum(cov) / len(cov)

eps = load_episodes(ROOT / "data" / "train_preds"); gt = load_gt()
random.Random(7).shuffle(eps); n = max(3, len(eps) // 5); te, tr = eps[:n], eps[n:]
for C in (0.12, 0.2):
    pp.CONF_PAIR = C
    print(f"CONF {C} baseline: tr {evaluate(tr, gt, baseline=True):.3f} te {evaluate(te, gt, baseline=True):.3f} all {evaluate(eps, gt, baseline=True):.3f}")
    for conf in (C,):
        for maxlen in (6, 8, 12):
            for pad in (0.0, 0.5):
                for sliding in (True, False):
                    kw = dict(conf=conf, maxlen=maxlen, pad=pad, sliding=sliding)
                    print(f"  {kw}: tr {evaluate(tr, gt, **kw):.3f} te {evaluate(te, gt, **kw):.3f} all {evaluate(eps, gt, **kw):.3f}")
