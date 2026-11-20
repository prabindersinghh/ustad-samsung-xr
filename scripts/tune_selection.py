"""Grid-search the selection constants on EPIC-100 *train* episodes only.

Tunes on 80% of the train episodes and reports the held-out 20%, so the chosen
values are checked on videos they were not picked on. Never touches the
scored manifest videos.

    python scripts/tune_selection.py
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_train import evaluate, load_episodes, load_gt  # noqa: E402

GRID = {
    "CONF_PAIR": [0.05, 0.08, 0.12, 0.2],
    "HL_LEN": [4.2, 5.0, 6.0, 8.0],
    "PAIR_TOPK": [2, 3, 4],
}


def main() -> int:
    eps = load_episodes(ROOT / "data" / "train_preds")
    gt = load_gt()
    random.Random(7).shuffle(eps)
    n_test = max(3, len(eps) // 5)
    test, train = eps[:n_test], eps[n_test:]
    base_tr, base_te = evaluate(train, gt), evaluate(test, gt)
    print(f"current: train cov {base_tr['coverage']:.3f} map {base_tr['action_map']:.3f} | "
          f"held-out cov {base_te['coverage']:.3f} map {base_te['action_map']:.3f}")
    rows = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        r = evaluate(train, gt, p)
        rows.append((r["coverage"] * 0.4 + r["action_map"] * 0.25, p, r))
    rows.sort(key=lambda x: -x[0])
    for s, p, r in rows[:5]:
        te = evaluate(test, gt, p)
        print(f"{p}: train cov {r['coverage']:.3f} map {r['action_map']:.3f} | "
              f"held-out cov {te['coverage']:.3f} map {te['action_map']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
