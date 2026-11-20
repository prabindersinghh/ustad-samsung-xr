"""Train the highlight-selection policy with reinforcement learning (REINFORCE).

Environment (one episode = one EPIC-100 *train* video):
    state   segments already picked, verb-noun pairs already covered, budget used
    action  pick the next highlight segment among those that still fit
    reward  at the end of the episode: Action-coverage of the ground-truth
            (verb, noun) pairs, with the 15% soft budget penalty - i.e. the
            Track 4 coverage metric itself (40% of the score)

Policy: a linear softmax over the state-action features in select.policy_features
(no ground truth in the features; GT appears only in the reward). Trained with
REINFORCE and a moving-average baseline; evaluated greedily on held-out train
videos against the hand-written greedy rule.

    python scripts/train_policy.py
writes weights/policy.json, which run.py uses when present.
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
from ustad_t4 import postprocess, scoring, select  # noqa: E402
from eval_train import load_episodes, load_gt  # noqa: E402

EPOCHS = 60
LR = 0.05
TEMP = 0.5
SEED = 7


def rollout(cands, duration, w, rng, greedy=False):
    cap = select.BUDGET * duration
    pool = [c for c in cands if c.length >= select.MIN_SEG]
    chosen, covered, used, traj = [], set(), 0.0, []
    while True:
        feas = [c for c in pool if used + c.length <= cap + 1e-6 and not select._overlaps(c, chosen)]
        if not feas:
            break
        frac = used / cap if cap > 0 else 1.0
        phi = np.array([select.policy_features(c, covered, frac) for c in feas])
        logits = phi @ w / TEMP
        if greedy:
            k = int(np.argmax(logits))
        else:
            p = np.exp(logits - logits.max())
            p /= p.sum()
            k = int(rng.choice(len(feas), p=p))
            traj.append((phi, p, k))
        chosen.append(feas[k])
        covered |= feas[k].pairs
        used += feas[k].length
    return select.finalize(chosen, duration, pool), traj


def episode_reward(hl, ep, gt):
    return scoring.coverage(gt.get(ep["video_id"], []), [[h.start, h.end] for h in hl], ep["duration"])


def main() -> int:
    eps = load_episodes(ROOT / "data" / "train_preds")
    gt = load_gt()
    if len(eps) < 6:
        print(f"need >= 6 train episodes, have {len(eps)}")
        return 1
    rng = np.random.default_rng(SEED)
    random.Random(SEED).shuffle(eps)
    n_test = max(3, len(eps) // 5)
    test, train = eps[:n_test], eps[n_test:]
    cands = {e["video_id"]: postprocess.highlight_candidates(e["windows"], e["duration"], e["motion"]) for e in eps}

    def heuristic(es):
        return float(np.mean([episode_reward(select.by_coverage(cands[e["video_id"]], e["duration"]), e, gt) for e in es]))

    def greedy_policy(es, w):
        return float(np.mean([episode_reward(rollout(cands[e["video_id"]], e["duration"], w, rng, greedy=True)[0], e, gt) for e in es]))

    # Start from the hand-written rule's preference: new pairs per second.
    dim = len(select.policy_features(select.Window(0, 4, 0), set(), 0.0))
    w = np.zeros(dim)
    w[0] = 1.0
    baseline = None
    history = []
    for epoch in range(EPOCHS):
        rewards = []
        for ep in train:
            hl, traj = rollout(cands[ep["video_id"]], ep["duration"], w, rng)
            r = episode_reward(hl, ep, gt)
            rewards.append(r)
            baseline = r if baseline is None else 0.9 * baseline + 0.1 * r
            adv = r - baseline
            for phi, p, k in traj:
                w += LR * adv * (phi[k] - p @ phi) / TEMP
        history.append(float(np.mean(rewards)))
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  mean train reward {history[-1]:.3f}")

    result = {
        "weights": w.tolist(),
        "features": ["new_pairs_per_s", "adds_new", "conf", "peak", "motion", "pairs_per_s", "budget_used", "bias"],
        "train_videos": [e["video_id"] for e in train],
        "heldout_videos": [e["video_id"] for e in test],
        "heldout_coverage_policy": greedy_policy(test, w),
        "heldout_coverage_greedy_rule": heuristic(test),
        "reward_curve": history,
        "algo": "REINFORCE, linear softmax policy, moving-average baseline",
    }
    print(f"held-out coverage: policy {result['heldout_coverage_policy']:.3f} "
          f"vs greedy rule {result['heldout_coverage_greedy_rule']:.3f}")
    (ROOT / "weights" / "policy.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
