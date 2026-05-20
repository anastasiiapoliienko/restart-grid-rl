"""Run baselines vs. a trained PPO policy over the same held-out seeds.

Usage:
    python scripts/evaluate.py --policy results/policy.zip --n 50
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import MaskablePPO

from restart.env import RestorationEnv
from restart.baselines import GreedyPolicy, SequentialPolicy
from restart.eval import run_policy, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default=None,
                    help="Path to a trained PPO .zip (optional)")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=10_000)
    args = ap.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.n))
    env = RestorationEnv(seed=0)

    policies = [
        ("greedy", GreedyPolicy(n_segments=env.N), False),
        ("sequential-dwell4", SequentialPolicy(n_segments=env.N, dwell_steps=4), False),
        ("sequential-dwell8", SequentialPolicy(n_segments=env.N, dwell_steps=8), False),
    ]
    if args.policy and Path(args.policy).exists():
        model = MaskablePPO.load(args.policy)
        policies.append(("ppo-masked", model, True))

    rows = []
    for name, pol, masked in policies:
        results = run_policy(env, pol, seeds, use_action_masks=masked)
        s = summarize(results)
        rows.append((name, s))

    # Print pretty table
    cols = ["policy", "trip%", "restore%", "mean_steps_99", "mean_peak_A", "mean_reward"]
    widths = [22, 8, 10, 14, 12, 12]
    print("".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * sum(widths))
    for name, s in rows:
        cells = [
            name,
            f"{100*s['trip_rate']:.1f}",
            f"{100*s['restore_rate']:.1f}",
            f"{s['mean_steps_to_99']:.1f}",
            f"{s['mean_peak_amps']:.1f}",
            f"{s['mean_reward']:.2f}",
        ]
        print("".join(c.ljust(w) for c, w in zip(cells, widths)))


if __name__ == "__main__":
    main()
