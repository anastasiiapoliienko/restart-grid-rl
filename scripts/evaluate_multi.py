"""Evaluate an ensemble of trained PPO policies vs baselines on the same
held-out scenario seeds, with per-policy spread reported across the ensemble.

Usage:
    python scripts/evaluate_multi.py \
        --policies results/policy_s1.zip results/policy_s2.zip results/policy_s3.zip \
        --n 1000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO

from restart.env import RestorationEnv
from restart.baselines import GreedyPolicy, SequentialPolicy
from restart.eval import run_policy, summarize


def fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def fmt_pm(values, fmt="{:.1f}"):
    """Mean ± std formatted string."""
    arr = np.array(values)
    return f"{fmt.format(arr.mean())} ± {fmt.format(arr.std())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, nargs="+", required=True,
                    help="Paths to one or more trained MaskablePPO .zip files")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=100_000)
    ap.add_argument("--baselines-only", action="store_true")
    args = ap.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.n))
    env = RestorationEnv(seed=0)

    print(f"Held-out scenarios: {args.n} (seeds {seeds[0]}…{seeds[-1]})")
    print(f"Ensemble size: {len(args.policies)} policy(ies)\n")

    # ---- Baselines ----
    baselines = [
        ("greedy",          GreedyPolicy(env.N),                       False),
        ("seq-dwell4",      SequentialPolicy(env.N, dwell_steps=4),    False),
        ("seq-dwell8",      SequentialPolicy(env.N, dwell_steps=8),    False),
        ("seq-dwell12",     SequentialPolicy(env.N, dwell_steps=12),   False),
    ]

    header = f"{'policy':<24}{'trip%':>10}{'restore%':>12}{'steps to 99%':>16}{'peak A':>12}{'reward':>12}"
    print(header)
    print("-" * len(header))

    for name, pol, masked in baselines:
        results = run_policy(env, pol, seeds, use_action_masks=masked)
        s = summarize(results)
        print(f"{name:<24}{fmt_pct(s['trip_rate']):>10}{fmt_pct(s['restore_rate']):>12}"
              f"{s['mean_steps_to_99']:>16.1f}{s['mean_peak_amps']:>12.1f}{s['mean_reward']:>12.2f}")

    if args.baselines_only:
        return

    # ---- PPO ensemble: each policy on the same seeds ----
    ppo_stats = []
    for path in args.policies:
        if not Path(path).exists():
            print(f"\nSkipping missing: {path}")
            continue
        model = MaskablePPO.load(path)
        results = run_policy(env, model, seeds, use_action_masks=True)
        s = summarize(results)
        ppo_stats.append((path, s))
        print(f"{'ppo '+Path(path).stem:<24}{fmt_pct(s['trip_rate']):>10}{fmt_pct(s['restore_rate']):>12}"
              f"{s['mean_steps_to_99']:>16.1f}{s['mean_peak_amps']:>12.1f}{s['mean_reward']:>12.2f}")

    if len(ppo_stats) >= 2:
        print()
        print(f"{'PPO ENSEMBLE (mean ± std)':<24}", end="")
        trip_vals    = [s['trip_rate']         for _, s in ppo_stats]
        rest_vals    = [s['restore_rate']      for _, s in ppo_stats]
        steps_vals   = [s['mean_steps_to_99']  for _, s in ppo_stats]
        peak_vals    = [s['mean_peak_amps']    for _, s in ppo_stats]
        rew_vals     = [s['mean_reward']       for _, s in ppo_stats]
        print(f"{fmt_pm(trip_vals,  fmt='{:.2%}'):>10}".replace("%", "").rjust(10)
              + f"{fmt_pm(rest_vals, fmt='{:.2%}').replace('%','').rjust(12)}"
              + f"{fmt_pm(steps_vals):>16}"
              + f"{fmt_pm(peak_vals):>12}"
              + f"{fmt_pm(rew_vals, fmt='{:.2f}'):>12}")


if __name__ == "__main__":
    main()
