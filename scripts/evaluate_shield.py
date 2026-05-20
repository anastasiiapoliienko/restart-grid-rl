"""Evaluate PPO with vs. without the SafetyShield, alongside baselines.

Usage:
    python scripts/evaluate_shield.py \
        --policies results/policy_s1.zip results/policy_s2.zip results/policy_s3.zip \
        --n 1000 --threshold-factor 0.95
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
from restart.shield import SafetyShield
from restart.eval import run_policy, summarize, RunResult


def run_shielded_once(env: RestorationEnv, shield: SafetyShield, seed: int) -> RunResult:
    obs, _ = env.reset(seed=seed)
    shield.reset()
    done = trunc = False
    total_r = 0.0
    peak_amps = 0.0
    steps_to_99 = env.max_steps
    while not (done or trunc):
        action, _ = shield.predict(obs, env=env, deterministic=True)
        action = int(action)
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        peak_amps = max(peak_amps, info["trunk_amps"])
        if info["n_energized"] >= int(0.99 * env.N) and steps_to_99 == env.max_steps:
            steps_to_99 = env.steps_taken
    return RunResult(
        seed=seed,
        trip=info["trip"],
        restored=(env.energized.all() and not info["trip"]),
        steps_to_99=steps_to_99,
        final_frac=float(env.energized.mean()),
        total_reward=total_r,
        peak_trunk_amps=peak_amps,
    )


def fmt_pm(values, fmt="{:.1f}"):
    arr = np.array(values)
    return f"{fmt.format(arr.mean())} ± {fmt.format(arr.std())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=100_000)
    ap.add_argument("--threshold-factor", type=float, default=0.95)
    args = ap.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.n))
    env = RestorationEnv(seed=0)

    print(f"Held-out scenarios: {args.n} (seeds {seeds[0]}…{seeds[-1]})")
    print(f"Shield threshold: {args.threshold_factor * env.trip_amps:.1f} A "
          f"({args.threshold_factor*100:.0f}% of trip {env.trip_amps:.0f} A)\n")

    header = f"{'policy':<28}{'trip%':>10}{'restore%':>12}{'steps to 99%':>16}{'peak A':>12}"
    print(header)
    print("-" * len(header))

    # --- Baselines ---
    for name, pol, masked in [
        ("greedy",          GreedyPolicy(env.N),                       False),
        ("sequential d=8",  SequentialPolicy(env.N, dwell_steps=8),    False),
        ("sequential d=12", SequentialPolicy(env.N, dwell_steps=12),   False),
    ]:
        rs = run_policy(env, pol, seeds, use_action_masks=masked)
        s = summarize(rs)
        print(f"{name:<28}{100*s['trip_rate']:>9.1f}%{100*s['restore_rate']:>11.1f}%"
              f"{s['mean_steps_to_99']:>16.1f}{s['mean_peak_amps']:>12.1f}")

    # --- PPO bare + shielded for each seed ---
    bare_stats, shield_stats = [], []
    override_rates = []
    for path in args.policies:
        if not Path(path).exists():
            print(f"missing: {path}"); continue
        model = MaskablePPO.load(path)
        rs_bare = run_policy(env, model, seeds, use_action_masks=True)
        sb = summarize(rs_bare); bare_stats.append(sb)
        print(f"{'ppo '+Path(path).stem+' (bare)':<28}{100*sb['trip_rate']:>9.1f}%"
              f"{100*sb['restore_rate']:>11.1f}%{sb['mean_steps_to_99']:>16.1f}{sb['mean_peak_amps']:>12.1f}")

        shield = SafetyShield(model, env, threshold_factor=args.threshold_factor, masked=True)
        rs_shield = [run_shielded_once(env, shield, s) for s in seeds]
        ss = summarize(rs_shield); shield_stats.append(ss)
        # Note shield stats persist after each run; for an aggregate over all
        # seeds we'd need to track per-episode counts. Just report final stats.
        print(f"{'ppo '+Path(path).stem+' + shield':<28}{100*ss['trip_rate']:>9.1f}%"
              f"{100*ss['restore_rate']:>11.1f}%{ss['mean_steps_to_99']:>16.1f}{ss['mean_peak_amps']:>12.1f}")

    if len(shield_stats) >= 2:
        print()
        print(f"{'PPO BARE (mean ± std)':<28}"
              f"{fmt_pm([s['trip_rate'] for s in bare_stats], '{:.2%}').replace('%',''):>10}"
              f"{fmt_pm([s['restore_rate'] for s in bare_stats], '{:.2%}').replace('%',''):>12}"
              f"{fmt_pm([s['mean_steps_to_99'] for s in bare_stats]):>16}"
              f"{fmt_pm([s['mean_peak_amps'] for s in bare_stats]):>12}")
        print(f"{'PPO + SHIELD (mean ± std)':<28}"
              f"{fmt_pm([s['trip_rate'] for s in shield_stats], '{:.2%}').replace('%',''):>10}"
              f"{fmt_pm([s['restore_rate'] for s in shield_stats], '{:.2%}').replace('%',''):>12}"
              f"{fmt_pm([s['mean_steps_to_99'] for s in shield_stats]):>16}"
              f"{fmt_pm([s['mean_peak_amps'] for s in shield_stats]):>12}")


if __name__ == "__main__":
    main()
