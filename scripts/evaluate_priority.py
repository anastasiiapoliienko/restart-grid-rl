"""Compare priority-aware PPO vs baselines and the bare 8-segment PPO on
priority-aware metrics: minutes-until-hospital-restored, minutes-until-all-
critical-restored, plus the usual trip/restore/steps numbers.

Note the bare-PPO policies live in the original (no-archetype) env and are
not evaluated here — only the priority policies see the archetype info. The
"naive" comparison is to a greedy policy and a sequential-d8 baseline running
on the priority env (they don't know about criticality; they just close in
segment order).

Usage:
    python scripts/evaluate_priority.py \
        --policies results/policy_priority_s1.zip results/policy_priority_s2.zip results/policy_priority_s3.zip \
        --n 500
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO

from restart.env_priority import PriorityRestorationEnv
from restart.baselines import GreedyPolicy, SequentialPolicy


def run_one(env: PriorityRestorationEnv, policy, seed: int, masked: bool):
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"): policy.reset()
    done = trunc = False
    peak = 0.0
    t_hosp = None; t_crit = None
    while not (done or trunc):
        if masked:
            action, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
        elif hasattr(policy, "predict"):
            try: action, _ = policy.predict(obs, deterministic=True)
            except TypeError: action, _ = policy.predict(obs)
        else: action = policy(obs)
        obs, r, done, trunc, info = env.step(int(action))
        peak = max(peak, info["trunk_amps"])
        if t_hosp is None and info["minutes_to_hospital"] is not None:
            t_hosp = info["minutes_to_hospital"]
        if t_crit is None and info["minutes_to_critical_all"] is not None:
            t_crit = info["minutes_to_critical_all"]
    return {
        "trip": info["trip"],
        "restored": env.energized.all() and not info["trip"],
        "peak": peak,
        "t_hospital_min": t_hosp,                # None if hospital never restored
        "t_critical_min": t_crit,
        "steps": env.steps_taken,
    }


def summarize(rs):
    n = len(rs)
    hosp_times = [r["t_hospital_min"] for r in rs if r["t_hospital_min"] is not None]
    crit_times = [r["t_critical_min"] for r in rs if r["t_critical_min"] is not None]
    return {
        "n": n,
        "trip%": 100 * sum(r["trip"] for r in rs) / n,
        "restore%": 100 * sum(r["restored"] for r in rs) / n,
        "hospital_done%": 100 * len(hosp_times) / n,
        "mean_t_hospital": float(np.mean(hosp_times)) if hosp_times else float("nan"),
        "mean_t_critical": float(np.mean(crit_times)) if crit_times else float("nan"),
        "mean_steps": float(np.mean([r["steps"] for r in rs])),
        "mean_peak": float(np.mean([r["peak"] for r in rs])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=500_000)
    args = ap.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.n))
    env = PriorityRestorationEnv(seed=0)

    print(f"Held-out scenarios: {args.n} (seeds {seeds[0]}…{seeds[-1]})\n")
    cols = ["policy", "trip%", "restore%", "hosp%", "min→hosp", "min→crit", "steps", "peak A"]
    widths = [28, 8, 10, 8, 10, 10, 8, 9]
    print("".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * sum(widths))

    def row(name, s):
        cells = [
            name,
            f"{s['trip%']:.1f}",
            f"{s['restore%']:.1f}",
            f"{s['hospital_done%']:.0f}",
            f"{s['mean_t_hospital']:.1f}" if s['mean_t_hospital'] == s['mean_t_hospital'] else "—",
            f"{s['mean_t_critical']:.1f}" if s['mean_t_critical'] == s['mean_t_critical'] else "—",
            f"{s['mean_steps']:.1f}",
            f"{s['mean_peak']:.0f}",
        ]
        print("".join(c.ljust(w) for c, w in zip(cells, widths)))

    # --- Baselines (no priority awareness) ---
    for name, pol, masked in [
        ("greedy (segment order)", GreedyPolicy(env.N), False),
        ("sequential d=8", SequentialPolicy(env.N, dwell_steps=8), False),
    ]:
        rs = [run_one(env, pol, s, masked) for s in seeds]
        row(name, summarize(rs))

    # --- Priority PPO ensemble ---
    for p in args.policies:
        if not Path(p).exists(): print(f"missing: {p}"); continue
        model = MaskablePPO.load(p)
        rs = [run_one(env, model, s, True) for s in seeds]
        row(f"priority {Path(p).stem}", summarize(rs))


if __name__ == "__main__":
    main()
