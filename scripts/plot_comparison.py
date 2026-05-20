"""Generate a side-by-side comparison plot of the trained PPO ensemble against
the fixed-dwell baselines on N held-out scenarios.

Produces results/comparison.png with three panels:
  1. Peak trunk current per policy (violin)
  2. Steps to 99% restoration per policy (violin)
  3. Trip-rate / restore-rate stacked bars

Usage:
    python scripts/plot_comparison.py \
        --policies results/policy_s1.zip results/policy_s2.zip results/policy_s3.zip \
        --n 500
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO

from restart.env import RestorationEnv
from restart.baselines import GreedyPolicy, SequentialPolicy
from restart.eval import run_policy_once


def collect(env, policy, seeds, masked):
    rs = [run_policy_once(env, policy, s, masked) for s in seeds]
    return {
        "peak": np.array([r.peak_trunk_amps for r in rs]),
        "steps99": np.array([r.steps_to_99 for r in rs]),
        "trip": np.array([r.trip for r in rs]),
        "restored": np.array([r.restored for r in rs]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="results/comparison.png")
    args = ap.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.n))
    env = RestorationEnv(seed=0)

    # Pool PPO ensemble results
    ppo_data = {"peak": [], "steps99": [], "trip": [], "restored": []}
    for p in args.policies:
        if not Path(p).exists():
            continue
        model = MaskablePPO.load(p)
        d = collect(env, model, seeds, masked=True)
        for k in ppo_data:
            ppo_data[k].append(d[k])
    for k in ppo_data:
        ppo_data[k] = np.concatenate(ppo_data[k]) if ppo_data[k] else np.array([])

    series = {
        "greedy":      collect(env, GreedyPolicy(env.N),                       seeds, False),
        "seq d=4":     collect(env, SequentialPolicy(env.N, dwell_steps=4),    seeds, False),
        "seq d=8":     collect(env, SequentialPolicy(env.N, dwell_steps=8),    seeds, False),
        "seq d=12":    collect(env, SequentialPolicy(env.N, dwell_steps=12),   seeds, False),
        "PPO":         ppo_data,
    }

    names = list(series.keys())
    colors = ["#c0392b", "#d68910", "#7f8c8d", "#5d6d7e", "#0b3a78"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    plt.subplots_adjust(wspace=0.32)

    # Panel 1: Peak trunk current
    ax = axes[0]
    parts = ax.violinplot([series[n]["peak"] for n in names], showmedians=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.7); body.set_edgecolor("black")
    ax.axhline(env.trip_amps, color="#c0392b", ls="--", lw=1.2, label=f"trip {env.trip_amps:.0f} A")
    ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("peak trunk current (A)")
    ax.set_title("Peak trunk current per restoration")
    ax.legend(loc="lower right", fontsize=9)

    # Panel 2: Steps to 99%
    ax = axes[1]
    data = [series[n]["steps99"] for n in names]
    parts = ax.violinplot(data, showmedians=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.7); body.set_edgecolor("black")
    ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("steps to 99% restored (lower = faster)")
    ax.set_title("Time to restore 99% of load")

    # Panel 3: Trip rate vs restore rate
    ax = axes[2]
    x = np.arange(len(names))
    trip_rates = [series[n]["trip"].mean()*100 for n in names]
    restore_rates = [series[n]["restored"].mean()*100 for n in names]
    width = 0.36
    b1 = ax.bar(x - width/2, trip_rates, width, color="#c0392b", label="trip %")
    b2 = ax.bar(x + width/2, restore_rates, width, color="#0f9d6b", label="restore %")
    for b, v in zip(b1, trip_rates):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}", ha="center", fontsize=9)
    for b, v in zip(b2, restore_rates):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("rate (%)"); ax.set_ylim(0, 110)
    ax.set_title("Trip vs restore rate")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Policy comparison on {args.n} held-out scenarios", fontsize=13, y=1.02)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
