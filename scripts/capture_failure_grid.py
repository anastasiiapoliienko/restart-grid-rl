"""Run N scenarios with the trained PPO ensemble, recording for each scenario
the (outage_hours, ambient_C, trip) tuple. Aggregate into a 2D grid and emit
a JSON file the website can render as a heatmap.

Usage:
    python scripts/capture_failure_grid.py \
        --policies results/policy_s1.zip results/policy_s2.zip results/policy_s3.zip \
        --n 800 --out results/failure_grid.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sb3_contrib import MaskablePPO

from restart.env import RestorationEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=str, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed-base", type=int, default=300_000)
    ap.add_argument("--out", type=str, default="results/failure_grid.json")
    # Binning
    ap.add_argument("--outage-bins", type=int, default=4)   # 4 to 12 h → 4 bins of 2h
    ap.add_argument("--ambient-bins", type=int, default=5)  # -5 to +5 C → 5 bins of 2C
    args = ap.parse_args()

    env_cfg = RestorationEnv(seed=0)
    out_lo, out_hi = env_cfg.outage_hours_range
    amb_lo, amb_hi = env_cfg.ambient_C_range

    out_edges = np.linspace(out_lo, out_hi, args.outage_bins + 1)
    amb_edges = np.linspace(amb_lo, amb_hi, args.ambient_bins + 1)

    grids_trip = [np.zeros((args.outage_bins, args.ambient_bins), dtype=int)
                  for _ in args.policies]
    grids_count = [np.zeros((args.outage_bins, args.ambient_bins), dtype=int)
                   for _ in args.policies]

    # Pre-determine scenario parameters by sampling reset() once per seed
    # (same RNG path each policy uses, since env.reset(seed=s) reseeds deterministically)
    scenarios = []
    rng_seeds = list(range(args.seed_base, args.seed_base + args.n))
    for s in rng_seeds:
        env_cfg.reset(seed=s)
        oh = env_cfg.cold_minutes.max() / 60.0  # outage hours actually used
        amb = env_cfg.ambient_C
        scenarios.append((s, float(oh), float(amb)))

    # Run each policy across all scenarios
    for pi, ppath in enumerate(args.policies):
        if not Path(ppath).exists():
            print(f"missing: {ppath}")
            continue
        model = MaskablePPO.load(ppath)
        env = RestorationEnv(seed=0)
        for (s, oh, amb) in scenarios:
            obs, _ = env.reset(seed=s)
            done = trunc = False
            tripped = False
            while not (done or trunc):
                a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                obs, r, done, trunc, info = env.step(int(a))
                if info["trip"]:
                    tripped = True
                    break
            i_out = min(args.outage_bins - 1, np.searchsorted(out_edges, oh, side="right") - 1)
            i_amb = min(args.ambient_bins - 1, np.searchsorted(amb_edges, amb, side="right") - 1)
            i_out = max(0, i_out); i_amb = max(0, i_amb)
            grids_count[pi][i_out, i_amb] += 1
            if tripped:
                grids_trip[pi][i_out, i_amb] += 1
        print(f"finished {ppath}")

    # Aggregate ensemble: sum trips and counts across seeds
    sum_trip = sum(grids_trip)
    sum_count = sum(grids_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(sum_count > 0, sum_trip / sum_count, 0.0)

    payload = {
        "outage_edges_h": list(out_edges.tolist()),
        "ambient_edges_C": list(amb_edges.tolist()),
        "trip_rate": rate.tolist(),
        "count": sum_count.tolist(),
        "n_scenarios": int(args.n),
        "n_policies": int(len(args.policies)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {args.out}")
    # Pretty-print
    print("\ntrip rate (rows: outage_h bins, cols: ambient_C bins):")
    for i, row in enumerate(rate):
        oh_label = f"{out_edges[i]:.1f}-{out_edges[i+1]:.1f}h"
        cells = "  ".join(f"{c*100:5.1f}%" for c in row)
        print(f"  {oh_label:<10s} {cells}")
    print("ambient bins:", "  ".join(f"{amb_edges[i]:.1f}-{amb_edges[i+1]:.1f}" for i in range(args.ambient_bins)))


if __name__ == "__main__":
    main()
