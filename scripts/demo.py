"""Run one restoration episode with a chosen policy and print a step-by-step
trace plus a load chart saved as a PNG.

Usage:
    python scripts/demo.py --policy greedy --seed 42
    python scripts/demo.py --policy sequential --seed 42
    python scripts/demo.py --policy ppo --model results/policy.zip --seed 42
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["greedy", "sequential", "ppo"], default="sequential")
    ap.add_argument("--model", type=str, default="results/policy.zip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/demo.png")
    args = ap.parse_args()

    env = RestorationEnv(seed=0)
    if args.policy == "greedy":
        pol = GreedyPolicy(n_segments=env.N)
        masked = False
    elif args.policy == "sequential":
        pol = SequentialPolicy(n_segments=env.N, dwell_steps=6)
        masked = False
    else:
        pol = MaskablePPO.load(args.model)
        masked = True

    obs, _ = env.reset(seed=args.seed)
    if hasattr(pol, "reset"):
        pol.reset()

    times = [0.0]
    amps = [env.trunk_amps_last]
    energized_count = [int(env.energized.sum())]

    print(f"# scenario seed={args.seed}  ambient={env.ambient_C:+.1f}°C")
    print(f"step  t(s)   action      trunk_A   energized   note")

    done = truncated = False
    while not (done or truncated):
        if masked:
            action, _ = pol.predict(obs, action_masks=env.action_masks(), deterministic=True)
        elif hasattr(pol, "predict"):
            action, _ = pol.predict(obs)
        else:
            action = pol(obs)
        action = int(action)

        obs, r, done, truncated, info = env.step(action)
        a_str = f"close S{action+1}" if action < env.N else "wait"
        note = "TRIP" if info["trip"] else ("done" if done else "")
        print(f"{env.steps_taken:4d}  {env.steps_taken*env.step_seconds:5.0f}  {a_str:<10}  "
              f"{info['trunk_amps']:7.1f}   {info['n_energized']}/{env.N}        {note}")

        times.append(env.steps_taken * env.step_seconds)
        amps.append(info["trunk_amps"])
        energized_count.append(info["n_energized"])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, amps, color="#0b3a78", lw=2, label="trunk current (A)")
    ax.axhline(env.trip_amps, color="#c0392b", ls="--", lw=1, label=f"trip {env.trip_amps:.0f} A")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trunk current (A)")
    ax.set_title(f"restoration — policy={args.policy}, seed={args.seed}")
    ax2 = ax.twinx()
    ax2.step(times, energized_count, where="post", color="#0f9d6b", alpha=0.6, label="segments on")
    ax2.set_ylabel("segments energized")
    ax2.set_ylim(0, env.N + 0.5)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
