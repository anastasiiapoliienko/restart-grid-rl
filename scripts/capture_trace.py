"""Run greedy, sequential-dwell8, and a trained PPO policy on the same scenario;
record per-step trunk current, action, and per-segment energization state;
emit a single JSON the website can animate.

Usage:
    python scripts/capture_trace.py --model results/policy.zip --seed 10042 \
        --out results/trace.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import MaskablePPO

from restart.env import RestorationEnv
from restart.baselines import GreedyPolicy, SequentialPolicy
from restart.shield import SafetyShield


def run_one(env: RestorationEnv, policy, seed: int, masked: bool, name: str,
            shielded: bool = False):
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    steps = []
    steps.append({
        "t": 0.0,
        "action": None,
        "trunk_a": float(env.trunk_amps_last),
        "energized": [bool(x) for x in env.energized],
        "n_energized": int(env.energized.sum()),
        "trip": False,
        "shield_veto": False,
    })
    done = trunc = False
    while not (done or trunc):
        shield_veto = False
        if shielded:
            # SafetyShield.predict already handles masked + forecast veto
            action, _ = policy.predict(obs, env=env, deterministic=True)
            # Detect a veto: shield returns N (no-op) when it overrode a close.
            # We can't easily know the original action from this trace, but the
            # shield's stats track it; we record an explicit veto flag when the
            # most recent stat counter advanced.
            cur_overrides = policy.stats.overrides
            shield_veto = cur_overrides > getattr(policy, "_prev_overrides", 0)
            policy._prev_overrides = cur_overrides
        elif masked:
            action, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
        elif hasattr(policy, "predict"):
            action, _ = policy.predict(obs)
        else:
            action = policy(obs)
        action = int(action)
        obs, r, done, trunc, info = env.step(action)
        steps.append({
            "t": float(env.steps_taken * env.step_seconds),
            "action": action if action < env.N else None,
            "trunk_a": float(info["trunk_amps"]),
            "energized": [bool(x) for x in env.energized],
            "n_energized": int(info["n_energized"]),
            "trip": bool(info["trip"]),
            "shield_veto": bool(shield_veto),
        })
        if done or trunc:
            break
    return {
        "name": name,
        "trip": bool(info["trip"]),
        "restored": bool(env.energized.all() and not info["trip"]),
        "final_t": float(env.steps_taken * env.step_seconds),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/policy.zip")
    ap.add_argument("--seed", type=int, default=10042)
    ap.add_argument("--out", type=str, default="results/trace.json")
    args = ap.parse_args()

    env = RestorationEnv(seed=0)
    model = MaskablePPO.load(args.model)

    runs = []
    runs.append(run_one(env, GreedyPolicy(env.N), args.seed, False, "greedy"))
    runs.append(run_one(env, SequentialPolicy(env.N, dwell_steps=8), args.seed, False, "sequential-d8"))
    runs.append(run_one(env, model, args.seed, True, "ppo"))
    shield = SafetyShield(model, env, threshold_factor=0.95, masked=True)
    shield._prev_overrides = 0
    runs.append(run_one(env, shield, args.seed, True, "ppo + shield", shielded=True))

    payload = {
        "seed": args.seed,
        "ambient_C": float(env.ambient_C),
        "n_segments": int(env.N),
        "trip_amps": float(env.trip_amps),
        "step_seconds": float(env.step_seconds),
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {args.out}")
    for r in runs:
        print(f"  {r['name']:<14s}  trip={r['trip']} restored={r['restored']} t_final={r['final_t']:.0f}s  steps={len(r['steps'])}")


if __name__ == "__main__":
    main()
