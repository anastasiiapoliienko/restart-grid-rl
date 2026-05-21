"""Dump a side-by-side trace JSON for the priority website.

Runs three policies on the same scenario:
  1. greedy in segment order (knows nothing about archetypes)
  2. sequential-d8 (knows nothing either)
  3. priority-aware PPO (reads criticality from obs and prioritizes)

Each run captures per-step state including the archetype tag of every
segment so the website can label them by hospital / water / bakery etc.

Usage:
    python scripts/capture_trace_priority.py \
        --model results/policy_priority_s1.zip \
        --seed 500042 --out results/trace_priority.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import MaskablePPO

from restart.env_priority import PriorityRestorationEnv
from restart.baselines import GreedyPolicy, SequentialPolicy
from restart.archetypes import ARCHETYPES, CRITICAL_THRESHOLD


def run_panic(env, seed):
    """Operator-panic worst case: force-energize all 8 segments in a single step.

    Represents pressing every breaker as fast as physically possible, ignoring
    diversity restoration. Useful as a peak-headroom benchmark.
    """
    obs, info0 = env.reset(seed=seed)
    archetypes = info0["archetype_keys"]
    criticality = info0["criticality"]
    icons = info0["archetype_icons"]
    names = info0["archetype_names"]

    steps = [{
        "t": 0.0, "action": None, "trunk_a": float(env.trunk_amps_last),
        "energized": [False]*env.N, "n_energized": 0, "trip": False,
        "t_hospital": None, "t_critical": None, "weighted_load_kw": 0.0,
    }]
    # Force-energize everything, advance one ETP step, push to OpenDSS, read trunk
    env.energized[:] = True
    for i, pop in enumerate(env.populations):
        pop.step(env.step_seconds, energized=True)
    loads_kw = [pop.power_kw(True) for pop in env.populations]
    env.feeder.set_state(loads_kw, [True]*env.N)
    env.feeder.solve()
    trunk_a = env.feeder.trunk_current_amps()
    env.trunk_amps_last = trunk_a
    env.steps_taken = 1
    tripped = trunk_a > env.trip_amps
    wkw = sum(kw * criticality[i] / 5.0 for i, kw in enumerate(loads_kw))
    steps.append({
        "t": float(env.step_seconds),
        "action": None,
        "trunk_a": float(trunk_a),
        "energized": [True]*env.N,
        "n_energized": env.N,
        "trip": tripped,
        "t_hospital": env.step_seconds / 60.0,
        "t_critical": env.step_seconds / 60.0,
        "weighted_load_kw": wkw,
    })

    return {
        "name": "panic close",
        "trip": tripped,
        "restored": not tripped,
        "final_t": float(env.step_seconds),
        "minutes_to_hospital": env.step_seconds / 60.0,
        "minutes_to_critical_all": env.step_seconds / 60.0,
        "archetypes": archetypes,
        "archetype_names": names,
        "icons": icons,
        "criticality": [float(c) for c in criticality],
        "hospital_segment": int(info0["hospital_segment"]),
        "steps": steps,
    }


def run_one(env, policy, seed, masked, name):
    obs, info0 = env.reset(seed=seed)
    if hasattr(policy, "reset"): policy.reset()
    archetypes = info0["archetype_keys"]
    criticality = info0["criticality"]
    icons = info0["archetype_icons"]
    names = info0["archetype_names"]

    steps = [{
        "t": 0.0, "action": None, "trunk_a": float(env.trunk_amps_last),
        "energized": [False]*env.N, "n_energized": 0, "trip": False,
        "t_hospital": None, "t_critical": None,
        "weighted_load_kw": 0.0,
    }]
    done = trunc = False
    while not (done or trunc):
        if masked:
            action, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
        elif hasattr(policy, "predict"):
            try: action, _ = policy.predict(obs, deterministic=True)
            except TypeError: action, _ = policy.predict(obs)
        else: action = policy(obs)
        action = int(action)
        obs, r, done, trunc, info = env.step(action)
        # Compute criticality-weighted load right now (informative)
        wkw = 0.0
        loads = info["loads_kw"]
        for i, kw in enumerate(loads):
            wkw += kw * criticality[i] / 5.0
        steps.append({
            "t": float(env.steps_taken * env.step_seconds),
            "action": action if action < env.N else None,
            "trunk_a": float(info["trunk_amps"]),
            "energized": [bool(x) for x in env.energized],
            "n_energized": int(info["n_energized"]),
            "trip": bool(info["trip"]),
            "t_hospital": info["minutes_to_hospital"],
            "t_critical": info["minutes_to_critical_all"],
            "weighted_load_kw": wkw,
        })
        if done or trunc: break

    return {
        "name": name,
        "trip": bool(info["trip"]),
        "restored": bool(env.energized.all() and not info["trip"]),
        "final_t": float(env.steps_taken * env.step_seconds),
        "minutes_to_hospital": info["minutes_to_hospital"],
        "minutes_to_critical_all": info["minutes_to_critical_all"],
        "archetypes": archetypes,
        "archetype_names": names,
        "icons": icons,
        "criticality": [float(c) for c in criticality],
        "hospital_segment": int(info["hospital_segment"]),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="results/policy_priority_s1.zip")
    ap.add_argument("--seed", type=int, default=500042)
    ap.add_argument("--out", type=str, default="results/trace_priority.json")
    ap.add_argument("--ambient", type=float, default=None,
                    help="Force ambient °C (else sampled in [-5, 5])")
    ap.add_argument("--outage", type=float, default=None,
                    help="Force outage duration in hours (else sampled in [4, 12])")
    args = ap.parse_args()

    kw = {"seed": 0}
    if args.ambient is not None:
        kw["ambient_C_range"] = (args.ambient, args.ambient)
    if args.outage is not None:
        kw["outage_hours_range"] = (args.outage, args.outage)
    env = PriorityRestorationEnv(**kw)
    model = MaskablePPO.load(args.model)

    runs = []
    runs.append(run_one(env, GreedyPolicy(env.N), args.seed, False, "greedy"))
    runs.append(run_one(env, SequentialPolicy(env.N, dwell_steps=8), args.seed, False, "sequential-d8"))
    runs.append(run_one(env, model, args.seed, True, "priority PPO"))
    runs.append(run_panic(env, args.seed))

    payload = {
        "seed": args.seed,
        "ambient_C": float(env.ambient_C),
        "n_segments": int(env.N),
        "trip_amps": float(env.trip_amps),
        "step_seconds": float(env.step_seconds),
        "critical_threshold": CRITICAL_THRESHOLD,
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")
    for r in runs:
        h = r["minutes_to_hospital"]
        c = r["minutes_to_critical_all"]
        print(f"  {r['name']:<18s}  trip={r['trip']}  restored={r['restored']}  "
              f"hospital@{(h or 0):.1f}min  critical@{(c or 0):.1f}min  t_final={r['final_t']:.0f}s")


if __name__ == "__main__":
    main()
