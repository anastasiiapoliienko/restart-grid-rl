"""Smoke test for VinnytsiaRestorationEnv.

Runs three rollouts:
  - greedy-now    : close all feeders immediately (likely to trip)
  - sequential    : close one feeder then wait until current settles
  - twin-replay   : same as sequential but uses a fixed seed for reproducibility
Prints summary stats. Useful as a sanity check after each repo update.
"""
from __future__ import annotations
import sys
from pathlib import Path

# vinnytsia-twin lives next to restart-grid-rl in the workspace layout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VINN = ROOT.parent / "vinnytsia-twin"
if (VINN / "twin").exists():
    sys.path.insert(0, str(VINN))

import numpy as np
from restart import VinnytsiaRestorationEnv

POP_JSON = VINN / "results" / "calibrated_population.json"


def rollout(env, policy, label, seed=0):
    obs, info = env.reset(seed=seed)
    total = 0.0
    for step in range(env.max_steps):
        a = policy(obs, env)
        obs, r, term, trunc, info = env.step(int(a))
        total += r
        if term or trunc:
            break
    print(f"  [{label:14s}] reward={total:7.1f}  "
          f"steps={env.steps_taken:3d}  "
          f"n_energized={info['n_energized']}/{env.N}  "
          f"trip={info['trip']}  ambient={info['ambient_C']:.1f}C")


def greedy_now(obs, env):
    if not env.energized.all():
        return int(np.argmin(env.energized.astype(int)))
    return env.N


def sequential(obs, env):
    # Close next feeder only if all currently energized are within safe trunk loading.
    if env.energized.all():
        return env.N
    safe = env.total_p_kw_last < 0.7 * env.substation_capacity_kw
    if safe:
        return int(np.argmin(env.energized.astype(int)))
    return env.N


# Smart restoration: rural -> peri -> urban, with a settle delay between
# closures. f5/f6 are rural (low CLPU), f3/f4 peri, f1/f2 urban (high CLPU).
SMART_ORDER = [4, 5, 2, 3, 0, 1]   # indices of f5, f6, f3, f4, f1, f2


def smart_restore(obs, env, _state={"idx": 0, "last_close": -10}):
    if env.energized.all():
        return env.N
    settle_steps = 20   # ~10 min at 30 s/step for CLPU diversity to recover
    if env.steps_taken - _state["last_close"] < settle_steps:
        return env.N
    while _state["idx"] < len(SMART_ORDER) and env.energized[SMART_ORDER[_state["idx"]]]:
        _state["idx"] += 1
    if _state["idx"] >= len(SMART_ORDER):
        return env.N
    a = SMART_ORDER[_state["idx"]]
    _state["last_close"] = env.steps_taken
    _state["idx"] += 1
    return int(a)


def reset_state():
    smart_restore.__defaults__[0].update({"idx": 0, "last_close": -10})


def main():
    if not POP_JSON.exists():
        print(f"missing {POP_JSON}; run vinnytsia-twin scripts/run_demo.py first")
        sys.exit(1)
    env = VinnytsiaRestorationEnv(population_json=POP_JSON, seed=0)
    print(f"VinnytsiaRestorationEnv: N={env.N} feeders, "
          f"capacity={env.substation_capacity_kw:.0f} kW")
    rollout(env, greedy_now,    "greedy-now",     seed=0)
    rollout(env, sequential,    "sequential",     seed=0)
    reset_state()
    rollout(env, smart_restore, "smart-rural-1st", seed=0)
    reset_state()
    rollout(env, smart_restore, "smart-replay",    seed=42)


if __name__ == "__main__":
    main()
