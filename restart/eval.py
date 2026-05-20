"""Evaluation harness — run any policy across N held-out scenarios and report.

Each scenario is identified by its seed, so the same seeds can be replayed
across different policies for paired comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Callable, Any
import numpy as np

from .env import RestorationEnv


@dataclass
class RunResult:
    seed: int
    trip: bool
    restored: bool
    steps_to_99: int        # steps until ≥99% restored (or max if never)
    final_frac: float
    total_reward: float
    peak_trunk_amps: float


def run_policy_once(
    env: RestorationEnv,
    policy,
    seed: int,
    use_action_masks: bool = False,
) -> RunResult:
    obs, _info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    done = False
    truncated = False
    total_r = 0.0
    peak_amps = 0.0
    steps_to_99 = env.max_steps
    while not (done or truncated):
        if use_action_masks and hasattr(policy, "predict") and "action_masks" in policy.predict.__code__.co_varnames:
            action, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
        elif hasattr(policy, "predict"):
            try:
                action, _ = policy.predict(obs, deterministic=True)
            except TypeError:
                action, _ = policy.predict(obs)
        else:
            action = policy(obs)
        action = int(action)
        obs, r, done, truncated, info = env.step(action)
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


def run_policy(env: RestorationEnv, policy, seeds: Sequence[int],
               use_action_masks: bool = False) -> list[RunResult]:
    return [run_policy_once(env, policy, s, use_action_masks) for s in seeds]


def summarize(results: list[RunResult]) -> dict:
    n = len(results)
    trips = sum(r.trip for r in results)
    restored = sum(r.restored for r in results)
    return {
        "n": n,
        "trip_rate": trips / n if n else 0.0,
        "restore_rate": restored / n if n else 0.0,
        "mean_steps_to_99": float(np.mean([r.steps_to_99 for r in results])) if n else 0.0,
        "mean_peak_amps": float(np.mean([r.peak_trunk_amps for r in results])) if n else 0.0,
        "mean_reward": float(np.mean([r.total_reward for r in results])) if n else 0.0,
    }
