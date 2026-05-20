"""Safety shield: a one-step-ahead forecast that vetoes PPO actions expected
to push the trunk over the trip threshold.

Workflow per env step:
  1. The trained PPO policy proposes an action a* given the observation.
  2. If a* is a close-segment action, the shield simulates one step ahead
     by deep-copying the ETP populations, applying a*, advancing them by
     `env.step_seconds`, and summing predicted load.
  3. If predicted total load (converted to trunk amps under a 3-phase
     10 kV / 0.97 PF assumption) exceeds `threshold_factor * env.trip_amps`,
     the action is replaced by the no-op (wait).
  4. Otherwise the action is passed through.

The shield is intentionally a thin layer — no learning, no shared parameters
with the policy. It uses the same ETP+CLPU dynamics that produced the
training data, so the forecast is consistent with what the env will compute
on the real step.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .env import RestorationEnv


# 3-phase 10 kV, PF ≈ 0.97 ⇒ amps = kW * 1000 / (sqrt(3) * 10000 * 0.97)
_KW_TO_AMPS = 1000.0 / (np.sqrt(3.0) * 10000.0 * 0.97)


@dataclass
class ShieldStats:
    overrides: int = 0           # times the shield said "wait" instead of agent's choice
    passes: int = 0              # times the agent's action was accepted unchanged
    waits_agent: int = 0         # times the agent chose to wait on its own
    forecast_amps_max: float = 0.0


class SafetyShield:
    """Thin one-step physics-forecast veto layer around any policy."""

    def __init__(self, inner_policy, env: RestorationEnv,
                 threshold_factor: float = 0.95,
                 masked: bool = True):
        self.inner = inner_policy
        self.env_ref = env
        self.threshold = float(env.trip_amps * threshold_factor)
        self.masked = masked
        self.stats = ShieldStats()

    def reset(self):
        self.stats = ShieldStats()
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def predict(self, obs, env: Optional[RestorationEnv] = None,
                deterministic: bool = True, **_):
        """Return (action, _) for compatibility with sb3 policy API.

        `env` must be the *live* env the action will be applied to — the shield
        reads its current state to simulate. If called via run_shielded_policy
        below, this is passed explicitly; if called via SB3's default loop, the
        shield falls back to the env reference captured at construction.
        """
        live_env = env if env is not None else self.env_ref

        # Ask the underlying policy
        if self.masked and hasattr(self.inner, "predict"):
            action, _ = self.inner.predict(
                obs, action_masks=live_env.action_masks(), deterministic=deterministic
            )
        elif hasattr(self.inner, "predict"):
            try:
                action, _ = self.inner.predict(obs, deterministic=deterministic)
            except TypeError:
                action, _ = self.inner.predict(obs)
        else:
            action = self.inner(obs)
        action = int(action)

        # Only veto close actions. Waits are always safe.
        if action >= live_env.N:
            self.stats.waits_agent += 1
            return action, None
        if live_env.energized[action]:
            # Already closed (shouldn't happen with masking, but be defensive)
            return action, None

        predicted_amps = self._forecast(live_env, action)
        self.stats.forecast_amps_max = max(self.stats.forecast_amps_max, predicted_amps)

        if predicted_amps > self.threshold:
            self.stats.overrides += 1
            return live_env.N, None  # force wait
        self.stats.passes += 1
        return action, None

    # ----- forecast -----

    def _forecast(self, env: RestorationEnv, candidate_action: int) -> float:
        """Predict trunk amps one env-step after applying `candidate_action`."""
        sim_energized = env.energized.copy()
        sim_energized[candidate_action] = True
        # Deep-copy ETP populations so the live env state is untouched.
        sim_pops = copy.deepcopy(env.populations)
        for i, pop in enumerate(sim_pops):
            pop.step(env.step_seconds, energized=bool(sim_energized[i]))
        total_kw = sum(
            pop.power_kw(bool(sim_energized[i])) for i, pop in enumerate(sim_pops)
        )
        return float(total_kw * _KW_TO_AMPS)
