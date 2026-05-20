"""Gymnasium environment: cold-load pickup restoration on an 8-segment feeder.

State (Box, shape=2N+3):
  - energized[N]            : 0/1
  - cold_minutes[N] / 60    : outage duration per segment in hours, clipped
  - ambient_C / 20.0        : normalized
  - trunk_load_pu           : current trunk load / rated
  - frac_restored           : fraction of segments currently stable+energized

Action (Discrete, N+1):
  - 0..N-1  : close switch on segment i (no-op if already energized)
  - N       : wait one step (advance time without changing state)

Each step advances simulated time by `step_seconds` (default 30 s).
Episode ends on:
  - upstream trunk breaker trip (large negative reward, terminate)
  - all segments stably restored (positive terminal bonus)
  - max steps reached (truncate)

Reward per step (concrete, see reward fn for exact form):
  + delta load reconnected (kW, normalized)
  − heavy penalty on trip
  − tiny urgency cost per step
  − voltage-violation penalty
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .etp import ETPPopulation
from .feeder import Feeder


FEEDER_DSS = Path(__file__).parent.parent / "feeders" / "feeder8.dss"


class RestorationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_segments: int = 8,
        houses_per_segment: int = 50,
        step_seconds: float = 30.0,
        max_steps: int = 240,                    # 240 * 30 s = 2 h cap
        outage_hours_range: tuple = (4.0, 12.0),
        ambient_C_range: tuple = (-5.0, 5.0),
        trip_amps: float = 240.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.N = n_segments
        self.houses_per_segment = houses_per_segment
        self.step_seconds = float(step_seconds)
        self.max_steps = int(max_steps)
        self.outage_hours_range = outage_hours_range
        self.ambient_C_range = ambient_C_range
        self.trip_amps = float(trip_amps)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.N + 1)
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(2 * self.N + 3,), dtype=np.float32
        )

        self.feeder = Feeder(FEEDER_DSS, n_segments=self.N, trip_amps=self.trip_amps)

        # State filled by reset()
        self.energized: np.ndarray
        self.cold_minutes: np.ndarray
        self.populations: list[ETPPopulation]
        self.ambient_C: float
        self.steps_taken: int
        self.trunk_amps_last: float
        self.prev_load_kw: float

    # ----- Gym API -----

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample scenario
        self.ambient_C = float(self.rng.uniform(*self.ambient_C_range))
        outage_h = float(self.rng.uniform(*self.outage_hours_range))

        # Build populations and cold-soak them for the outage duration
        self.populations = []
        for i in range(self.N):
            pop = ETPPopulation(
                n_houses=self.houses_per_segment,
                ambient_C=self.ambient_C,
                seed=int(self.rng.integers(1 << 30)),
            )
            pop.cool_for(outage_h)
            self.populations.append(pop)

        self.energized = np.zeros(self.N, dtype=bool)
        self.cold_minutes = np.full(self.N, outage_h * 60.0, dtype=np.float32)
        self.steps_taken = 0
        self.trunk_amps_last = 0.0
        self.prev_load_kw = 0.0

        self.feeder.reset()
        self._sync_feeder()  # initial state, all open
        self.feeder.solve()

        return self._obs(), self._info(trip=False, just_closed=None)

    def step(self, action: int):
        self.steps_taken += 1
        just_closed = None
        prev_n_energized = int(self.energized.sum())

        if 0 <= action < self.N:
            if not self.energized[action]:
                self.energized[action] = True
                just_closed = action

        for i, pop in enumerate(self.populations):
            pop.step(self.step_seconds, energized=bool(self.energized[i]))
            if self.energized[i]:
                self.cold_minutes[i] = 0.0
            else:
                self.cold_minutes[i] += self.step_seconds / 60.0

        loads_kw = [pop.power_kw(bool(en)) for pop, en in zip(self.populations, self.energized)]
        self.feeder.set_state(loads_kw, list(map(bool, self.energized)))
        self.feeder.solve()

        trunk_a = self.feeder.trunk_current_amps()
        self.trunk_amps_last = trunk_a
        vmin_pu = self.feeder.min_voltage_pu()
        tripped = trunk_a > self.trip_amps

        # --- Reward shaped so completion-fast strictly dominates partial-and-wait ---
        n_now = int(self.energized.sum())
        new_energized = n_now - prev_n_energized
        reward = 0.0
        reward += 10.0 * new_energized                # +10 each time a segment is newly closed
        reward -= 1.0                                 # -1 per step (urgency)
        if vmin_pu < 0.93:
            reward -= 2.0 * (0.93 - vmin_pu) * 10.0
        self.prev_load_kw = float(sum(loads_kw))

        terminated = False
        truncated = False
        if tripped:
            reward -= 30.0
            terminated = True
        elif self.energized.all():
            reward += 30.0
            terminated = True
        if self.steps_taken >= self.max_steps:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, self._info(trip=tripped, just_closed=just_closed)

    # ----- helpers -----

    def _sync_feeder(self) -> None:
        loads_kw = [pop.power_kw(bool(en)) for pop, en in zip(self.populations, self.energized)]
        self.feeder.set_state(loads_kw, list(map(bool, self.energized)))

    def _obs(self) -> np.ndarray:
        cold_h = np.clip(self.cold_minutes / 60.0, 0, 24.0).astype(np.float32) / 12.0
        amb_norm = np.float32(self.ambient_C / 20.0)
        # Trunk load p.u.: amps over trip threshold
        trunk_pu = np.float32(self.trunk_amps_last / self.trip_amps)
        frac_restored = np.float32(self.energized.mean())
        obs = np.concatenate([
            self.energized.astype(np.float32),
            cold_h,
            np.array([amb_norm, trunk_pu, frac_restored], dtype=np.float32),
        ])
        return obs

    def _info(self, trip: bool, just_closed: Optional[int]) -> dict:
        return {
            "trunk_amps": float(self.trunk_amps_last),
            "trip": bool(trip),
            "just_closed": just_closed,
            "ambient_C": float(self.ambient_C),
            "n_energized": int(self.energized.sum()),
            "loads_kw": [pop.power_kw(bool(en)) for pop, en in zip(self.populations, self.energized)],
        }

    def action_masks(self) -> np.ndarray:
        """For MaskablePPO: mask out closing already-energized segments."""
        mask = np.ones(self.N + 1, dtype=bool)
        mask[: self.N] = ~self.energized
        return mask
