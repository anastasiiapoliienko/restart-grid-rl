"""Priority-aware restoration environment.

Subclasses RestorationEnv. The 8 archetypes (hospital, water pumping, bakery,
elderly apartments, pharmacy, residential, school, warehouse) are shuffled
across the 8 segments each reset, so the policy must consult the criticality
vector in its observation rather than memorize a segment index.

Differences from the base env:

* Observation gains 8 floats: per-segment criticality_weight / 5.0.
  New shape = 3*N + 3 (was 2*N + 3).
* Reward shaping multiplies the +10 per new-energized term by the segment's
  criticality / mean_criticality, so closing the hospital gives a much larger
  reward than closing the warehouse — without changing the absolute scale.
* `info` reports `archetype_keys`, `criticality`, `minutes_to_hospital`,
  `minutes_to_critical` (time until every criticality>=3.0 segment is on).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import RestorationEnv, FEEDER_DSS
from .etp import ETPPopulation
from .feeder import Feeder
from .archetypes import ARCHETYPES, CRITICAL_THRESHOLD


class PriorityRestorationEnv(RestorationEnv):

    def __init__(
        self,
        step_seconds: float = 30.0,
        max_steps: int = 240,
        outage_hours_range: tuple = (4.0, 12.0),
        ambient_C_range: tuple = (-5.0, 5.0),
        trip_amps: float = 240.0,
        seed: Optional[int] = None,
        shuffle_archetypes: bool = True,
    ):
        # Build base env (8 segments, default houses ignored — we resample per-archetype)
        super().__init__(
            n_segments=len(ARCHETYPES),
            houses_per_segment=30,
            step_seconds=step_seconds,
            max_steps=max_steps,
            outage_hours_range=outage_hours_range,
            ambient_C_range=ambient_C_range,
            trip_amps=trip_amps,
            seed=seed,
        )
        self.shuffle_archetypes = shuffle_archetypes
        # Re-declare observation space (extra N floats for criticality)
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0,
            shape=(3 * self.N + 3,),
            dtype=np.float32,
        )
        # placeholders populated by reset()
        self.archetype_idx: np.ndarray              # which Archetype is at segment i
        self.criticality: np.ndarray                # weights aligned to segments
        self._mean_crit: float = 1.0
        self._t_hospital: Optional[float] = None    # seconds until hospital energized
        self._t_critical_all: Optional[float] = None
        self._hospital_seg: int = 0

    # ── Override reset to seed the archetype assignment + custom-param populations ──
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.ambient_C = float(self.rng.uniform(*self.ambient_C_range))
        outage_h = float(self.rng.uniform(*self.outage_hours_range))

        # Shuffle archetypes across segments
        perm = np.arange(len(ARCHETYPES))
        if self.shuffle_archetypes:
            self.rng.shuffle(perm)
        self.archetype_idx = perm.copy()
        self.criticality = np.array(
            [ARCHETYPES[i].criticality for i in self.archetype_idx], dtype=np.float32
        )
        self._mean_crit = float(self.criticality.mean())
        # Find hospital
        for i, ai in enumerate(self.archetype_idx):
            if ARCHETYPES[ai].key == "hospital":
                self._hospital_seg = i
                break
        self._t_hospital = None
        self._t_critical_all = None

        # Build a custom-parameter population per segment
        self.populations = []
        for i in range(self.N):
            arch = ARCHETYPES[self.archetype_idx[i]]
            pop = ETPPopulation(
                n_houses=arch.house_count,
                ambient_C=self.ambient_C,
                seed=int(self.rng.integers(1 << 30)),
            )
            # Override CLPU + Q for this archetype
            rng2 = np.random.default_rng(int(self.rng.integers(1 << 30)))
            pop.clpu_mag = float(rng2.uniform(*arch.clpu_mag_range))
            pop.clpu_tau = float(rng2.uniform(*arch.clpu_tau_range))
            pop.Q = rng2.uniform(arch.q_range[0], arch.q_range[1], pop.n_houses)
            # Resample heater state under new Q distribution
            pop.heater_on = (pop.T < pop.setpoint).astype(bool)
            pop.cool_for(outage_h)
            self.populations.append(pop)

        self.energized = np.zeros(self.N, dtype=bool)
        self.cold_minutes = np.full(self.N, outage_h * 60.0, dtype=np.float32)
        self.steps_taken = 0
        self.trunk_amps_last = 0.0
        self.prev_load_kw = 0.0

        self.feeder.reset()
        self._sync_feeder()
        self.feeder.solve()

        return self._obs(), self._info(trip=False, just_closed=None)

    # ── Override step to weight reward by criticality and track priority metrics ──
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

        # Update priority timing metrics
        elapsed_s = self.steps_taken * self.step_seconds
        if self._t_hospital is None and self.energized[self._hospital_seg]:
            self._t_hospital = elapsed_s
        if self._t_critical_all is None:
            critical_mask = self.criticality >= CRITICAL_THRESHOLD
            if critical_mask.any() and all(self.energized[critical_mask]):
                self._t_critical_all = elapsed_s

        # Criticality-weighted reward
        n_now = int(self.energized.sum())
        if just_closed is not None:
            close_weight = self.criticality[just_closed] / self._mean_crit
        else:
            close_weight = 0.0
        reward = 0.0
        reward += 10.0 * close_weight                  # weighted close bonus
        reward -= 1.0                                  # urgency
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

    # ── Augmented observation: append criticality / 5.0 per segment ──
    def _obs(self) -> np.ndarray:
        base = super()._obs()
        crit_norm = (self.criticality / 5.0).astype(np.float32)
        return np.concatenate([base, crit_norm])

    # ── Augmented info ──
    def _info(self, trip: bool, just_closed):
        info = super()._info(trip=trip, just_closed=just_closed)
        info["archetype_keys"] = [ARCHETYPES[i].key for i in self.archetype_idx]
        info["archetype_names"] = [ARCHETYPES[i].name for i in self.archetype_idx]
        info["archetype_icons"] = [ARCHETYPES[i].icon for i in self.archetype_idx]
        info["criticality"] = [float(c) for c in self.criticality]
        info["minutes_to_hospital"] = (self._t_hospital / 60.0) if self._t_hospital is not None else None
        info["minutes_to_critical_all"] = (self._t_critical_all / 60.0) if self._t_critical_all is not None else None
        info["hospital_segment"] = int(self._hospital_seg)
        return info

    # Action mask is unchanged (inherited)
