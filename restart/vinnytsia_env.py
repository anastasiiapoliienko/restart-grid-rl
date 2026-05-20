"""Gymnasium environment: restoration on the Vinnytsia 35/10 kV substation.

This is Phase 1 of the Restart project. The substation topology and the
per-DTR house population are loaded from a JSON file produced by
vinnytsia-twin's calibration pipeline. Cold-load pickup is purely
emergent from the thermostatic dynamics (no multipliers).

State (Box):
  - feeder_energized[6]
  - feeder_cold_h[6]            : outage duration per feeder in hours / 12
  - ambient_C / 20.0
  - trunk_load_pu               : substation supply / nameplate capacity
  - frac_restored               : feeders currently energized / 6

Action (Discrete, 7):
  - 0..5 : close switch on feeder Fi (no-op if already energized)
  - 6    : wait one step

Termination:
  - if any feeder trip exceeds threshold (heavy negative reward)
  - all 6 feeders energized (positive bonus)
  - max_steps reached (truncate)

The env imports vinnytsia-twin as a regular package (install editable
with `pip install -e ../vinnytsia-twin`).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from twin.etp import HousePopulation
    from twin.substation import Substation, FEEDERS, DTR_LOADS, FEEDER_ZONES
    from twin.persist import population_from_json
    _HAVE_TWIN = True
except ImportError:
    _HAVE_TWIN = False


# Map each DTR load to its feeder.
def _feeder_of(load_name: str) -> str:
    return load_name.split("_")[0]


class VinnytsiaRestorationEnv(gym.Env):
    """6-feeder substation restoration env using the calibrated twin."""
    metadata = {"render_modes": []}

    def __init__(
        self,
        population_json: str | Path,
        dss_path: str | Path | None = None,
        step_seconds: float = 30.0,
        max_steps: int = 240,
        outage_hours_range: tuple = (2.0, 5.0),
        ambient_C_range: tuple = (-15.0, 0.0),
        trip_kw_per_feeder: float = 6000.0,    # ~600 A on a 10 kV feeder
        substation_capacity_kw: float = 32000.0,    # 2 x 16 MVA at PF 1
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not _HAVE_TWIN:
            raise ImportError(
                "vinnytsia-twin is not importable. Install with "
                "`pip install -e <path-to-vinnytsia-twin>`."
            )
        self.population_json = Path(population_json)
        if dss_path is None:
            payload = json.loads(self.population_json.read_text())
            dss_path = payload.get("dss_path")
            if dss_path is None:
                raise ValueError("dss_path not in JSON; pass explicitly.")
        self.dss_path = str(Path(dss_path).resolve())

        self.N = len(FEEDERS)   # 6
        self.step_seconds = float(step_seconds)
        self.max_steps = int(max_steps)
        self.outage_hours_range = outage_hours_range
        self.ambient_C_range = ambient_C_range
        self.trip_kw_per_feeder = float(trip_kw_per_feeder)
        self.substation_capacity_kw = float(substation_capacity_kw)
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.N + 1)
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(2 * self.N + 3,), dtype=np.float32
        )

        # Built lazily on first reset() to avoid the OpenDSS handle being
        # held by a wrapped/copied env.
        self._sub: Substation | None = None
        self._pops_template: Dict[str, HousePopulation] | None = None

        # State filled by reset()
        self.energized: np.ndarray
        self.cold_minutes: np.ndarray
        self.populations: Dict[str, HousePopulation]
        self.ambient_C: float
        self.steps_taken: int
        self.feeder_load_kw_last: np.ndarray
        self.total_p_kw_last: float

    def _lazy_init(self) -> None:
        if self._sub is None:
            self._sub = Substation(self.dss_path)
        if self._pops_template is None:
            self._pops_template = population_from_json(self.population_json)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._lazy_init()
        self.ambient_C = float(self.rng.uniform(*self.ambient_C_range))
        outage_h = float(self.rng.uniform(*self.outage_hours_range))

        # Fresh populations: cold-soak each one for outage_h at ambient_C.
        self.populations = {}
        for ld, template in self._pops_template.items():
            pop = HousePopulation(
                n_houses=template.n_houses,
                seed=int(self.rng.integers(1 << 30)),
                R_mean=template.R_mean, C_mean=template.C_mean,
                Q_mean=template.Q_mean, P_base_mean=template.P_base_mean,
                setpoint_mean=template.setpoint_mean, deadband=template.deadband,
            )
            # Cool through the outage with energized=False.
            n_cool_steps = int(round(outage_h * 3600.0 / self.step_seconds))
            for _ in range(n_cool_steps):
                pop.step(self.step_seconds, self.ambient_C, energized=False)
            self.populations[ld] = pop

        self.energized = np.zeros(self.N, dtype=bool)
        self.cold_minutes = np.full(self.N, outage_h * 60.0, dtype=np.float32)
        self.steps_taken = 0
        self.feeder_load_kw_last = np.zeros(self.N, dtype=np.float32)
        self.total_p_kw_last = 0.0

        self._sub.reset()
        self._push_state_and_solve()
        return self._obs(), self._info(trip=False, just_closed=None)

    def step(self, action: int):
        self.steps_taken += 1
        prev_n_energized = int(self.energized.sum())
        just_closed = None

        if 0 <= action < self.N and not self.energized[action]:
            self.energized[action] = True
            just_closed = action

        # Advance dynamics.
        for ld, pop in self.populations.items():
            fid = FEEDERS.index(_feeder_of(ld))
            pop.step(self.step_seconds, self.ambient_C, energized=bool(self.energized[fid]))
        for i, en in enumerate(self.energized):
            if en:
                self.cold_minutes[i] = 0.0
            else:
                self.cold_minutes[i] += self.step_seconds / 60.0

        res = self._push_state_and_solve()
        # Per-feeder trip detection.
        tripped_any = False
        for i, f in enumerate(FEEDERS):
            if self.energized[i] and self.feeder_load_kw_last[i] > self.trip_kw_per_feeder:
                tripped_any = True
                break

        # Reward
        n_now = int(self.energized.sum())
        new_energized = n_now - prev_n_energized
        reward = 10.0 * new_energized - 1.0
        # Voltage penalty.
        if res.min_voltage_pu < 0.93:
            reward -= 20.0 * (0.93 - res.min_voltage_pu)
        terminated = False
        truncated = False
        if tripped_any:
            reward -= 30.0
            terminated = True
        elif self.energized.all():
            reward += 30.0
            terminated = True
        if self.steps_taken >= self.max_steps:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, self._info(trip=tripped_any, just_closed=just_closed)

    def _push_state_and_solve(self):
        # Aggregate per-DTR load + push to OpenDSS.
        kw_by_load = {}
        feeder_load = np.zeros(self.N)
        for ld, pop in self.populations.items():
            fid = FEEDERS.index(_feeder_of(ld))
            en = bool(self.energized[fid])
            kw = pop.power_kw(energized=en)
            kw_by_load[ld] = kw
            if en:
                feeder_load[fid] += kw
        energized_feeders = {FEEDERS[i]: bool(self.energized[i]) for i in range(self.N)}
        self._sub.set_dtr_loads(kw_by_load, energized_feeders=energized_feeders)
        res = self._sub.solve()
        self.total_p_kw_last = res.total_p_kw
        self.feeder_load_kw_last = feeder_load.astype(np.float32)
        return res

    def _obs(self) -> np.ndarray:
        cold_h = np.clip(self.cold_minutes / 60.0, 0, 24.0).astype(np.float32) / 12.0
        amb_norm = np.float32(self.ambient_C / 20.0)
        trunk_pu = np.float32(self.total_p_kw_last / self.substation_capacity_kw)
        frac = np.float32(self.energized.mean())
        return np.concatenate([
            self.energized.astype(np.float32),
            cold_h,
            np.array([amb_norm, trunk_pu, frac], dtype=np.float32),
        ])

    def _info(self, trip: bool, just_closed: Optional[int]) -> dict:
        return {
            "total_p_kw": float(self.total_p_kw_last),
            "feeder_load_kw": self.feeder_load_kw_last.tolist(),
            "trip": bool(trip),
            "just_closed": just_closed,
            "ambient_C": float(self.ambient_C),
            "n_energized": int(self.energized.sum()),
        }

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.N + 1, dtype=bool)
        mask[: self.N] = ~self.energized
        return mask
