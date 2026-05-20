"""Equivalent Thermal Parameter (ETP) load population with CLPU multiplier.

Two physical effects are modeled:

  1. Slow thermostatic dynamics (R, C, Q per house). Determines the warm-state
     vs. cold-state demand and the time it takes to recover diversity over hours.
  2. A short-timescale CLPU multiplier representing motor starts, refrigeration
     compressors, lighting surge, and other inrush that decays in 3-8 minutes:

         load_kw(t) = etp_load_kw * (1 + clpu_mag * exp(-t_since_energized/clpu_tau))

The two together reproduce the classic cold-load-pickup shape: a sharp spike at
re-energization, then a multi-minute decay to a warm-but-elevated plateau.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ETPPopulation:
    n_houses: int = 30
    ambient_C: float = 0.0
    seed: int = 0

    R: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)
    setpoint: np.ndarray = field(init=False)
    deadband: float = 1.0

    clpu_mag: float = field(init=False)
    clpu_tau: float = field(init=False)

    T: np.ndarray = field(init=False)
    heater_on: np.ndarray = field(init=False)
    t_since_energized: float = field(init=False, default=float("inf"))

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        n = self.n_houses
        self.R = rng.uniform(3.0, 5.0, n)
        self.C = rng.uniform(8.0, 15.0, n)
        self.Q = rng.uniform(6.0, 10.0, n)
        self.setpoint = rng.uniform(18.5, 21.0, n)
        self.T = self.setpoint + rng.normal(0.0, 0.3, n)
        self.heater_on = (self.T < self.setpoint).astype(bool)
        self.clpu_mag = float(rng.uniform(1.0, 2.0))
        self.clpu_tau = float(rng.uniform(60.0, 180.0))
        self.t_since_energized = float("inf")

    def step(self, dt_seconds: float, energized: bool) -> None:
        if energized:
            if not np.isfinite(self.t_since_energized):
                self.t_since_energized = 0.0
            dt_h = dt_seconds / 3600.0
            dT = dt_h * ((self.ambient_C - self.T) / (self.R * self.C)
                         + self.Q * self.heater_on / self.C)
            self.T += dT
            on_now = self.T < (self.setpoint - self.deadband / 2.0)
            off_now = self.T > (self.setpoint + self.deadband / 2.0)
            self.heater_on = np.where(on_now, True, np.where(off_now, False, self.heater_on))
            self.t_since_energized += dt_seconds
        else:
            dt_h = dt_seconds / 3600.0
            self.T += dt_h * (self.ambient_C - self.T) / (self.R * self.C)
            self.heater_on[:] = False
            self.t_since_energized = float("inf")

    def power_kw(self, energized: bool) -> float:
        if not energized:
            return 0.0
        base = float(np.sum(self.heater_on.astype(float) * self.Q))
        if np.isfinite(self.t_since_energized):
            clpu = 1.0 + self.clpu_mag * np.exp(-self.t_since_energized / self.clpu_tau)
        else:
            clpu = 1.0
        return base * clpu

    def cool_for(self, hours: float) -> None:
        steps = max(1, int(hours * 60))
        for _ in range(steps):
            self.step(60.0, energized=False)

    def mean_temp(self) -> float:
        return float(np.mean(self.T))

    def fraction_below_setpoint(self) -> float:
        return float(np.mean(self.T < self.setpoint))
