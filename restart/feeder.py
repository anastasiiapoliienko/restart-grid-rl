"""Thin wrapper around OpenDSSDirect for the 8-segment feeder.

The Python side owns the load values (driven by ETP populations) and the
switch states; this module just pushes those into OpenDSS, solves a snapshot
power flow, and reads back trunk current and segment voltages.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence

import opendssdirect as dss


class Feeder:
    """8-segment 10 kV radial feeder."""

    def __init__(self, dss_file: str | Path, n_segments: int = 8,
                 trip_amps: float = 200.0):
        self.dss_file = str(Path(dss_file).resolve())
        self.n = n_segments
        self.trip_amps = trip_amps
        self._compile()

    def _compile(self) -> None:
        cwd = os.getcwd()
        dss.Text.Command("Clear")
        dss.Text.Command(f'Compile "{self.dss_file}"')
        dss.Text.Command("set mode=snapshot")
        os.chdir(cwd)  # OpenDSS chdir's into the .dss file's dir; restore.

    def reset(self) -> None:
        """Reload the feeder from the .dss file (fresh state)."""
        self._compile()

    def set_state(self, loads_kw: Sequence[float], energized: Sequence[bool]) -> None:
        """Push current load values and switch states into OpenDSS."""
        for i in range(self.n):
            k = i + 1
            kw = max(0.01, float(loads_kw[i]))
            kvar = 0.25 * kw  # power factor ~0.97 lagging
            dss.Text.Command(f"Edit Load.ld{k} kW={kw:.3f} kvar={kvar:.3f}")
            state = "yes" if energized[i] else "no"
            dss.Text.Command(f"Edit Line.sw{k} enabled={state}")
            dss.Text.Command(f"Edit Load.ld{k} enabled={state}")

    def solve(self) -> bool:
        dss.Solution.Solve()
        return bool(dss.Solution.Converged())

    def trunk_current_amps(self) -> float:
        dss.Circuit.SetActiveElement("Line.trunk")
        currents = dss.CktElement.CurrentsMagAng()
        # CurrentsMagAng returns [mag1, ang1, mag2, ang2, ...] for terminal 1 then 2.
        # For a 3-phase line, indices 0,2,4 are |I| on phases A,B,C at terminal 1.
        mags = [currents[0], currents[2], currents[4]]
        return float(max(mags))

    def min_voltage_pu(self) -> float:
        """Worst per-unit voltage across energized buses (excluding source)."""
        vmags = dss.Circuit.AllBusMagPu()
        if not vmags:
            return 1.0
        # Filter out sourcebus (~1.0) and zero-voltage de-energized buses
        nonzero = [v for v in vmags if v > 0.05]
        return float(min(nonzero)) if nonzero else 1.0

    def is_tripped(self, current_amps: float | None = None) -> bool:
        i = self.trunk_current_amps() if current_amps is None else current_amps
        return i > self.trip_amps
