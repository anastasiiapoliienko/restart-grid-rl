"""Eight feeder-segment archetypes with criticality, ETP profile, and human framing.

These describe what's actually on each segment of a real distribution feeder —
hospitals, water pumping, bakeries, vulnerable housing — and quantify both the
*priority* (criticality weight 1.0–5.0) and the *load shape* (load level, CLPU
magnitude and decay time constant, house count) that change how each archetype
behaves during restoration.

The criticality weight is what the priority RL reward function multiplies the
per-close reward by, so the agent learns to restore high-criticality archetypes
first when safe.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class Archetype:
    key: str                                # short identifier
    name: str                                # display name
    criticality: float                       # 1.0 (low) … 5.0 (life-critical)
    icon: str                                # emoji/symbol for UI
    blurb: str                               # one-sentence human framing
    q_range: Tuple[float, float]             # heater kW per "house" / unit
    house_count: int                         # # of units on this segment
    clpu_mag_range: Tuple[float, float]      # short-timescale inrush multiplier
    clpu_tau_range: Tuple[float, float]      # CLPU decay tau, seconds


# Order is the canonical demo ordering used by the website (S1=hospital, …).
# During training, archetypes are shuffled across the 8 segments each episode
# so the policy must read the criticality from the observation, not memorize
# the segment index.
ARCHETYPES: List[Archetype] = [
    Archetype(
        key="hospital", name="Regional hospital",
        criticality=5.0, icon="🏥",
        blurb="ICU, four surgeries in progress, 12 dialysis patients. Backup generator running low.",
        q_range=(9.0, 13.0), house_count=44,
        clpu_mag_range=(0.8, 1.4), clpu_tau_range=(60.0, 150.0),
    ),
    Archetype(
        key="water", name="District water pumping",
        criticality=4.5, icon="💧",
        blurb="Pumps water for ~80,000 households. About six hours of gravity buffer left in elevated tanks.",
        q_range=(11.0, 16.0), house_count=28,
        clpu_mag_range=(1.0, 1.6), clpu_tau_range=(60.0, 150.0),
    ),
    Archetype(
        key="bakery", name="Bakery + grocery cluster",
        criticality=3.5, icon="🍞",
        blurb="District bread shifts and refrigerated grocery. Spoilage clock starts on outage.",
        q_range=(9.0, 13.0), house_count=22,
        clpu_mag_range=(1.8, 2.8), clpu_tau_range=(90.0, 210.0),
    ),
    Archetype(
        key="elderly", name="Apartment block, elderly residents",
        criticality=3.5, icon="🏢",
        blurb="320 residents, mean age 71. Heating is the difference between safe and unsafe.",
        q_range=(7.0, 11.0), house_count=50,
        clpu_mag_range=(1.6, 2.6), clpu_tau_range=(90.0, 210.0),
    ),
    Archetype(
        key="pharmacy", name="Pharmacy cluster",
        criticality=3.0, icon="💊",
        blurb="Refrigerated medicines, insulin stocks, regional vaccine inventory.",
        q_range=(5.0, 9.0), house_count=17,
        clpu_mag_range=(1.5, 2.5), clpu_tau_range=(60.0, 180.0),
    ),
    Archetype(
        key="residential", name="Mixed residential block",
        criticality=2.0, icon="🏘️",
        blurb="350 households, mixed demographics, electric heating dominant.",
        q_range=(7.0, 11.0), house_count=44,
        clpu_mag_range=(1.4, 2.4), clpu_tau_range=(90.0, 210.0),
    ),
    Archetype(
        key="school", name="School",
        criticality=1.5, icon="🏫",
        blurb="Closed off-hours; heating maintenance only. Pipes at risk if outage is long.",
        q_range=(7.0, 11.0), house_count=22,
        clpu_mag_range=(1.1, 2.0), clpu_tau_range=(90.0, 210.0),
    ),
    Archetype(
        key="warehouse", name="Light commercial / cold storage",
        criticality=1.0, icon="🏬",
        blurb="Freezers and refrigerated retail. Tolerates short outages; non-critical short-term.",
        q_range=(7.0, 11.0), house_count=28,
        clpu_mag_range=(1.8, 2.8), clpu_tau_range=(120.0, 270.0),
    ),
]

# Threshold above which a segment is "critical" for the priority metric.
CRITICAL_THRESHOLD = 3.0


def by_key(k: str) -> Archetype:
    for a in ARCHETYPES:
        if a.key == k: return a
    raise KeyError(k)
