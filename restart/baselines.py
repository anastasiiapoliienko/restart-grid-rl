"""Non-learned restoration baselines we compare the RL policy against.

GreedyPolicy:
  Close every still-open segment as fast as possible, no dwell. This is the
  "what an inexperienced operator under pressure would do" baseline. Trips often.

SequentialPolicy:
  Close one segment, then wait `dwell_steps` env-steps before closing the next.
  This is roughly the paper-procedure baseline: spread the load out in time.
  Works in most scenarios at the cost of restoration time.
"""
from __future__ import annotations
import numpy as np


class GreedyPolicy:
    name = "greedy"

    def __init__(self, n_segments: int = 8):
        self.N = n_segments

    def reset(self):
        pass

    def predict(self, obs: np.ndarray):
        energized = obs[: self.N].astype(bool)
        # close lowest-index still-open segment, else no-op
        for i in range(self.N):
            if not energized[i]:
                return i, None
        return self.N, None  # no-op (all already on)


class SequentialPolicy:
    """Close-then-wait policy with a fixed dwell between closures."""
    name = "sequential"

    def __init__(self, n_segments: int = 8, dwell_steps: int = 4):
        self.N = n_segments
        self.dwell_steps = int(dwell_steps)
        self._wait_left = 0

    def reset(self):
        self._wait_left = 0

    def predict(self, obs: np.ndarray):
        energized = obs[: self.N].astype(bool)
        if energized.all():
            return self.N, None
        if self._wait_left > 0:
            self._wait_left -= 1
            return self.N, None  # no-op (wait)
        # otherwise close the next segment and set dwell
        for i in range(self.N):
            if not energized[i]:
                self._wait_left = self.dwell_steps
                return i, None
        return self.N, None
