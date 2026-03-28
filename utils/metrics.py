"""Runtime metrics utilities: FPS counter and latency tracking.

Example:
    >>> from utils.metrics import FPSCounter
    >>> fps = FPSCounter()
    >>> fps.tick()
    0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Deque
from collections import deque


@dataclass
class FPSCounter:
    """Compute FPS using a sliding window."""

    window_size: int = 30
    _times: Deque[float] = field(default_factory=lambda: deque(maxlen=30))

    def tick(self) -> float:
        """Record a frame and return estimated FPS.

        Example:
            >>> c = FPSCounter(window_size=3)
            >>> _ = c.tick()
            >>> isinstance(c.tick(), float)
            True
        """

        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0:
            return 0.0
        return float((len(self._times) - 1) / dt)


@dataclass
class LatencyTracker:
    """Track latest and average latency values in milliseconds."""

    window_size: int = 50
    _values_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def add(self, value_ms: float) -> None:
        """Add a latency value in milliseconds.

        Example:
            >>> t = LatencyTracker()
            >>> t.add(10.0)
        """

        self._values_ms.append(float(value_ms))

    @property
    def last(self) -> float:
        """Return last value or 0.0."""

        return float(self._values_ms[-1]) if self._values_ms else 0.0

    @property
    def mean(self) -> float:
        """Return mean latency or 0.0."""

        if not self._values_ms:
            return 0.0
        return float(sum(self._values_ms) / len(self._values_ms))

