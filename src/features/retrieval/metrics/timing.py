"""Shared stage timing utility for retrieval observability."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class StageTimer:
    """Accumulates per-stage latency in milliseconds without duplicating timing code."""

    _stages: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.record(name, elapsed_ms)

    def record(self, name: str, elapsed_ms: float) -> None:
        self._stages[name] = round(float(self._stages.get(name, 0.0) + elapsed_ms), 2)

    def get(self, name: str) -> float:
        return float(self._stages.get(name, 0.0))

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000.0, 2)

    def as_dict(self) -> dict[str, float]:
        return dict(self._stages)

    def merge(self, other: dict[str, float | int]) -> None:
        for key, value in other.items():
            if isinstance(value, (int, float)):
                self.record(key, float(value))
