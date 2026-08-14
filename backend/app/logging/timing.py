"""Timing utilities for pipeline stages and benchmarks."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class StageTimer:
    name: str
    budget_ms: int | None = None
    elapsed_ms: float = field(default=0.0, init=False)
    exceeded_budget: bool = field(default=False, init=False)

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.budget_ms is not None and self.elapsed_ms > self.budget_ms:
            self.exceeded_budget = True


@dataclass
class PipelineTimings:
    stt: float = 0.0
    pre_guardrail: float = 0.0
    retrieval: float = 0.0
    llm: float = 0.0
    post_guardrail: float = 0.0

    @property
    def total(self) -> float:
        return self.stt + self.pre_guardrail + self.retrieval + self.llm + self.post_guardrail


@contextmanager
def timed_stage(timings: PipelineTimings, field_name: str) -> Generator[StageTimer, None, None]:
    timer = StageTimer(field_name)
    with timer:
        yield timer
    setattr(timings, field_name, timer.elapsed_ms)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def compute_stats(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p100": percentile(values, 100),
    }
