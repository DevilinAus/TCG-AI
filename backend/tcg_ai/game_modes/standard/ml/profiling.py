from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator

_ACTIVE_PROFILE: ContextVar["DecisionProfile | None"] = ContextVar(
    "tcg_ai_standard_ml_profile",
    default=None,
)


@dataclass
class DecisionProfile:
    timing_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    maxima: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_ms(self, name: str, value_ms: float) -> None:
        if value_ms <= 0:
            return
        self.timing_ms[name] += float(value_ms)

    def increment(self, name: str, amount: int = 1) -> None:
        if amount == 0:
            return
        self.counters[name] += int(amount)

    def observe_max(self, name: str, value: float) -> None:
        current = self.maxima.get(name, 0.0)
        if value > current:
            self.maxima[name] = float(value)

    def set_metadata(self, name: str, value: Any) -> None:
        self.metadata[name] = value

    def snapshot(self) -> dict[str, Any]:
        return {
            "timing_ms": {
                key: round(value, 3)
                for key, value in sorted(self.timing_ms.items())
                if value > 0
            },
            "counters": {
                key: int(value)
                for key, value in sorted(self.counters.items())
                if value
            },
            "maxima": {
                key: round(value, 3)
                for key, value in sorted(self.maxima.items())
                if value > 0
            },
            "metadata": dict(sorted(self.metadata.items())),
        }


def current_profile() -> DecisionProfile | None:
    return _ACTIVE_PROFILE.get()


@contextmanager
def use_profile(profile: DecisionProfile) -> Iterator[DecisionProfile]:
    token = _ACTIVE_PROFILE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE_PROFILE.reset(token)


def record_ms(name: str, value_ms: float) -> None:
    profile = current_profile()
    if profile is not None:
        profile.add_ms(name, value_ms)


def record_counter(name: str, amount: int = 1) -> None:
    profile = current_profile()
    if profile is not None:
        profile.increment(name, amount)


def observe_max(name: str, value: float) -> None:
    profile = current_profile()
    if profile is not None:
        profile.observe_max(name, value)


def set_metadata(name: str, value: Any) -> None:
    profile = current_profile()
    if profile is not None:
        profile.set_metadata(name, value)


@contextmanager
def time_metric(name: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        record_ms(name, (perf_counter() - started) * 1000)
