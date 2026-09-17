"""Retry policies and circuit breakers for resilient outbound calls."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration-driven retry with exponential backoff and jitter."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.2
    max_delay_seconds: float = 5.0
    multiplier: float = 2.0
    jitter: float = 0.1
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    give_up_on: tuple[type[BaseException], ...] = ()

    @classmethod
    def from_env(cls, prefix: str = "INFRA_RETRY") -> "RetryPolicy":
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(f"{prefix}_{name}")
            return float(raw) if raw is not None else default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(f"{prefix}_{name}")
            return int(raw) if raw is not None else default

        return cls(
            max_attempts=max(1, _int("MAX_ATTEMPTS", 3)),
            initial_delay_seconds=_float("INITIAL_DELAY_SECONDS", 0.2),
            max_delay_seconds=_float("MAX_DELAY_SECONDS", 5.0),
            multiplier=_float("MULTIPLIER", 2.0),
            jitter=_float("JITTER", 0.1),
        )

    def delay_for_attempt(self, attempt: int) -> float:
        # attempt is 0-based after a failure
        base = min(
            self.max_delay_seconds,
            self.initial_delay_seconds * (self.multiplier ** attempt),
        )
        if self.jitter <= 0:
            return base
        return max(0.0, base * (1.0 + random.uniform(-self.jitter, self.jitter)))

    def run(self, fn: Callable[[], T], *, on_retry: Callable[[int, BaseException, float], None] | None = None) -> T:
        last_exc: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                return fn()
            except self.give_up_on as exc:  # type: ignore[misc]
                raise exc
            except self.retry_on as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt >= self.max_attempts - 1:
                    break
                delay = self.delay_for_attempt(attempt)
                if on_retry is not None:
                    on_retry(attempt + 1, exc, delay)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.run(lambda: fn(*args, **kwargs))

        return wrapper


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    def __init__(self, name: str, retry_after_seconds: float) -> None:
        self.name = name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Circuit '{name}' is open; retry after {retry_after_seconds:.2f}s")


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    success_threshold: int = 1

    @classmethod
    def from_env(cls, prefix: str = "INFRA_CIRCUIT") -> "CircuitBreakerConfig":
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(f"{prefix}_{name}")
            return float(raw) if raw is not None else default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(f"{prefix}_{name}")
            return int(raw) if raw is not None else default

        return cls(
            failure_threshold=max(1, _int("FAILURE_THRESHOLD", 5)),
            recovery_timeout_seconds=_float("RECOVERY_TIMEOUT_SECONDS", 30.0),
            success_threshold=max(1, _int("SUCCESS_THRESHOLD", 1)),
        )


class CircuitBreaker:
    """Simple thread-safe circuit breaker."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig.from_env()
        self._lock = threading.RLock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._maybe_transition_to_half_open()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "opened_at": self._opened_at,
            }

    def _maybe_transition_to_half_open(self) -> None:
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return
        if time.monotonic() - self._opened_at >= self.config.recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

    def allow(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._opened_at = None
            else:
                self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    def run(self, fn: Callable[[], T]) -> T:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                remaining = 0.0
                if self._opened_at is not None:
                    remaining = max(
                        0.0,
                        self.config.recovery_timeout_seconds - (time.monotonic() - self._opened_at),
                    )
                raise CircuitOpenError(self.name, remaining)
        try:
            result = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.run(lambda: fn(*args, **kwargs))

        return wrapper


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(name, config)
        return _breakers[name]


def reset_circuit_breakers() -> None:
    with _breakers_lock:
        _breakers.clear()
