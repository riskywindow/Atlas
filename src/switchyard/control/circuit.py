"""In-process backend circuit-breaker protection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Protocol

from switchyard.config import CircuitBreakerSettings
from switchyard.schemas.admin import CircuitBreakerRuntimeSummary
from switchyard.schemas.routing import CircuitBreakerPhase, CircuitBreakerState


class CircuitBreakerClock(Protocol):
    """Clock abstraction for deterministic breaker tests."""

    def now(self) -> float: ...


@dataclass(frozen=True, slots=True)
class MonotonicCircuitBreakerClock:
    """Default monotonic clock."""

    def now(self) -> float:
        return monotonic()


@dataclass(slots=True)
class _BreakerRecord:
    phase: CircuitBreakerPhase = CircuitBreakerPhase.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    opened_at: float | None = None
    cooldown_until: float | None = None
    last_failure_at: float | None = None
    reason: str | None = None
    probe_in_flight: bool = False


@dataclass(frozen=True, slots=True)
class CircuitProbe:
    """Probe lease for a half-open backend."""

    backend_name: str
    phase: CircuitBreakerPhase


class CircuitBreakerService:
    """Small local breaker manager for backend invocation failures."""

    def __init__(
        self,
        settings: CircuitBreakerSettings,
        *,
        clock: CircuitBreakerClock | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or MonotonicCircuitBreakerClock()
        self._records: dict[str, _BreakerRecord] = {}

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def state_for(self, backend_name: str) -> CircuitBreakerState:
        """Return the current breaker state for one backend."""

        record = self._records.get(backend_name)
        if record is None:
            return CircuitBreakerState(backend_name=backend_name)
        self._advance_to_half_open_if_ready(record)
        return CircuitBreakerState(
            backend_name=backend_name,
            phase=record.phase,
            failure_count=record.consecutive_failures,
            success_count=record.consecutive_successes,
            last_failure_at=_epoch_to_datetime(record.last_failure_at),
            opened_at=_epoch_to_datetime(record.opened_at),
            cooldown_until=_epoch_to_datetime(record.cooldown_until),
            reason=record.reason,
        )

    def allow_routing(self, backend_name: str) -> tuple[bool, CircuitBreakerState, str | None]:
        """Return whether the backend may be considered for routing."""

        if not self.enabled:
            state = CircuitBreakerState(backend_name=backend_name)
            return True, state, None

        record = self._records.setdefault(backend_name, _BreakerRecord())
        self._advance_to_half_open_if_ready(record)
        state = self.state_for(backend_name)
        if record.phase is CircuitBreakerPhase.OPEN:
            reason = record.reason or "backend circuit is open"
            return False, state, reason
        return True, state, "backend circuit is half_open and eligible for a recovery probe"

    def begin_execution(self, backend_name: str) -> tuple[bool, CircuitBreakerState, str | None]:
        """Reserve execution for a backend, including a half-open probe slot."""

        if not self.enabled:
            state = CircuitBreakerState(backend_name=backend_name)
            return True, state, None

        record = self._records.setdefault(backend_name, _BreakerRecord())
        self._advance_to_half_open_if_ready(record)
        state = self.state_for(backend_name)
        if record.phase is CircuitBreakerPhase.CLOSED:
            return True, state, None
        if record.phase is CircuitBreakerPhase.OPEN:
            reason = record.reason or "backend circuit is open"
            return False, state, reason
        if record.probe_in_flight:
            return False, state, "backend circuit is half_open and awaiting probe outcome"
        record.probe_in_flight = True
        return True, state, "backend circuit is half_open and allowing a recovery probe"

    def record_failure(
        self,
        backend_name: str,
        *,
        reason: str,
    ) -> CircuitBreakerState:
        """Record a backend failure and update breaker state."""

        if not self.enabled:
            return CircuitBreakerState(backend_name=backend_name)

        record = self._records.setdefault(backend_name, _BreakerRecord())
        now = self._clock.now()
        record.last_failure_at = now
        record.reason = reason
        record.consecutive_successes = 0
        record.probe_in_flight = False
        record.consecutive_failures += 1
        if record.consecutive_failures >= self._settings.failure_threshold:
            record.phase = CircuitBreakerPhase.OPEN
            record.opened_at = now
            record.cooldown_until = now + self._settings.open_cooldown_seconds
        elif record.phase is CircuitBreakerPhase.HALF_OPEN:
            record.phase = CircuitBreakerPhase.OPEN
            record.opened_at = now
            record.cooldown_until = now + self._settings.open_cooldown_seconds
        return self.state_for(backend_name)

    def record_success(self, backend_name: str) -> CircuitBreakerState:
        """Record a backend success and update breaker state."""

        if not self.enabled:
            return CircuitBreakerState(backend_name=backend_name)

        record = self._records.setdefault(backend_name, _BreakerRecord())
        self._advance_to_half_open_if_ready(record)
        record.probe_in_flight = False
        if record.phase is CircuitBreakerPhase.HALF_OPEN:
            record.consecutive_successes += 1
            if record.consecutive_successes >= self._settings.recovery_success_threshold:
                record.phase = CircuitBreakerPhase.CLOSED
                record.consecutive_failures = 0
                record.consecutive_successes = 0
                record.opened_at = None
                record.cooldown_until = None
                record.reason = None
        else:
            record.consecutive_failures = 0
            record.consecutive_successes = 0
            record.reason = None
        return self.state_for(backend_name)

    def inspect_state(self) -> CircuitBreakerRuntimeSummary:
        """Return a runtime summary for all tracked backend breakers."""

        backends = [self.state_for(name) for name in sorted(self._records)]
        return CircuitBreakerRuntimeSummary(enabled=self.enabled, backends=backends)

    def _advance_to_half_open_if_ready(self, record: _BreakerRecord) -> None:
        if record.phase is not CircuitBreakerPhase.OPEN:
            return
        cooldown_until = record.cooldown_until
        if cooldown_until is None or self._clock.now() < cooldown_until:
            return
        record.phase = CircuitBreakerPhase.HALF_OPEN
        record.consecutive_successes = 0
        record.probe_in_flight = False


def _epoch_to_datetime(value: float | None) -> datetime | None:
    if value is None:
        return None
    from datetime import UTC, datetime

    return datetime.fromtimestamp(value, tz=UTC)
