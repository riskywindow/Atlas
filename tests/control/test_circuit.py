from __future__ import annotations

from switchyard.config import CircuitBreakerSettings
from switchyard.control.circuit import CircuitBreakerService
from switchyard.schemas.routing import CircuitBreakerPhase


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_circuit_breaker_opens_after_repeated_failures() -> None:
    service = CircuitBreakerService(
        CircuitBreakerSettings(enabled=True, failure_threshold=2, open_cooldown_seconds=10.0)
    )

    service.record_failure("backend-a", reason="invocation_failure")
    state = service.record_failure("backend-a", reason="invocation_failure")

    assert state.phase is CircuitBreakerPhase.OPEN
    assert state.failure_count == 2
    allowed, _, reason = service.allow_routing("backend-a")
    assert allowed is False
    assert reason == "invocation_failure"


def test_circuit_breaker_transitions_to_half_open_after_cooldown() -> None:
    clock = FakeClock()
    service = CircuitBreakerService(
        CircuitBreakerSettings(enabled=True, failure_threshold=1, open_cooldown_seconds=5.0),
        clock=clock,
    )

    service.record_failure("backend-a", reason="timeout_like_failure")
    assert service.state_for("backend-a").phase is CircuitBreakerPhase.OPEN

    clock.advance(6.0)
    allowed, state, reason = service.allow_routing("backend-a")

    assert allowed is True
    assert state.phase is CircuitBreakerPhase.HALF_OPEN
    assert reason == "backend circuit is half_open and eligible for a recovery probe"


def test_circuit_breaker_allows_only_one_half_open_probe() -> None:
    clock = FakeClock()
    service = CircuitBreakerService(
        CircuitBreakerSettings(enabled=True, failure_threshold=1, open_cooldown_seconds=1.0),
        clock=clock,
    )
    service.record_failure("backend-a", reason="invocation_failure")
    clock.advance(2.0)

    first_allowed, first_state, _ = service.begin_execution("backend-a")
    second_allowed, second_state, second_reason = service.begin_execution("backend-a")

    assert first_allowed is True
    assert first_state.phase is CircuitBreakerPhase.HALF_OPEN
    assert second_allowed is False
    assert second_state.phase is CircuitBreakerPhase.HALF_OPEN
    assert second_reason == "backend circuit is half_open and awaiting probe outcome"


def test_circuit_breaker_closes_after_successful_probe() -> None:
    clock = FakeClock()
    service = CircuitBreakerService(
        CircuitBreakerSettings(
            enabled=True,
            failure_threshold=1,
            recovery_success_threshold=1,
            open_cooldown_seconds=1.0,
        ),
        clock=clock,
    )
    service.record_failure("backend-a", reason="invocation_failure")
    clock.advance(2.0)

    allowed, _, _ = service.begin_execution("backend-a")
    assert allowed is True
    state = service.record_success("backend-a")

    assert state.phase is CircuitBreakerPhase.CLOSED
    assert state.failure_count == 0


def test_circuit_breaker_reopens_when_half_open_probe_fails() -> None:
    clock = FakeClock()
    service = CircuitBreakerService(
        CircuitBreakerSettings(enabled=True, failure_threshold=1, open_cooldown_seconds=3.0),
        clock=clock,
    )
    service.record_failure("backend-a", reason="invocation_failure")
    clock.advance(4.0)

    allowed, _, _ = service.begin_execution("backend-a")
    assert allowed is True
    state = service.record_failure("backend-a", reason="timeout_like_failure")

    assert state.phase is CircuitBreakerPhase.OPEN
    assert state.reason == "timeout_like_failure"
