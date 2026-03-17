from datetime import UTC, datetime, timedelta

from switchyard.config import SessionAffinitySettings
from switchyard.control.affinity import SessionAffinityService
from switchyard.schemas.routing import AffinityDisposition, SessionAffinityKey


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def build_key(session_id: str) -> SessionAffinityKey:
    return SessionAffinityKey(
        tenant_id="tenant-a",
        session_id=session_id,
        serving_target="mock-chat",
    )


def test_session_affinity_reuses_existing_binding() -> None:
    clock = FakeClock()
    service = SessionAffinityService(
        SessionAffinitySettings(enabled=True, ttl_seconds=30.0, max_sessions=4),
        clock=clock,
    )

    bound = service.bind(build_key("session-1"), backend_name="mock-a")
    lookup = service.lookup(build_key("session-1"))

    assert lookup.disposition is AffinityDisposition.REUSED
    assert lookup.sticky_route == bound
    assert len(service) == 1


def test_session_affinity_expires_bindings() -> None:
    clock = FakeClock()
    service = SessionAffinityService(
        SessionAffinitySettings(enabled=True, ttl_seconds=5.0, max_sessions=4),
        clock=clock,
    )

    service.bind(build_key("session-1"), backend_name="mock-a")
    clock.advance(6.0)
    lookup = service.lookup(build_key("session-1"))

    assert lookup.disposition is AffinityDisposition.EXPIRED
    assert lookup.sticky_route is None
    assert len(service) == 0


def test_session_affinity_evicts_oldest_binding_when_capacity_is_reached() -> None:
    clock = FakeClock()
    service = SessionAffinityService(
        SessionAffinitySettings(enabled=True, ttl_seconds=30.0, max_sessions=2),
        clock=clock,
    )

    service.bind(build_key("session-1"), backend_name="mock-a")
    clock.advance(1.0)
    service.bind(build_key("session-2"), backend_name="mock-b")
    clock.advance(1.0)
    service.bind(build_key("session-3"), backend_name="mock-c")

    assert service.lookup(build_key("session-1")).disposition is AffinityDisposition.MISSED
    assert service.lookup(build_key("session-2")).sticky_route is not None
    assert service.lookup(build_key("session-3")).sticky_route is not None


def test_session_affinity_refreshes_existing_binding() -> None:
    clock = FakeClock()
    service = SessionAffinityService(
        SessionAffinitySettings(enabled=True, ttl_seconds=10.0, max_sessions=4),
        clock=clock,
    )

    first = service.bind(build_key("session-1"), backend_name="mock-a")
    clock.advance(4.0)
    refreshed = service.bind(build_key("session-1"), backend_name="mock-a")

    assert refreshed.bound_at > first.bound_at
    assert refreshed.expires_at > first.expires_at
