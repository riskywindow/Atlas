from pydantic import ValidationError

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
)


def test_backend_snapshot_serializes() -> None:
    snapshot = BackendStatusSnapshot(
        name="mock-a",
        capabilities=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            max_context_tokens=8192,
            concurrency_limit=2,
        ),
        health=BackendHealth(
            state=BackendHealthState.HEALTHY,
            latency_ms=12.5,
            error_rate=0.0,
        ),
        active_requests=1,
    )

    payload = snapshot.model_dump(mode="json")

    assert payload["capabilities"]["backend_type"] == "mock"
    assert payload["health"]["state"] == "healthy"


def test_backend_health_rejects_invalid_error_rate() -> None:
    try:
        BackendHealth(state=BackendHealthState.DEGRADED, error_rate=1.5)
    except ValidationError as exc:
        assert "error_rate" in str(exc)
    else:
        raise AssertionError("BackendHealth should reject error_rate values above 1.0")
