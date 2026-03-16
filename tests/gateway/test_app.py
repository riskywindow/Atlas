from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pytest import CaptureFixture

from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import AppEnvironment, Settings
from switchyard.gateway import create_app
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendType,
    DeviceClass,
)
from switchyard.telemetry import Telemetry, configure_telemetry


def build_client(*, unavailable: bool = False) -> tuple[TestClient, Telemetry]:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-gateway",
            health_state=(
                BackendHealthState.UNAVAILABLE if unavailable else BackendHealthState.HEALTHY
            ),
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=3,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} said={user_message}"
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    return TestClient(app), telemetry


def test_health_and_readiness_endpoints() -> None:
    client, telemetry = build_client()

    health = client.get("/healthz")
    ready = client.get("/readyz", headers={"x-request-id": "req-ready"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert "x-request-id" in health.headers
    assert ready.status_code == 200
    assert ready.headers["x-request-id"] == "req-ready"
    assert ready.json()["status"] == "ready"
    assert ready.json()["adapters"] == ["mock-gateway"]
    assert telemetry.state.request_count == 2
    assert telemetry.state.backend_health_snapshots[0]["backend_name"] == "mock-gateway"


def test_chat_completions_returns_deterministic_mock_response(capsys: CaptureFixture[str]) -> None:
    client, telemetry = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-123"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert payload["backend_name"] == "mock-gateway"
    assert payload["id"] == "mockcmpl_c4c89bb9a8139e6f"
    assert payload["choices"][0]["message"]["content"] == (
        "backend=mock-gateway request=req-123 said=Hello gateway"
    )
    assert telemetry.state.route_decision_count == 1

    log_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"event":' in line
    ]
    route_log = next(line for line in log_lines if line["event"] == "route_decision")
    completion_log = next(
        line for line in log_lines if line["event"] == "chat_completion_succeeded"
    )

    assert route_log["request_id"] == "req-123"
    assert route_log["chosen_backend"] == "mock-gateway"
    assert completion_log["request_id"] == "req-123"
    assert completion_log["chosen_backend"] == "mock-gateway"


def test_chat_completions_returns_typed_validation_error() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-invalid"},
        json={"model": "", "messages": []},
    )

    payload = response.json()

    assert response.status_code == 422
    assert response.headers["x-request-id"] == "req-invalid"
    assert payload["code"] == "invalid_request"
    assert payload["request_id"] == "req-invalid"


def test_chat_completions_returns_backend_unavailable_error() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-miss"},
        json={
            "model": "unsupported-model",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 503
    assert response.headers["x-request-id"] == "req-miss"
    assert payload["code"] == "backend_unavailable"
    assert payload["request_id"] == "req-miss"


def test_readiness_fails_when_all_backends_are_unavailable() -> None:
    client, _ = build_client(unavailable=True)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
