from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

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
from switchyard.schemas.benchmark import TraceCaptureMode
from switchyard.telemetry import configure_telemetry


def _build_trace_client(
    *,
    trace_capture_mode: TraceCaptureMode,
    trace_output_path: Path,
) -> TestClient:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-trace",
            health_state=BackendHealthState.HEALTHY,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
            ),
            response_template=MockResponseTemplate(content="trace-content"),
        )
    )
    telemetry = configure_telemetry("switchyard-trace-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            trace_capture_mode=trace_capture_mode,
            trace_capture_output_path=trace_output_path,
        ),
    )
    return TestClient(app)


def test_trace_capture_metadata_only_writes_safe_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "trace.jsonl"
    client = _build_trace_client(
        trace_capture_mode=TraceCaptureMode.METADATA_ONLY,
        trace_output_path=output_path,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "trace-meta-1",
            "x-switchyard-tenant-id": "tenant-trace",
            "x-switchyard-request-class": "bulk",
            "x-switchyard-session-id": "session-trace",
            "authorization": "Bearer secret-token",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "secret prompt"}],
        },
    )

    assert response.status_code == 200
    lines = output_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["capture_mode"] == "metadata_only"
    assert payload["request_id"] == "trace-meta-1"
    assert payload["logical_alias"] == "mock-chat"
    assert payload["tenant_id"] == "tenant-trace"
    assert payload["request_class"] == "bulk"
    assert payload["session_id"] == "session-trace"
    assert payload["normalized_request_payload"]["message_count"] == 1
    assert "messages" not in payload["normalized_request_payload"]
    assert payload["chosen_backend"] == "mock-trace"
    assert "authorization" not in json.dumps(payload).lower()


def test_trace_capture_redacts_content(tmp_path: Path) -> None:
    output_path = tmp_path / "trace.jsonl"
    client = _build_trace_client(
        trace_capture_mode=TraceCaptureMode.REDACTED_CONTENT,
        trace_output_path=output_path,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "trace-redacted-1"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "top secret"}],
        },
    )

    assert response.status_code == 200
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["normalized_request_payload"]["messages"][0]["content"].startswith("[redacted")
    assert payload["normalized_response_payload"]["choices"][0]["message"]["content"].startswith(
        "[redacted"
    )


def test_trace_capture_full_content_writes_replayable_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "trace.jsonl"
    client = _build_trace_client(
        trace_capture_mode=TraceCaptureMode.FULL_CONTENT,
        trace_output_path=output_path,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "trace-full-1",
            "x-switchyard-tenant-id": "tenant-full",
            "x-switchyard-request-class": "latency_sensitive",
            "x-switchyard-internal-backend-pin": "mock-trace",
        },
        json={
            "model": "mock-chat",
            "stream": False,
            "messages": [{"role": "user", "content": "store this prompt"}],
        },
    )

    assert response.status_code == 200
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["capture_mode"] == "full_content"
    assert payload["execution_target"]["pinned_backend"] == "mock-trace"
    assert payload["tenant_id"] == "tenant-full"
    assert payload["request_class"] == "latency_sensitive"
    assert payload["normalized_request_payload"]["messages"][0]["content"] == "store this prompt"
    assert payload["normalized_response_payload"]["choices"][0]["message"]["content"] == (
        "trace-content"
    )


def test_trace_capture_off_writes_nothing(tmp_path: Path) -> None:
    output_path = tmp_path / "trace.jsonl"
    client = _build_trace_client(
        trace_capture_mode=TraceCaptureMode.OFF,
        trace_output_path=output_path,
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "trace-off-1"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "ignore me"}],
        },
    )

    assert response.status_code == 200
    assert not output_path.exists()
