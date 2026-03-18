from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo

import pytest
from fastapi.testclient import TestClient

from switchyard.config import (
    Phase7ControlPlaneSettings,
    RemoteWorkerLifecycleSettings,
    Settings,
)
from switchyard.control.remote_workers import build_signed_enrollment_token
from switchyard.gateway.app import create_app
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendNetworkEndpoint,
    BackendType,
    DeviceClass,
    EngineType,
    WorkerLifecycleState,
    WorkerTransportType,
)
from switchyard.schemas.worker import RemoteWorkerAuthMode, RemoteWorkerRegistrationRequest


def _app(
    *,
    remote_worker_settings: RemoteWorkerLifecycleSettings,
) -> TestClient:
    app = create_app(
        settings=Settings(
            phase7=Phase7ControlPlaneSettings(
                remote_workers=remote_worker_settings,
            )
        )
    )
    return TestClient(app)


def _registration_payload() -> dict[str, object]:
    return {
        "worker_id": "worker-1",
        "worker_name": "remote-a",
        "backend_type": "vllm_cuda",
        "model_identifier": "meta-llama/Llama-3.1-8B-Instruct",
        "serving_targets": ["chat-shared"],
        "endpoint": {
            "base_url": "https://remote-a.internal",
            "transport": "https",
        },
        "capabilities": {
            "backend_type": "vllm_cuda",
            "engine_type": "vllm",
            "device_class": "nvidia_gpu",
            "model_ids": ["meta-llama/Llama-3.1-8B-Instruct"],
            "serving_targets": ["chat-shared"],
            "max_context_tokens": 8192,
            "supports_streaming": True,
            "concurrency_limit": 8,
        },
        "device_class": "nvidia_gpu",
        "environment": "staging",
        "placement": {"provider": "aws", "region": "us-east-1"},
        "lifecycle_state": "registering",
        "ready": False,
        "tags": ["remote", "gpu"],
    }


def test_remote_worker_admin_endpoints_track_lifecycle_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 3, 17, tzinfo=UTC)

    class FrozenDateTime:
        current: datetime

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            assert tz is not None
            return cls.current.astimezone(tz)

    FrozenDateTime.current = now

    monkeypatch.setenv("SWITCHYARD_REMOTE_REGISTRATION_TOKEN", "secret-token")
    monkeypatch.setattr("switchyard.control.remote_workers.datetime", FrozenDateTime)
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            secure_registration_required=True,
            auth_mode=RemoteWorkerAuthMode.STATIC_TOKEN,
            registration_token_name="SWITCHYARD_REMOTE_REGISTRATION_TOKEN",
            heartbeat_timeout_seconds=30.0,
            stale_eviction_seconds=20.0,
        )
    )

    register = client.post(
        "/internal/control-plane/remote-workers/register",
        headers={"x-switchyard-registration-token": "secret-token"},
        json=_registration_payload(),
    )
    assert register.status_code == 200
    lease_token = register.json()["lease_token"]
    assert lease_token

    heartbeat = client.post(
        "/internal/control-plane/remote-workers/heartbeat",
        headers={"x-switchyard-lease-token": lease_token},
        json={
            "worker_id": "worker-1",
            "lifecycle_state": "warming",
            "ready": False,
            "active_requests": 2,
            "queue_depth": 1,
            "health": {
                "state": "healthy",
                "load_state": "warming",
                "latency_ms": 12.0,
            },
            "metadata": {"gpu_type": "l4"},
        },
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["lifecycle_state"] == "warming"

    ready = client.post(
        "/internal/control-plane/remote-workers/heartbeat",
        headers={"x-switchyard-lease-token": lease_token},
        json={
            "worker_id": "worker-1",
            "ready": True,
            "active_requests": 0,
            "queue_depth": 0,
            "health": {
                "state": "healthy",
                "load_state": "ready",
                "latency_ms": 8.0,
            },
        },
    )
    assert ready.status_code == 200
    assert ready.json()["lifecycle_state"] == "ready"
    assert ready.json()["ready"] is True

    snapshot = client.get("/admin/remote-workers")
    assert snapshot.status_code == 200
    payload = snapshot.json()
    assert payload["worker_count"] == 1
    assert payload["ready_worker_count"] == 1
    assert payload["workers"][0]["heartbeat_count"] == 3
    assert payload["workers"][0]["lifecycle_state"] == "ready"
    assert payload["workers"][0]["metadata"]["gpu_type"] == "l4"
    assert payload["recent_events"][0]["event_type"] == "registered"

    FrozenDateTime.current = now + timedelta(seconds=31)
    stale = client.get("/admin/remote-workers").json()
    assert stale["stale_worker_count"] == 1
    assert stale["lost_worker_count"] == 1
    assert stale["workers"][0]["lifecycle_state"] == "lost"

    FrozenDateTime.current = now + timedelta(seconds=52)
    cleanup = client.post("/admin/remote-workers/cleanup")
    assert cleanup.status_code == 200
    assert cleanup.json()["evicted_worker_ids"] == ["worker-1"]


def test_remote_worker_admin_supports_graceful_deregister() -> None:
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
        )
    )
    register = client.post(
        "/internal/control-plane/remote-workers/register",
        json=_registration_payload(),
    )
    lease_token = register.json()["lease_token"]

    drain = client.post(
        "/internal/control-plane/remote-workers/heartbeat",
        headers={"x-switchyard-lease-token": lease_token},
        json={
            "worker_id": "worker-1",
            "lifecycle_state": "draining",
            "ready": False,
        },
    )
    assert drain.status_code == 200
    assert drain.json()["lifecycle_state"] == "draining"

    retire = client.post(
        "/internal/control-plane/remote-workers/deregister",
        headers={"x-switchyard-lease-token": lease_token},
        json={"worker_id": "worker-1", "reason": "deploy rotation"},
    )
    assert retire.status_code == 200
    assert retire.json()["lifecycle_state"] == "retired"
    assert retire.json()["live"] is False


def test_remote_worker_registration_returns_400_for_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_REMOTE_REGISTRATION_TOKEN", "secret-token")
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            secure_registration_required=True,
            auth_mode=RemoteWorkerAuthMode.STATIC_TOKEN,
            registration_token_name="SWITCHYARD_REMOTE_REGISTRATION_TOKEN",
        )
    )

    response = client.post(
        "/internal/control-plane/remote-workers/register",
        headers={"x-switchyard-registration-token": "wrong-token"},
        json=_registration_payload(),
    )

    assert response.status_code == 400
    assert response.json()["code"] == "remote_worker_registration_error"


def test_remote_worker_registration_accepts_signed_enrollment_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_ENROLLMENT_SECRET", "secret-value")
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            secure_registration_required=True,
            auth_mode=RemoteWorkerAuthMode.SIGNED_ENROLLMENT,
            enrollment_secret_name="SWITCHYARD_ENROLLMENT_SECRET",
        )
    )
    request = RemoteWorkerRegistrationRequest(
        worker_id="worker-signed",
        worker_name="remote-signed",
        backend_type=BackendType.VLLM_CUDA,
        model_identifier="meta-llama/Llama-3.1-8B-Instruct",
        serving_targets=["chat-shared"],
        endpoint=BackendNetworkEndpoint(
            base_url="https://worker-signed.internal",
            transport=WorkerTransportType.HTTPS,
        ),
        capabilities=BackendCapabilities(
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM,
            device_class=DeviceClass.NVIDIA_GPU,
            model_ids=["meta-llama/Llama-3.1-8B-Instruct"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=True,
            concurrency_limit=8,
        ),
        device_class=DeviceClass.NVIDIA_GPU,
        lifecycle_state=WorkerLifecycleState.REGISTERING,
    )
    token = build_signed_enrollment_token(
        request=request,
        secret="secret-value",
        expires_at=datetime(2027, 3, 17, 0, 5, tzinfo=UTC),
    )

    response = client.post(
        "/internal/control-plane/remote-workers/register",
        headers={"x-switchyard-registration-token": token},
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json()["auth_mode"] == "signed_enrollment"
