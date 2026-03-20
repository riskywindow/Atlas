from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from switchyard.adapters.registry import AdapterRegistry
from switchyard.adapters.remote_worker import RemoteWorkerAdapter
from switchyard.config import (
    AppEnvironment,
    BackendInstanceConfig,
    GenerationDefaults,
    HybridExecutionSettings,
    LocalModelConfig,
    Phase7ControlPlaneSettings,
    RemoteWorkerLifecycleSettings,
    Settings,
    WarmupSettings,
)
from switchyard.control.remote_workers import build_signed_enrollment_token
from switchyard.gateway.app import create_app
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendNetworkEndpoint,
    BackendType,
    DeviceClass,
    EngineType,
    GPUDeviceMetadata,
    RuntimeIdentity,
    WorkerLifecycleState,
    WorkerTransportType,
)
from switchyard.schemas.routing import RoutingPolicy
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


def _registration_payload() -> dict[str, Any]:
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
            "runtime": {
                "runtime_family": "vllm_cuda",
                "runtime_label": "vllm_cuda",
                "runtime_version": "0.6.5",
                "engine_type": "vllm_cuda",
                "backend_type": "vllm_cuda",
            },
            "gpu": {
                "accelerator_type": "cuda",
                "vendor": "nvidia",
                "model": "L4",
                "count": 1,
                "memory_per_device_gib": 24.0,
                "cuda_version": "12.4",
            },
            "model_ids": ["meta-llama/Llama-3.1-8B-Instruct"],
            "serving_targets": ["chat-shared"],
            "max_context_tokens": 8192,
            "supports_streaming": True,
            "concurrency_limit": 8,
        },
        "device_class": "nvidia_gpu",
        "runtime": {
            "runtime_family": "vllm_cuda",
            "runtime_label": "vllm_cuda",
            "runtime_version": "0.6.5",
            "engine_type": "vllm_cuda",
            "backend_type": "vllm_cuda",
        },
        "gpu": {
            "accelerator_type": "cuda",
            "vendor": "nvidia",
            "model": "L4",
            "count": 1,
            "memory_per_device_gib": 24.0,
            "cuda_version": "12.4",
        },
        "environment": "staging",
        "placement": {"provider": "aws", "region": "us-east-1"},
        "cost_profile": {"profile": "premium", "budget_bucket": "gpu-staging"},
        "lifecycle_state": "registering",
        "ready": False,
        "tags": ["remote", "gpu"],
        "metadata": {"instance_type": "g6e.xlarge"},
    }


def _remote_model_config() -> LocalModelConfig:
    return LocalModelConfig(
        alias="remote-chat",
        serving_target="chat-shared",
        model_identifier="mock-chat",
        backend_type=BackendType.MOCK,
        worker_transport=WorkerTransportType.HTTP,
        instances=(
            BackendInstanceConfig(
                instance_id="worker-1",
                base_url="http://worker.internal",
                transport=WorkerTransportType.HTTP,
            ),
        ),
        generation_defaults=GenerationDefaults(),
        warmup=WarmupSettings(),
    )


def _worker_app(*, failing_generate: bool = False) -> FastAPI:
    app = FastAPI()
    health = BackendHealth(
        state=BackendHealthState.HEALTHY,
        load_state=BackendLoadState.READY,
        warmed_models=["chat-shared"],
    )
    capabilities = BackendCapabilities(
        backend_type=BackendType.MOCK,
        engine_type=EngineType.MOCK,
        device_class=DeviceClass.REMOTE,
        model_ids=["chat-shared", "mock-chat"],
        serving_targets=["chat-shared"],
        max_context_tokens=8192,
        supports_streaming=False,
        concurrency_limit=2,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"worker_name": "worker-1", "health": health.model_dump(mode="json")}

    @app.get("/internal/worker/ready")
    async def ready() -> dict[str, object]:
        return {
            "worker_name": "worker-1",
            "ready": True,
            "health": health.model_dump(mode="json"),
        }

    @app.get("/internal/worker/capabilities")
    async def worker_capabilities() -> dict[str, object]:
        return {
            "worker_name": "worker-1",
            "backend_type": BackendType.MOCK.value,
            "capabilities": capabilities.model_dump(mode="json"),
        }

    @app.post("/internal/worker/generate")
    async def generate() -> Response:
        if failing_generate:
            return Response(
                content='{"detail":"remote transport failed"}',
                status_code=503,
                media_type="application/json",
            )
        return Response(
            content=(
                '{"worker_name":"worker-1","response":{"id":"remote-1","object":"chat.completion",'
                '"created_at":"2026-03-17T00:00:00Z","model":"chat-shared",'
                '"choices":[{"index":0,"message":{"role":"assistant","content":"remote ok"},'
                '"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5},'
                '"backend_name":"worker-1"}}'
            ),
            media_type="application/json",
        )

    return app


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
    assert payload["usable_worker_count"] == 1
    assert payload["workers"][0]["heartbeat_count"] == 3
    assert payload["workers"][0]["lifecycle_state"] == "ready"
    assert payload["workers"][0]["usable"] is True
    assert payload["workers"][0]["quarantined"] is False
    assert payload["workers"][0]["runtime"]["runtime_version"] == "0.6.5"
    assert payload["workers"][0]["gpu"]["model"] == "L4"
    assert "provider:aws" in payload["workers"][0]["tags"]
    assert "instance-type:g6e.xlarge" in payload["workers"][0]["tags"]
    assert payload["workers"][0]["metadata"]["gpu_type"] == "l4"
    assert payload["recent_events"][0]["event_type"] == "heartbeat"
    assert payload["recent_events"][-1]["event_type"] == "registered"

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


def test_remote_worker_admin_operator_controls_mutate_worker_posture() -> None:
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
        )
    )
    register = client.post(
        "/internal/control-plane/remote-workers/register",
        json=_registration_payload(),
    )
    assert register.status_code == 200

    drain = client.post(
        "/admin/remote-workers/worker-1/drain",
        json={"reason": "rotate node"},
    )
    assert drain.status_code == 200
    assert drain.json()["lifecycle_state"] == "draining"

    quarantine = client.post(
        "/admin/remote-workers/worker-1/quarantine",
        json={"enabled": True, "reason": "transport instability"},
    )
    assert quarantine.status_code == 200
    assert quarantine.json()["lifecycle_state"] == "unhealthy"

    canary_only = client.post(
        "/admin/remote-workers/worker-1/canary-only",
        json={"enabled": True, "reason": "limit blast radius"},
    )
    assert canary_only.status_code == 200

    snapshot = client.get("/admin/remote-workers")
    assert snapshot.status_code == 200
    tags = snapshot.json()["workers"][0]["instance"]["tags"]
    assert "quarantined" in tags
    assert "canary-only" in tags


@pytest.mark.asyncio
async def test_admin_hybrid_endpoints_expose_remote_controls_and_transport_errors() -> None:
    worker_app = _worker_app(failing_generate=True)
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(RemoteWorkerAdapter(_remote_model_config(), client=worker_client))
    app = create_app(
        registry=registry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            local_models=(_remote_model_config(),),
            default_routing_policy=RoutingPolicy.BURST_TO_REMOTE,
            phase7=Phase7ControlPlaneSettings(
                hybrid_execution=HybridExecutionSettings(
                    enabled=True,
                    spillover_enabled=True,
                    remote_request_budget_per_minute=2,
                ),
                remote_workers=RemoteWorkerLifecycleSettings(
                    dynamic_registration_enabled=True,
                ),
            ),
        )
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        failure = await gateway_client.post(
            "/v1/chat/completions",
            headers={
                "x-request-id": "req-remote-failure",
                "x-switchyard-routing-policy": "burst_to_remote",
            },
            json={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "force remote"}],
            },
        )
        hybrid = await gateway_client.get("/admin/hybrid")
        routes = await gateway_client.get("/admin/hybrid/routes")
        disabled = await gateway_client.post(
            "/admin/hybrid/remote-enabled",
            json={"enabled": False, "reason": "incident response"},
        )
        reset = await gateway_client.post("/admin/hybrid/budget/reset")
        runtime = await gateway_client.get("/admin/runtime")
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert failure.status_code == 503
    assert hybrid.status_code == 200
    hybrid_payload = hybrid.json()
    assert hybrid_payload["recent_route_example_count"] == 1
    assert hybrid_payload["recent_placement_distribution"]["remote_count"] == 1
    assert hybrid_payload["recent_cloud_evidence"]["estimated_placement_count"] == 0
    assert hybrid_payload["recent_cloud_evidence"]["estimated_cost_count"] == 0
    assert hybrid_payload["recent_cloud_evidence"]["remote_provider_counts"] == {}
    assert hybrid_payload["recent_cloud_evidence"]["observed_budget_bucket_counts"] == {}
    assert hybrid_payload["recent_cloud_evidence"]["estimated_budget_bucket_counts"] == {}
    assert hybrid_payload["recent_cloud_evidence"]["total_observed_relative_cost_index"] is None
    assert hybrid_payload["recent_route_examples"][0]["request_id"] == "req-remote-failure"
    assert hybrid_payload["recent_route_examples"][0]["execution_path"] == "remote"
    assert hybrid_payload["recent_route_examples"][0]["remote_candidate_count"] == 1
    assert hybrid_payload["recent_route_examples"][0]["placement_provider"] is None
    assert hybrid_payload["recent_route_examples"][0]["placement_evidence_source"] is None
    assert hybrid_payload["recent_route_examples"][0]["budget_bucket"] is None
    assert hybrid_payload["recent_route_examples"][0]["relative_cost_index"] is None
    assert hybrid_payload["recent_remote_transport_errors"][0]["request_id"] == "req-remote-failure"
    assert hybrid_payload["recent_remote_transport_errors"][0]["cooldown_triggered"] is False

    assert routes.status_code == 200
    assert routes.json()[0]["request_id"] == "req-remote-failure"

    assert disabled.status_code == 200
    assert disabled.json()["remote_enabled_override"] is False
    assert disabled.json()["remote_effectively_enabled"] is False

    assert reset.status_code == 200
    assert reset.json()["remote_budget_requests_used"] == 0
    assert reset.json()["remote_budget_requests_remaining"] == 2

    assert runtime.status_code == 200
    runtime_payload = runtime.json()
    assert runtime_payload["hybrid_execution"]["enabled"] is False
    assert runtime_payload["hybrid_execution"]["remote_budget_requests_used"] == 0
    assert runtime_payload["hybrid_operator"]["remote_enabled_override"] is False
    assert (
        runtime_payload["hybrid_operator"]["recent_cloud_evidence"][
            "remote_provider_counts"
        ]
        == {}
    )
    assert (
        runtime_payload["hybrid_operator"]["recent_remote_transport_errors"][0][
            "backend_name"
        ]
        == "remote-worker:remote-chat"
    )


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


def test_remote_worker_registration_quarantines_incompatible_runtime_version() -> None:
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
        )
    )
    payload = _registration_payload()
    runtime = dict(payload["runtime"])
    runtime["runtime_version"] = "0.5.9"
    payload["runtime"] = runtime
    capabilities = dict(payload["capabilities"])
    capabilities_runtime = dict(capabilities["runtime"])
    capabilities_runtime["runtime_version"] = "0.5.9"
    capabilities["runtime"] = capabilities_runtime
    payload["capabilities"] = capabilities

    response = client.post(
        "/internal/control-plane/remote-workers/register",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["lifecycle_state"] == "unhealthy"
    assert response.json()["quarantined"] is True
    assert response.json()["usable"] is False
    assert "minimum supported is 0.6.0" in (response.json()["detail"] or "")

    snapshot = client.get("/admin/remote-workers").json()
    assert snapshot["quarantined_worker_count"] == 1
    assert snapshot["usable_worker_count"] == 0
    assert snapshot["workers"][0]["quarantined"] is True
    assert snapshot["workers"][0]["eligibility_reasons"] == [
        "worker reported unsupported vllm_cuda runtime_version 0.5.9; minimum supported is 0.6.0"
    ]
    assert snapshot["recent_events"][0]["event_type"] == "quarantined"


def test_remote_worker_registration_rejects_bad_capability_inventory() -> None:
    client = _app(
        remote_worker_settings=RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
        )
    )
    payload = _registration_payload()
    capabilities = dict(payload["capabilities"])
    capabilities["model_ids"] = ["different-model"]
    payload["capabilities"] = capabilities

    response = client.post(
        "/internal/control-plane/remote-workers/register",
        json=payload,
    )

    assert response.status_code == 400
    assert "model_identifier is not present" in response.json()["message"]

    snapshot = client.get("/admin/remote-workers").json()
    assert snapshot["worker_count"] == 0
    assert snapshot["recent_events"][0]["event_type"] == "registration_rejected"


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
            runtime=RuntimeIdentity(
                runtime_family="vllm_cuda",
                runtime_label="vllm_cuda",
                runtime_version="0.6.5",
                engine_type=EngineType.VLLM_CUDA,
                backend_type=BackendType.VLLM_CUDA,
            ),
            gpu=GPUDeviceMetadata(
                accelerator_type="cuda",
                vendor="nvidia",
                model="L4",
                count=1,
                memory_per_device_gib=24.0,
                cuda_version="12.4",
            ),
            model_ids=["meta-llama/Llama-3.1-8B-Instruct"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=True,
            concurrency_limit=8,
        ),
        device_class=DeviceClass.NVIDIA_GPU,
        runtime=RuntimeIdentity(
            runtime_family="vllm_cuda",
            runtime_label="vllm_cuda",
            runtime_version="0.6.5",
            engine_type=EngineType.VLLM_CUDA,
            backend_type=BackendType.VLLM_CUDA,
        ),
        gpu=GPUDeviceMetadata(
            accelerator_type="cuda",
            vendor="nvidia",
            model="L4",
            count=1,
            memory_per_device_gib=24.0,
            cuda_version="12.4",
        ),
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
