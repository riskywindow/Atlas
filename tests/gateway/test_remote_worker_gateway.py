from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI

from switchyard.adapters.registry import AdapterRegistry
from switchyard.adapters.remote_worker import RemoteWorkerAdapter
from switchyard.config import (
    AppEnvironment,
    BackendInstanceConfig,
    GenerationDefaults,
    LocalModelConfig,
    Phase7ControlPlaneSettings,
    Settings,
    WarmupSettings,
)
from switchyard.gateway import create_app
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendType,
    DeviceClass,
    EngineType,
    WorkerTransportType,
)
from switchyard.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    FinishReason,
    UsageStats,
)
from switchyard.schemas.worker import (
    WorkerCapabilitiesResponse,
    WorkerGenerateResponse,
    WorkerHealthResponse,
    WorkerReadinessResponse,
)
from switchyard.telemetry import configure_telemetry


def _build_remote_model_config() -> LocalModelConfig:
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


def _build_worker_app() -> FastAPI:
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
        return WorkerHealthResponse(worker_name="worker-1", health=health).model_dump(mode="json")

    @app.get("/internal/worker/ready")
    async def ready() -> dict[str, object]:
        return WorkerReadinessResponse(
            worker_name="worker-1",
            ready=True,
            health=health,
        ).model_dump(mode="json")

    @app.get("/internal/worker/capabilities")
    async def worker_capabilities() -> dict[str, object]:
        return WorkerCapabilitiesResponse(
            worker_name="worker-1",
            backend_type=BackendType.MOCK,
            capabilities=capabilities,
        ).model_dump(mode="json")

    @app.post("/internal/worker/generate")
    async def generate() -> dict[str, object]:
        return WorkerGenerateResponse(
            worker_name="worker-1",
            response=ChatCompletionResponse(
                id="remote-gateway-1",
                created_at=datetime(2026, 3, 17, tzinfo=UTC),
                model="chat-shared",
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role=ChatRole.ASSISTANT, content="gateway via remote"),
                        finish_reason=FinishReason.STOP,
                    )
                ],
                usage=UsageStats(prompt_tokens=3, completion_tokens=3, total_tokens=6),
                backend_name="worker-1",
            ),
        ).model_dump(mode="json")

    return app


@pytest.mark.asyncio
async def test_gateway_routes_to_remote_worker_adapter() -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(RemoteWorkerAdapter(_build_remote_model_config(), client=worker_client))
    telemetry = configure_telemetry("switchyard-remote-worker-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        response = await gateway_client.post(
            "/v1/chat/completions",
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="hello")],
            ).model_dump(mode="json"),
        )
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend_name"] == "remote-worker:remote-chat"
    assert telemetry.state.backend_execution_records[0].execution_mode == "remote_worker"
    assert telemetry.state.backend_execution_records[0].worker_transport == "http"


@pytest.mark.asyncio
async def test_admin_runtime_includes_instance_inventory_for_remote_workers() -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(RemoteWorkerAdapter(_build_remote_model_config(), client=worker_client))
    app = create_app(
        registry=registry,
        telemetry=configure_telemetry("switchyard-remote-worker-admin-test"),
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        response = await gateway_client.get("/admin/runtime")
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert response.status_code == 200
    payload = response.json()
    assert payload["backends"][0]["instances"][0]["instance_id"] == "worker-1"
    assert payload["backends"][0]["instances"][0]["transport"] == "http"
    assert payload["backends"][0]["instances"][0]["device_class"] == "remote"
    assert payload["backends"][0]["instances"][0]["registration_state"] == "static"
    assert payload["backends"][0]["instances"][0]["health_state"] == "healthy"
    assert payload["hybrid_execution"]["remote_capable_backends"] == 1
    assert payload["hybrid_execution"]["healthy_remote_backends"] == 1
    assert payload["remote_workers"]["static_instance_count"] == 0


@pytest.mark.asyncio
async def test_admin_deployment_reports_runtime_diagnostics_for_remote_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(RemoteWorkerAdapter(_build_remote_model_config(), client=worker_client))
    app = create_app(
        registry=registry,
        telemetry=configure_telemetry("switchyard-remote-worker-diagnostics-test"),
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            local_models=(_build_remote_model_config(),),
            phase7=Phase7ControlPlaneSettings.model_validate(
                {
                    "hybrid_execution": {
                        "enabled": True,
                        "spillover_enabled": True,
                        "max_remote_share_percent": 25.0,
                        "remote_request_budget_per_minute": 60,
                        "allowed_remote_environments": ["staging"],
                    },
                    "remote_workers": {
                        "secure_registration_required": True,
                        "dynamic_registration_enabled": True,
                        "heartbeat_timeout_seconds": 45.0,
                        "registration_token_name": "SWITCHYARD_WORKER_TOKEN",
                    },
                }
            ),
        ),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    async def fake_probe_worker_health(**_: object) -> object:
        from switchyard.schemas.admin import DiagnosticStatus, ProbeKind, ProbeResult

        return ProbeResult(
            target="http://worker.internal/healthz",
            probe_kind=ProbeKind.HTTP,
            status=DiagnosticStatus.OK,
        )

    async def fake_probe_supporting_services(**_: object) -> list[object]:
        return []

    monkeypatch.setattr(
        "switchyard.diagnostics._probe_worker_health",
        fake_probe_worker_health,
    )
    monkeypatch.setattr(
        "switchyard.diagnostics._probe_supporting_services",
        fake_probe_supporting_services,
    )

    try:
        response = await gateway_client.get("/admin/deployment")
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostics_source"] == "runtime"
    assert payload["worker_deployments"][0]["alias"] == "remote-chat"
    assert payload["worker_deployments"][0]["configured_instances"][0]["probe"]["status"] == "ok"
    assert payload["runtime_backends"][0]["backend_name"] == "remote-worker:remote-chat"
    assert payload["hybrid_execution"]["enabled"] is True
    assert payload["hybrid_execution"]["remote_capable_backends"] == 1
    assert payload["hybrid_execution"]["max_remote_share_percent"] == 25.0
    assert payload["remote_workers"]["secure_registration_required"] is True
    assert payload["remote_workers"]["registration_token_name"] == "SWITCHYARD_WORKER_TOKEN"
