from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.adapters.remote_worker import RemoteWorkerAdapter
from switchyard.config import (
    AppEnvironment,
    BackendInstanceConfig,
    CanaryRoutingSettings,
    GenerationDefaults,
    LocalModelConfig,
    Phase4ControlPlaneSettings,
    Phase7ControlPlaneSettings,
    SessionAffinitySettings,
    Settings,
    ShadowRoutingSettings,
    WarmupSettings,
)
from switchyard.gateway import create_app
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendType,
    CloudPlacementMetadata,
    CostBudgetProfile,
    CostProfileClass,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
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
from switchyard.schemas.routing import CanaryPolicy, ShadowPolicy, WeightedBackendAllocation
from switchyard.schemas.worker import (
    WorkerCapabilitiesResponse,
    WorkerGenerateResponse,
    WorkerHealthResponse,
    WorkerReadinessResponse,
)
from switchyard.telemetry import configure_telemetry


def _build_remote_model_config(
    *,
    with_observed_cloud_metadata: bool = False,
    canary_only: bool = False,
) -> LocalModelConfig:
    return LocalModelConfig(
        alias="remote-chat",
        serving_target="chat-shared",
        model_identifier="mock-chat",
        backend_type=BackendType.MOCK,
        worker_transport=WorkerTransportType.HTTP,
        execution_mode=ExecutionModeLabel.REMOTE_WORKER,
        instances=(
            BackendInstanceConfig(
                instance_id="worker-1",
                base_url="http://worker.internal",
                transport=WorkerTransportType.HTTP,
                execution_mode=ExecutionModeLabel.REMOTE_WORKER,
                placement=(
                    CloudPlacementMetadata(
                        provider="aws",
                        region="us-east-1",
                        zone="us-east-1b",
                    )
                    if with_observed_cloud_metadata
                    else CloudPlacementMetadata()
                ),
                cost_profile=(
                    CostBudgetProfile(
                        profile=CostProfileClass.PREMIUM,
                        budget_bucket="gpu-canary",
                        relative_cost_index=0.73,
                        currency="usd",
                    )
                    if with_observed_cloud_metadata
                    else CostBudgetProfile()
                ),
                tags=("canary-only",) if canary_only else (),
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


def _build_local_adapter(*, latency_ms: float = 1.0) -> MockBackendAdapter:
    return MockBackendAdapter(
        name="local-chat",
        simulated_latency_ms=latency_ms,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            engine_type=EngineType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["chat-shared"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=False,
            concurrency_limit=4,
            default_model="chat-shared",
            model_aliases={"chat-shared": "chat-shared"},
        ),
    )


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
async def test_alias_override_controls_can_pin_or_disable_remote_backend() -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(_build_local_adapter())
    registry.register(RemoteWorkerAdapter(_build_remote_model_config(), client=worker_client))
    app = create_app(
        registry=registry,
        telemetry=configure_telemetry("switchyard-alias-override-test"),
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            phase4=Phase4ControlPlaneSettings(
                session_affinity=SessionAffinitySettings(
                    enabled=True,
                    ttl_seconds=120.0,
                    max_sessions=32,
                )
            ),
        ),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        pin = await gateway_client.post(
            "/admin/hybrid/aliases/chat-shared/override",
            json={
                "pinned_backend": "remote-worker:remote-chat",
                "reason": "exercise the real cloud path",
            },
        )
        pinned = await gateway_client.post(
            "/v1/chat/completions",
            headers={"x-switchyard-session-id": "session-override"},
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="hello")],
            ).model_dump(mode="json"),
        )
        runtime = await gateway_client.get("/admin/runtime")
        disable = await gateway_client.post(
            "/admin/hybrid/aliases/chat-shared/override",
            json={
                "disabled_backends": ["remote-worker:remote-chat"],
                "reason": "rollback the cloud worker for this alias",
            },
        )
        disabled = await gateway_client.post(
            "/v1/chat/completions",
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="hello again")],
            ).model_dump(mode="json"),
        )
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert pin.status_code == 200
    assert disable.status_code == 200
    assert pinned.status_code == 200
    assert pinned.json()["backend_name"] == "remote-worker:remote-chat"
    pinned_route = json.loads(pinned.headers["x-switchyard-route-decision"])
    assert pinned_route["annotations"]["operator_override_applied"] is True
    assert pinned_route["annotations"]["cloud_routing_reason"] == "operator_override"
    assert "operator_override" in pinned_route["explanation"]["selection_reason_codes"]
    assert "session affinity bypassed by internal backend pin" in pinned_route["annotations"][
        "notes"
    ]
    assert runtime.status_code == 200
    runtime_payload = runtime.json()
    assert runtime_payload["session_affinity"]["active_bindings"] == 0
    assert runtime_payload["hybrid_operator"]["recent_route_examples"][0][
        "operator_override_applied"
    ] is True
    assert (
        runtime_payload["hybrid_operator"]["recent_route_examples"][0]["cloud_routing_reason"]
        == "operator_override"
    )
    assert disabled.status_code == 200
    assert disabled.json()["backend_name"] == "local-chat"


@pytest.mark.asyncio
async def test_hybrid_diagnostics_surface_shadow_validation_for_remote_target() -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(_build_local_adapter())
    registry.register(RemoteWorkerAdapter(_build_remote_model_config(), client=worker_client))
    app = create_app(
        registry=registry,
        telemetry=configure_telemetry("switchyard-remote-shadow-validation-test"),
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            phase4=Phase4ControlPlaneSettings(
                shadow_routing=ShadowRoutingSettings(
                    enabled=True,
                    policies=(
                        ShadowPolicy(
                            policy_name="remote-shadow-validation",
                            enabled=True,
                            serving_target="chat-shared",
                            target_backend="remote-worker:remote-chat",
                            sampling_rate=1.0,
                        ),
                    ),
                )
            ),
        ),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        completion = await gateway_client.post(
            "/v1/chat/completions",
            headers={"x-request-id": "req-shadow-remote-validation"},
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="validate remotely first")],
            ).model_dump(mode="json"),
        )
        hybrid = await gateway_client.get("/admin/hybrid")
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert completion.status_code == 200
    assert completion.json()["backend_name"] == "local-chat"
    route_header = json.loads(completion.headers["x-switchyard-route-decision"])
    assert route_header["annotations"]["shadow_disposition"] == "shadowed"
    assert hybrid.status_code == 200
    route_example = hybrid.json()["recent_route_examples"][0]
    assert route_example["request_id"] == "req-shadow-remote-validation"
    assert route_example["shadow_disposition"] == "shadowed"
    assert route_example["shadow_target_backend"] == "remote-worker:remote-chat"


@pytest.mark.asyncio
async def test_hybrid_admin_reports_observed_cloud_evidence_for_remote_execution() -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(
        RemoteWorkerAdapter(
            _build_remote_model_config(with_observed_cloud_metadata=True),
            client=worker_client,
        )
    )
    app = create_app(
        registry=registry,
        telemetry=configure_telemetry("switchyard-remote-worker-observed-cloud-test"),
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        completion = await gateway_client.post(
            "/v1/chat/completions",
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="hello")],
            ).model_dump(mode="json"),
        )
        hybrid = await gateway_client.get("/admin/hybrid")
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert completion.status_code == 200
    assert hybrid.status_code == 200
    payload = hybrid.json()
    assert payload["recent_cloud_evidence"]["observed_placement_count"] == 1
    assert payload["recent_cloud_evidence"]["observed_cost_count"] == 1
    assert payload["recent_cloud_evidence"]["estimated_placement_count"] == 0
    assert payload["recent_cloud_evidence"]["estimated_cost_count"] == 0
    assert payload["recent_cloud_evidence"]["observed_budget_bucket_counts"] == {
        "gpu-canary": 1
    }
    assert payload["recent_cloud_evidence"]["total_observed_relative_cost_index"] == 0.73
    assert payload["recent_route_examples"][0]["placement_provider"] == "aws"
    assert payload["recent_route_examples"][0]["placement_zone"] == "us-east-1b"
    assert payload["recent_route_examples"][0]["budget_bucket"] == "gpu-canary"
    assert payload["recent_route_examples"][0]["relative_cost_index"] == 0.73
    assert payload["recent_route_examples"][0]["placement_evidence_source"] == "observed_runtime"
    assert payload["recent_route_examples"][0]["cost_evidence_source"] == "observed_runtime"


@pytest.mark.asyncio
async def test_canary_only_remote_backend_requires_explicit_cloud_rollout() -> None:
    worker_app = _build_worker_app()
    worker_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app),
        base_url="http://worker.internal",
    )
    registry = AdapterRegistry()
    registry.register(_build_local_adapter(latency_ms=120.0))
    registry.register(
        RemoteWorkerAdapter(
            _build_remote_model_config(canary_only=True),
            client=worker_client,
        )
    )
    app = create_app(
        registry=registry,
        telemetry=configure_telemetry("switchyard-cloud-rollout-test"),
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            phase4=Phase4ControlPlaneSettings(
                canary_routing=CanaryRoutingSettings(
                    enabled=True,
                    policies=(
                        CanaryPolicy(
                            policy_name="cloud-remote-rollout",
                            serving_target="chat-shared",
                            enabled=True,
                            baseline_backend="local-chat",
                            allocations=[
                                WeightedBackendAllocation(
                                    backend_name="remote-worker:remote-chat",
                                    percentage=100.0,
                                ),
                            ],
                        ),
                    ),
                )
            ),
        ),
    )
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )

    try:
        baseline = await gateway_client.post(
            "/v1/chat/completions",
            headers={
                "x-request-id": "req-cloud-rollout-baseline",
                "x-switchyard-routing-policy": "burst_to_remote",
            },
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="hello")],
            ).model_dump(mode="json"),
        )
        rollout = await gateway_client.post(
            "/admin/hybrid/cloud-rollout",
            json={"enabled": True, "canary_percentage": 100.0},
        )
        canary = await gateway_client.post(
            "/v1/chat/completions",
            headers={
                "x-request-id": "req-cloud-rollout-canary",
                "x-switchyard-routing-policy": "burst_to_remote",
            },
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="hello again")],
            ).model_dump(mode="json"),
        )
        kill_switch = await gateway_client.post(
            "/admin/hybrid/cloud-rollout",
            json={"kill_switch_enabled": True},
        )
        rolled_back = await gateway_client.post(
            "/v1/chat/completions",
            headers={
                "x-request-id": "req-cloud-rollout-rollback",
                "x-switchyard-routing-policy": "burst_to_remote",
            },
            json=ChatCompletionRequest(
                model="chat-shared",
                messages=[ChatMessage(role=ChatRole.USER, content="rollback")],
            ).model_dump(mode="json"),
        )
        hybrid = await gateway_client.get("/admin/hybrid")
    finally:
        await gateway_client.aclose()
        await worker_client.aclose()

    assert baseline.status_code == 200
    assert baseline.json()["backend_name"] == "local-chat"
    baseline_route = json.loads(baseline.headers["x-switchyard-route-decision"])
    assert baseline_route["protected_backends"] == {
        "remote-worker:remote-chat": "cloud rollout is disabled for canary-only backend"
    }
    assert baseline_route["annotations"]["cloud_rollout_disposition"] == "disabled"
    assert rollout.status_code == 200
    assert rollout.json()["enabled"] is True
    assert rollout.json()["canary_percentage"] == 100.0
    assert canary.status_code == 200
    assert canary.json()["backend_name"] == "remote-worker:remote-chat"
    canary_route = json.loads(canary.headers["x-switchyard-route-decision"])
    assert canary_route["protected_backends"] == {}
    assert "remote-worker:remote-chat" in canary_route["considered_backends"]
    assert canary_route["annotations"]["cloud_rollout_disposition"] == "explicit_canary"
    assert canary_route["annotations"]["cloud_routing_reason"] == "cloud_rollout"
    assert "cloud_rollout" in canary_route["explanation"]["selection_reason_codes"]
    assert kill_switch.status_code == 200
    assert kill_switch.json()["kill_switch_enabled"] is True
    assert rolled_back.status_code == 200
    assert rolled_back.json()["backend_name"] == "local-chat"
    rolled_back_route = json.loads(rolled_back.headers["x-switchyard-route-decision"])
    assert rolled_back_route["protected_backends"] == {
        "remote-worker:remote-chat": "cloud rollout kill switch blocked canary-only backend"
    }
    assert rolled_back_route["annotations"]["cloud_rollout_disposition"] == "kill_switch"
    assert hybrid.status_code == 200
    assert hybrid.json()["cloud_rollout"]["recent_allowed_count"] >= 1
    assert hybrid.json()["cloud_rollout"]["recent_blocked_count"] >= 2
    assert hybrid.json()["cloud_rollout"]["kill_switch_enabled"] is True
    route_examples = {
        example["request_id"]: example for example in hybrid.json()["recent_route_examples"]
    }
    assert route_examples["req-cloud-rollout-canary"]["cloud_rollout_disposition"] == (
        "explicit_canary"
    )
    assert route_examples["req-cloud-rollout-canary"]["cloud_routing_reason"] == "cloud_rollout"
    assert route_examples["req-cloud-rollout-rollback"]["cloud_rollout_disposition"] == (
        "kill_switch"
    )


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
