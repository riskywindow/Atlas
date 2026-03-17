from __future__ import annotations

from datetime import UTC, datetime

import pytest

from switchyard.config import (
    BackendInstanceConfig,
    DeploymentLayerConfig,
    DeploymentTopologySettings,
    GenerationDefaults,
    LocalModelConfig,
    Settings,
    WarmupSettings,
)
from switchyard.diagnostics import collect_deployment_diagnostics
from switchyard.schemas.admin import DiagnosticStatus, ProbeKind, ProbeResult
from switchyard.schemas.backend import (
    BackendImageMetadata,
    BackendType,
    DeploymentProfile,
    DeviceClass,
    WorkerTransportType,
)


def _settings_with_remote_worker() -> Settings:
    return Settings(
        topology=DeploymentTopologySettings(
            active_environment="compose-smoke",
            deployment_profile=DeploymentProfile.COMPOSE,
            control_plane_image=BackendImageMetadata(image_tag="switchyard/control-plane:dev"),
            layers=(
                DeploymentLayerConfig(
                    name="compose-smoke",
                    deployment_profile=DeploymentProfile.COMPOSE,
                    gateway_base_url="http://switchyard-gateway:8000",
                ),
            ),
        ),
        local_models=(
            LocalModelConfig(
                alias="chat-smoke",
                serving_target="chat-shared",
                environment="compose-smoke",
                deployment_profile=DeploymentProfile.COMPOSE,
                model_identifier="mock-chat",
                backend_type=BackendType.MOCK,
                worker_transport=WorkerTransportType.HTTP,
                instances=(
                    BackendInstanceConfig(
                        instance_id="worker-1",
                        base_url="http://host.docker.internal:8101",
                        transport=WorkerTransportType.HTTP,
                        device_class=DeviceClass.CPU,
                        image_metadata=BackendImageMetadata(image_tag="worker:dev"),
                    ),
                ),
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_collect_deployment_diagnostics_reports_worker_and_supporting_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_with_remote_worker()

    async def fake_probe_worker_health(**_: object) -> ProbeResult:
        return ProbeResult(
            target="http://host.docker.internal:8101/healthz",
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

    diagnostics = await collect_deployment_diagnostics(settings)

    assert diagnostics.diagnostics_source == "config_preflight"
    assert diagnostics.effective_deployment_profile is DeploymentProfile.COMPOSE
    assert diagnostics.gateway_base_url == "http://switchyard-gateway:8000"
    assert (
        diagnostics.worker_deployments[0].configured_instances[0].probe.status
        is DiagnosticStatus.OK
    )
    assert diagnostics.worker_deployments[0].configured_instances[0].image_metadata is not None
    assert diagnostics.runtime_backends == []
    assert any("config preflight" in note for note in diagnostics.notes)


@pytest.mark.asyncio
async def test_collect_deployment_diagnostics_does_not_guess_compose_services_for_local_preflight(
) -> None:
    settings = _settings_with_remote_worker()

    diagnostics = await collect_deployment_diagnostics(settings)

    assert diagnostics.supporting_services == []
    assert any("no supporting service endpoints" in note for note in diagnostics.notes)


@pytest.mark.asyncio
async def test_collect_deployment_diagnostics_marks_in_process_workers_not_verifiable() -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="chat-local",
                model_identifier="mock-chat",
                backend_type=BackendType.MOCK,
                worker_transport=WorkerTransportType.IN_PROCESS,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        ),
    )

    diagnostics = await collect_deployment_diagnostics(settings)

    instance = diagnostics.worker_deployments[0].configured_instances[0]
    assert instance.probe.status is DiagnosticStatus.NOT_VERIFIABLE
    assert "in-process workers" in (instance.probe.detail or "")


@pytest.mark.asyncio
async def test_collect_deployment_diagnostics_carries_runtime_backend_inventory() -> None:
    settings = _settings_with_remote_worker()
    runtime_backend = {
        "backend_name": "remote-worker:chat-smoke",
        "backend_type": "mock",
        "health_state": "healthy",
        "load_state": "ready",
        "latency_ms": 1.0,
        "active_requests": 0,
        "queue_depth": 0,
        "circuit_open": False,
        "circuit_reason": None,
        "instances": [
            {
                "instance_id": "worker-1",
                "source_of_truth": "static_config",
                "endpoint": "http://host.docker.internal:8101",
                "transport": "http",
                "health_state": "healthy",
                "load_state": "ready",
                "last_seen_at": datetime(2026, 3, 17, tzinfo=UTC),
                "tags": ["local"],
            }
        ],
    }

    async def fake_runtime_backends(_: object) -> list[object]:
        from switchyard.schemas.admin import BackendRuntimeSummary

        return [BackendRuntimeSummary.model_validate(runtime_backend)]

    async def fake_probe_supporting_services(**_: object) -> list[object]:
        return []

    async def fake_probe_worker_health(**_: object) -> ProbeResult:
        return ProbeResult(
            target="http://host.docker.internal:8101/healthz",
            probe_kind=ProbeKind.HTTP,
            status=DiagnosticStatus.OK,
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "switchyard.diagnostics.collect_runtime_backend_summaries",
        fake_runtime_backends,
    )
    monkeypatch.setattr(
        "switchyard.diagnostics._probe_supporting_services",
        fake_probe_supporting_services,
    )
    monkeypatch.setattr(
        "switchyard.diagnostics._probe_worker_health",
        fake_probe_worker_health,
    )
    try:
        diagnostics = await collect_deployment_diagnostics(settings, registry=object())  # type: ignore[arg-type]
    finally:
        monkeypatch.undo()

    instance = diagnostics.worker_deployments[0].configured_instances[0]
    assert diagnostics.diagnostics_source == "runtime"
    assert diagnostics.runtime_backends[0].backend_name == "remote-worker:chat-smoke"
    assert instance.runtime_health_state == "healthy"
