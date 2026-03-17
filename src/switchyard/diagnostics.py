"""Deployment diagnostics for local and deployed control-plane workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

import httpx

from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import DeploymentLayerConfig, LocalModelConfig, Settings
from switchyard.schemas.admin import (
    BackendInstanceRuntimeSummary,
    BackendRuntimeSummary,
    DeploymentDiagnosticsResponse,
    DiagnosticStatus,
    ProbeKind,
    ProbeResult,
    SupportingServiceDiagnostic,
    WorkerDeploymentDiagnostic,
    WorkerInstanceDiagnostic,
)
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendImageMetadata,
    BackendLoadState,
    BackendType,
    DeploymentProfile,
    WorkerTransportType,
)

_DEFAULT_HTTP_TIMEOUT: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class SupportServiceTarget:
    """One supporting service endpoint to probe."""

    service_name: str
    target: str
    probe_kind: ProbeKind


async def collect_deployment_diagnostics(
    settings: Settings,
    *,
    registry: AdapterRegistry | None = None,
) -> DeploymentDiagnosticsResponse:
    """Collect a deployment-aware diagnostics report from config and optional runtime."""

    active_layer = _active_layer(settings)
    runtime_backends = (
        await collect_runtime_backend_summaries(registry) if registry is not None else []
    )
    runtime_by_deployment = {backend.backend_name: backend for backend in runtime_backends}
    worker_deployments = [
        await _diagnose_worker_deployment(
            model_config=model_config,
            runtime_backend=runtime_by_deployment.get(_deployment_name_for_model(model_config)),
        )
        for model_config in settings.local_models
    ]
    supporting_services = await _probe_supporting_services(
        settings=settings,
        active_layer=active_layer,
        allow_inferred_targets=registry is not None,
    )
    notes: list[str] = []
    if registry is None:
        notes.append("runtime backend inventory was not collected; this is a config preflight view")
    if active_layer is None:
        notes.append("no active topology layer matched settings.topology.active_environment")
    if not settings.local_models:
        notes.append("no local_models are configured")
    if not supporting_services:
        notes.append("no supporting service endpoints were configured or inferred for probing")
    return DeploymentDiagnosticsResponse(
        diagnostics_source="runtime" if registry is not None else "config_preflight",
        active_environment=settings.topology.active_environment,
        effective_deployment_profile=settings.topology.deployment_profile,
        gateway_base_url=active_layer.gateway_base_url if active_layer is not None else None,
        active_layer_name=active_layer.name if active_layer is not None else None,
        control_plane_image=settings.topology.control_plane_image,
        worker_deployments=worker_deployments,
        runtime_backends=runtime_backends,
        supporting_services=supporting_services,
        notes=notes,
    )


async def collect_runtime_backend_summaries(
    registry: AdapterRegistry,
) -> list[BackendRuntimeSummary]:
    """Build the runtime backend summary used by admin inspection and diagnostics."""

    summaries: list[BackendRuntimeSummary] = []
    for adapter in registry.list():
        status_snapshot = await adapter.status()
        summaries.append(
            BackendRuntimeSummary(
                backend_name=status_snapshot.name,
                backend_type=status_snapshot.capabilities.backend_type.value,
                health_state=status_snapshot.health.state.value,
                load_state=status_snapshot.health.load_state.value,
                latency_ms=status_snapshot.health.latency_ms,
                active_requests=status_snapshot.active_requests,
                queue_depth=status_snapshot.queue_depth,
                circuit_open=status_snapshot.health.circuit_open,
                circuit_reason=status_snapshot.health.circuit_reason,
                instances=[
                    BackendInstanceRuntimeSummary(
                        instance_id=instance.instance_id,
                        source_of_truth=instance.source_of_truth.value,
                        endpoint=instance.endpoint.base_url,
                        transport=instance.endpoint.transport.value,
                        health_state=(
                            instance.health.state.value
                            if instance.health is not None
                            else BackendHealthState.UNAVAILABLE.value
                        ),
                        load_state=(
                            instance.health.load_state.value
                            if instance.health is not None
                            else BackendLoadState.COLD.value
                        ),
                        last_seen_at=instance.last_seen_at,
                        tags=list(instance.tags),
                    )
                    for instance in status_snapshot.instance_inventory
                ],
            )
        )
    return summaries


async def _diagnose_worker_deployment(
    *,
    model_config: LocalModelConfig,
    runtime_backend: BackendRuntimeSummary | None,
) -> WorkerDeploymentDiagnostic:
    serving_target = model_config.serving_target or model_config.alias
    image_metadata = _image_metadata_for_model(model_config)
    runtime_instances = {
        instance.instance_id: instance
        for instance in (runtime_backend.instances if runtime_backend is not None else [])
    }
    configured_instances: list[WorkerInstanceDiagnostic] = []
    if model_config.instances:
        for instance in model_config.instances:
            runtime_instance = runtime_instances.get(instance.instance_id)
            configured_instances.append(
                await _diagnose_worker_instance(
                    model_config=model_config,
                    runtime_instance=runtime_instance,
                    instance_id=instance.instance_id,
                    endpoint=instance.base_url,
                    transport=instance.transport.value,
                    health_path=instance.health_path,
                    source_of_truth=instance.source_of_truth.value,
                    tags=list(instance.tags),
                    image_metadata=instance.image_metadata,
                    connect_timeout_seconds=instance.connect_timeout_seconds,
                    request_timeout_seconds=instance.request_timeout_seconds,
                )
            )
    else:
        configured_instances.append(
            await _diagnose_worker_instance(
                model_config=model_config,
                runtime_instance=None,
                instance_id=f"{model_config.alias}-in-process",
                endpoint=None,
                transport=model_config.worker_transport.value,
                health_path=None,
                source_of_truth=None,
                tags=[],
                image_metadata=None,
                connect_timeout_seconds=_DEFAULT_HTTP_TIMEOUT,
                request_timeout_seconds=_DEFAULT_HTTP_TIMEOUT,
            )
        )
    return WorkerDeploymentDiagnostic(
        alias=model_config.alias,
        serving_target=serving_target,
        backend_type=model_config.backend_type.value,
        deployment_profile=model_config.deployment_profile,
        worker_transport=model_config.worker_transport.value,
        model_identifier=model_config.model_identifier,
        image_metadata=image_metadata,
        build_metadata=dict(model_config.build_metadata),
        configured_instances=configured_instances,
    )


async def _diagnose_worker_instance(
    *,
    model_config: LocalModelConfig,
    runtime_instance: BackendInstanceRuntimeSummary | None,
    instance_id: str,
    endpoint: str | None,
    transport: str,
    health_path: str | None,
    source_of_truth: str | None,
    tags: list[str],
    image_metadata: BackendImageMetadata | None,
    connect_timeout_seconds: float,
    request_timeout_seconds: float,
) -> WorkerInstanceDiagnostic:
    if endpoint is None or health_path is None:
        probe = ProbeResult(
            target=f"{model_config.alias}:{transport}",
            probe_kind=ProbeKind.CONFIG,
            status=DiagnosticStatus.NOT_VERIFIABLE,
            detail="in-process workers do not expose a network health endpoint",
        )
    elif model_config.worker_transport is WorkerTransportType.IN_PROCESS:
        probe = ProbeResult(
            target=f"{endpoint.rstrip('/')}{health_path}",
            probe_kind=ProbeKind.CONFIG,
            status=DiagnosticStatus.NOT_VERIFIABLE,
            detail="worker transport is in_process even though a static endpoint is present",
        )
    else:
        probe = await _probe_worker_health(
            endpoint=endpoint,
            health_path=health_path,
            connect_timeout_seconds=connect_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    return WorkerInstanceDiagnostic(
        instance_id=instance_id,
        endpoint=endpoint,
        transport=transport,
        source_of_truth=source_of_truth,
        tags=tags,
        image_metadata=image_metadata,
        probe=probe,
        runtime_health_state=(
            runtime_instance.health_state if runtime_instance is not None else None
        ),
        runtime_load_state=(runtime_instance.load_state if runtime_instance is not None else None),
        runtime_last_seen_at=(
            runtime_instance.last_seen_at if runtime_instance is not None else None
        ),
    )


async def _probe_worker_health(
    *,
    endpoint: str,
    health_path: str,
    connect_timeout_seconds: float,
    request_timeout_seconds: float,
) -> ProbeResult:
    target = f"{endpoint.rstrip('/')}{health_path}"
    timeout = httpx.Timeout(
        timeout=request_timeout_seconds,
        connect=connect_timeout_seconds,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(target)
        status = (
            DiagnosticStatus.OK
            if response.status_code < 400
            else DiagnosticStatus.UNREACHABLE
        )
        detail = None if status is DiagnosticStatus.OK else "worker health endpoint returned error"
        return ProbeResult(
            target=target,
            probe_kind=ProbeKind.HTTP,
            status=status,
            http_status_code=response.status_code,
            detail=detail,
        )
    except httpx.HTTPError as exc:
        return ProbeResult(
            target=target,
            probe_kind=ProbeKind.HTTP,
            status=DiagnosticStatus.UNREACHABLE,
            detail=str(exc),
        )


async def _probe_supporting_services(
    *,
    settings: Settings,
    active_layer: DeploymentLayerConfig | None,
    allow_inferred_targets: bool,
) -> list[SupportingServiceDiagnostic]:
    service_targets = _supporting_service_targets(
        deployment_profile=settings.topology.deployment_profile,
        active_layer=active_layer,
        allow_inferred_targets=allow_inferred_targets,
    )
    diagnostics: list[SupportingServiceDiagnostic] = []
    for target in service_targets:
        if target.probe_kind is ProbeKind.HTTP:
            probe = await _probe_http_endpoint(target.target)
        else:
            probe = await _probe_tcp_endpoint(target.target)
        diagnostics.append(
            SupportingServiceDiagnostic(
                service_name=target.service_name,
                target=target.target,
                probe_kind=probe.probe_kind,
                status=probe.status,
                verified_at=probe.verified_at,
                detail=probe.detail,
                http_status_code=probe.http_status_code,
                metadata=probe.metadata,
            )
        )
    return diagnostics


def _supporting_service_targets(
    *,
    deployment_profile: DeploymentProfile,
    active_layer: DeploymentLayerConfig | None,
    allow_inferred_targets: bool,
) -> list[SupportServiceTarget]:
    metadata = active_layer.metadata if active_layer is not None else {}
    configured: list[SupportServiceTarget] = []
    for key, service_name, probe_kind in (
        ("postgres_address", "postgres", ProbeKind.TCP),
        ("redis_address", "redis", ProbeKind.TCP),
        ("otel_collector_address", "otel-collector", ProbeKind.TCP),
        ("prometheus_url", "prometheus", ProbeKind.HTTP),
        ("grafana_url", "grafana", ProbeKind.HTTP),
    ):
        target = metadata.get(key)
        if target:
            configured.append(
                SupportServiceTarget(
                    service_name=service_name,
                    target=target,
                    probe_kind=probe_kind,
                )
            )
    if configured:
        return configured
    if not allow_inferred_targets or deployment_profile is not DeploymentProfile.COMPOSE:
        return []
    return [
        SupportServiceTarget("postgres", "postgres:5432", ProbeKind.TCP),
        SupportServiceTarget("redis", "redis:6379", ProbeKind.TCP),
        SupportServiceTarget("otel-collector", "otel-collector:4318", ProbeKind.TCP),
        SupportServiceTarget("prometheus", "http://prometheus:9090/-/healthy", ProbeKind.HTTP),
        SupportServiceTarget("grafana", "http://grafana:3000/api/health", ProbeKind.HTTP),
    ]


async def _probe_http_endpoint(target: str) -> ProbeResult:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(_DEFAULT_HTTP_TIMEOUT)) as client:
            response = await client.get(target)
        status = DiagnosticStatus.OK if response.status_code < 400 else DiagnosticStatus.UNREACHABLE
        detail = None if status is DiagnosticStatus.OK else "supporting service returned error"
        return ProbeResult(
            target=target,
            probe_kind=ProbeKind.HTTP,
            status=status,
            http_status_code=response.status_code,
            detail=detail,
        )
    except httpx.HTTPError as exc:
        return ProbeResult(
            target=target,
            probe_kind=ProbeKind.HTTP,
            status=DiagnosticStatus.UNREACHABLE,
            detail=str(exc),
        )


async def _probe_tcp_endpoint(target: str) -> ProbeResult:
    host, port = _parse_host_port(target)
    if host is None or port is None:
        return ProbeResult(
            target=target,
            probe_kind=ProbeKind.TCP,
            status=DiagnosticStatus.ERROR,
            detail="tcp target must be formatted as host:port",
        )
    try:
        connection = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=_DEFAULT_HTTP_TIMEOUT,
        )
    except (TimeoutError, OSError) as exc:
        return ProbeResult(
            target=target,
            probe_kind=ProbeKind.TCP,
            status=DiagnosticStatus.UNREACHABLE,
            detail=str(exc),
        )
    reader, writer = connection
    _ = reader
    writer.close()
    await writer.wait_closed()
    return ProbeResult(
        target=target,
        probe_kind=ProbeKind.TCP,
        status=DiagnosticStatus.OK,
    )


def _parse_host_port(target: str) -> tuple[str | None, int | None]:
    if "://" in target:
        parsed = urlparse(target)
        return parsed.hostname, parsed.port
    host, separator, port_text = target.rpartition(":")
    if not separator:
        return None, None
    try:
        return host, int(port_text)
    except ValueError:
        return None, None


def _active_layer(settings: Settings) -> DeploymentLayerConfig | None:
    for layer in settings.topology.layers:
        if layer.name == settings.topology.active_environment:
            return layer
    return None


def _deployment_name_for_model(model_config: LocalModelConfig) -> str:
    if model_config.worker_transport is WorkerTransportType.IN_PROCESS:
        backend_prefix = (
            "mlx-lm"
            if model_config.backend_type is BackendType.MLX_LM
            else "vllm-metal"
            if model_config.backend_type is BackendType.VLLM_METAL
            else model_config.backend_type.value
        )
        return f"{backend_prefix}:{model_config.alias}"
    return f"remote-worker:{model_config.alias}"


def _image_metadata_for_model(model_config: LocalModelConfig) -> BackendImageMetadata | None:
    if model_config.image_tag is None and not model_config.build_metadata:
        return None
    return BackendImageMetadata(
        image_tag=model_config.image_tag,
        build_metadata=dict(model_config.build_metadata),
    )
