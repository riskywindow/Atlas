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
    HybridExecutionRuntimeSummary,
    ProbeKind,
    ProbeResult,
    RemoteWorkerLifecycleRuntimeSummary,
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
    DeviceClass,
    WorkerTransportType,
)
from switchyard.schemas.worker import RegisteredRemoteWorkerSnapshot, RemoteWorkerAuthMode

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
    remote_worker_registry: RegisteredRemoteWorkerSnapshot | None = None,
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
        hybrid_execution=summarize_hybrid_execution(
            settings=settings,
            runtime_backends=runtime_backends,
        ),
        remote_workers=summarize_remote_worker_lifecycle(
            settings=settings,
            runtime_backends=runtime_backends,
            remote_worker_registry=remote_worker_registry,
        ),
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
        deployment = status_snapshot.deployment
        deployment_profile = (
            deployment.deployment_profile.value
            if deployment is not None
            else DeploymentProfile.HOST_NATIVE.value
        )
        environment = deployment.environment if deployment is not None else "local"
        summaries.append(
            BackendRuntimeSummary(
                backend_name=status_snapshot.name,
                backend_type=status_snapshot.capabilities.backend_type.value,
                deployment_profile=deployment_profile,
                execution_mode=(
                    deployment.execution_mode.value if deployment is not None else None
                ),
                environment=environment,
                provider=(deployment.placement.provider if deployment is not None else None),
                region=(deployment.placement.region if deployment is not None else None),
                zone=(deployment.placement.zone if deployment is not None else None),
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
                        device_class=(
                            instance.device_class.value
                            if instance.device_class is not None
                            else None
                        ),
                        locality=instance.locality,
                        locality_class=instance.locality_class.value,
                        execution_mode=instance.execution_mode.value,
                        provider=instance.placement.provider,
                        region=instance.placement.region,
                        zone=instance.placement.zone,
                        network_profile=instance.network_characteristics.profile.value,
                        auth_state=instance.trust.auth_state.value,
                        trust_state=instance.trust.trust_state.value,
                        registration_state=instance.registration.state.value,
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


def summarize_hybrid_execution(
    *,
    settings: Settings,
    runtime_backends: list[BackendRuntimeSummary],
) -> HybridExecutionRuntimeSummary:
    """Summarize Phase 7 hybrid execution posture from config and runtime truth."""

    local_capable_backends = 0
    remote_capable_backends = 0
    healthy_local_backends = 0
    healthy_remote_backends = 0
    degraded_remote_backends = 0
    unavailable_remote_backends = 0
    remote_instance_count = 0

    for backend in runtime_backends:
        has_local = any(not _is_remote_runtime_instance(instance) for instance in backend.instances)
        has_remote = any(_is_remote_runtime_instance(instance) for instance in backend.instances)
        if not backend.instances and backend.deployment_profile != DeploymentProfile.REMOTE.value:
            has_local = True
        if has_local:
            local_capable_backends += 1
            if backend.health_state == BackendHealthState.HEALTHY.value:
                healthy_local_backends += 1
        if has_remote:
            remote_capable_backends += 1
            remote_instance_count += sum(
                1 for instance in backend.instances if _is_remote_runtime_instance(instance)
            )
            if backend.health_state == BackendHealthState.HEALTHY.value:
                healthy_remote_backends += 1
            elif backend.health_state == BackendHealthState.DEGRADED.value:
                degraded_remote_backends += 1
            elif backend.health_state == BackendHealthState.UNAVAILABLE.value:
                unavailable_remote_backends += 1

    notes: list[str] = []
    if not runtime_backends:
        notes.append("runtime backend inventory is empty; hybrid counts reflect config only")
    if (
        settings.phase7.hybrid_execution.enabled
        and settings.phase7.hybrid_execution.spillover_enabled
        and remote_capable_backends == 0
    ):
        notes.append(
            "hybrid spillover is enabled but no remote-capable backends are "
            "currently registered"
        )

    return HybridExecutionRuntimeSummary(
        enabled=settings.phase7.hybrid_execution.enabled,
        prefer_local=settings.phase7.hybrid_execution.prefer_local,
        spillover_enabled=settings.phase7.hybrid_execution.spillover_enabled,
        require_healthy_local_backends=(
            settings.phase7.hybrid_execution.require_healthy_local_backends
        ),
        max_remote_share_percent=settings.phase7.hybrid_execution.max_remote_share_percent,
        remote_request_budget_per_minute=(
            settings.phase7.hybrid_execution.remote_request_budget_per_minute
        ),
        allowed_remote_environments=list(
            settings.phase7.hybrid_execution.allowed_remote_environments
        ),
        local_capable_backends=local_capable_backends,
        remote_capable_backends=remote_capable_backends,
        healthy_local_backends=healthy_local_backends,
        healthy_remote_backends=healthy_remote_backends,
        degraded_remote_backends=degraded_remote_backends,
        unavailable_remote_backends=unavailable_remote_backends,
        remote_instance_count=remote_instance_count,
        notes=notes,
    )


def summarize_remote_worker_lifecycle(
    *,
    settings: Settings,
    runtime_backends: list[BackendRuntimeSummary],
    remote_worker_registry: RegisteredRemoteWorkerSnapshot | None = None,
) -> RemoteWorkerLifecycleRuntimeSummary:
    """Summarize Phase 7 registration posture from config and runtime inventory."""

    static_instance_count = sum(
        len(model_config.instances)
        for model_config in settings.local_models
        for instance in model_config.instances
        if instance.source_of_truth.value == "static_config"
    )
    registered_instance_count = 0
    discovered_instance_count = 0
    stale_instance_count = 0
    ready_instance_count = 0
    draining_instance_count = 0
    unhealthy_instance_count = 0
    lost_instance_count = 0
    retired_instance_count = 0
    for backend in runtime_backends:
        for instance in backend.instances:
            if instance.registration_state == "registered":
                registered_instance_count += 1
            elif instance.registration_state == "discovered":
                discovered_instance_count += 1
            elif instance.registration_state == "stale":
                stale_instance_count += 1
    if remote_worker_registry is not None:
        registered_instance_count = remote_worker_registry.worker_count
        stale_instance_count = remote_worker_registry.stale_worker_count
        ready_instance_count = remote_worker_registry.ready_worker_count
        draining_instance_count = remote_worker_registry.draining_worker_count
        unhealthy_instance_count = remote_worker_registry.unhealthy_worker_count
        lost_instance_count = remote_worker_registry.lost_worker_count
        retired_instance_count = remote_worker_registry.retired_worker_count

    notes: list[str] = []
    if (
        settings.phase7.remote_workers.secure_registration_required
        and settings.phase7.remote_workers.registration_token_name is None
    ):
        notes.append("secure registration is required but no registration_token_name is configured")
    if not runtime_backends:
        notes.append(
            "runtime backend inventory is empty; registration counts only include "
            "static config"
        )

    return RemoteWorkerLifecycleRuntimeSummary(
        secure_registration_required=settings.phase7.remote_workers.secure_registration_required,
        auth_mode=(
            settings.phase7.remote_workers.auth_mode
            if settings.phase7.remote_workers.secure_registration_required
            else RemoteWorkerAuthMode.NONE
        ),
        dynamic_registration_enabled=settings.phase7.remote_workers.dynamic_registration_enabled,
        heartbeat_timeout_seconds=settings.phase7.remote_workers.heartbeat_timeout_seconds,
        stale_eviction_seconds=settings.phase7.remote_workers.stale_eviction_seconds,
        registration_token_name=settings.phase7.remote_workers.registration_token_name,
        allow_static_instances=settings.phase7.remote_workers.allow_static_instances,
        static_instance_count=static_instance_count,
        registered_instance_count=registered_instance_count,
        discovered_instance_count=discovered_instance_count,
        stale_instance_count=stale_instance_count,
        ready_instance_count=ready_instance_count,
        draining_instance_count=draining_instance_count,
        unhealthy_instance_count=unhealthy_instance_count,
        lost_instance_count=lost_instance_count,
        retired_instance_count=retired_instance_count,
        notes=notes,
    )


def _is_remote_runtime_instance(instance: BackendInstanceRuntimeSummary) -> bool:
    return (
        instance.device_class == DeviceClass.REMOTE.value
        or instance.device_class == DeviceClass.REMOTE_UNKNOWN.value
        or instance.locality_class in {"remote_private", "remote_cloud", "external_service"}
        or instance.execution_mode in {"remote_worker", "external_service"}
        or instance.locality == "remote"
        or "remote" in instance.tags
    )


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
