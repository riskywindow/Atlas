"""Typed runtime inspection schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.backend import BackendImageMetadata, DeploymentProfile
from switchyard.schemas.routing import (
    CanaryPolicy,
    CircuitBreakerState,
    HistoryDepthBucket,
    InputLengthBucket,
    ShadowPolicy,
    WorkloadTag,
)


class BackendRuntimeSummary(BaseModel):
    """Live backend status summary for admin inspection."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1, max_length=128)
    backend_type: str = Field(min_length=1, max_length=64)
    health_state: str = Field(min_length=1, max_length=64)
    load_state: str = Field(min_length=1, max_length=64)
    latency_ms: float | None = Field(default=None, ge=0.0)
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    circuit_open: bool = False
    circuit_reason: str | None = Field(default=None, min_length=1, max_length=256)
    instances: list[BackendInstanceRuntimeSummary] = Field(default_factory=list)


class BackendInstanceRuntimeSummary(BaseModel):
    """Live instance-level status summary for one backend deployment."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=128)
    source_of_truth: str = Field(min_length=1, max_length=64)
    endpoint: str = Field(min_length=1, max_length=512)
    transport: str = Field(min_length=1, max_length=64)
    health_state: str = Field(min_length=1, max_length=64)
    load_state: str = Field(min_length=1, max_length=64)
    last_seen_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class TenantLimiterRuntimeSummary(BaseModel):
    """Current limiter summary for one tenant/request class pair."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=128)
    request_class: str | None = Field(default=None, min_length=1, max_length=64)
    in_flight_requests: int = Field(default=0, ge=0)
    concurrency_cap: int = Field(ge=1)


class AdmissionRuntimeSummary(BaseModel):
    """Live admission-control state summary."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    global_concurrency_cap: int = Field(ge=1)
    global_queue_size: int = Field(ge=0)
    in_flight_total: int = Field(default=0, ge=0)
    queued_requests: int = Field(default=0, ge=0)
    oldest_queue_age_ms: float | None = Field(default=None, ge=0.0)
    tenant_limiters: list[TenantLimiterRuntimeSummary] = Field(default_factory=list)


class CircuitBreakerRuntimeSummary(BaseModel):
    """Live circuit-breaker state summary."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backends: list[CircuitBreakerState] = Field(default_factory=list)


class CanaryRoutingRuntimeSummary(BaseModel):
    """Active canary-routing configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    policies: list[CanaryPolicy] = Field(default_factory=list)


class ShadowRoutingRuntimeSummary(BaseModel):
    """Active shadow-routing configuration and task count."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    active_tasks: int = Field(default=0, ge=0)
    policies: list[ShadowPolicy] = Field(default_factory=list)


class SessionAffinityRuntimeSummary(BaseModel):
    """Current session-affinity cache summary."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ttl_seconds: float = Field(gt=0.0, le=86_400.0)
    max_sessions: int = Field(ge=1)
    active_bindings: int = Field(default=0, ge=0)
    bindings_by_target: dict[str, int] = Field(default_factory=dict)


class RoutingFeatureRuntimeSummary(BaseModel):
    """Stable request-feature extraction contract exposed at runtime."""

    model_config = ConfigDict(extra="forbid")

    feature_version: str = Field(min_length=1, max_length=32)
    input_length_buckets: list[InputLengthBucket] = Field(default_factory=list)
    history_depth_buckets: list[HistoryDepthBucket] = Field(default_factory=list)
    workload_tags: list[WorkloadTag] = Field(default_factory=list)
    prefix_fingerprint_algorithm: str = Field(min_length=1, max_length=64)
    prefix_plaintext_retained: bool = False


class RuntimeInspectionResponse(BaseModel):
    """Top-level runtime inspection payload."""

    model_config = ConfigDict(extra="forbid")

    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    backends: list[BackendRuntimeSummary] = Field(default_factory=list)
    admission: AdmissionRuntimeSummary
    circuit_breakers: CircuitBreakerRuntimeSummary
    canary_routing: CanaryRoutingRuntimeSummary
    shadow_routing: ShadowRoutingRuntimeSummary
    session_affinity: SessionAffinityRuntimeSummary
    routing_features: RoutingFeatureRuntimeSummary


class DiagnosticStatus(StrEnum):
    """Honest probe outcome categories for deployment diagnostics."""

    OK = "ok"
    UNREACHABLE = "unreachable"
    NOT_CONFIGURED = "not_configured"
    NOT_VERIFIABLE = "not_verifiable"
    ERROR = "error"


class ProbeKind(StrEnum):
    """Probe transport used by a deployment diagnostic."""

    HTTP = "http"
    TCP = "tcp"
    CONFIG = "config"


class ProbeResult(BaseModel):
    """Result of one diagnostic probe."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1, max_length=512)
    probe_kind: ProbeKind
    status: DiagnosticStatus
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detail: str | None = Field(default=None, max_length=512)
    http_status_code: int | None = Field(default=None, ge=100, le=599)
    metadata: dict[str, str] = Field(default_factory=dict)


class WorkerInstanceDiagnostic(BaseModel):
    """Configured worker instance plus current probe/runtime information."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=128)
    endpoint: str | None = Field(default=None, min_length=1, max_length=512)
    transport: str = Field(min_length=1, max_length=64)
    source_of_truth: str | None = Field(default=None, min_length=1, max_length=64)
    tags: list[str] = Field(default_factory=list)
    image_metadata: BackendImageMetadata | None = None
    probe: ProbeResult
    runtime_health_state: str | None = Field(default=None, min_length=1, max_length=64)
    runtime_load_state: str | None = Field(default=None, min_length=1, max_length=64)
    runtime_last_seen_at: datetime | None = None


class WorkerDeploymentDiagnostic(BaseModel):
    """Deployment-level diagnostic summary for one configured backend."""

    model_config = ConfigDict(extra="forbid")

    alias: str = Field(min_length=1, max_length=128)
    serving_target: str = Field(min_length=1, max_length=128)
    backend_type: str = Field(min_length=1, max_length=64)
    deployment_profile: DeploymentProfile
    worker_transport: str = Field(min_length=1, max_length=64)
    model_identifier: str = Field(min_length=1, max_length=512)
    image_metadata: BackendImageMetadata | None = None
    build_metadata: dict[str, str] = Field(default_factory=dict)
    configured_instances: list[WorkerInstanceDiagnostic] = Field(default_factory=list)


class SupportingServiceDiagnostic(BaseModel):
    """Diagnostic result for one supporting deployment service."""

    model_config = ConfigDict(extra="forbid")

    service_name: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=512)
    probe_kind: ProbeKind
    status: DiagnosticStatus
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detail: str | None = Field(default=None, max_length=512)
    http_status_code: int | None = Field(default=None, ge=100, le=599)
    metadata: dict[str, str] = Field(default_factory=dict)


class DeploymentDiagnosticsResponse(BaseModel):
    """Deployment-aware diagnostic report for local or deployed control planes."""

    model_config = ConfigDict(extra="forbid")

    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    diagnostics_source: str = Field(min_length=1, max_length=64)
    active_environment: str = Field(min_length=1, max_length=64)
    effective_deployment_profile: DeploymentProfile
    gateway_base_url: str | None = Field(default=None, min_length=1, max_length=256)
    active_layer_name: str | None = Field(default=None, min_length=1, max_length=64)
    control_plane_image: BackendImageMetadata | None = None
    worker_deployments: list[WorkerDeploymentDiagnostic] = Field(default_factory=list)
    runtime_backends: list[BackendRuntimeSummary] = Field(default_factory=list)
    supporting_services: list[SupportingServiceDiagnostic] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
