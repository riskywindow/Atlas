"""Typed runtime inspection schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.backend import (
    BackendImageMetadata,
    CapacitySnapshot,
    DeploymentProfile,
    GPUDeviceMetadata,
    LogicalModelTarget,
    RequestFeatureSupport,
    RuntimeIdentity,
)
from switchyard.schemas.routing import (
    CanaryPolicy,
    CircuitBreakerState,
    HistoryDepthBucket,
    InputLengthBucket,
    PolicyReference,
    PolicyRolloutMode,
    PrefixHotness,
    ShadowPolicy,
    WorkloadTag,
)
from switchyard.schemas.worker import RegisteredRemoteWorkerSnapshot, RemoteWorkerAuthMode


class BackendRuntimeSummary(BaseModel):
    """Live backend status summary for admin inspection."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1, max_length=128)
    backend_type: str = Field(min_length=1, max_length=64)
    deployment_profile: str = Field(min_length=1, max_length=64)
    execution_mode: str | None = Field(default=None, min_length=1, max_length=64)
    environment: str = Field(min_length=1, max_length=64)
    runtime: RuntimeIdentity | None = None
    gpu: GPUDeviceMetadata | None = None
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    region: str | None = Field(default=None, min_length=1, max_length=128)
    zone: str | None = Field(default=None, min_length=1, max_length=128)
    health_state: str = Field(min_length=1, max_length=64)
    load_state: str = Field(min_length=1, max_length=64)
    latency_ms: float | None = Field(default=None, ge=0.0)
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    request_features: RequestFeatureSupport = Field(default_factory=RequestFeatureSupport)
    logical_targets: list[LogicalModelTarget] = Field(default_factory=list)
    observed_capacity: CapacitySnapshot | None = None
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
    device_class: str | None = Field(default=None, min_length=1, max_length=64)
    runtime: RuntimeIdentity | None = None
    gpu: GPUDeviceMetadata | None = None
    locality: str = Field(default="local", min_length=1, max_length=64)
    locality_class: str | None = Field(default=None, min_length=1, max_length=64)
    execution_mode: str | None = Field(default=None, min_length=1, max_length=64)
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    region: str | None = Field(default=None, min_length=1, max_length=128)
    zone: str | None = Field(default=None, min_length=1, max_length=128)
    network_profile: str | None = Field(default=None, min_length=1, max_length=64)
    auth_state: str | None = Field(default=None, min_length=1, max_length=64)
    trust_state: str | None = Field(default=None, min_length=1, max_length=64)
    registration_state: str | None = Field(default=None, min_length=1, max_length=64)
    health_state: str = Field(min_length=1, max_length=64)
    load_state: str = Field(min_length=1, max_length=64)
    observed_capacity: CapacitySnapshot | None = None
    last_seen_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class HybridExecutionRuntimeSummary(BaseModel):
    """Configured hybrid local/remote posture plus current runtime health counts."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    prefer_local: bool = True
    spillover_enabled: bool = False
    require_healthy_local_backends: bool = True
    max_remote_share_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    remote_request_budget_per_minute: int | None = Field(default=None, ge=1)
    remote_concurrency_cap: int | None = Field(default=None, ge=1)
    remote_kill_switch_enabled: bool = False
    remote_cooldown_seconds: float = Field(default=0.0, ge=0.0)
    allow_high_priority_remote_escalation: bool = True
    allowed_remote_environments: list[str] = Field(default_factory=list)
    tenant_remote_policy_count: int = Field(default=0, ge=0)
    local_capable_backends: int = Field(default=0, ge=0)
    remote_capable_backends: int = Field(default=0, ge=0)
    healthy_local_backends: int = Field(default=0, ge=0)
    healthy_remote_backends: int = Field(default=0, ge=0)
    degraded_remote_backends: int = Field(default=0, ge=0)
    unavailable_remote_backends: int = Field(default=0, ge=0)
    remote_instance_count: int = Field(default=0, ge=0)
    remote_budget_window_started_at: datetime | None = None
    remote_budget_requests_used: int = Field(default=0, ge=0)
    remote_budget_requests_remaining: int | None = Field(default=None, ge=0)
    remote_in_flight_requests: int = Field(default=0, ge=0)
    cooldown_active: bool = False
    cooldown_until: datetime | None = None
    remote_policy_eligible: list[str] = Field(default_factory=list)
    remote_policy_ineligible: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HybridRemoteToggleRequest(BaseModel):
    """Mutable operator request for remote enable/disable posture."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    reason: str | None = Field(default=None, min_length=1, max_length=256)


class HybridBudgetResetResponse(BaseModel):
    """Result of resetting the current remote budget window."""

    model_config = ConfigDict(extra="forbid")

    budget_window_started_at: datetime
    remote_budget_requests_used: int = Field(ge=0)
    remote_budget_requests_remaining: int | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


class HybridBudgetRuntimeSummary(BaseModel):
    """Operator-facing remote budget, spend bucket, and cooldown posture."""

    model_config = ConfigDict(extra="forbid")

    remote_request_budget_per_minute: int | None = Field(default=None, ge=1)
    remote_budget_window_started_at: datetime | None = None
    remote_budget_requests_used: int = Field(default=0, ge=0)
    remote_budget_requests_remaining: int | None = Field(default=None, ge=0)
    remote_in_flight_requests: int = Field(default=0, ge=0)
    remote_concurrency_cap: int | None = Field(default=None, ge=1)
    cooldown_active: bool = False
    cooldown_until: datetime | None = None
    recent_observed_budget_bucket_counts: dict[str, int] = Field(default_factory=dict)
    recent_estimated_budget_bucket_counts: dict[str, int] = Field(default_factory=dict)
    total_observed_relative_cost_index: float | None = Field(default=None, ge=0.0)
    total_estimated_relative_cost_index: float | None = Field(default=None, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class CloudWorkerControlRuntimeEntry(BaseModel):
    """Compact operator-facing control and runtime view for one cloud worker."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(min_length=1, max_length=128)
    backend_name: str = Field(min_length=1, max_length=128)
    serving_targets: list[str] = Field(default_factory=list)
    lifecycle_state: str = Field(min_length=1, max_length=64)
    last_heartbeat_at: datetime | None = None
    runtime: RuntimeIdentity | None = None
    gpu: GPUDeviceMetadata | None = None
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    region: str | None = Field(default=None, min_length=1, max_length=128)
    zone: str | None = Field(default=None, min_length=1, max_length=128)
    ready: bool = False
    usable: bool = False
    quarantined: bool = False
    canary_only: bool = False
    draining: bool = False
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    eligibility_reasons: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class AliasRoutingOverrideState(BaseModel):
    """Current operator override for one logical alias or serving target."""

    model_config = ConfigDict(extra="forbid")

    serving_target: str = Field(min_length=1, max_length=128)
    pinned_backend: str | None = Field(default=None, min_length=1, max_length=128)
    disabled_backends: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reason: str | None = Field(default=None, min_length=1, max_length=256)


class AliasRoutingOverrideRequest(BaseModel):
    """Operator mutation request for one alias-scoped routing override."""

    model_config = ConfigDict(extra="forbid")

    pinned_backend: str | None = Field(default=None, min_length=1, max_length=128)
    disabled_backends: list[str] = Field(default_factory=list)
    reason: str | None = Field(default=None, min_length=1, max_length=256)


class AliasCompatibilityRuntimeEntry(BaseModel):
    """Remote-worker eligibility and override posture for one logical alias."""

    model_config = ConfigDict(extra="forbid")

    serving_target: str = Field(min_length=1, max_length=128)
    eligible_remote_backends: list[str] = Field(default_factory=list)
    ineligible_remote_backends: dict[str, list[str]] = Field(default_factory=dict)
    pinned_backend: str | None = Field(default=None, min_length=1, max_length=128)
    disabled_backends: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HybridRouteExample(BaseModel):
    """Recent operator-facing route example for hybrid placement decisions."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=128)
    chosen_backend: str = Field(min_length=1, max_length=128)
    executed_backend: str | None = Field(default=None, min_length=1, max_length=128)
    execution_path: str = Field(min_length=1, max_length=64)
    fallback_used: bool = False
    final_outcome: str | None = Field(default=None, min_length=1, max_length=128)
    route_reason_codes: list[str] = Field(default_factory=list)
    admission_reason_code: str | None = Field(default=None, min_length=1, max_length=128)
    remote_candidate_count: int = Field(default=0, ge=0)
    placement_provider: str | None = Field(default=None, min_length=1, max_length=128)
    placement_region: str | None = Field(default=None, min_length=1, max_length=128)
    placement_zone: str | None = Field(default=None, min_length=1, max_length=128)
    placement_evidence_source: str | None = Field(default=None, min_length=1, max_length=64)
    budget_bucket: str | None = Field(default=None, min_length=1, max_length=128)
    relative_cost_index: float | None = Field(default=None, ge=0.0)
    cost_evidence_source: str | None = Field(default=None, min_length=1, max_length=64)
    notes: list[str] = Field(default_factory=list)


class PlacementDistributionRuntimeSummary(BaseModel):
    """Recent local-vs-remote placement distribution for operator inspection."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(default=0, ge=0)
    local_count: int = Field(default=0, ge=0)
    remote_count: int = Field(default=0, ge=0)
    remote_blocked_count: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)


class RemoteTransportErrorRuntimeEntry(BaseModel):
    """Bounded recent remote transport error entry."""

    model_config = ConfigDict(extra="forbid")

    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str = Field(min_length=1, max_length=128)
    backend_name: str = Field(min_length=1, max_length=128)
    error: str = Field(min_length=1, max_length=512)
    cooldown_triggered: bool = False


class CloudRouteEvidenceRuntimeSummary(BaseModel):
    """Recent operator-facing summary of cloud placement and spend evidence."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(default=0, ge=0)
    observed_placement_count: int = Field(default=0, ge=0)
    estimated_placement_count: int = Field(default=0, ge=0)
    observed_cost_count: int = Field(default=0, ge=0)
    estimated_cost_count: int = Field(default=0, ge=0)
    remote_provider_counts: dict[str, int] = Field(default_factory=dict)
    observed_budget_bucket_counts: dict[str, int] = Field(default_factory=dict)
    estimated_budget_bucket_counts: dict[str, int] = Field(default_factory=dict)
    total_observed_relative_cost_index: float | None = Field(default=None, ge=0.0)
    total_estimated_relative_cost_index: float | None = Field(default=None, ge=0.0)


class HybridOperatorRuntimeSummary(BaseModel):
    """Operator-facing recent hybrid-routing decisions and controls."""

    model_config = ConfigDict(extra="forbid")

    remote_enabled_override: bool | None = None
    remote_effectively_enabled: bool = True
    recent_route_examples: list[HybridRouteExample] = Field(default_factory=list)
    recent_route_example_count: int = Field(default=0, ge=0)
    recent_placement_distribution: PlacementDistributionRuntimeSummary = Field(
        default_factory=PlacementDistributionRuntimeSummary
    )
    recent_cloud_evidence: CloudRouteEvidenceRuntimeSummary = Field(
        default_factory=CloudRouteEvidenceRuntimeSummary
    )
    budget_state: HybridBudgetRuntimeSummary = Field(default_factory=HybridBudgetRuntimeSummary)
    cloud_workers: list[CloudWorkerControlRuntimeEntry] = Field(default_factory=list)
    alias_compatibility: list[AliasCompatibilityRuntimeEntry] = Field(default_factory=list)
    alias_overrides: list[AliasRoutingOverrideState] = Field(default_factory=list)
    recent_remote_transport_errors: list[RemoteTransportErrorRuntimeEntry] = Field(
        default_factory=list
    )
    notes: list[str] = Field(default_factory=list)


class RemoteWorkerOperatorRequest(BaseModel):
    """Operator request to mutate remote worker posture."""

    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, min_length=1, max_length=256)
    enabled: bool = True


class RemoteWorkerLifecycleRuntimeSummary(BaseModel):
    """Operator-facing registration and heartbeat view for remote workers."""

    model_config = ConfigDict(extra="forbid")

    secure_registration_required: bool = False
    auth_mode: RemoteWorkerAuthMode = RemoteWorkerAuthMode.NONE
    dynamic_registration_enabled: bool = False
    heartbeat_timeout_seconds: float = Field(gt=0.0, le=3600.0)
    stale_eviction_seconds: float = Field(gt=0.0, le=86_400.0)
    registration_token_name: str | None = Field(default=None, min_length=1, max_length=128)
    allow_static_instances: bool = True
    static_instance_count: int = Field(default=0, ge=0)
    registered_instance_count: int = Field(default=0, ge=0)
    discovered_instance_count: int = Field(default=0, ge=0)
    stale_instance_count: int = Field(default=0, ge=0)
    ready_instance_count: int = Field(default=0, ge=0)
    usable_instance_count: int = Field(default=0, ge=0)
    draining_instance_count: int = Field(default=0, ge=0)
    unhealthy_instance_count: int = Field(default=0, ge=0)
    quarantined_instance_count: int = Field(default=0, ge=0)
    lost_instance_count: int = Field(default=0, ge=0)
    retired_instance_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


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


class PolicyDecisionRuntimeEntry(BaseModel):
    """Bounded recent policy-decision record for runtime inspection."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requested_policy: str = Field(min_length=1, max_length=64)
    rollout_mode: PolicyRolloutMode
    selected_policy: PolicyReference
    selected_backend: str = Field(min_length=1, max_length=128)
    shadow_policy: PolicyReference | None = None
    abstained: bool = False
    exploration_used: bool = False
    canary_selected: bool = False
    guardrail_triggers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PolicyRolloutRuntimeSummary(BaseModel):
    """Current intelligent-policy rollout state and recent bounded diagnostics."""

    model_config = ConfigDict(extra="forbid")

    mode: PolicyRolloutMode = PolicyRolloutMode.DISABLED
    candidate_policy: PolicyReference | None = None
    active_policy: PolicyReference | None = None
    shadow_policy: PolicyReference | None = None
    compatibility_policy: str | None = Field(default=None, min_length=1, max_length=64)
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    kill_switch_enabled: bool = False
    learning_frozen: bool = False
    exploration_enabled: bool = False
    recent_decisions: list[PolicyDecisionRuntimeEntry] = Field(default_factory=list)
    recent_abstentions: int = Field(default=0, ge=0)
    recent_guardrail_triggers: list[str] = Field(default_factory=list)
    last_policy_update_at: datetime | None = None
    last_learning_event_at: datetime | None = None
    last_learning_event: str | None = Field(default=None, min_length=1, max_length=256)
    last_guardrail_trigger: str | None = Field(default=None, min_length=1, max_length=256)
    notes: list[str] = Field(default_factory=list)


class PolicyRolloutUpdateRequest(BaseModel):
    """Runtime mutation request for local policy-rollout controls."""

    model_config = ConfigDict(extra="forbid")

    mode: PolicyRolloutMode | None = None
    canary_percentage: float | None = Field(default=None, ge=0.0, le=100.0)
    kill_switch_enabled: bool | None = None
    learning_frozen: bool | None = None


class PolicyRolloutStateSnapshot(BaseModel):
    """Serializable policy-rollout state for local export/import workflows."""

    model_config = ConfigDict(extra="forbid")

    mode: PolicyRolloutMode = PolicyRolloutMode.DISABLED
    candidate_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    shadow_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    kill_switch_enabled: bool = False
    learning_frozen: bool = False
    last_policy_update_at: datetime | None = None
    last_learning_event_at: datetime | None = None
    last_learning_event: str | None = Field(default=None, min_length=1, max_length=256)
    recent_decisions: list[PolicyDecisionRuntimeEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


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


class TrackedPrefixRuntimeSummary(BaseModel):
    """Bounded runtime summary for one tracked prefix digest."""

    model_config = ConfigDict(extra="forbid")

    serving_target: str = Field(min_length=1, max_length=128)
    locality_key: str = Field(min_length=8, max_length=64)
    prefix_fingerprint: str = Field(min_length=8, max_length=64)
    recent_request_count: int = Field(default=0, ge=0)
    hotness: PrefixHotness = PrefixHotness.COLD
    preferred_backend: str | None = Field(default=None, min_length=1, max_length=128)
    preferred_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    last_seen_at: datetime | None = None


class PrefixLocalityRuntimeSummary(BaseModel):
    """Current repeated-prefix and locality tracker summary."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ttl_seconds: float = Field(gt=0.0, le=86_400.0)
    max_prefixes: int = Field(ge=1)
    active_prefixes: int = Field(default=0, ge=0)
    hot_prefixes: int = Field(default=0, ge=0)
    tracked_serving_targets: dict[str, int] = Field(default_factory=dict)
    hottest_prefixes: list[TrackedPrefixRuntimeSummary] = Field(default_factory=list)
    prefix_fingerprint_algorithm: str = Field(min_length=1, max_length=64)
    prefix_plaintext_retained: bool = False
    collision_scope: str = Field(min_length=1, max_length=128)


class RuntimeInspectionResponse(BaseModel):
    """Top-level runtime inspection payload."""

    model_config = ConfigDict(extra="forbid")

    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    backends: list[BackendRuntimeSummary] = Field(default_factory=list)
    admission: AdmissionRuntimeSummary
    circuit_breakers: CircuitBreakerRuntimeSummary
    canary_routing: CanaryRoutingRuntimeSummary
    shadow_routing: ShadowRoutingRuntimeSummary
    policy_rollout: PolicyRolloutRuntimeSummary
    session_affinity: SessionAffinityRuntimeSummary
    hybrid_execution: HybridExecutionRuntimeSummary
    hybrid_operator: HybridOperatorRuntimeSummary = Field(
        default_factory=HybridOperatorRuntimeSummary
    )
    remote_workers: RemoteWorkerLifecycleRuntimeSummary
    remote_worker_registry: RegisteredRemoteWorkerSnapshot | None = None
    routing_features: RoutingFeatureRuntimeSummary
    prefix_locality: PrefixLocalityRuntimeSummary


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
    hybrid_execution: HybridExecutionRuntimeSummary
    remote_workers: RemoteWorkerLifecycleRuntimeSummary
    supporting_services: list[SupportingServiceDiagnostic] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
