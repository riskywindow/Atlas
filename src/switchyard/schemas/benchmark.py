"""Benchmark and replay artifact schemas."""

from __future__ import annotations

import platform as platform_module
import sys
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from switchyard.schemas.admin import (
    HybridExecutionRuntimeSummary,
    RemoteWorkerLifecycleRuntimeSummary,
)
from switchyard.schemas.backend import (
    BackendImageMetadata,
    BackendInstance,
    DeploymentProfile,
    ExecutionModeLabel,
    NetworkProfile,
    TopologySchemaVersion,
    WorkerLocalityClass,
    WorkerTransportType,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import (
    AdmissionDecision,
    CanaryPolicy,
    CircuitBreakerState,
    HistoryDepthBucket,
    InputLengthBucket,
    PolicyReference,
    PrefixHotness,
    RequestClass,
    RequestFeatureVector,
    RouteDecision,
    RouteExecutionObservation,
    RouteExplanation,
    RouteTelemetryMetadata,
    RoutingPolicy,
    ShadowPolicy,
    ShadowRouteEvidence,
    StickyRouteRecord,
    TopologySnapshotReference,
    WorkloadShape,
    WorkloadTag,
)
from switchyard.schemas.worker import RegisteredRemoteWorkerSnapshot


class BenchmarkArtifactSchemaVersion(StrEnum):
    """Explicit artifact schema versions for benchmark/replay data."""

    V1 = "switchyard.benchmark.v1"
    V2 = "switchyard.benchmark.v2"
    V3 = "switchyard.benchmark.v3"


class WorkloadPattern(StrEnum):
    """Synthetic workload shapes supported by the Phase 3 benchmark layer."""

    UNIFORM = "uniform"
    REPEATED_PREFIX = "repeated_prefix"
    BURSTY = "bursty"


class WorkloadScenarioFamily(StrEnum):
    """Small library of built-in workload scenario families."""

    SHORT_CHAT = "short_chat"
    LONG_PROMPT = "long_prompt"
    REPEATED_PREFIX = "repeated_prefix"
    CONCURRENCY_BURST = "concurrency_burst"
    QUEUE_SATURATION = "queue_saturation"
    TENANT_CONTENTION = "tenant_contention"
    BACKEND_FLAKINESS = "backend_flakiness"
    SESSION_STICKINESS = "session_stickiness"
    CANARY_ROLLOUT = "canary_rollout"
    SHADOW_TRAFFIC = "shadow_traffic"
    HYBRID_SPILLOVER = "hybrid_spillover"
    REMOTE_COLD_WARM = "remote_cold_warm"
    REMOTE_BUDGET_GUARDRAIL = "remote_budget_guardrail"
    MIXED = "mixed"


class ExecutionTargetType(StrEnum):
    """Target selection mode for a benchmark or replay execution."""

    LOGICAL_ALIAS = "logical_alias"
    ROUTING_POLICY = "routing_policy"
    PINNED_BACKEND = "pinned_backend"


class TraceCaptureMode(StrEnum):
    """Trace capture detail level."""

    OFF = "off"
    METADATA_ONLY = "metadata_only"
    REDACTED_CONTENT = "redacted_content"
    FULL_CONTENT = "full_content"


class ReplayMode(StrEnum):
    """Replay dispatch strategy."""

    SEQUENTIAL = "sequential"
    FIXED_CONCURRENCY = "fixed_concurrency"
    PRESERVE_ORDER_WITHOUT_ORIGINAL_TIMING = "preserve_order_without_original_timing"


class ReportFormat(StrEnum):
    """Supported derived report formats."""

    MARKDOWN = "markdown"
    JSON = "json"


class ReportSourceOfTruth(StrEnum):
    """Authoritative data source for generated reports."""

    BENCHMARK_ARTIFACT = "benchmark_artifact"


class ComparisonSourceKind(StrEnum):
    """Source set used for a two-target comparison."""

    WORKLOAD_MANIFEST = "workload_manifest"
    TRACE_SET = "trace_set"


class HybridExecutionPath(StrEnum):
    """Observed execution posture for one request."""

    LOCAL_ONLY = "local_only"
    HYBRID_SPILLOVER = "hybrid_spillover"
    REMOTE_ONLY = "remote_only"
    REMOTE_BLOCKED = "remote_blocked"
    UNKNOWN = "unknown"


class RemoteTemperature(StrEnum):
    """Cold-versus-warm posture for remote execution when known."""

    COLD = "cold"
    WARM = "warm"
    UNKNOWN = "unknown"


class RemoteBudgetOutcome(StrEnum):
    """Typed budget/admission posture for remote execution."""

    WITHIN_BUDGET = "within_budget"
    EXHAUSTED = "exhausted"
    DISABLED = "disabled"
    NOT_MODELED = "not_modeled"
    UNKNOWN = "unknown"


class HybridConditionSource(StrEnum):
    """Where a remote/local hybrid condition came from."""

    OBSERVED_RUNTIME = "observed_runtime"
    INJECTED_MOCK = "injected_mock"
    PREDICTOR_ESTIMATE = "predictor_estimate"


class HybridComparisonOutcome(StrEnum):
    """Human-facing interpretation of a hybrid comparison delta."""

    BENEFICIAL = "beneficial"
    HARMFUL = "harmful"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


class HybridConditionProfile(BaseModel):
    """Injected or estimated remote-condition profile attached to a request."""

    model_config = ConfigDict(extra="forbid")

    source: HybridConditionSource
    execution_path: HybridExecutionPath = HybridExecutionPath.UNKNOWN
    remote_temperature: RemoteTemperature = RemoteTemperature.UNKNOWN
    budget_outcome: RemoteBudgetOutcome = RemoteBudgetOutcome.UNKNOWN
    network_penalty_ms: float | None = Field(default=None, ge=0.0)
    cold_start_penalty_ms: float | None = Field(default=None, ge=0.0)
    modeled_cost: float | None = Field(default=None, ge=0.0)
    confidence: RecommendationConfidence | None = None
    notes: list[str] = Field(default_factory=list)


class HybridExecutionContext(BaseModel):
    """Observed plus modeled hybrid execution context for one request."""

    model_config = ConfigDict(extra="forbid")

    observed_execution_path: HybridExecutionPath = HybridExecutionPath.UNKNOWN
    observed_remote_temperature: RemoteTemperature = RemoteTemperature.UNKNOWN
    observed_budget_outcome: RemoteBudgetOutcome = RemoteBudgetOutcome.UNKNOWN
    observed_network_penalty_ms: float | None = Field(default=None, ge=0.0)
    observed_modeled_cost: float | None = Field(default=None, ge=0.0)
    reason_codes: list[str] = Field(default_factory=list)
    injected_condition: HybridConditionProfile | None = None
    predictor_condition: HybridConditionProfile | None = None


class HybridBenchmarkSummary(BaseModel):
    """Aggregate hybrid-routing evidence summary for one run."""

    model_config = ConfigDict(extra="forbid")

    local_only_count: int = Field(default=0, ge=0)
    hybrid_spillover_count: int = Field(default=0, ge=0)
    remote_only_count: int = Field(default=0, ge=0)
    remote_blocked_count: int = Field(default=0, ge=0)
    remote_cold_count: int = Field(default=0, ge=0)
    remote_warm_count: int = Field(default=0, ge=0)
    observed_runtime_count: int = Field(default=0, ge=0)
    injected_condition_count: int = Field(default=0, ge=0)
    predictor_estimate_count: int = Field(default=0, ge=0)
    low_confidence_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    budget_exhausted_count: int = Field(default=0, ge=0)
    budget_disabled_count: int = Field(default=0, ge=0)
    avg_observed_network_penalty_ms: float | None = Field(default=None, ge=0.0)
    avg_injected_network_penalty_ms: float | None = Field(default=None, ge=0.0)
    avg_predicted_network_penalty_ms: float | None = Field(default=None, ge=0.0)
    total_modeled_cost: float | None = Field(default=None, ge=0.0)
    avg_modeled_cost: float | None = Field(default=None, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class HybridComparisonSummary(BaseModel):
    """Hybrid-specific comparison summary between two benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    beneficial_count: int = Field(default=0, ge=0)
    harmful_count: int = Field(default=0, ge=0)
    inconclusive_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    direct_observation_count: int = Field(default=0, ge=0)
    predictor_estimate_count: int = Field(default=0, ge=0)
    low_confidence_count: int = Field(default=0, ge=0)
    observed_network_penalty_delta_ms: float | None = Field(default=None)
    modeled_cost_delta: float | None = Field(default=None)
    budget_exhausted_delta: int = 0
    notes: list[str] = Field(default_factory=list)


class BenchmarkDeploymentTarget(StrEnum):
    """Deployment shape exercised by a benchmark or replay run."""

    LOCAL_DEV = "local_dev"
    COMPOSE = "compose"
    KIND = "kind"
    REMOTE = "remote"


class Phase4SchemaCompatibility(StrEnum):
    """Compatibility note for Phase 4 typed extensions on top of benchmark v2."""

    V2_EXTENDED = "phase4_v2_extended"


class ControlPlaneReportMetadata(BaseModel):
    """Relevant Phase 4 metadata surfaced in reports and traces."""

    model_config = ConfigDict(extra="forbid")

    compatibility: Phase4SchemaCompatibility = Phase4SchemaCompatibility.V2_EXTENDED
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    admission_decision: AdmissionDecision | None = None
    circuit_breaker_state: CircuitBreakerState | None = None
    sticky_route: StickyRouteRecord | None = None
    canary_policy: CanaryPolicy | None = None
    shadow_policy: ShadowPolicy | None = None
    shadow_decision: ShadowRouteEvidence | None = None
    policy_reference: PolicyReference | None = None
    topology_reference: TopologySnapshotReference | None = None
    execution_observation: RouteExecutionObservation | None = None
    telemetry_metadata: RouteTelemetryMetadata | None = None


class DeployedTopologyEndpoint(BaseModel):
    """One explicit addressable endpoint recorded for a benchmark environment."""

    model_config = ConfigDict(extra="forbid")

    endpoint_id: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1, max_length=64)
    address: str = Field(min_length=1, max_length=512)
    topology_schema_version: TopologySchemaVersion = TopologySchemaVersion.V1
    transport: WorkerTransportType | None = None
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.HOST_NATIVE
    locality_class: WorkerLocalityClass = WorkerLocalityClass.UNKNOWN
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    region: str | None = Field(default=None, min_length=1, max_length=128)
    zone: str | None = Field(default=None, min_length=1, max_length=128)
    network_profile: NetworkProfile = NetworkProfile.UNKNOWN
    metadata: dict[str, str] = Field(default_factory=dict)


class WorkloadGenerationConfig(BaseModel):
    """Deterministic synthetic workload generation settings."""

    model_config = ConfigDict(extra="forbid")

    pattern: WorkloadPattern = WorkloadPattern.UNIFORM
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    shared_prefix: str | None = Field(default=None, min_length=1, max_length=512)
    burst_size: int = Field(default=1, ge=1, le=1024)

    @model_validator(mode="after")
    def validate_pattern_settings(self) -> WorkloadGenerationConfig:
        if self.pattern is WorkloadPattern.REPEATED_PREFIX and self.shared_prefix is None:
            msg = "repeated_prefix workloads require shared_prefix"
            raise ValueError(msg)
        if self.pattern is not WorkloadPattern.BURSTY and self.burst_size != 1:
            msg = "burst_size may only be greater than 1 for bursty workloads"
            raise ValueError(msg)
        return self


class WorkloadItem(BaseModel):
    """One logical workload item inside a benchmark or replay scenario."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=128)
    family: WorkloadScenarioFamily
    prompt: str = Field(min_length=1, max_length=8192)
    metadata: dict[str, str] = Field(default_factory=dict)
    shared_prefix: str | None = Field(default=None, min_length=1, max_length=512)
    burst_index: int | None = Field(default=None, ge=1)
    burst_size: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_burst_settings(self) -> WorkloadItem:
        if (self.burst_index is None) != (self.burst_size is None):
            msg = "burst_index and burst_size must either both be set or both be omitted"
            raise ValueError(msg)
        return self


class ExecutionTarget(BaseModel):
    """A concrete benchmark or replay execution target."""

    model_config = ConfigDict(extra="forbid")

    target_type: ExecutionTargetType = ExecutionTargetType.ROUTING_POLICY
    model_alias: str = Field(min_length=1, max_length=128)
    routing_policy: RoutingPolicy | None = None
    pinned_backend: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_target(self) -> ExecutionTarget:
        if self.target_type is ExecutionTargetType.ROUTING_POLICY and self.routing_policy is None:
            msg = "routing_policy targets require routing_policy"
            raise ValueError(msg)
        if self.target_type is ExecutionTargetType.PINNED_BACKEND and self.pinned_backend is None:
            msg = "pinned_backend targets require pinned_backend"
            raise ValueError(msg)
        if self.target_type is ExecutionTargetType.LOGICAL_ALIAS:
            self.routing_policy = None
            self.pinned_backend = None
        if self.target_type is ExecutionTargetType.ROUTING_POLICY:
            self.pinned_backend = None
        return self


class BenchmarkWarmupConfig(BaseModel):
    """Warmup settings for a benchmark or replay run."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    request_count: int = Field(default=0, ge=0, le=1024)
    concurrency: int = Field(default=1, ge=1, le=1024)

    @model_validator(mode="after")
    def validate_enabled_shape(self) -> BenchmarkWarmupConfig:
        if not self.enabled and self.request_count != 0:
            msg = "warmup.request_count must be 0 when warmup is disabled"
            raise ValueError(msg)
        return self


class BenchmarkConfigKnobCategory(StrEnum):
    """High-level grouping for one benchmark-relevant config knob."""

    SERVING = "serving"
    ROUTING = "routing"
    SCHEDULING = "scheduling"
    HYBRID_EXECUTION = "hybrid_execution"
    WORKER_LAUNCH = "worker_launch"
    SEARCH_SPACE = "search_space"


class BenchmarkConfigKnob(BaseModel):
    """One benchmark-relevant knob captured in a canonical config snapshot."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=256)
    category: BenchmarkConfigKnobCategory
    config_path: str = Field(min_length=1, max_length=256)
    value: bool | int | float | str | list[str] | None = None
    source_scope: str = Field(default="global", min_length=1, max_length=128)
    notes: list[str] = Field(default_factory=list)


class BenchmarkConfigFingerprint(BaseModel):
    """Canonical fingerprint for an immutable benchmark or replay config snapshot."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = Field(default="sha256_canonical_json", min_length=1, max_length=64)
    value: str = Field(min_length=16, max_length=128)


class BenchmarkConfigSnapshot(BaseModel):
    """Immutable configuration truth recorded with one benchmark or replay run."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    fingerprint: BenchmarkConfigFingerprint
    knobs: list[BenchmarkConfigKnob] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BenchmarkRunConfig(BaseModel):
    """Execution configuration for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    benchmark_mode: str = Field(default="synthetic", min_length=1, max_length=64)
    execution_target: ExecutionTarget | None = None
    concurrency: int = Field(default=1, ge=1, le=1024)
    warmup: BenchmarkWarmupConfig = Field(default_factory=BenchmarkWarmupConfig)
    replay_mode: ReplayMode | None = None
    trace_capture_mode: TraceCaptureMode = TraceCaptureMode.OFF
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    shadow_sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    session_affinity_ttl_seconds: float | None = Field(default=None, gt=0.0, le=86_400.0)
    config_fingerprint: BenchmarkConfigFingerprint | None = None
    immutable_config: BenchmarkConfigSnapshot | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class WorkloadScenario(BaseModel):
    """Configuration and generated items for a benchmark workload."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    family: WorkloadScenarioFamily = WorkloadScenarioFamily.SHORT_CHAT
    policy: RoutingPolicy
    workload_shape: WorkloadShape
    request_count: int = Field(ge=1)
    input_messages_per_request: int = Field(default=1, ge=1)
    stream: bool = False
    max_output_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    prompt_template: str | None = Field(default=None, min_length=1, max_length=512)
    workload_generation: WorkloadGenerationConfig = Field(
        default_factory=WorkloadGenerationConfig
    )
    items: list[WorkloadItem] = Field(default_factory=list)
    scenario_seed: int | None = Field(default=None, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def validate_workload_shape(self) -> WorkloadScenario:
        if self.model_alias is None:
            self.model_alias = self.model
        if self.scenario_seed is None:
            self.scenario_seed = self.workload_generation.seed
        if self.items and len(self.items) != self.request_count:
            msg = "items length must match request_count when items are provided"
            raise ValueError(msg)
        return self


class BenchmarkScenario(WorkloadScenario):
    """Backward-compatible workload scenario name used by Phase 2 helpers."""


class CapturedTraceRecord(BaseModel):
    """A captured trace record suitable for replay planning."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1, max_length=128)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    execution_target: ExecutionTarget
    logical_alias: str | None = Field(default=None, min_length=1, max_length=128)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    request_class: RequestClass = RequestClass.STANDARD
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_features: RequestFeatureVector | None = None
    policy_reference: PolicyReference | None = None
    topology_reference: TopologySnapshotReference | None = None
    route_decision: RouteDecision | None = None
    chosen_backend: str | None = Field(default=None, min_length=1, max_length=128)
    stream: bool = False
    fallback_used: bool = False
    status_code: int | None = Field(default=None, ge=100, le=599)
    latency_ms: float | None = Field(default=None, ge=0.0)
    ttft_ms: float | None = Field(default=None, ge=0.0)
    output_tokens: int | None = Field(default=None, ge=0)
    error: str | None = Field(default=None, max_length=512)
    error_category: str | None = Field(default=None, min_length=1, max_length=128)
    capture_mode: TraceCaptureMode = TraceCaptureMode.METADATA_ONLY
    normalized_request_payload: dict[str, object] | None = None
    normalized_response_payload: dict[str, object] | None = None
    control_plane_metadata: ControlPlaneReportMetadata | None = None
    hybrid_context: HybridExecutionContext | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trace_outcome(self) -> CapturedTraceRecord:
        if self.logical_alias is None:
            self.logical_alias = self.execution_target.model_alias
        if self.route_decision is not None and self.route_decision.telemetry_metadata is not None:
            self.tenant_id = self.route_decision.telemetry_metadata.tenant_id
            self.request_class = self.route_decision.telemetry_metadata.request_class
            if self.route_decision.session_affinity_key is not None:
                self.session_id = self.route_decision.session_affinity_key.session_id
        if self.route_decision is not None:
            if self.request_features is None:
                self.request_features = self.route_decision.request_features
            if self.policy_reference is None:
                self.policy_reference = self.route_decision.policy_reference
            if self.topology_reference is None:
                self.topology_reference = self.route_decision.topology_reference
        if self.status_code is not None and self.status_code >= 400 and self.error is None:
            msg = "failed trace records must include an error message"
            raise ValueError(msg)
        return self


class ReplayPlan(BaseModel):
    """A typed replay plan derived from benchmark artifacts or captured traces."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_run_id: str = Field(min_length=1, max_length=128)
    source_schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    execution_target: ExecutionTarget
    replay_mode: ReplayMode = ReplayMode.SEQUENTIAL
    concurrency: int = Field(default=1, ge=1, le=1024)
    warmup: BenchmarkWarmupConfig = Field(default_factory=BenchmarkWarmupConfig)
    config_fingerprint: BenchmarkConfigFingerprint | None = None
    immutable_config: BenchmarkConfigSnapshot | None = None
    request_ids: list[str] = Field(default_factory=list)
    trace_record_ids: list[str] = Field(default_factory=list)
    requests: list[ReplayRequest] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_requests(self) -> ReplayPlan:
        if self.requests:
            request_ids = [request.source_request_id for request in self.requests]
            trace_record_ids = [
                request.source_trace_record_id for request in self.requests
            ]
            if self.request_ids and self.request_ids != request_ids:
                msg = "request_ids must match requests when both are provided"
                raise ValueError(msg)
            if self.trace_record_ids and self.trace_record_ids != trace_record_ids:
                msg = "trace_record_ids must match requests when both are provided"
                raise ValueError(msg)
            if not self.request_ids:
                self.request_ids = request_ids
            if not self.trace_record_ids:
                self.trace_record_ids = trace_record_ids
        if self.config_fingerprint is None and self.immutable_config is not None:
            self.config_fingerprint = self.immutable_config.fingerprint
        return self


class ReplayRequest(BaseModel):
    """One replayable request entry derived from a captured trace."""

    model_config = ConfigDict(extra="forbid")

    replay_request_id: str = Field(min_length=1, max_length=128)
    source_request_id: str = Field(min_length=1, max_length=128)
    source_trace_record_id: str = Field(min_length=1, max_length=128)
    order_index: int = Field(ge=0)
    original_request_timestamp: datetime | None = None
    original_interarrival_ms: float | None = Field(default=None, ge=0.0)
    scheduled_offset_ms: float | None = Field(default=None, ge=0.0)
    stream: bool = False
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    request_class: RequestClass = RequestClass.STANDARD
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_features: RequestFeatureVector | None = None
    policy_reference: PolicyReference | None = None
    topology_reference: TopologySnapshotReference | None = None
    hybrid_context: HybridExecutionContext | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ComparisonRunSummary(BaseModel):
    """Aggregate summary for a comparative run."""

    model_config = ConfigDict(extra="forbid")

    compared_run_ids: list[str] = Field(min_length=1)
    compared_targets: list[ExecutionTarget] = Field(min_length=1)
    result_count: int = Field(ge=1)
    best_result_by_latency: str = Field(min_length=1, max_length=128)
    best_result_by_throughput: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_counts(self) -> ComparisonRunSummary:
        if self.result_count != len(self.compared_run_ids):
            msg = "result_count must match compared_run_ids length"
            raise ValueError(msg)
        if self.result_count != len(self.compared_targets):
            msg = "result_count must match compared_targets length"
            raise ValueError(msg)
        return self


class BenchmarkComparisonSideSummary(BaseModel):
    """Comparable summary for one side of a two-target evaluation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    execution_target: ExecutionTarget
    request_count: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    p50_ttft_ms: float | None = Field(default=None, ge=0.0)
    p95_ttft_ms: float | None = Field(default=None, ge=0.0)
    avg_tokens_per_second: float | None = Field(default=None, ge=0.0)
    p95_tokens_per_second: float | None = Field(default=None, ge=0.0)
    route_distribution: dict[str, int] = Field(default_factory=dict)
    backend_distribution: dict[str, int] = Field(default_factory=dict)
    hybrid_summary: HybridBenchmarkSummary | None = None


class ScenarioDelta(BaseModel):
    """Per-item or per-source delta between two runs."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=256)
    left_request_id: str = Field(min_length=1, max_length=128)
    right_request_id: str = Field(min_length=1, max_length=128)
    latency_delta_ms: float = 0.0
    ttft_delta_ms: float | None = Field(default=None)
    tokens_per_second_delta: float | None = Field(default=None)
    modeled_cost_delta: float | None = Field(default=None)
    success_changed: bool = False
    backend_changed: bool = False
    route_changed: bool = False
    evidence_kind: SimulationEvidenceKind = Field(
        default_factory=lambda: SimulationEvidenceKind.UNSUPPORTED
    )
    hybrid_outcome: HybridComparisonOutcome = HybridComparisonOutcome.UNSUPPORTED
    condition_sources: list[HybridConditionSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BenchmarkComparisonDeltaSummary(BaseModel):
    """Aggregated diff summary for a two-target evaluation."""

    model_config = ConfigDict(extra="forbid")

    request_count_delta: int = 0
    success_rate_delta: float = 0.0
    error_rate_delta: float = 0.0
    fallback_rate_delta: float = 0.0
    p50_latency_delta_ms: float = 0.0
    p95_latency_delta_ms: float = 0.0
    p50_ttft_delta_ms: float | None = None
    p95_ttft_delta_ms: float | None = None
    avg_tokens_per_second_delta: float | None = None
    p95_tokens_per_second_delta: float | None = None
    route_distribution_delta: dict[str, int] = Field(default_factory=dict)
    backend_distribution_delta: dict[str, int] = Field(default_factory=dict)
    hybrid_summary: HybridComparisonSummary | None = None
    notable_scenario_deltas: list[ScenarioDelta] = Field(default_factory=list)


class BenchmarkTargetComparisonArtifact(BaseModel):
    """Serializable side-by-side comparison for two benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    comparison_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_kind: ComparisonSourceKind
    source_name: str = Field(min_length=1, max_length=256)
    request_count: int = Field(ge=0)
    left: BenchmarkComparisonSideSummary
    right: BenchmarkComparisonSideSummary
    delta: BenchmarkComparisonDeltaSummary
    comparison_summary: ComparisonRunSummary | None = None

    @model_validator(mode="after")
    def validate_sides(self) -> BenchmarkTargetComparisonArtifact:
        if self.comparison_summary is None:
            self.comparison_summary = ComparisonRunSummary(
                compared_run_ids=[self.left.run_id, self.right.run_id],
                compared_targets=[self.left.execution_target, self.right.execution_target],
                result_count=2,
                best_result_by_latency=(
                    self.left.run_id
                    if self.left.p50_latency_ms <= self.right.p50_latency_ms
                    else self.right.run_id
                ),
                best_result_by_throughput=_best_run_id_by_optional_metric(
                    left_run_id=self.left.run_id,
                    right_run_id=self.right.run_id,
                    left_value=self.left.avg_tokens_per_second,
                    right_value=self.right.avg_tokens_per_second,
                ),
            )
        return self


def _best_run_id_by_optional_metric(
    *,
    left_run_id: str,
    right_run_id: str,
    left_value: float | None,
    right_value: float | None,
) -> str | None:
    if left_value is None and right_value is None:
        return None
    if left_value is None:
        return right_run_id
    if right_value is None:
        return left_run_id
    return left_run_id if left_value >= right_value else right_run_id


class ReportMetadata(BaseModel):
    """Metadata for a derived benchmark report."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    report_format: ReportFormat
    source_of_truth: ReportSourceOfTruth = ReportSourceOfTruth.BENCHMARK_ARTIFACT
    source_run_ids: list[str] = Field(min_length=1)
    source_schema_versions: list[BenchmarkArtifactSchemaVersion] = Field(min_length=1)
    control_plane_metadata: ControlPlaneReportMetadata | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class EnvironmentSnapshot(BaseModel):
    """Portable host/platform summary for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    host_name: str = Field(default_factory=platform_module.node)
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform_module.platform)
    machine: str = Field(default_factory=platform_module.machine)
    processor: str = Field(default_factory=platform_module.processor)

    @classmethod
    def from_environment(
        cls,
        environment: BenchmarkEnvironmentMetadata,
    ) -> EnvironmentSnapshot:
        return cls(
            python_version=environment.python_version,
            platform=environment.platform,
            machine=environment.machine,
        )


class BenchmarkRequestRecord(BaseModel):
    """Per-request benchmark result."""

    request_id: str = Field(min_length=1, max_length=128)
    workload_item_id: str | None = Field(default=None, min_length=1, max_length=128)
    scenario_family: WorkloadScenarioFamily | None = None
    source_request_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_trace_record_id: str | None = Field(default=None, min_length=1, max_length=128)
    replay_correlation_id: str | None = Field(default=None, min_length=1, max_length=128)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    request_class: RequestClass = RequestClass.STANDARD
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_features: RequestFeatureVector | None = None
    policy_reference: PolicyReference | None = None
    topology_reference: TopologySnapshotReference | None = None
    backend_name: str = Field(min_length=1, max_length=128)
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    backend_type: str | None = Field(default=None, min_length=1, max_length=64)
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    model_identifier: str | None = Field(default=None, min_length=1, max_length=512)
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0.0)
    ttft_ms: float | None = Field(default=None, ge=0.0)
    output_tokens: int | None = Field(default=None, ge=0)
    tokens_per_second: float | None = Field(default=None, ge=0.0)
    queue_delay_ms: float | None = Field(default=None, ge=0.0)
    routing_policy: RoutingPolicy | None = None
    route_candidate_count: int | None = Field(default=None, ge=1)
    fallback_used: bool = False
    fallback_backend_name: str | None = Field(default=None, min_length=1, max_length=128)
    route_reason: str | None = Field(default=None, min_length=1, max_length=512)
    route_explanation: RouteExplanation | None = None
    route_decision: RouteDecision | None = None
    execution_observation: RouteExecutionObservation | None = None
    control_plane_metadata: ControlPlaneReportMetadata | None = None
    hybrid_context: HybridExecutionContext | None = None
    success: bool
    status_code: int = Field(ge=100, le=599)
    usage: UsageStats | None = None
    cache_observation: CacheObservation | None = None
    error: str | None = Field(default=None, max_length=512)
    error_category: str | None = Field(default=None, min_length=1, max_length=128)
    error_details: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_fields(self) -> BenchmarkRequestRecord:
        if self.completed_at < self.started_at:
            msg = "completed_at must be greater than or equal to started_at"
            raise ValueError(msg)
        if self.success and self.error is not None:
            msg = "successful benchmark records cannot include an error message"
            raise ValueError(msg)
        if not self.success and self.error is None:
            msg = "failed benchmark records must include an error message"
            raise ValueError(msg)
        if self.route_decision is not None:
            if self.routing_policy is None:
                self.routing_policy = self.route_decision.policy
            if self.request_features is None:
                self.request_features = self.route_decision.request_features
            if self.policy_reference is None:
                self.policy_reference = self.route_decision.policy_reference
            if self.topology_reference is None:
                self.topology_reference = self.route_decision.topology_reference
            if self.execution_observation is None:
                self.execution_observation = self.route_decision.execution_observation
            if self.route_candidate_count is None:
                self.route_candidate_count = len(self.route_decision.considered_backends)
            if self.route_explanation is None:
                self.route_explanation = self.route_decision.explanation
            if self.route_explanation is not None and self.route_reason is None:
                self.route_reason = self.route_explanation.compact_reason()
            if self.backend_name != self.route_decision.backend_name:
                self.fallback_used = True
                if self.fallback_backend_name is None:
                    self.fallback_backend_name = self.backend_name
            telemetry_metadata = self.route_decision.telemetry_metadata
            if telemetry_metadata is not None:
                self.tenant_id = telemetry_metadata.tenant_id
                self.request_class = telemetry_metadata.request_class
                if self.route_decision.session_affinity_key is not None:
                    self.session_id = self.route_decision.session_affinity_key.session_id
                if self.control_plane_metadata is None:
                    self.control_plane_metadata = ControlPlaneReportMetadata(
                        tenant_id=telemetry_metadata.tenant_id,
                        session_id=self.session_id,
                        admission_decision=self.route_decision.admission_decision,
                        circuit_breaker_state=self.route_decision.circuit_breaker_state,
                        sticky_route=self.route_decision.sticky_route,
                        canary_policy=self.route_decision.canary_policy,
                        shadow_policy=self.route_decision.shadow_policy,
                        shadow_decision=self.route_decision.shadow_decision,
                        policy_reference=self.route_decision.policy_reference,
                        topology_reference=self.route_decision.topology_reference,
                        execution_observation=self.route_decision.execution_observation,
                        telemetry_metadata=telemetry_metadata,
                    )
        if self.execution_observation is not None:
            if self.backend_instance_id is None:
                self.backend_instance_id = self.execution_observation.backend_instance_id
            if self.queue_delay_ms is None:
                self.queue_delay_ms = self.execution_observation.queue_delay_ms
        if self.route_explanation is not None:
            compact_reason = self.route_explanation.compact_reason()
            if self.route_reason is None:
                self.route_reason = compact_reason
            if self.route_candidate_count is None:
                self.route_candidate_count = len(self.route_explanation.candidates)
        if self.fallback_backend_name is not None:
            self.fallback_used = True
        return self


class CacheObservation(BaseModel):
    """Cache-related hints or metrics recorded per request when available."""

    model_config = ConfigDict(extra="forbid")

    supports_prefix_cache: bool | None = None
    supports_prompt_cache_read: bool | None = None
    supports_prompt_cache_write: bool | None = None
    supports_kv_cache_reuse: bool | None = None
    prefix_cache_hit: bool | None = None
    prompt_cache_read_hit: bool | None = None
    prompt_cache_write_performed: bool | None = None


class FamilyBenchmarkSummary(BaseModel):
    """Aggregate summary for one workload scenario family."""

    model_config = ConfigDict(extra="forbid")

    family: WorkloadScenarioFamily
    request_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    avg_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    avg_ttft_ms: float | None = Field(default=None, ge=0.0)
    p50_ttft_ms: float | None = Field(default=None, ge=0.0)
    p95_ttft_ms: float | None = Field(default=None, ge=0.0)
    total_output_tokens: int = Field(default=0, ge=0)
    avg_output_tokens: float = Field(default=0.0, ge=0.0)
    avg_tokens_per_second: float | None = Field(default=None, ge=0.0)
    p95_tokens_per_second: float | None = Field(default=None, ge=0.0)
    fallback_count: int = Field(default=0, ge=0)
    chosen_backend_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> FamilyBenchmarkSummary:
        if self.success_count + self.failure_count != self.request_count:
            msg = "family success_count + failure_count must equal request_count"
            raise ValueError(msg)
        return self


class BenchmarkSummary(BaseModel):
    """Summary statistics for a benchmark run."""

    request_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    avg_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    avg_ttft_ms: float | None = Field(default=None, ge=0.0)
    p50_ttft_ms: float | None = Field(default=None, ge=0.0)
    p95_ttft_ms: float | None = Field(default=None, ge=0.0)
    total_output_tokens: int = Field(default=0, ge=0)
    avg_output_tokens: float = Field(default=0.0, ge=0.0)
    avg_tokens_per_second: float | None = Field(default=None, ge=0.0)
    p95_tokens_per_second: float | None = Field(default=None, ge=0.0)
    fallback_count: int = Field(default=0, ge=0)
    chosen_backend_counts: dict[str, int] = Field(default_factory=dict)
    hybrid_summary: HybridBenchmarkSummary | None = None
    family_summaries: dict[WorkloadScenarioFamily, FamilyBenchmarkSummary] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_counts(self) -> BenchmarkSummary:
        if self.success_count + self.failure_count != self.request_count:
            msg = "success_count + failure_count must equal request_count"
            raise ValueError(msg)
        return self


class HistoricalDimension(StrEnum):
    """Typed dimensions that historical summaries can group by."""

    MODEL_ALIAS = "model_alias"
    BACKEND_TYPE = "backend_type"
    BACKEND_NAME = "backend_name"
    BACKEND_INSTANCE_ID = "backend_instance_id"
    POLICY_ID = "policy_id"
    REQUEST_CLASS = "request_class"
    TENANT_ID = "tenant_id"
    INPUT_LENGTH_BUCKET = "input_length_bucket"
    HISTORY_DEPTH_BUCKET = "history_depth_bucket"
    WORKLOAD_TAG = "workload_tag"
    PREFIX_HOTNESS = "prefix_hotness"
    CACHE_OPPORTUNITY = "cache_opportunity"
    LOCALITY_BENEFIT = "locality_benefit"


class HistoricalSummaryKey(BaseModel):
    """Canonical group key for historical performance summaries."""

    model_config = ConfigDict(extra="forbid")

    dimensions: list[HistoricalDimension] = Field(default_factory=list)
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    backend_type: str | None = Field(default=None, min_length=1, max_length=64)
    backend_name: str | None = Field(default=None, min_length=1, max_length=128)
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_class: RequestClass | None = None
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    input_length_bucket: InputLengthBucket | None = None
    history_depth_bucket: HistoryDepthBucket | None = None
    workload_tag: WorkloadTag | None = None
    prefix_hotness: PrefixHotness | None = None
    cache_opportunity: bool | None = None
    locality_benefit: bool | None = None


class HistoricalSummaryQuery(BaseModel):
    """Query for aggregating historical benchmark records."""

    model_config = ConfigDict(extra="forbid")

    group_by: list[HistoricalDimension] = Field(default_factory=list)
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    backend_type: str | None = Field(default=None, min_length=1, max_length=64)
    backend_name: str | None = Field(default=None, min_length=1, max_length=128)
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_class: RequestClass | None = None
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    input_length_bucket: InputLengthBucket | None = None
    history_depth_bucket: HistoryDepthBucket | None = None
    workload_tag: WorkloadTag | None = None
    prefix_hotness: PrefixHotness | None = None
    cache_opportunity: bool | None = None
    locality_benefit: bool | None = None


class PerformanceBucketSummary(BaseModel):
    """Count of observations inside one transparent metric bucket."""

    model_config = ConfigDict(extra="forbid")

    bucket_label: str = Field(min_length=1, max_length=64)
    lower_bound: float = Field(ge=0.0)
    upper_bound: float | None = Field(default=None, ge=0.0)
    count: int = Field(default=0, ge=0)


class HistoricalMetricSummary(BaseModel):
    """Transparent numeric summary used by historical aggregations."""

    model_config = ConfigDict(extra="forbid")

    observation_count: int = Field(default=0, ge=0)
    avg: float | None = Field(default=None, ge=0.0)
    ewma: float | None = Field(default=None, ge=0.0)
    p50: float | None = Field(default=None, ge=0.0)
    p95: float | None = Field(default=None, ge=0.0)
    buckets: list[PerformanceBucketSummary] = Field(default_factory=list)


class HistoricalPerformanceSummary(BaseModel):
    """Aggregated historical outcome summary for one query/group slice."""

    model_config = ConfigDict(extra="forbid")

    key: HistoricalSummaryKey
    request_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    cache_opportunity_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    locality_benefit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    error_category_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: HistoricalMetricSummary
    ttft_ms: HistoricalMetricSummary
    tokens_per_second: HistoricalMetricSummary
    queue_delay_ms: HistoricalMetricSummary

    @model_validator(mode="after")
    def validate_counts(self) -> HistoricalPerformanceSummary:
        if self.success_count + self.failure_count != self.request_count:
            msg = "success_count + failure_count must equal request_count"
            raise ValueError(msg)
        return self


class HistoricalPerformanceIndex(BaseModel):
    """Typed result of a historical aggregation query."""

    model_config = ConfigDict(extra="forbid")

    query: HistoricalSummaryQuery
    source_record_count: int = Field(ge=0)
    matched_record_count: int = Field(ge=0)
    summaries: list[HistoricalPerformanceSummary] = Field(default_factory=list)


class CandidateRouteEstimateContext(BaseModel):
    """Candidate route features used for transparent historical estimation."""

    model_config = ConfigDict(extra="forbid")

    model_alias: str = Field(min_length=1, max_length=128)
    backend_name: str = Field(min_length=1, max_length=128)
    backend_type: str | None = Field(default=None, min_length=1, max_length=64)
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_class: RequestClass = RequestClass.STANDARD
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    input_length_bucket: InputLengthBucket | None = None
    history_depth_bucket: HistoryDepthBucket | None = None
    workload_tags: list[WorkloadTag] = Field(default_factory=list)
    prefix_hotness: PrefixHotness | None = None
    cache_opportunity: bool | None = None
    locality_benefit: bool | None = None


class HistoricalRouteEstimate(BaseModel):
    """Transparent expected-outcome estimate for one candidate route."""

    model_config = ConfigDict(extra="forbid")

    context: CandidateRouteEstimateContext
    evidence_key: HistoricalSummaryKey | None = None
    evidence_count: int = Field(default=0, ge=0)
    sufficient_data: bool = False
    insufficiency_reason: str | None = Field(default=None, min_length=1, max_length=256)
    expected_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_latency_ms: float | None = Field(default=None, ge=0.0)
    expected_ttft_ms: float | None = Field(default=None, ge=0.0)
    expected_tokens_per_second: float | None = Field(default=None, ge=0.0)
    expected_queue_delay_ms: float | None = Field(default=None, ge=0.0)
    rationale: list[str] = Field(default_factory=list)


class CounterfactualObjective(StrEnum):
    """Explainable objectives for offline policy comparison."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    BALANCED = "balanced"


class AdaptivePolicyMode(StrEnum):
    """Safe rollout posture for adaptive policy recommendations."""

    SHADOW = "shadow"
    RECOMMEND = "recommend"
    GUARDED = "guarded"


class SimulationSourceKind(StrEnum):
    """Authoritative source family for offline simulation cases."""

    BENCHMARK_RUN = "benchmark_run"
    CAPTURED_TRACE = "captured_trace"


class SimulationEvidenceKind(StrEnum):
    """Evidence quality for one simulated candidate or recommendation."""

    DIRECT_OBSERVATION = "direct_observation"
    PREDICTOR_ESTIMATE = "predictor_estimate"
    LOW_CONFIDENCE_ESTIMATE = "low_confidence_estimate"
    UNSUPPORTED = "unsupported"


class SimulationBucketDimension(StrEnum):
    """Bucket dimensions used for offline simulation summaries."""

    MODEL_ALIAS = "model_alias"
    TENANT_ID = "tenant_id"
    INPUT_LENGTH_BUCKET = "input_length_bucket"
    BACKEND_INSTANCE_ID = "backend_instance_id"


class RecommendationScopeKind(StrEnum):
    """Scope of a human-facing routing recommendation."""

    MODEL_ALIAS = "model_alias"
    REQUEST_CLASS = "request_class"
    REPEATED_PREFIX_BACKEND = "repeated_prefix_backend"
    REPEATED_PREFIX_INSTANCE = "repeated_prefix_instance"


class RecommendationDisposition(StrEnum):
    """Operator-facing recommendation posture."""

    PREFER_POLICY = "prefer_policy"
    KEEP_SHADOW_ONLY = "keep_shadow_only"
    NO_CHANGE = "no_change"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PREFER_BACKEND = "prefer_backend"
    AVOID_BACKEND = "avoid_backend"


class RecommendationConfidence(StrEnum):
    """Honest confidence categories for guidance output."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class AdaptivePolicyGuardrails(BaseModel):
    """Portable guardrails for safe adaptive recommendations."""

    model_config = ConfigDict(extra="forbid")

    require_sufficient_data: bool = True
    max_predicted_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_predicted_latency_regression_ms: float | None = Field(default=None, ge=0.0)
    require_observed_backend_evidence: bool = False


class ExplainablePolicySpec(BaseModel):
    """Serializable offline policy/scorer configuration."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(default="phase6.v1", min_length=1, max_length=64)
    objective: CounterfactualObjective = CounterfactualObjective.BALANCED
    mode: AdaptivePolicyMode = AdaptivePolicyMode.RECOMMEND
    min_evidence_count: int = Field(default=3, ge=1, le=100_000)
    guardrails: AdaptivePolicyGuardrails = Field(default_factory=AdaptivePolicyGuardrails)
    rationale: list[str] = Field(default_factory=list)


class CounterfactualCandidateScore(BaseModel):
    """Explainable counterfactual score for one backend candidate."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1, max_length=128)
    score: float | None = None
    eligible: bool = False
    evidence_kind: SimulationEvidenceKind = SimulationEvidenceKind.UNSUPPORTED
    evidence_count: int = Field(default=0, ge=0)
    rejection_reason: str | None = Field(default=None, min_length=1, max_length=256)
    estimate: HistoricalRouteEstimate | None = None
    directly_observed: bool = False
    observed_latency_ms: float | None = Field(default=None, ge=0.0)
    observed_success: bool | None = None
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    confidence_note: str | None = Field(default=None, min_length=1, max_length=256)
    rationale: list[str] = Field(default_factory=list)


class PolicyRecommendation(BaseModel):
    """One explainable recommendation for a recorded request."""

    model_config = ConfigDict(extra="forbid")

    observed_backend: str = Field(min_length=1, max_length=128)
    recommended_backend: str = Field(min_length=1, max_length=128)
    recommendation_changed: bool = False
    guardrail_blocked: bool = False
    insufficient_data: bool = False
    evidence_kind: SimulationEvidenceKind = SimulationEvidenceKind.UNSUPPORTED
    rationale: list[str] = Field(default_factory=list)


class SimulationBucketSummary(BaseModel):
    """Aggregate summary for one bucket inside a simulated policy result."""

    model_config = ConfigDict(extra="forbid")

    dimension: SimulationBucketDimension
    bucket_key: str = Field(min_length=1, max_length=256)
    request_count: int = Field(ge=0)
    changed_count: int = Field(default=0, ge=0)
    direct_observation_count: int = Field(default=0, ge=0)
    predictor_estimate_count: int = Field(default=0, ge=0)
    low_confidence_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    avg_projected_latency_ms: float | None = Field(default=None, ge=0.0)


class CounterfactualSimulationRecord(BaseModel):
    """Per-request offline simulation result for one explainable policy."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    source_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_kind: SimulationSourceKind = SimulationSourceKind.BENCHMARK_RUN
    source_record_id: str | None = Field(default=None, min_length=1, max_length=128)
    workload_item_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_trace_record_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    request_class: RequestClass = RequestClass.STANDARD
    request_features: RequestFeatureVector | None = None
    observed_backend: str = Field(min_length=1, max_length=128)
    observed_backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    observed_latency_ms: float | None = Field(default=None, ge=0.0)
    observed_success: bool | None = None
    topology_reference: TopologySnapshotReference | None = None
    candidate_scores: list[CounterfactualCandidateScore] = Field(min_length=1)
    recommendation: PolicyRecommendation


class CounterfactualSimulationSummary(BaseModel):
    """Aggregate summary for one offline simulation run."""

    model_config = ConfigDict(extra="forbid")

    request_count: int = Field(ge=0)
    changed_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    direct_observation_count: int = Field(default=0, ge=0)
    predictor_estimate_count: int = Field(default=0, ge=0)
    low_confidence_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    insufficient_data_count: int = Field(ge=0)
    guardrail_block_count: int = Field(ge=0)
    observed_backend_counts: dict[str, int] = Field(default_factory=dict)
    recommended_backend_counts: dict[str, int] = Field(default_factory=dict)
    projected_avg_latency_ms: float | None = Field(default=None, ge=0.0)
    projected_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    projected_avg_tokens_per_second: float | None = Field(default=None, ge=0.0)
    bucket_summaries: list[SimulationBucketSummary] = Field(default_factory=list)
    limitation_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> CounterfactualSimulationSummary:
        if self.changed_count + self.unchanged_count != self.request_count:
            msg = "changed_count + unchanged_count must equal request_count"
            raise ValueError(msg)
        return self


class CounterfactualSimulationArtifact(BaseModel):
    """Serializable offline simulation artifact for policy comparison."""

    model_config = ConfigDict(extra="forbid")

    schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    simulation_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_run_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    historical_source_run_ids: list[str] = Field(default_factory=list)
    historical_source_trace_ids: list[str] = Field(default_factory=list)
    policy: ExplainablePolicySpec
    summary: CounterfactualSimulationSummary
    records: list[CounterfactualSimulationRecord] = Field(default_factory=list)
    topology_references: list[TopologySnapshotReference] = Field(default_factory=list)
    deployed_topology: list[DeployedTopologyEndpoint] = Field(default_factory=list)
    worker_instance_inventory: list[BackendInstance] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_records(self) -> CounterfactualSimulationArtifact:
        if self.summary.request_count != len(self.records):
            msg = "summary.request_count must match the number of simulation records"
            raise ValueError(msg)
        self.source_run_ids = sorted(set(self.source_run_ids))
        self.source_trace_ids = sorted(set(self.source_trace_ids))
        self.historical_source_run_ids = sorted(set(self.historical_source_run_ids))
        self.historical_source_trace_ids = sorted(set(self.historical_source_trace_ids))
        return self


class CounterfactualSimulationComparisonArtifact(BaseModel):
    """Comparable offline simulation results across several candidate policies."""

    model_config = ConfigDict(extra="forbid")

    schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    simulation_comparison_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_run_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    historical_source_run_ids: list[str] = Field(default_factory=list)
    historical_source_trace_ids: list[str] = Field(default_factory=list)
    policies: list[ExplainablePolicySpec] = Field(min_length=1)
    evaluations: list[CounterfactualSimulationArtifact] = Field(min_length=1)
    topology_references: list[TopologySnapshotReference] = Field(default_factory=list)
    deployed_topology: list[DeployedTopologyEndpoint] = Field(default_factory=list)
    worker_instance_inventory: list[BackendInstance] = Field(default_factory=list)
    limitation_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_evaluations(self) -> CounterfactualSimulationComparisonArtifact:
        self.source_run_ids = sorted(set(self.source_run_ids))
        self.source_trace_ids = sorted(set(self.source_trace_ids))
        self.historical_source_run_ids = sorted(set(self.historical_source_run_ids))
        self.historical_source_trace_ids = sorted(set(self.historical_source_trace_ids))
        policy_ids = [policy.policy_id for policy in self.policies]
        evaluation_ids = [evaluation.policy.policy_id for evaluation in self.evaluations]
        if policy_ids != evaluation_ids:
            msg = "policies must align one-to-one with evaluations in the same order"
            raise ValueError(msg)
        return self


class RecommendationEvidenceWindow(BaseModel):
    """Source and time bounds for one recommendation report."""

    model_config = ConfigDict(extra="forbid")

    source_run_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    historical_source_run_ids: list[str] = Field(default_factory=list)
    historical_source_trace_ids: list[str] = Field(default_factory=list)
    window_started_at: datetime | None = None
    window_ended_at: datetime | None = None


class RoutingPolicyGuidance(BaseModel):
    """One scoped, evidence-based routing recommendation."""

    model_config = ConfigDict(extra="forbid")

    scope_kind: RecommendationScopeKind
    scope_key: str = Field(min_length=1, max_length=256)
    recommendation: RecommendationDisposition
    confidence: RecommendationConfidence = RecommendationConfidence.INSUFFICIENT
    sample_size: int = Field(default=0, ge=0)
    workload_buckets: list[str] = Field(default_factory=list)
    recommended_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    recommended_target: str | None = Field(default=None, min_length=1, max_length=128)
    recommended_target_type: str | None = Field(default=None, min_length=1, max_length=64)
    evidence_summary: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    confidence_notes: list[str] = Field(default_factory=list)
    notable_regressions: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)


class PolicyRecommendationReportArtifact(BaseModel):
    """Human-facing routing-policy recommendation report derived from artifacts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    recommendation_report_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence_window: RecommendationEvidenceWindow
    recommendations: list[RoutingPolicyGuidance] = Field(default_factory=list)
    notable_limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class BenchmarkEnvironmentMetadata(EnvironmentSnapshot):
    """Environment and config details needed to reproduce a benchmark run."""

    benchmark_mode: str = Field(min_length=1, max_length=64)
    gateway_base_url: str | None = Field(default=None, min_length=1, max_length=256)
    deployment_target: BenchmarkDeploymentTarget | None = None
    deployment_profile: DeploymentProfile | None = None
    config_profile_name: str | None = Field(default=None, min_length=1, max_length=128)
    metrics_url: str | None = Field(default=None, min_length=1, max_length=256)
    stream: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    session_affinity_ttl_seconds: float | None = Field(default=None, gt=0.0, le=86_400.0)
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    shadow_sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    deployed_topology: list[DeployedTopologyEndpoint] = Field(default_factory=list)
    worker_instance_inventory: list[BackendInstance] = Field(default_factory=list)
    hybrid_execution: HybridExecutionRuntimeSummary | None = None
    remote_workers: RemoteWorkerLifecycleRuntimeSummary | None = None
    remote_worker_snapshot: RegisteredRemoteWorkerSnapshot | None = None
    control_plane_image: BackendImageMetadata | None = None
    topology_capture_source: str | None = Field(default=None, min_length=1, max_length=128)
    topology_reference: TopologySnapshotReference | None = None
    control_plane_metadata: ControlPlaneReportMetadata | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class BenchmarkRunArtifact(BaseModel):
    """Serializable benchmark artifact for reproducible analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    run_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str | None = Field(default=None, min_length=7, max_length=40)
    git_revision: str | None = Field(default=None, min_length=7, max_length=40)
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    scenario: BenchmarkScenario
    policy: RoutingPolicy
    execution_target: ExecutionTarget | None = None
    run_config: BenchmarkRunConfig = Field(default_factory=BenchmarkRunConfig)
    backends_involved: list[str] = Field(min_length=1)
    backend_types_involved: list[str] = Field(default_factory=list)
    model_aliases_involved: list[str] = Field(default_factory=list)
    request_count: int = Field(ge=0)
    summary: BenchmarkSummary
    environment: BenchmarkEnvironmentMetadata = Field(
        default_factory=lambda: BenchmarkEnvironmentMetadata(benchmark_mode="synthetic")
    )
    environment_snapshot: EnvironmentSnapshot | None = None
    records: list[BenchmarkRequestRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> BenchmarkRunArtifact:
        if self.request_count != len(self.records):
            msg = "request_count must match the number of records"
            raise ValueError(msg)
        if self.summary.request_count != self.request_count:
            msg = "summary.request_count must match request_count"
            raise ValueError(msg)
        if self.backend_types_involved != sorted(set(self.backend_types_involved)):
            msg = "backend_types_involved must be sorted and unique"
            raise ValueError(msg)
        if self.model_aliases_involved != sorted(set(self.model_aliases_involved)):
            msg = "model_aliases_involved must be sorted and unique"
            raise ValueError(msg)
        if self.git_revision is None and self.git_sha is not None:
            self.git_revision = self.git_sha
        if self.git_sha is None and self.git_revision is not None:
            self.git_sha = self.git_revision
        if self.execution_target is None:
            self.execution_target = ExecutionTarget(
                target_type=ExecutionTargetType.ROUTING_POLICY,
                model_alias=self.scenario.model_alias or self.scenario.model,
                routing_policy=self.policy,
            )
        if self.run_config.execution_target is None:
            self.run_config.execution_target = self.execution_target
        if self.run_config.benchmark_mode != self.environment.benchmark_mode:
            self.run_config.benchmark_mode = self.environment.benchmark_mode
        if self.run_config.timeout_seconds != self.environment.timeout_seconds:
            self.run_config.timeout_seconds = self.environment.timeout_seconds
        if self.model_alias is None:
            self.model_alias = self.execution_target.model_alias
        if self.environment_snapshot is None:
            self.environment_snapshot = EnvironmentSnapshot.from_environment(self.environment)
        if (
            self.environment.topology_reference is None
            and self.environment.topology_capture_source is not None
        ):
            self.environment.topology_reference = TopologySnapshotReference(
                topology_snapshot_id=self.run_id,
                capture_source=self.environment.topology_capture_source,
                artifact_run_id=self.run_id,
            )
        if self.environment.topology_reference is not None:
            for record in self.records:
                if record.topology_reference is None:
                    record.topology_reference = self.environment.topology_reference
        return self


class BenchmarkPolicyComparison(BaseModel):
    """Comparable benchmark result for a single routing policy."""

    comparison_label: str = Field(min_length=1, max_length=128)
    policy: RoutingPolicy
    internal_backend_pin: str | None = Field(default=None, min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    backends_involved: list[str] = Field(min_length=1)
    execution_target: ExecutionTarget | None = None
    summary: BenchmarkSummary

    @model_validator(mode="after")
    def validate_execution_target(self) -> BenchmarkPolicyComparison:
        if self.execution_target is None:
            target_type = ExecutionTargetType.ROUTING_POLICY
            if self.internal_backend_pin is not None:
                target_type = ExecutionTargetType.PINNED_BACKEND
            self.execution_target = ExecutionTarget(
                target_type=target_type,
                model_alias="comparison-target",
                routing_policy=None if self.internal_backend_pin else self.policy,
                pinned_backend=self.internal_backend_pin,
            )
        return self


class BenchmarkComparisonArtifact(BaseModel):
    """Serializable comparison artifact across several routing policies."""

    model_config = ConfigDict(extra="forbid")

    schema_version: BenchmarkArtifactSchemaVersion = BenchmarkArtifactSchemaVersion.V3
    run_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scenario_name: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    workload_shape: WorkloadShape
    request_count: int = Field(ge=1)
    results: list[BenchmarkPolicyComparison] = Field(min_length=1)
    comparison_summary: ComparisonRunSummary | None = None
    best_policy_by_latency: RoutingPolicy
    best_policy_by_throughput: RoutingPolicy | None = None
    best_result_by_latency: str = Field(min_length=1, max_length=128)
    best_result_by_throughput: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_policies(self) -> BenchmarkComparisonArtifact:
        labels = [result.comparison_label for result in self.results]
        if len(labels) != len(set(labels)):
            msg = "results must contain unique comparison_label values"
            raise ValueError(msg)
        if self.comparison_summary is None:
            self.comparison_summary = ComparisonRunSummary(
                compared_run_ids=[result.run_id for result in self.results],
                compared_targets=[
                    ExecutionTarget(
                        target_type=(
                            ExecutionTargetType.PINNED_BACKEND
                            if result.internal_backend_pin is not None
                            else ExecutionTargetType.ROUTING_POLICY
                        ),
                        model_alias=self.model,
                        routing_policy=(
                            None if result.internal_backend_pin is not None else result.policy
                        ),
                        pinned_backend=result.internal_backend_pin,
                    )
                    for result in self.results
                ],
                result_count=len(self.results),
                best_result_by_latency=self.best_result_by_latency,
                best_result_by_throughput=self.best_result_by_throughput,
            )
        return self
