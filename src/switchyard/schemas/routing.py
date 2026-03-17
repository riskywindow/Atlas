"""Routing-related schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from switchyard.schemas.backend import BackendDeployment, EngineType


class RoutingPolicy(StrEnum):
    """Routing policies available in Phase 0."""

    LATENCY_FIRST = "latency_first"
    BALANCED = "balanced"
    QUALITY_FIRST = "quality_first"
    LOCAL_ONLY = "local_only"


class TenantTier(StrEnum):
    """Coarse tenant classes used for bounded admission decisions."""

    STANDARD = "standard"
    PRIORITY = "priority"


class RequestClass(StrEnum):
    """Logical request classes used by Phase 4 control-plane policy."""

    STANDARD = "standard"
    LATENCY_SENSITIVE = "latency_sensitive"
    BULK = "bulk"


class WorkloadShape(StrEnum):
    """Broad request shapes the router can reason about."""

    INTERACTIVE = "interactive"
    BATCH = "batch"
    EVALUATION = "evaluation"


class InputLengthBucket(StrEnum):
    """Coarse prompt-size buckets for deterministic routing features."""

    TINY = "tiny"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VERY_LONG = "very_long"


class HistoryDepthBucket(StrEnum):
    """Coarse conversation-history buckets for deterministic routing features."""

    SINGLE_TURN = "single_turn"
    SHORT_HISTORY = "short_history"
    DEEP_HISTORY = "deep_history"


class WorkloadTag(StrEnum):
    """Explainable workload tags derived without model inference."""

    SHORT_CHAT = "short_chat"
    LONG_CONTEXT = "long_context"
    REPEATED_PREFIX = "repeated_prefix"
    BURST_CANDIDATE = "burst_candidate"
    SESSION_CONTINUATION = "session_continuation"
    STREAMING = "streaming"
    LATENCY_SENSITIVE = "latency_sensitive"
    BULK = "bulk"
    PRIORITY_TENANT = "priority_tenant"


class PrefixHotness(StrEnum):
    """Coarse recent-prefix hotness for cache/locality-aware analysis."""

    COLD = "cold"
    WARM = "warm"
    HOT = "hot"


class RouteEligibilityState(StrEnum):
    """Eligibility state for a concrete backend deployment during routing."""

    ELIGIBLE = "eligible"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class AdmissionDecisionState(StrEnum):
    """Admission outcome for a request at route-selection time."""

    ADMITTED = "admitted"
    QUEUED = "queued"
    REJECTED = "rejected"
    BYPASSED = "bypassed"


class AdmissionReasonCode(StrEnum):
    """Stable reason codes for overload and queue outcomes."""

    DISABLED = "disabled"
    GLOBAL_CONCURRENCY_LIMIT = "global_concurrency_limit"
    TENANT_CONCURRENCY_LIMIT = "tenant_concurrency_limit"
    QUEUE_FULL = "queue_full"
    QUEUE_TIMEOUT = "queue_timeout"
    STALE_REQUEST = "stale_request"


class LimiterMode(StrEnum):
    """Generic limiter operating mode."""

    DISABLED = "disabled"
    ENFORCING = "enforcing"
    COOLDOWN = "cooldown"


class CircuitBreakerPhase(StrEnum):
    """Portable circuit-breaker phases."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AffinityDisposition(StrEnum):
    """Outcome for session-affinity evaluation."""

    NOT_REQUESTED = "not_requested"
    CREATED = "created"
    REUSED = "reused"
    MISSED = "missed"
    EXPIRED = "expired"


class RolloutDisposition(StrEnum):
    """Outcome for canary or weighted rollout selection."""

    NONE = "none"
    BASELINE = "baseline"
    CANARY = "canary"


class ShadowDisposition(StrEnum):
    """Outcome for shadow-routing evaluation."""

    DISABLED = "disabled"
    SKIPPED = "skipped"
    SHADOWED = "shadowed"


class RouteSelectionReasonCode(StrEnum):
    """Stable reason codes for route selection and analysis."""

    POLICY_SCORE = "policy_score"
    SHADOW_POLICY_SCORE = "shadow_policy_score"
    SESSION_AFFINITY = "session_affinity"
    CANARY_BASELINE = "canary_baseline"
    CANARY_SELECTED = "canary_selected"
    SHADOW_SKIPPED = "shadow_skipped"
    SHADOW_LAUNCHED = "shadow_launched"
    FALLBACK_EXECUTION = "fallback_execution"


class TenantIdentity(BaseModel):
    """Portable tenant identity and request class metadata."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=128)
    tenant_tier: TenantTier = TenantTier.STANDARD
    request_class: RequestClass = RequestClass.STANDARD
    attributes: dict[str, str] = Field(default_factory=dict)


class QueueSnapshot(BaseModel):
    """Point-in-time bounded queue state."""

    model_config = ConfigDict(extra="forbid")

    queue_name: str = Field(min_length=1, max_length=128)
    current_depth: int = Field(ge=0)
    max_depth: int = Field(ge=0)
    queued_requests: int = Field(default=0, ge=0)
    queue_timeout_ms: int | None = Field(default=None, ge=1)
    oldest_entry_age_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_depths(self) -> QueueSnapshot:
        if self.current_depth > self.max_depth:
            msg = "current_depth must not exceed max_depth"
            raise ValueError(msg)
        return self


class LimiterState(BaseModel):
    """Snapshot of a tenant or route limiter."""

    model_config = ConfigDict(extra="forbid")

    limiter_key: str = Field(min_length=1, max_length=256)
    mode: LimiterMode = LimiterMode.ENFORCING
    in_flight_requests: int = Field(default=0, ge=0)
    concurrency_limit: int = Field(ge=1)
    queue_snapshot: QueueSnapshot | None = None
    cooldown_until: datetime | None = None

    @model_validator(mode="after")
    def validate_inflight(self) -> LimiterState:
        if self.in_flight_requests > self.concurrency_limit and self.mode is LimiterMode.DISABLED:
            msg = "disabled limiters must not report in_flight_requests above concurrency_limit"
            raise ValueError(msg)
        return self


class AdmissionDecision(BaseModel):
    """Explicit admission decision recorded by the control plane."""

    model_config = ConfigDict(extra="forbid")

    state: AdmissionDecisionState
    reason_code: AdmissionReasonCode | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=256)
    limiter_key: str | None = Field(default=None, min_length=1, max_length=256)
    queue_snapshot: QueueSnapshot | None = None
    queue_position: int | None = Field(default=None, ge=1)
    request_timeout_ms: int | None = Field(default=None, ge=1)
    cooldown_until: datetime | None = None
    queue_wait_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_reason(self) -> AdmissionDecision:
        if self.state is AdmissionDecisionState.REJECTED and self.reason is None:
            msg = "rejected admission decisions must include a reason"
            raise ValueError(msg)
        return self


class CircuitBreakerState(BaseModel):
    """Portable state snapshot for backend protection."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1, max_length=128)
    phase: CircuitBreakerPhase = CircuitBreakerPhase.CLOSED
    failure_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    last_failure_at: datetime | None = None
    opened_at: datetime | None = None
    cooldown_until: datetime | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=256)


class SessionAffinityKey(BaseModel):
    """Stable key for sticky multi-turn routing."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    serving_target: str = Field(min_length=1, max_length=128)


class StickyRouteRecord(BaseModel):
    """Sticky-route binding for a session affinity key."""

    model_config = ConfigDict(extra="forbid")

    affinity_key: SessionAffinityKey
    backend_name: str = Field(min_length=1, max_length=128)
    bound_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    reason: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_ttl(self) -> StickyRouteRecord:
        if self.expires_at <= self.bound_at:
            msg = "expires_at must be greater than bound_at"
            raise ValueError(msg)
        return self


class WeightedBackendAllocation(BaseModel):
    """One weighted rollout target."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1, max_length=128)
    percentage: float = Field(gt=0.0, le=100.0)


class CanaryPolicy(BaseModel):
    """Explainable canary or weighted rollout rule."""

    model_config = ConfigDict(extra="forbid")

    policy_name: str = Field(min_length=1, max_length=128)
    serving_target: str = Field(min_length=1, max_length=128)
    enabled: bool = False
    allocations: list[WeightedBackendAllocation] = Field(default_factory=list)
    baseline_backend: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_allocations(self) -> CanaryPolicy:
        total = sum(allocation.percentage for allocation in self.allocations)
        if total > 100.0:
            msg = "canary allocation percentages must sum to 100 or less"
            raise ValueError(msg)
        return self


class ShadowPolicy(BaseModel):
    """Opt-in shadow-routing policy."""

    model_config = ConfigDict(extra="forbid")

    policy_name: str = Field(min_length=1, max_length=128)
    enabled: bool = False
    serving_target: str | None = Field(default=None, min_length=1, max_length=128)
    tenant_id: str | None = Field(default=None, min_length=1, max_length=128)
    request_class: RequestClass | None = None
    target_alias: str | None = Field(default=None, min_length=1, max_length=128)
    target_backend: str | None = Field(default=None, min_length=1, max_length=128)
    sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    capture_response: bool = False

    @model_validator(mode="after")
    def validate_enabled_shape(self) -> ShadowPolicy:
        if self.enabled and self.target_backend is None and self.target_alias is None:
            msg = "enabled shadow policies must set target_backend or target_alias"
            raise ValueError(msg)
        if self.target_backend is not None and self.target_alias is not None:
            msg = "shadow policies may set either target_backend or target_alias, not both"
            raise ValueError(msg)
        return self


class RouteAnnotations(BaseModel):
    """Typed annotations for overload, breaker, affinity, and rollout decisions."""

    model_config = ConfigDict(extra="forbid")

    overload_state: AdmissionDecisionState | None = None
    breaker_phase: CircuitBreakerPhase | None = None
    affinity_disposition: AffinityDisposition = AffinityDisposition.NOT_REQUESTED
    rollout_disposition: RolloutDisposition = RolloutDisposition.NONE
    shadow_disposition: ShadowDisposition = ShadowDisposition.DISABLED
    notes: list[str] = Field(default_factory=list)


class RouteTelemetryMetadata(BaseModel):
    """Portable route metadata for logs, telemetry, traces, and reports."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=128)
    tenant_tier: TenantTier = TenantTier.STANDARD
    request_class: RequestClass = RequestClass.STANDARD
    session_affinity_enabled: bool = False
    admission_control_enabled: bool = False
    circuit_breaker_enabled: bool = False
    canary_enabled: bool = False
    shadow_enabled: bool = False
    labels: dict[str, str] = Field(default_factory=dict)


class PolicyReference(BaseModel):
    """Stable policy identifier captured in route and artifact records."""

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(default="phase6.v1", min_length=1, max_length=64)


class TopologySnapshotReference(BaseModel):
    """Reference to the deployment topology snapshot used for analysis."""

    model_config = ConfigDict(extra="forbid")

    topology_snapshot_id: str = Field(min_length=1, max_length=128)
    capture_source: str = Field(min_length=1, max_length=128)
    captured_at: datetime | None = None
    artifact_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    metadata: dict[str, str] = Field(default_factory=dict)


class RequestFeatureVector(BaseModel):
    """Deterministic request and locality signals extracted before scoring."""

    model_config = ConfigDict(extra="forbid")

    feature_version: str = Field(default="phase6.v2", min_length=1, max_length=32)
    message_count: int = Field(ge=1)
    system_message_count: int = Field(default=0, ge=0)
    user_message_count: int = Field(default=0, ge=0)
    assistant_message_count: int = Field(default=0, ge=0)
    tool_message_count: int = Field(default=0, ge=0)
    prompt_character_count: int = Field(ge=0)
    prompt_token_estimate: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    expected_total_tokens: int = Field(ge=1)
    input_length_bucket: InputLengthBucket = InputLengthBucket.SHORT
    history_depth_bucket: HistoryDepthBucket = HistoryDepthBucket.SINGLE_TURN
    workload_tags: list[WorkloadTag] = Field(default_factory=list)
    stream: bool = False
    request_class: RequestClass = RequestClass.STANDARD
    tenant_tier: TenantTier = TenantTier.STANDARD
    internal_backend_pinned: bool = False
    conversation_continuation: bool = False
    repeated_prefix_candidate: bool = False
    prefix_character_count: int = Field(default=0, ge=0)
    prefix_fingerprint: str | None = Field(default=None, min_length=8, max_length=64)
    locality_key: str = Field(min_length=8, max_length=64)
    session_affinity_expected: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> RequestFeatureVector:
        counted_messages = (
            self.system_message_count
            + self.user_message_count
            + self.assistant_message_count
            + self.tool_message_count
        )
        if counted_messages != self.message_count:
            msg = "role-specific message counts must sum to message_count"
            raise ValueError(msg)
        if self.expected_total_tokens < self.prompt_token_estimate:
            msg = "expected_total_tokens must be greater than or equal to prompt_token_estimate"
            raise ValueError(msg)
        if self.prefix_character_count == 0 and self.prefix_fingerprint is not None:
            msg = "prefix_fingerprint requires a positive prefix_character_count"
            raise ValueError(msg)
        if self.prefix_character_count > 0 and self.prefix_fingerprint is None:
            msg = "prefix_character_count requires prefix_fingerprint"
            raise ValueError(msg)
        return self


class PrefixLocalitySignal(BaseModel):
    """Decision-time repeated-prefix and warm-locality evidence."""

    model_config = ConfigDict(extra="forbid")

    signal_version: str = Field(default="phase6.v1", min_length=1, max_length=32)
    serving_target: str = Field(min_length=1, max_length=128)
    locality_key: str = Field(min_length=8, max_length=64)
    prefix_fingerprint: str | None = Field(default=None, min_length=8, max_length=64)
    repeated_prefix_detected: bool = False
    recent_request_count: int = Field(default=0, ge=0)
    hotness: PrefixHotness = PrefixHotness.COLD
    cache_opportunity: bool = False
    likely_benefits_from_locality: bool = False
    preferred_backend: str | None = Field(default=None, min_length=1, max_length=128)
    preferred_backend_request_count: int = Field(default=0, ge=0)
    preferred_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    preferred_instance_request_count: int = Field(default=0, ge=0)
    candidate_local_backend: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_local_backend_request_count: int = Field(default=0, ge=0)
    recent_backend_counts: dict[str, int] = Field(default_factory=dict)
    recent_instance_counts: dict[str, int] = Field(default_factory=dict)
    session_affinity_enabled: bool = False
    session_affinity_backend: str | None = Field(default=None, min_length=1, max_length=128)
    affinity_conflict: bool = False
    last_seen_at: datetime | None = None

    @model_validator(mode="after")
    def validate_signal(self) -> PrefixLocalitySignal:
        if self.repeated_prefix_detected and self.prefix_fingerprint is None:
            msg = "repeated_prefix_detected requires prefix_fingerprint"
            raise ValueError(msg)
        if self.affinity_conflict and self.session_affinity_backend is None:
            msg = "affinity_conflict requires session_affinity_backend"
            raise ValueError(msg)
        return self


class ShadowRouteEvidence(BaseModel):
    """Explainable shadow-routing decision captured alongside the primary route."""

    model_config = ConfigDict(extra="forbid")

    policy_name: str = Field(min_length=1, max_length=128)
    disposition: ShadowDisposition
    target_backend: str | None = Field(default=None, min_length=1, max_length=128)
    target_alias: str | None = Field(default=None, min_length=1, max_length=128)
    sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_reason: str | None = Field(default=None, min_length=1, max_length=256)
    score: float | None = None

    @model_validator(mode="after")
    def validate_targets(self) -> ShadowRouteEvidence:
        if self.target_backend is not None and self.target_alias is not None:
            msg = "shadow evidence may set target_backend or target_alias, not both"
            raise ValueError(msg)
        return self


class RouteExecutionObservation(BaseModel):
    """Runtime outcomes learned after the route decision was made."""

    model_config = ConfigDict(extra="forbid")

    executed_backend: str | None = Field(default=None, min_length=1, max_length=128)
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)
    queue_delay_ms: float | None = Field(default=None, ge=0.0)
    ttft_ms: float | None = Field(default=None, ge=0.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    output_tokens: int | None = Field(default=None, ge=0)
    status_code: int | None = Field(default=None, ge=100, le=599)
    error_category: str | None = Field(default=None, min_length=1, max_length=128)
    final_outcome: str | None = Field(default=None, min_length=1, max_length=128)


class RouteCandidateExplanation(BaseModel):
    """Deterministic explanation for one backend candidate."""

    backend_name: str = Field(min_length=1, max_length=128)
    serving_target: str = Field(min_length=1, max_length=128)
    eligibility_state: RouteEligibilityState
    score: float | None = None
    reason_codes: list[RouteSelectionReasonCode] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    rejection_reason: str | None = Field(default=None, min_length=1, max_length=256)
    deployment: BackendDeployment | None = None
    engine_type: EngineType | None = None
    backend_instance_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_state_reason(self) -> RouteCandidateExplanation:
        if (
            self.eligibility_state is RouteEligibilityState.REJECTED
            and self.rejection_reason is None
        ):
            msg = "rejected candidates must include a rejection_reason"
            raise ValueError(msg)
        return self


class RouteExplanation(BaseModel):
    """Structured routing explanation for logs, tests, and benchmarks."""

    serving_target: str = Field(min_length=1, max_length=128)
    candidates: list[RouteCandidateExplanation] = Field(min_length=1)
    request_features: RequestFeatureVector | None = None
    prefix_locality_signal: PrefixLocalitySignal | None = None
    policy_reference: PolicyReference | None = None
    selected_backend: str = Field(min_length=1, max_length=128)
    selection_reason_codes: list[RouteSelectionReasonCode] = Field(default_factory=list)
    selected_reason: list[str] = Field(default_factory=list)
    tie_breaker: str | None = Field(default=None, min_length=1, max_length=128)
    executed_backend: str | None = Field(default=None, min_length=1, max_length=128)
    fallback_used: bool = False
    execution_events: list[str] = Field(default_factory=list)
    final_outcome: str | None = Field(default=None, min_length=1, max_length=128)
    shadow_evaluations: list[ShadowPolicyExplanation] = Field(default_factory=list)

    def compact_reason(self) -> str:
        """Return a stable, compact explanation string for logs and metrics labels."""

        selected_reason = "; ".join(self.selected_reason) if self.selected_reason else "selected"
        parts = [
            f"target={self.serving_target}",
            f"selected={self.selected_backend}",
            f"reason={selected_reason}",
        ]
        if self.tie_breaker is not None:
            parts.append(f"tie_breaker={self.tie_breaker}")
        if self.final_outcome is not None:
            parts.append(f"outcome={self.final_outcome}")
        return " | ".join(parts)


class ShadowPolicyExplanation(BaseModel):
    """Non-binding shadow policy/scorer evaluation for the same request."""

    model_config = ConfigDict(extra="forbid")

    policy_reference: PolicyReference
    selected_backend: str = Field(min_length=1, max_length=128)
    candidates: list[RouteCandidateExplanation] = Field(min_length=1)
    selection_reason_codes: list[RouteSelectionReasonCode] = Field(default_factory=list)
    selected_reason: list[str] = Field(default_factory=list)
    tie_breaker: str | None = Field(default=None, min_length=1, max_length=128)


class RequestContext(BaseModel):
    """Routing-relevant metadata attached to a request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    policy: RoutingPolicy = RoutingPolicy.BALANCED
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE
    request_class: RequestClass = RequestClass.STANDARD
    max_latency_ms: int | None = Field(default=None, ge=1)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    internal_backend_pin: str | None = Field(default=None, min_length=1, max_length=128)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    tenant_tier: TenantTier = TenantTier.STANDARD
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    tenant: TenantIdentity | None = None
    session_affinity_key: SessionAffinityKey | None = None
    request_features: RequestFeatureVector | None = None
    prefix_locality_signal: PrefixLocalitySignal | None = None

    @model_validator(mode="after")
    def validate_phase4_context(self) -> RequestContext:
        if self.tenant is None:
            self.tenant = TenantIdentity(
                tenant_id=self.tenant_id,
                tenant_tier=self.tenant_tier,
                request_class=self.request_class,
            )
        if self.tenant.tenant_id != self.tenant_id:
            msg = "tenant.tenant_id must match tenant_id"
            raise ValueError(msg)
        if self.tenant.tenant_tier is not self.tenant_tier:
            msg = "tenant.tenant_tier must match tenant_tier"
            raise ValueError(msg)
        if self.tenant.request_class is not self.request_class:
            msg = "tenant.request_class must match request_class"
            raise ValueError(msg)
        if self.session_affinity_key is not None and self.session_id is None:
            msg = "session_id must be set when session_affinity_key is provided"
            raise ValueError(msg)
        if (
            self.session_affinity_key is not None
            and self.session_affinity_key.session_id != self.session_id
        ):
            msg = "session_affinity_key.session_id must match session_id"
            raise ValueError(msg)
        return self


class RouteDecision(BaseModel):
    """Result of a routing decision."""

    backend_name: str = Field(min_length=1, max_length=128)
    serving_target: str = Field(min_length=1, max_length=128)
    policy: RoutingPolicy
    request_id: str = Field(min_length=1, max_length=128)
    workload_shape: WorkloadShape
    rationale: list[str] = Field(min_length=1)
    score: float | None = None
    considered_backends: list[str] = Field(min_length=1)
    rejected_backends: dict[str, str] = Field(default_factory=dict)
    admission_limited_backends: dict[str, str] = Field(default_factory=dict)
    protected_backends: dict[str, str] = Field(default_factory=dict)
    degraded_backends: list[str] = Field(default_factory=list)
    fallback_backends: list[str] = Field(default_factory=list)
    admission_decision: AdmissionDecision | None = None
    queue_snapshot: QueueSnapshot | None = None
    limiter_state: LimiterState | None = None
    circuit_breaker_state: CircuitBreakerState | None = None
    session_affinity_key: SessionAffinityKey | None = None
    sticky_route: StickyRouteRecord | None = None
    canary_policy: CanaryPolicy | None = None
    shadow_policy: ShadowPolicy | None = None
    shadow_decision: ShadowRouteEvidence | None = None
    annotations: RouteAnnotations | None = None
    telemetry_metadata: RouteTelemetryMetadata | None = None
    request_features: RequestFeatureVector | None = None
    prefix_locality_signal: PrefixLocalitySignal | None = None
    policy_reference: PolicyReference | None = None
    topology_reference: TopologySnapshotReference | None = None
    selected_deployment: BackendDeployment | None = None
    execution_observation: RouteExecutionObservation | None = None
    explanation: RouteExplanation | None = None

    @model_validator(mode="after")
    def validate_fallback_backends(self) -> RouteDecision:
        if self.policy_reference is None:
            self.policy_reference = PolicyReference(policy_id=self.policy.value)
        if self.backend_name in self.fallback_backends:
            msg = "fallback_backends must not include the chosen backend"
            raise ValueError(msg)
        if self.backend_name not in self.considered_backends:
            msg = "considered_backends must include the chosen backend"
            raise ValueError(msg)
        if self.explanation is not None and self.explanation.selected_backend != self.backend_name:
            msg = "explanation.selected_backend must match backend_name"
            raise ValueError(msg)
        if (
            self.request_features is not None
            and self.explanation is not None
            and self.explanation.request_features is not None
            and self.explanation.request_features != self.request_features
        ):
            msg = "explanation.request_features must match request_features"
            raise ValueError(msg)
        if (
            self.prefix_locality_signal is not None
            and self.explanation is not None
            and self.explanation.prefix_locality_signal is not None
            and self.explanation.prefix_locality_signal != self.prefix_locality_signal
        ):
            msg = "explanation.prefix_locality_signal must match prefix_locality_signal"
            raise ValueError(msg)
        if self.explanation is not None:
            if self.explanation.policy_reference is None:
                self.explanation.policy_reference = self.policy_reference
            elif self.explanation.policy_reference != self.policy_reference:
                msg = "explanation.policy_reference must match policy_reference"
                raise ValueError(msg)
        if (
            self.selected_deployment is not None
            and self.selected_deployment.name != self.backend_name
        ):
            msg = "selected_deployment.name must match backend_name"
            raise ValueError(msg)
        if not set(self.admission_limited_backends).issubset(self.rejected_backends):
            msg = "admission_limited_backends must be a subset of rejected_backends"
            raise ValueError(msg)
        if not set(self.protected_backends).issubset(self.rejected_backends):
            msg = "protected_backends must be a subset of rejected_backends"
            raise ValueError(msg)
        if self.sticky_route is not None and self.sticky_route.backend_name != self.backend_name:
            msg = "sticky_route.backend_name must match backend_name"
            raise ValueError(msg)
        if (
            self.session_affinity_key is not None
            and self.sticky_route is not None
            and self.sticky_route.affinity_key != self.session_affinity_key
        ):
            msg = "sticky_route.affinity_key must match session_affinity_key"
            raise ValueError(msg)
        return self
