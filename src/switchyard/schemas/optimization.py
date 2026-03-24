"""Typed optimization-ready control-plane surfaces."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.backend import BackendInstance
from switchyard.schemas.benchmark import (
    CounterfactualObjective,
    DeployedTopologyEndpoint,
    RecommendationConfidence,
    WorkloadScenarioFamily,
)
from switchyard.schemas.routing import (
    PolicyRolloutMode,
    RequestClass,
    RoutingPolicy,
    TopologySnapshotReference,
    WorkloadTag,
)


class OptimizationKnobType(StrEnum):
    """Portable type label for an exported optimization knob."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    ENUM = "enum"
    STRING_LIST = "string_list"


class ForgeEvidenceSourceKind(StrEnum):
    """Authoritative evidence source families for Forge Stage A evaluation."""

    OBSERVED_RUNTIME = "observed_runtime"
    REPLAYED_BENCHMARK = "replayed_benchmark"
    REPLAYED_TRACE = "replayed_trace"
    COUNTERFACTUAL_SIMULATION = "counterfactual_simulation"


class ForgeTrialRole(StrEnum):
    """Lineage role for one Forge Stage A trial."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"


class ForgeCandidateKind(StrEnum):
    """Typed candidate family for one Forge Stage A trial."""

    ROUTING_POLICY = "routing_policy"
    ROLLOUT_POLICY = "rollout_policy"
    CONFIG_PROFILE = "config_profile"


class OptimizationScopeKind(StrEnum):
    """Where one tuning surface element applies."""

    GLOBAL = "global"
    SERVING_TARGET = "serving_target"
    WORKER_CLASS = "worker_class"
    SCENARIO_FAMILY = "scenario_family"


class OptimizationKnobGroup(StrEnum):
    """High-level grouping for one safely exposed tuning knob."""

    ROUTING_POLICY = "routing_policy"
    POLICY_ROLLOUT = "policy_rollout"
    ADMISSION_CONTROL = "admission_control"
    BACKEND_PROTECTION = "backend_protection"
    SESSION_AFFINITY = "session_affinity"
    HYBRID_EXECUTION = "hybrid_execution"
    RUNTIME_PROFILE = "runtime_profile"


class OptimizationDomainKind(StrEnum):
    """Supported domain shapes for one tunable knob."""

    BOOLEAN = "boolean"
    INTEGER_RANGE = "integer_range"
    FLOAT_RANGE = "float_range"
    ENUM = "enum"
    STRING_LIST = "string_list"


class OptimizationObjectiveMetric(StrEnum):
    """Measurable outputs that Phase 9 can optimize against explicitly."""

    LATENCY_MS = "latency_ms"
    QUEUE_DELAY_MS = "queue_delay_ms"
    TTFT_MS = "ttft_ms"
    TOKENS_PER_SECOND = "tokens_per_second"
    ERROR_RATE = "error_rate"
    REMOTE_SHARE_PERCENT = "remote_share_percent"


class OptimizationGoal(StrEnum):
    """Optimization direction or target style for one objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    AT_MOST = "at_most"
    AT_LEAST = "at_least"


class OptimizationConstraintStrength(StrEnum):
    """Whether a constraint is mandatory or advisory."""

    HARD = "hard"
    SOFT = "soft"


class OptimizationComparisonOperator(StrEnum):
    """Comparison operator for one constraint."""

    LTE = "lte"
    GTE = "gte"
    EQ = "eq"


class OptimizationConstraintDimension(StrEnum):
    """Bounded dimensions that Forge Stage A may constrain explicitly."""

    PREDICTED_ERROR_RATE = "predicted_error_rate"
    PREDICTED_LATENCY_REGRESSION_MS = "predicted_latency_regression_ms"
    CANARY_PERCENTAGE = "canary_percentage"
    SHADOW_SAMPLING_RATE = "shadow_sampling_rate"
    REMOTE_SHARE_PERCENT = "remote_share_percent"
    REMOTE_REQUEST_BUDGET_PER_MINUTE = "remote_request_budget_per_minute"
    REMOTE_CONCURRENCY_CAP = "remote_concurrency_cap"
    OPERATOR_REVIEW_REQUIRED = "operator_review_required"
    LOCAL_PREFERENCE_ENABLED = "local_preference_enabled"


class OptimizationWorkloadSourceKind(StrEnum):
    """Where one workload bundle comes from."""

    BUILT_IN_SCENARIO_FAMILY = "built_in_scenario_family"
    REPLAY_PLAN = "replay_plan"
    TRACE_SET = "trace_set"
    MIXED = "mixed"


class OptimizationScope(BaseModel):
    """Explicit applicability boundary for one optimization element."""

    model_config = ConfigDict(extra="forbid")

    kind: OptimizationScopeKind
    target: str | None = Field(default=None, min_length=1, max_length=128)

    def model_post_init(self, __context: object) -> None:
        if self.kind is OptimizationScopeKind.GLOBAL and self.target is not None:
            msg = "global scopes must not set target"
            raise ValueError(msg)
        if self.kind is not OptimizationScopeKind.GLOBAL and self.target is None:
            msg = "non-global scopes must set target"
            raise ValueError(msg)


class OptimizationKnobDomain(BaseModel):
    """Validation domain for one exposed tuning knob."""

    model_config = ConfigDict(extra="forbid")

    domain_kind: OptimizationDomainKind
    min_value: int | float | None = None
    max_value: int | float | None = None
    step_value: int | float | None = None
    allowed_values: list[str] = Field(default_factory=list)
    nullable: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.domain_kind in {
            OptimizationDomainKind.BOOLEAN,
            OptimizationDomainKind.STRING_LIST,
        }:
            if (
                self.min_value is not None
                or self.max_value is not None
                or self.step_value is not None
                or self.allowed_values
            ):
                msg = (
                    "boolean and string_list domains do not accept numeric bounds or "
                    "allowed_values"
                )
                raise ValueError(msg)
            return
        if self.domain_kind is OptimizationDomainKind.ENUM:
            if not self.allowed_values:
                msg = "enum domains require allowed_values"
                raise ValueError(msg)
            if (
                self.min_value is not None
                or self.max_value is not None
                or self.step_value is not None
            ):
                msg = "enum domains do not accept numeric bounds"
                raise ValueError(msg)
            return
        if not isinstance(self.min_value, (int, float)) or not isinstance(
            self.max_value, (int, float)
        ):
            msg = "numeric range domains require min_value and max_value"
            raise ValueError(msg)
        if self.min_value > self.max_value:
            msg = "numeric range domains require min_value <= max_value"
            raise ValueError(msg)
        if self.allowed_values:
            msg = "numeric range domains must not set allowed_values"
            raise ValueError(msg)
        if self.step_value is not None and self.step_value <= 0:
            msg = "numeric range domains require positive step_value when set"
            raise ValueError(msg)


class OptimizationKnobSurface(BaseModel):
    """One exported optimization-ready knob with current value and bounds."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(min_length=1, max_length=256)
    group: OptimizationKnobGroup = OptimizationKnobGroup.ROUTING_POLICY
    knob_type: OptimizationKnobType
    domain: OptimizationKnobDomain
    current_value: bool | int | float | str | list[str] | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    allowed_values: list[str] = Field(default_factory=list)
    mutable_at_runtime: bool = False
    applies_to: list[OptimizationScope] = Field(
        default_factory=lambda: [OptimizationScope(kind=OptimizationScopeKind.GLOBAL)]
    )
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        expected_domain_kind = {
            OptimizationKnobType.BOOLEAN: OptimizationDomainKind.BOOLEAN,
            OptimizationKnobType.INTEGER: OptimizationDomainKind.INTEGER_RANGE,
            OptimizationKnobType.FLOAT: OptimizationDomainKind.FLOAT_RANGE,
            OptimizationKnobType.ENUM: OptimizationDomainKind.ENUM,
            OptimizationKnobType.STRING_LIST: OptimizationDomainKind.STRING_LIST,
        }[self.knob_type]
        if self.domain.domain_kind is not expected_domain_kind:
            msg = "knob_type must align with domain.domain_kind"
            raise ValueError(msg)
        if self.min_value != self.domain.min_value:
            self.min_value = self.domain.min_value
        if self.max_value != self.domain.max_value:
            self.max_value = self.domain.max_value
        if self.allowed_values != self.domain.allowed_values:
            self.allowed_values = list(self.domain.allowed_values)


class WorkerLaunchPresetScope(StrEnum):
    """Scope for one worker launch preset."""

    HOST_NATIVE = "host_native"
    REMOTE_WORKER = "remote_worker"


class WorkerLaunchPreset(BaseModel):
    """Typed worker launch preset that later benchmark loops may select from."""

    model_config = ConfigDict(extra="forbid")

    preset_name: str = Field(min_length=1, max_length=128)
    scope: WorkerLaunchPresetScope
    warmup_mode: str | None = Field(default=None, min_length=1, max_length=32)
    concurrency_limit: int | None = Field(default=None, ge=1, le=100_000)
    supports_streaming: bool | None = None
    stream_chunk_size: int | None = Field(default=None, ge=1, le=4096)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class OptimizationObjectiveTarget(BaseModel):
    """One explicit Forge Stage A objective target."""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(min_length=1, max_length=128)
    metric: OptimizationObjectiveMetric
    goal: OptimizationGoal
    target_value: float | None = None
    weight: float = Field(default=1.0, gt=0.0, le=1000.0)
    applies_to: list[OptimizationScope] = Field(
        default_factory=lambda: [OptimizationScope(kind=OptimizationScopeKind.GLOBAL)]
    )
    workload_set_ids: list[str] = Field(default_factory=list)
    evidence_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.goal in {OptimizationGoal.AT_MOST, OptimizationGoal.AT_LEAST}:
            if self.target_value is None:
                msg = "target_value is required for bounded objective goals"
                raise ValueError(msg)


class OptimizationConstraint(BaseModel):
    """One hard or soft constraint attached to the optimization contract."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1, max_length=128)
    dimension: OptimizationConstraintDimension
    strength: OptimizationConstraintStrength
    operator: OptimizationComparisonOperator
    threshold_value: bool | int | float | str
    applies_to: list[OptimizationScope] = Field(
        default_factory=lambda: [OptimizationScope(kind=OptimizationScopeKind.GLOBAL)]
    )
    notes: list[str] = Field(default_factory=list)


class OptimizationWorkloadSet(BaseModel):
    """Workload bundle used to evaluate one tuning campaign or objective."""

    model_config = ConfigDict(extra="forbid")

    workload_set_id: str = Field(min_length=1, max_length=128)
    source_kind: OptimizationWorkloadSourceKind
    serving_targets: list[str] = Field(default_factory=list)
    scenario_families: list[WorkloadScenarioFamily] = Field(default_factory=list)
    request_classes: list[RequestClass] = Field(default_factory=list)
    workload_tags: list[WorkloadTag] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if not any(
            (
                self.serving_targets,
                self.scenario_families,
                self.request_classes,
                self.workload_tags,
            )
        ):
            msg = (
                "workload sets must select at least one serving target, scenario "
                "family, request class, or workload tag"
            )
            raise ValueError(msg)


class OptimizationCampaignMetadata(BaseModel):
    """Campaign-level metadata for the formal Phase 9 optimization surface."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1, max_length=128)
    optimization_profile_id: str = Field(min_length=1, max_length=128)
    objective: CounterfactualObjective = CounterfactualObjective.BALANCED
    evidence_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    required_evidence_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    default_workload_set_ids: list[str] = Field(default_factory=list)
    promotion_requires_operator_review: bool = True
    notes: list[str] = Field(default_factory=list)


class OptimizationTrialIdentity(BaseModel):
    """Typed identity for one baseline or candidate trial."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    candidate_kind: ForgeCandidateKind
    config_profile_id: str = Field(min_length=1, max_length=128)
    routing_policy: RoutingPolicy | None = None
    rollout_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    worker_launch_preset: str | None = Field(default=None, min_length=1, max_length=128)
    applies_to: list[OptimizationScope] = Field(
        default_factory=lambda: [OptimizationScope(kind=OptimizationScopeKind.GLOBAL)]
    )
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if (
            self.candidate_kind is ForgeCandidateKind.ROUTING_POLICY
            and self.routing_policy is None
        ):
            msg = "routing_policy candidates require routing_policy"
            raise ValueError(msg)
        if (
            self.candidate_kind is ForgeCandidateKind.ROLLOUT_POLICY
            and self.rollout_policy_id is None
        ):
            msg = "rollout_policy candidates require rollout_policy_id"
            raise ValueError(msg)


class OptimizationExcludedDimension(BaseModel):
    """Explicitly unsupported tuning dimension for the current phase."""

    model_config = ConfigDict(extra="forbid")

    dimension_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=256)
    notes: list[str] = Field(default_factory=list)


class OptimizationEvidenceProfile(BaseModel):
    """Offline evidence posture for Phase 9 Forge Stage A policy search."""

    model_config = ConfigDict(extra="forbid")

    objective: CounterfactualObjective = CounterfactualObjective.BALANCED
    min_evidence_count: int = Field(default=3, ge=1, le=100_000)
    max_predicted_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_predicted_latency_regression_ms: float | None = Field(default=None, ge=0.0)
    require_observed_backend_evidence: bool = False
    promotion_requires_operator_review: bool = True
    notes: list[str] = Field(default_factory=list)


class OptimizationCandidateGenerationStrategy(StrEnum):
    """Explicit first-pass Forge Stage A search strategies."""

    FIXED_BASELINE = "fixed_baseline"
    ONE_FACTOR_AT_A_TIME = "one_factor_at_a_time"
    BOUNDED_GRID_SEARCH = "bounded_grid_search"
    RANDOM_SEARCH = "random_search"
    HEURISTIC_SEED = "heuristic_seed"


class OptimizationCandidateEligibilityStatus(StrEnum):
    """Whether a generated candidate is safe to execute."""

    ELIGIBLE = "eligible"
    PRUNED = "pruned"
    REJECTED = "rejected"


class OptimizationCandidateGenerationMetadata(BaseModel):
    """How one candidate was produced by the search layer."""

    model_config = ConfigDict(extra="forbid")

    strategy: OptimizationCandidateGenerationStrategy
    strategy_index: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    parent_candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    varied_knob_ids: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationCandidateEligibilityRecord(BaseModel):
    """Early safety and constraint decision for a generated candidate."""

    model_config = ConfigDict(extra="forbid")

    status: OptimizationCandidateEligibilityStatus
    eligible: bool = False
    blocking_constraint_ids: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self.eligible = self.status is OptimizationCandidateEligibilityStatus.ELIGIBLE


class OptimizationGeneratedCandidate(BaseModel):
    """Generated candidate plus explicit provenance and eligibility posture."""

    model_config = ConfigDict(extra="forbid")

    trial: OptimizationTrialIdentity
    knob_changes: list[OptimizationKnobChange] = Field(default_factory=list)
    generation: OptimizationCandidateGenerationMetadata
    eligibility: OptimizationCandidateEligibilityRecord
    notes: list[str] = Field(default_factory=list)


class OptimizationCandidateGenerationConfig(BaseModel):
    """Search configuration for simple explicit Forge Stage A candidate generation."""

    model_config = ConfigDict(extra="forbid")

    strategies: list[OptimizationCandidateGenerationStrategy] = Field(
        default_factory=lambda: [
            OptimizationCandidateGenerationStrategy.FIXED_BASELINE,
            OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
            OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH,
            OptimizationCandidateGenerationStrategy.RANDOM_SEARCH,
            OptimizationCandidateGenerationStrategy.HEURISTIC_SEED,
        ]
    )
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    allowed_knob_ids: list[str] = Field(default_factory=list)
    max_one_factor_variants_per_knob: int = Field(default=3, ge=1, le=64)
    max_grid_dimensions: int = Field(default=2, ge=1, le=8)
    max_grid_candidates: int = Field(default=8, ge=1, le=256)
    max_random_candidates: int = Field(default=6, ge=0, le=256)
    heuristic_seed_limit: int = Field(default=3, ge=0, le=64)
    notes: list[str] = Field(default_factory=list)


class OptimizationCandidateGenerationResult(BaseModel):
    """Typed output of the first Forge Stage A candidate-generation layer."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    baseline_trial: OptimizationTrialIdentity
    baseline_generation: OptimizationCandidateGenerationMetadata
    generation_config: OptimizationCandidateGenerationConfig
    eligible_candidates: list[OptimizationGeneratedCandidate] = Field(default_factory=list)
    rejected_candidates: list[OptimizationGeneratedCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationProfile(BaseModel):
    """Exportable optimization surface for later Forge-style tuning."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    active_routing_policy: RoutingPolicy
    active_rollout_mode: PolicyRolloutMode
    allowlisted_routing_policies: list[RoutingPolicy] = Field(default_factory=list)
    allowlisted_rollout_modes: list[PolicyRolloutMode] = Field(default_factory=list)
    candidate_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    shadow_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    hybrid_remote_enabled: bool = False
    worker_launch_presets: list[WorkerLaunchPreset] = Field(default_factory=list)
    evidence: OptimizationEvidenceProfile = Field(default_factory=OptimizationEvidenceProfile)
    knobs: list[OptimizationKnobSurface] = Field(default_factory=list)
    objectives: list[OptimizationObjectiveTarget] = Field(default_factory=list)
    constraints: list[OptimizationConstraint] = Field(default_factory=list)
    workload_sets: list[OptimizationWorkloadSet] = Field(default_factory=list)
    campaign: OptimizationCampaignMetadata | None = None
    baseline_trial: OptimizationTrialIdentity | None = None
    candidate_trials: list[OptimizationTrialIdentity] = Field(default_factory=list)
    excluded_dimensions: list[OptimizationExcludedDimension] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionPlan(BaseModel):
    """Safe promotion posture for one Forge Stage A campaign."""

    model_config = ConfigDict(extra="forbid")

    config_profile_id: str = Field(min_length=1, max_length=128)
    rollout_mode: PolicyRolloutMode
    max_canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    requires_operator_review: bool = True
    reversible_controls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgeTrialLineage(BaseModel):
    """One baseline or candidate trial inside a Forge Stage A campaign."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str = Field(min_length=1, max_length=128)
    parent_trial_id: str | None = Field(default=None, min_length=1, max_length=128)
    trial_role: ForgeTrialRole
    candidate_kind: ForgeCandidateKind
    config_profile_id: str = Field(min_length=1, max_length=128)
    routing_policy: RoutingPolicy | None = None
    rollout_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    rollout_mode: PolicyRolloutMode
    evaluation_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    required_evaluation_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    explainable_recommendations_only: bool = True
    notes: list[str] = Field(default_factory=list)


class ForgeStageACampaign(BaseModel):
    """Typed inspection surface for the first Forge Stage A campaign slice."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    optimization_profile_id: str = Field(min_length=1, max_length=128)
    objective: CounterfactualObjective = CounterfactualObjective.BALANCED
    active_routing_policy: RoutingPolicy
    active_rollout_mode: PolicyRolloutMode
    baseline_trial_id: str = Field(min_length=1, max_length=128)
    evaluation_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    required_evaluation_sources: list[ForgeEvidenceSourceKind] = Field(default_factory=list)
    candidate_count: int = Field(default=0, ge=0)
    explainable_recommendations_only: bool = True
    automatic_promotion_enabled: bool = False
    promotion: ForgePromotionPlan
    trial_lineage: list[ForgeTrialLineage] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class OptimizationArtifactSchemaVersion(StrEnum):
    """Schema versions for optimization campaign and trial artifacts."""

    V1 = "switchyard.optimization.v1"


class OptimizationArtifactStatus(StrEnum):
    """Materialization or freshness status for one optimization artifact."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    STALE = "stale"
    INVALIDATED = "invalidated"


class OptimizationArtifactEvidenceKind(StrEnum):
    """Top-level evidence kind carried by optimization artifacts."""

    OBSERVED = "observed"
    REPLAYED = "replayed"
    SIMULATED = "simulated"
    ESTIMATED = "estimated"


class OptimizationArtifactSourceType(StrEnum):
    """Upstream artifact family referenced by one optimization evidence record."""

    BENCHMARK_RUN = "benchmark_run"
    REPLAY_PLAN = "replay_plan"
    CAPTURED_TRACE = "captured_trace"
    SIMULATION = "simulation"
    SIMULATION_COMPARISON = "simulation_comparison"
    RECOMMENDATION_REPORT = "recommendation_report"
    ESTIMATE_SUMMARY = "estimate_summary"


class OptimizationRecommendationDisposition(StrEnum):
    """Conservative recommendation posture for one trial or campaign."""

    PROMOTE_CANDIDATE = "promote_candidate"
    KEEP_BASELINE = "keep_baseline"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    INVALIDATE_TRIAL = "invalidate_trial"
    NO_CHANGE = "no_change"


class OptimizationPromotionDisposition(StrEnum):
    """Promotion outcome for one candidate after trial evaluation."""

    NO_ACTION = "no_action"
    RECOMMEND_CANARY = "recommend_canary"
    APPROVED_CANARY = "approved_canary"
    PROMOTED_DEFAULT = "promoted_default"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class OptimizationRecommendationLabel(StrEnum):
    """Offline review posture for one candidate recommendation."""

    PROMOTION_ELIGIBLE = "promotion_eligible"
    REVIEW_ONLY = "review_only"
    REJECTED = "rejected"


class OptimizationRecommendationReasonCode(StrEnum):
    """Explainable reason codes for one candidate recommendation."""

    PRIMARY_OBJECTIVE_IMPROVED = "primary_objective_improved"
    PRIMARY_OBJECTIVE_REGRESSED = "primary_objective_regressed"
    SECONDARY_OBJECTIVE_IMPROVED = "secondary_objective_improved"
    SECONDARY_OBJECTIVE_REGRESSED = "secondary_objective_regressed"
    HARD_CONSTRAINT_VIOLATED = "hard_constraint_violated"
    SOFT_CONSTRAINT_VIOLATED = "soft_constraint_violated"
    NON_DOMINATED = "non_dominated"
    DOMINATED = "dominated"
    MIXED_WORKLOAD_TRADEOFF = "mixed_workload_tradeoff"
    WORKLOAD_FAMILY_BENEFIT = "workload_family_benefit"
    WORKLOAD_FAMILY_REGRESSION = "workload_family_regression"
    OBSERVED_EVIDENCE_PRESENT = "observed_evidence_present"
    OBSERVED_EVIDENCE_MISSING = "observed_evidence_missing"
    ESTIMATED_EVIDENCE_PRESENT = "estimated_evidence_present"
    TIED_WITH_PEER = "tied_with_peer"
    NO_MEANINGFUL_DELTA = "no_meaningful_delta"
    PROMOTION_REQUIRES_REVIEW = "promotion_requires_review"


class OptimizationConfigProfileRole(StrEnum):
    """Lifecycle posture for one explicit optimization config profile."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"


class OptimizationConfigProfileSourceKind(StrEnum):
    """Authoritative source used to materialize one config profile artifact."""

    SETTINGS_BASELINE = "settings_baseline"
    CANDIDATE_CONFIGURATION = "candidate_configuration"
    REVIEWED_TRIAL = "reviewed_trial"


class OptimizationConfigProfileValidationIssueKind(StrEnum):
    """Compatibility problem discovered while materializing one config profile."""

    UNKNOWN_KNOB = "unknown_knob"
    CONFIG_PATH_MISMATCH = "config_path_mismatch"
    SCOPE_NOT_ALLOWED = "scope_not_allowed"
    DOMAIN_VIOLATION = "domain_violation"
    ELIGIBILITY_BLOCKED = "eligibility_blocked"
    PROVENANCE_MISMATCH = "provenance_mismatch"


class OptimizationConfigProfileValidationIssue(BaseModel):
    """One validation or compatibility issue found in a config profile."""

    model_config = ConfigDict(extra="forbid")

    issue_kind: OptimizationConfigProfileValidationIssueKind
    knob_id: str | None = Field(default=None, min_length=1, max_length=128)
    detail: str = Field(min_length=1, max_length=256)
    notes: list[str] = Field(default_factory=list)


class OptimizationConfigProfileValidation(BaseModel):
    """Compatibility posture for one explicit optimization config profile."""

    model_config = ConfigDict(extra="forbid")

    compatible: bool = True
    validated_knob_ids: list[str] = Field(default_factory=list)
    issues: list[OptimizationConfigProfileValidationIssue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self.validated_knob_ids = sorted(set(self.validated_knob_ids))
        self.compatible = not self.issues


class OptimizationConfigProfileProvenance(BaseModel):
    """Provenance and recommendation lineage behind one config profile artifact."""

    model_config = ConfigDict(extra="forbid")

    source_kind: OptimizationConfigProfileSourceKind
    optimization_profile_id: str = Field(min_length=1, max_length=128)
    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    parent_config_profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    campaign_id: str | None = Field(default=None, min_length=1, max_length=128)
    campaign_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    trial_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_configuration_id: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_kind: ForgeCandidateKind | None = None
    recommendation_summary_id: str | None = Field(default=None, min_length=1, max_length=128)
    recommendation_disposition: OptimizationRecommendationDisposition | None = None
    promotion_decision_id: str | None = Field(default=None, min_length=1, max_length=128)
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationConfigProfileChange(BaseModel):
    """One explicitly supported config delta inside an optimization profile."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(min_length=1, max_length=256)
    group: OptimizationKnobGroup
    knob_type: OptimizationKnobType
    applies_to: list[OptimizationScope] = Field(default_factory=list)
    supported_scopes: list[OptimizationScope] = Field(default_factory=list)
    baseline_value: bool | int | float | str | list[str] | None = None
    candidate_value: bool | int | float | str | list[str] | None = None
    mutable_at_runtime: bool = False
    notes: list[str] = Field(default_factory=list)


class OptimizationConfigProfileDiff(BaseModel):
    """Compact introspection summary for one baseline-versus-profile diff."""

    model_config = ConfigDict(extra="forbid")

    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    changed_knob_ids: list[str] = Field(default_factory=list)
    changed_groups: list[OptimizationKnobGroup] = Field(default_factory=list)
    mutable_runtime_knob_ids: list[str] = Field(default_factory=list)
    immutable_knob_ids: list[str] = Field(default_factory=list)
    profile_scope: list[OptimizationScope] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationConfigProfile(BaseModel):
    """Promotion-ready explicit config profile derived from one candidate or baseline."""

    model_config = ConfigDict(extra="forbid")

    config_profile_id: str = Field(min_length=1, max_length=128)
    profile_version: int = Field(default=1, ge=1, le=1_000_000)
    profile_role: OptimizationConfigProfileRole
    optimization_profile_id: str = Field(min_length=1, max_length=128)
    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    applies_to: list[OptimizationScope] = Field(
        default_factory=lambda: [OptimizationScope(kind=OptimizationScopeKind.GLOBAL)]
    )
    candidate_kind: ForgeCandidateKind | None = None
    routing_policy: RoutingPolicy | None = None
    rollout_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    worker_launch_preset: str | None = Field(default=None, min_length=1, max_length=128)
    changes: list[OptimizationConfigProfileChange] = Field(default_factory=list)
    validation: OptimizationConfigProfileValidation = Field(
        default_factory=OptimizationConfigProfileValidation
    )
    diff: OptimizationConfigProfileDiff
    provenance: OptimizationConfigProfileProvenance
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        knob_ids = [change.knob_id for change in self.changes]
        if len(knob_ids) != len(set(knob_ids)):
            msg = "changes must not contain duplicate knob_id values"
            raise ValueError(msg)
        if self.profile_role is OptimizationConfigProfileRole.BASELINE:
            if self.config_profile_id != self.baseline_config_profile_id:
                msg = "baseline profiles must use baseline_config_profile_id as config_profile_id"
                raise ValueError(msg)
        if self.diff.config_profile_id != self.config_profile_id:
            msg = "diff.config_profile_id must match config_profile_id"
            raise ValueError(msg)
        if self.diff.baseline_config_profile_id != self.baseline_config_profile_id:
            msg = "diff.baseline_config_profile_id must match baseline_config_profile_id"
            raise ValueError(msg)
        if self.provenance.optimization_profile_id != self.optimization_profile_id:
            msg = "provenance.optimization_profile_id must match optimization_profile_id"
            raise ValueError(msg)
        if self.provenance.baseline_config_profile_id != self.baseline_config_profile_id:
            msg = "provenance.baseline_config_profile_id must match baseline_config_profile_id"
            raise ValueError(msg)


class OptimizationKnobChange(BaseModel):
    """One explicit knob delta from the baseline configuration."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(min_length=1, max_length=256)
    applies_to: list[OptimizationScope] = Field(
        default_factory=lambda: [OptimizationScope(kind=OptimizationScopeKind.GLOBAL)]
    )
    baseline_value: bool | int | float | str | list[str] | None = None
    candidate_value: bool | int | float | str | list[str] | None = None
    changed: bool = True
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self.changed = self.baseline_value != self.candidate_value


class OptimizationEvidenceRecord(BaseModel):
    """Authoritative evidence input referenced by a campaign or trial artifact."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=128)
    evidence_kind: OptimizationArtifactEvidenceKind
    source_type: OptimizationArtifactSourceType
    source_artifact_id: str = Field(min_length=1, max_length=128)
    source_run_ids: list[str] = Field(default_factory=list)
    source_trace_ids: list[str] = Field(default_factory=list)
    source_simulation_ids: list[str] = Field(default_factory=list)
    status: OptimizationArtifactStatus = OptimizationArtifactStatus.COMPLETE
    window_started_at: datetime | None = None
    window_ended_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        self.source_run_ids = sorted(set(self.source_run_ids))
        self.source_trace_ids = sorted(set(self.source_trace_ids))
        self.source_simulation_ids = sorted(set(self.source_simulation_ids))
        if self.window_started_at and self.window_ended_at:
            if self.window_started_at > self.window_ended_at:
                msg = "window_started_at must be <= window_ended_at"
                raise ValueError(msg)


class OptimizationTopologyLineage(BaseModel):
    """Topology and worker inventory captured for one optimization artifact."""

    model_config = ConfigDict(extra="forbid")

    topology_references: list[TopologySnapshotReference] = Field(default_factory=list)
    deployed_topology: list[DeployedTopologyEndpoint] = Field(default_factory=list)
    worker_instance_inventory: list[BackendInstance] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationObjectiveAssessment(BaseModel):
    """Measured or estimated outcome for one objective under one trial."""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(min_length=1, max_length=128)
    metric: OptimizationObjectiveMetric
    goal: OptimizationGoal
    measured_value: float | None = None
    target_value: float | None = None
    satisfied: bool | None = None
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationConstraintAssessment(BaseModel):
    """Constraint evaluation result for one trial."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str = Field(min_length=1, max_length=128)
    dimension: OptimizationConstraintDimension
    strength: OptimizationConstraintStrength
    operator: OptimizationComparisonOperator
    threshold_value: bool | int | float | str
    evaluated_value: bool | int | float | str | None = None
    satisfied: bool | None = None
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationEvidenceMixSummary(BaseModel):
    """Explicit evidence posture behind one offline recommendation."""

    model_config = ConfigDict(extra="forbid")

    total_request_count: int = Field(default=0, ge=0)
    replay_backed_request_count: int = Field(default=0, ge=0)
    simulated_request_count: int = Field(default=0, ge=0)
    direct_observation_count: int = Field(default=0, ge=0)
    estimated_request_count: int = Field(default=0, ge=0)
    unsupported_request_count: int = Field(default=0, ge=0)
    observed_share: float | None = Field(default=None, ge=0.0, le=1.0)
    replayed_share: float | None = Field(default=None, ge=0.0, le=1.0)
    simulated_share: float | None = Field(default=None, ge=0.0, le=1.0)
    estimated_share: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class OptimizationRecommendationSummary(BaseModel):
    """Operator-facing recommendation distilled from one trial or campaign."""

    model_config = ConfigDict(extra="forbid")

    recommendation_summary_id: str = Field(min_length=1, max_length=128)
    disposition: OptimizationRecommendationDisposition
    recommendation_label: OptimizationRecommendationLabel = (
        OptimizationRecommendationLabel.REVIEW_ONLY
    )
    confidence: RecommendationConfidence = RecommendationConfidence.INSUFFICIENT
    candidate_configuration_id: str | None = Field(default=None, min_length=1, max_length=128)
    config_profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    reason_codes: list[OptimizationRecommendationReasonCode] = Field(default_factory=list)
    improved_objective_ids: list[str] = Field(default_factory=list)
    regressed_objective_ids: list[str] = Field(default_factory=list)
    satisfied_constraint_ids: list[str] = Field(default_factory=list)
    violated_constraint_ids: list[str] = Field(default_factory=list)
    benefited_workload_families: list[str] = Field(default_factory=list)
    regressed_workload_families: list[str] = Field(default_factory=list)
    evidence_mix: OptimizationEvidenceMixSummary | None = None
    rationale: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationPromotionDecision(BaseModel):
    """Explicit promotion or rollback decision for one candidate."""

    model_config = ConfigDict(extra="forbid")

    promotion_decision_id: str = Field(min_length=1, max_length=128)
    disposition: OptimizationPromotionDisposition
    candidate_configuration_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    rollout_mode: PolicyRolloutMode
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    rollback_supported: bool = True
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rationale: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationObjectiveDelta(BaseModel):
    """Baseline-versus-candidate delta for one objective metric."""

    model_config = ConfigDict(extra="forbid")

    objective_id: str = Field(min_length=1, max_length=128)
    metric: OptimizationObjectiveMetric
    goal: OptimizationGoal
    baseline_value: float | None = None
    candidate_value: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    normalized_tradeoff: float | None = None
    improved: bool | None = None
    notes: list[str] = Field(default_factory=list)


class OptimizationWorkloadMetricDelta(BaseModel):
    """One workload-family metric delta derived from simulation truth."""

    model_config = ConfigDict(extra="forbid")

    metric: OptimizationObjectiveMetric
    baseline_value: float | None = None
    candidate_value: float | None = None
    absolute_delta: float | None = None
    improved: bool | None = None


class OptimizationWorkloadImpactSummary(BaseModel):
    """Workload-family benefit or regression summary for one candidate."""

    model_config = ConfigDict(extra="forbid")

    workload_family: str = Field(min_length=1, max_length=128)
    request_count: int = Field(default=0, ge=0)
    metric_deltas: list[OptimizationWorkloadMetricDelta] = Field(default_factory=list)
    improved_metrics: list[OptimizationObjectiveMetric] = Field(default_factory=list)
    regressed_metrics: list[OptimizationObjectiveMetric] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationCandidateComparisonArtifact(BaseModel):
    """Offline comparison summary for one candidate against the baseline."""

    model_config = ConfigDict(extra="forbid")

    candidate_configuration_id: str = Field(min_length=1, max_length=128)
    trial_artifact_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    rank: int = Field(default=0, ge=0)
    tied_candidate_configuration_ids: list[str] = Field(default_factory=list)
    objective_deltas: list[OptimizationObjectiveDelta] = Field(default_factory=list)
    normalized_tradeoff_score: float = 0.0
    hard_constraint_violations: list[str] = Field(default_factory=list)
    soft_constraint_violations: list[str] = Field(default_factory=list)
    pareto_optimal: bool = False
    dominated: bool = False
    dominated_by_candidate_configuration_ids: list[str] = Field(default_factory=list)
    workload_impacts: list[OptimizationWorkloadImpactSummary] = Field(default_factory=list)
    recommendation_summary: OptimizationRecommendationSummary
    notes: list[str] = Field(default_factory=list)


class OptimizationParetoSummary(BaseModel):
    """Pareto frontier posture across one campaign's compared candidates."""

    model_config = ConfigDict(extra="forbid")

    frontier_candidate_configuration_ids: list[str] = Field(default_factory=list)
    dominated_candidate_configuration_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OptimizationCampaignComparisonArtifact(BaseModel):
    """Comparable offline ranking and recommendation output for one campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: OptimizationArtifactSchemaVersion = OptimizationArtifactSchemaVersion.V1
    comparison_artifact_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    campaign_id: str = Field(min_length=1, max_length=128)
    baseline_candidate_configuration_id: str = Field(min_length=1, max_length=128)
    compared_candidate_configuration_ids: list[str] = Field(default_factory=list)
    candidate_comparisons: list[OptimizationCandidateComparisonArtifact] = Field(
        default_factory=list
    )
    pareto_summary: OptimizationParetoSummary = Field(
        default_factory=OptimizationParetoSummary
    )
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        expected_ids = {
            comparison.candidate_configuration_id for comparison in self.candidate_comparisons
        }
        if self.compared_candidate_configuration_ids:
            if expected_ids != set(self.compared_candidate_configuration_ids):
                msg = (
                    "compared_candidate_configuration_ids must match candidate_comparisons"
                )
                raise ValueError(msg)
        else:
            self.compared_candidate_configuration_ids = sorted(expected_ids)


class OptimizationCandidateConfigurationArtifact(BaseModel):
    """Authoritative candidate configuration captured for one campaign."""

    model_config = ConfigDict(extra="forbid")

    schema_version: OptimizationArtifactSchemaVersion = OptimizationArtifactSchemaVersion.V1
    candidate_configuration_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    campaign_id: str = Field(min_length=1, max_length=128)
    candidate: OptimizationTrialIdentity
    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    knob_changes: list[OptimizationKnobChange] = Field(default_factory=list)
    objectives_in_scope: list[OptimizationObjectiveTarget] = Field(default_factory=list)
    constraints_in_scope: list[OptimizationConstraint] = Field(default_factory=list)
    workload_sets: list[OptimizationWorkloadSet] = Field(default_factory=list)
    expected_evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    generation: OptimizationCandidateGenerationMetadata | None = None
    eligibility: OptimizationCandidateEligibilityRecord | None = None
    topology_lineage: OptimizationTopologyLineage | None = None
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        knob_ids = [change.knob_id for change in self.knob_changes]
        if len(knob_ids) != len(set(knob_ids)):
            msg = "knob_changes must not contain duplicate knob_id values"
            raise ValueError(msg)
        objective_ids = [objective.objective_id for objective in self.objectives_in_scope]
        if len(objective_ids) != len(set(objective_ids)):
            msg = "objectives_in_scope must not contain duplicate objective_id values"
            raise ValueError(msg)
        constraint_ids = [
            constraint.constraint_id for constraint in self.constraints_in_scope
        ]
        if len(constraint_ids) != len(set(constraint_ids)):
            msg = "constraints_in_scope must not contain duplicate constraint_id values"
            raise ValueError(msg)


class OptimizationTrialArtifact(BaseModel):
    """Authoritative result artifact for one candidate trial."""

    model_config = ConfigDict(extra="forbid")

    schema_version: OptimizationArtifactSchemaVersion = OptimizationArtifactSchemaVersion.V1
    trial_artifact_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    campaign_id: str = Field(min_length=1, max_length=128)
    baseline_candidate_configuration_id: str = Field(min_length=1, max_length=128)
    candidate_configuration: OptimizationCandidateConfigurationArtifact
    trial_identity: OptimizationTrialIdentity
    evidence_records: list[OptimizationEvidenceRecord] = Field(default_factory=list)
    topology_lineage: OptimizationTopologyLineage | None = None
    result_status: OptimizationArtifactStatus = OptimizationArtifactStatus.COMPLETE
    stale_reason: str | None = Field(default=None, min_length=1, max_length=256)
    invalidation_reason: str | None = Field(default=None, min_length=1, max_length=256)
    objective_assessments: list[OptimizationObjectiveAssessment] = Field(default_factory=list)
    constraint_assessments: list[OptimizationConstraintAssessment] = Field(default_factory=list)
    recommendation_summary: OptimizationRecommendationSummary | None = None
    promotion_decision: OptimizationPromotionDecision | None = None
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if self.candidate_configuration.campaign_id != self.campaign_id:
            msg = "candidate_configuration.campaign_id must match campaign_id"
            raise ValueError(msg)
        if self.candidate_configuration.candidate != self.trial_identity:
            msg = "candidate_configuration.candidate must match trial_identity"
            raise ValueError(msg)
        if (
            self.result_status is OptimizationArtifactStatus.STALE
            and self.stale_reason is None
        ):
            msg = "stale artifacts require stale_reason"
            raise ValueError(msg)
        if (
            self.result_status is OptimizationArtifactStatus.INVALIDATED
            and self.invalidation_reason is None
        ):
            msg = "invalidated artifacts require invalidation_reason"
            raise ValueError(msg)
        if self.promotion_decision is not None:
            if (
                self.promotion_decision.candidate_configuration_id
                != self.candidate_configuration.candidate_configuration_id
            ):
                msg = "promotion_decision must reference candidate_configuration"
                raise ValueError(msg)


class OptimizationCampaignArtifact(BaseModel):
    """Authoritative campaign artifact for later reporting and lineage lookup."""

    model_config = ConfigDict(extra="forbid")

    schema_version: OptimizationArtifactSchemaVersion = OptimizationArtifactSchemaVersion.V1
    campaign_artifact_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    campaign: OptimizationCampaignMetadata
    baseline_candidate_configuration: OptimizationCandidateConfigurationArtifact
    candidate_configurations: list[OptimizationCandidateConfigurationArtifact] = Field(
        default_factory=list
    )
    trials: list[OptimizationTrialArtifact] = Field(default_factory=list)
    evidence_records: list[OptimizationEvidenceRecord] = Field(default_factory=list)
    topology_lineage: OptimizationTopologyLineage | None = None
    recommendation_summaries: list[OptimizationRecommendationSummary] = Field(
        default_factory=list
    )
    promotion_decisions: list[OptimizationPromotionDecision] = Field(default_factory=list)
    result_status: OptimizationArtifactStatus = OptimizationArtifactStatus.COMPLETE
    stale_reason: str | None = Field(default=None, min_length=1, max_length=256)
    invalidation_reason: str | None = Field(default=None, min_length=1, max_length=256)
    notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        campaign_id = self.campaign.campaign_id
        if self.baseline_candidate_configuration.campaign_id != campaign_id:
            msg = "baseline_candidate_configuration.campaign_id must match campaign.campaign_id"
            raise ValueError(msg)
        candidate_ids = {
            self.baseline_candidate_configuration.candidate_configuration_id,
            *(
                candidate.candidate_configuration_id
                for candidate in self.candidate_configurations
            ),
        }
        for candidate in self.candidate_configurations:
            if candidate.campaign_id != campaign_id:
                msg = "candidate_configurations must all match campaign.campaign_id"
                raise ValueError(msg)
        trial_ids = [trial.trial_artifact_id for trial in self.trials]
        if len(trial_ids) != len(set(trial_ids)):
            msg = "trials must not contain duplicate trial_artifact_id values"
            raise ValueError(msg)
        for trial in self.trials:
            if trial.campaign_id != campaign_id:
                msg = "trials must all match campaign.campaign_id"
                raise ValueError(msg)
            candidate_id = trial.candidate_configuration.candidate_configuration_id
            if candidate_id not in candidate_ids:
                msg = (
                    "trials must reference a candidate configuration "
                    "present in the campaign artifact"
                )
                raise ValueError(msg)
        for recommendation in self.recommendation_summaries:
            if (
                recommendation.candidate_configuration_id is not None
                and recommendation.candidate_configuration_id not in candidate_ids
            ):
                msg = (
                    "recommendation_summaries must reference a candidate configuration "
                    "present in the campaign artifact"
                )
                raise ValueError(msg)
        for decision in self.promotion_decisions:
            if decision.candidate_configuration_id not in candidate_ids:
                msg = (
                    "promotion_decisions must reference a candidate configuration "
                    "present in the campaign artifact"
                )
                raise ValueError(msg)
        if (
            self.result_status is OptimizationArtifactStatus.STALE
            and self.stale_reason is None
        ):
            msg = "stale campaign artifacts require stale_reason"
            raise ValueError(msg)
        if (
            self.result_status is OptimizationArtifactStatus.INVALIDATED
            and self.invalidation_reason is None
        ):
            msg = "invalidated campaign artifacts require invalidation_reason"
            raise ValueError(msg)
