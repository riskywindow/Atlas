"""Typed Forge Stage A rollout and promotion schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactStatus,
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationConfigProfile,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationTrialArtifact,
)
from switchyard.schemas.routing import PolicyRolloutMode


class ForgeHonestyWarningKind(StrEnum):
    """Category of a campaign-honesty warning surfaced in inspection views."""

    BUDGET_BOUND_EXCEEDED = "budget_bound_exceeded"
    TOPOLOGY_DRIFT = "topology_drift"
    STALE_EVIDENCE = "stale_evidence"
    NARROW_WORKLOAD_COVERAGE = "narrow_workload_coverage"
    EVIDENCE_INCONSISTENCY = "evidence_inconsistency"
    OBSERVED_EVIDENCE_MISSING = "observed_evidence_missing"
    COST_SIGNAL_MISMATCH = "cost_signal_mismatch"


class ForgeHonestyWarningSummary(BaseModel):
    """Operator-facing honesty warning surfaced in a campaign inspection."""

    model_config = ConfigDict(extra="forbid")

    kind: ForgeHonestyWarningKind
    severity: str = Field(default="warning", pattern=r"^(info|warning|error)$")
    message: str = Field(min_length=1, max_length=512)
    affected_trial_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionLifecycleState(StrEnum):
    """Bounded lifecycle state for one promoted optimization profile."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    CANARY_ACTIVE = "canary_active"
    COMPARED = "compared"
    PROMOTED_DEFAULT = "promoted_default"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class ForgePromotionAppliedKnobChange(BaseModel):
    """One config-profile knob change and whether runtime application succeeded."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(min_length=1, max_length=256)
    runtime_mutable: bool
    applied: bool
    baseline_value: bool | int | float | str | list[str] | None = None
    candidate_value: bool | int | float | str | list[str] | None = None
    notes: list[str] = Field(default_factory=list)


class ForgePromotionComparisonSummary(BaseModel):
    """Artifact-backed canary-versus-baseline comparison for one active profile."""

    model_config = ConfigDict(extra="forbid")

    comparison_artifact_id: str = Field(min_length=1, max_length=128)
    campaign_id: str = Field(min_length=1, max_length=128)
    candidate_configuration_id: str = Field(min_length=1, max_length=128)
    baseline_candidate_configuration_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    compared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rank: int = Field(default=0, ge=0)
    pareto_optimal: bool = False
    dominated: bool = False
    recommendation_disposition: OptimizationRecommendationDisposition
    recommendation_label: OptimizationRecommendationLabel
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    improved_objective_ids: list[str] = Field(default_factory=list)
    regressed_objective_ids: list[str] = Field(default_factory=list)
    satisfied_constraint_ids: list[str] = Field(default_factory=list)
    violated_constraint_ids: list[str] = Field(default_factory=list)
    benefited_workload_families: list[str] = Field(default_factory=list)
    regressed_workload_families: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionLifecycleEvent(BaseModel):
    """One explicit lifecycle decision recorded for auditability."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=128)
    lifecycle_state: ForgePromotionLifecycleState
    promotion_disposition: OptimizationPromotionDisposition
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: list[str] = Field(default_factory=list)


class ForgePromotionRuntimeSummary(BaseModel):
    """Authoritative serializable state for one bounded Forge rollout."""

    model_config = ConfigDict(extra="forbid")

    rollout_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    active_config_profile_id: str = Field(min_length=1, max_length=128)
    candidate_config_profile_id: str | None = Field(default=None, min_length=1, max_length=128)
    lifecycle_state: ForgePromotionLifecycleState | None = None
    applied: bool = False
    campaign_id: str | None = Field(default=None, min_length=1, max_length=128)
    campaign_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    trial_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    baseline_candidate_configuration_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    candidate_configuration_id: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_kind: ForgeCandidateKind | None = None
    routing_policy: str | None = Field(default=None, min_length=1, max_length=64)
    rollout_mode: PolicyRolloutMode = PolicyRolloutMode.DISABLED
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    recommendation_disposition: OptimizationRecommendationDisposition | None = None
    promotion_disposition: OptimizationPromotionDisposition = (
        OptimizationPromotionDisposition.NO_ACTION
    )
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    config_profile: OptimizationConfigProfile | None = None
    comparison: ForgePromotionComparisonSummary | None = None
    applied_knob_changes: list[ForgePromotionAppliedKnobChange] = Field(default_factory=list)
    blocked_knob_changes: list[ForgePromotionAppliedKnobChange] = Field(default_factory=list)
    rollback_available: bool = False
    requires_operator_review: bool = True
    applied_at: datetime | None = Field(default_factory=lambda: None)
    last_reset_at: datetime | None = Field(default_factory=lambda: None)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    lifecycle_events: list[ForgePromotionLifecycleEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgeCandidateDiffEntry(BaseModel):
    """Operator-facing diff entry for one candidate knob change."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(min_length=1, max_length=256)
    baseline_value: bool | int | float | str | list[str] | None = None
    candidate_value: bool | int | float | str | list[str] | None = None
    notes: list[str] = Field(default_factory=list)


class ForgeCandidateProvenanceSummary(BaseModel):
    """Compact provenance summary for one candidate or trial."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(min_length=1, max_length=128)
    trial_artifact_id: str = Field(min_length=1, max_length=128)
    candidate_configuration_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    generation_strategy: str | None = Field(default=None, min_length=1, max_length=128)
    eligibility_status: str | None = Field(default=None, min_length=1, max_length=64)
    topology_reference_count: int = Field(default=0, ge=0)
    topology_endpoint_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class ForgeTrialInspectionSummary(BaseModel):
    """Operator-facing summary for one Forge Stage A trial."""

    model_config = ConfigDict(extra="forbid")

    trial_artifact_id: str = Field(min_length=1, max_length=128)
    candidate_configuration_id: str = Field(min_length=1, max_length=128)
    config_profile_id: str = Field(min_length=1, max_length=128)
    baseline_config_profile_id: str = Field(min_length=1, max_length=128)
    trial_status: OptimizationArtifactStatus
    candidate_kind: ForgeCandidateKind
    routing_policy: str | None = Field(default=None, min_length=1, max_length=64)
    recommendation_disposition: OptimizationRecommendationDisposition | None = None
    recommendation_label: OptimizationRecommendationLabel | None = None
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    helped_workload_families: list[str] = Field(default_factory=list)
    hurt_workload_families: list[str] = Field(default_factory=list)
    comparison_rank: int | None = Field(default=None, ge=0)
    pareto_optimal: bool | None = None
    dominated: bool | None = None
    remote_budget_involved: bool = False
    remote_budget_constraint_ids: list[str] = Field(default_factory=list)
    remote_budget_constraint_outcomes: list[str] = Field(default_factory=list)
    diff_entries: list[ForgeCandidateDiffEntry] = Field(default_factory=list)
    provenance: ForgeCandidateProvenanceSummary
    notes: list[str] = Field(default_factory=list)


class ForgeCampaignInspectionSummary(BaseModel):
    """Operator-facing summary for one Forge Stage A campaign artifact."""

    model_config = ConfigDict(extra="forbid")

    campaign_artifact_id: str = Field(min_length=1, max_length=128)
    campaign_id: str = Field(min_length=1, max_length=128)
    optimization_profile_id: str = Field(min_length=1, max_length=128)
    result_status: OptimizationArtifactStatus
    objective: str = Field(min_length=1, max_length=64)
    evidence_kinds: list[OptimizationArtifactEvidenceKind] = Field(default_factory=list)
    recommendation_status_counts: dict[str, int] = Field(default_factory=dict)
    helped_workload_families: list[str] = Field(default_factory=list)
    hurt_workload_families: list[str] = Field(default_factory=list)
    remote_budget_involved: bool = False
    remote_budget_constraint_ids: list[str] = Field(default_factory=list)
    honesty_warnings: list[ForgeHonestyWarningSummary] = Field(default_factory=list)
    trustworthy: bool = True
    comparison_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    trials: list[ForgeTrialInspectionSummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgeCampaignInspectionRequest(BaseModel):
    """Artifact-backed request to inspect one or more Forge campaigns."""

    model_config = ConfigDict(extra="forbid")

    campaign_artifacts: list[OptimizationCampaignArtifact] = Field(default_factory=list)
    comparison_artifacts: list[OptimizationCampaignComparisonArtifact] = Field(default_factory=list)


class ForgeCampaignInspectionResponse(BaseModel):
    """Operator-facing response listing one or more Forge campaigns and trials."""

    model_config = ConfigDict(extra="forbid")

    campaigns: list[ForgeCampaignInspectionSummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionProposeRequest(BaseModel):
    """Typed admin request to propose one reviewed Forge Stage A trial."""

    model_config = ConfigDict(extra="forbid")

    trial_artifact: OptimizationTrialArtifact
    campaign_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionApplyRequest(BaseModel):
    """Typed admin request to activate an approved canary."""

    model_config = ConfigDict(extra="forbid")

    rollout_artifact_id: str = Field(min_length=1, max_length=128)
    canary_percentage: float | None = Field(default=None, ge=0.0, le=100.0)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionDecisionRequest(BaseModel):
    """Typed admin request for approve, reject, promote-default, or rollback steps."""

    model_config = ConfigDict(extra="forbid")

    rollout_artifact_id: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, min_length=1, max_length=256)
    notes: list[str] = Field(default_factory=list)


class ForgePromotionCompareRequest(BaseModel):
    """Typed admin request to attach artifact-backed canary comparison evidence."""

    model_config = ConfigDict(extra="forbid")

    rollout_artifact_id: str = Field(min_length=1, max_length=128)
    comparison_artifact: OptimizationCampaignComparisonArtifact
    notes: list[str] = Field(default_factory=list)
