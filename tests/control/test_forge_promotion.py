from __future__ import annotations

import pytest

from switchyard.config import HybridExecutionSettings, PolicyRolloutSettings, Settings
from switchyard.control.forge_promotion import ForgePromotionService
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.spillover import RemoteSpilloverControlService
from switchyard.optimization import build_baseline_optimization_config_profile
from switchyard.schemas.benchmark import CounterfactualObjective, RecommendationConfidence
from switchyard.schemas.forge import (
    ForgePromotionApplyRequest,
    ForgePromotionCompareRequest,
    ForgePromotionDecisionRequest,
    ForgePromotionLifecycleState,
    ForgePromotionProposeRequest,
)
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    OptimizationArtifactEvidenceKind,
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationCampaignMetadata,
    OptimizationCandidateComparisonArtifact,
    OptimizationCandidateConfigurationArtifact,
    OptimizationKnobChange,
    OptimizationPromotionDecision,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import PolicyRolloutMode, RoutingPolicy


def test_forge_promotion_lifecycle_canary_compare_promote_and_rollback() -> None:
    settings = Settings()
    policy_rollout = PolicyRolloutService(PolicyRolloutSettings())
    spillover = RemoteSpilloverControlService(HybridExecutionSettings())
    service = ForgePromotionService(
        settings=settings,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        max_canary_percentage=15.0,
        requires_operator_review=True,
        policy_rollout=policy_rollout,
        spillover=spillover,
    )

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(
                settings=settings,
                canary_percentage=20.0,
            ),
        )
    )
    assert proposed.lifecycle_state is ForgePromotionLifecycleState.PROPOSED
    assert proposed.active_config_profile_id == _baseline_config_profile_id(settings)
    assert proposed.candidate_config_profile_id == "phase9-local-preferred"
    assert proposed.config_profile is not None
    assert proposed.rollout_artifact_id is not None
    rollout_artifact_id = proposed.rollout_artifact_id

    approved = service.approve(
        ForgePromotionDecisionRequest(rollout_artifact_id=rollout_artifact_id)
    )
    assert approved.lifecycle_state is ForgePromotionLifecycleState.APPROVED

    canary = service.apply(
        ForgePromotionApplyRequest(
            rollout_artifact_id=rollout_artifact_id,
            canary_percentage=25.0,
        )
    )
    assert canary.lifecycle_state is ForgePromotionLifecycleState.CANARY_ACTIVE
    assert canary.applied is True
    assert canary.active_config_profile_id == "phase9-local-preferred"
    assert canary.canary_percentage == 15.0
    assert canary.promotion_disposition is OptimizationPromotionDisposition.APPROVED_CANARY
    assert canary.applied_knob_changes[0].knob_id == "default_routing_policy"

    rollout_state = policy_rollout.inspect_state()
    assert rollout_state.mode is PolicyRolloutMode.CANARY
    assert rollout_state.canary_percentage == 15.0
    assert rollout_state.candidate_policy is not None
    assert rollout_state.candidate_policy.policy_id == "phase9-local-preferred"

    compared = service.compare(
        ForgePromotionCompareRequest(
            rollout_artifact_id=rollout_artifact_id,
            comparison_artifact=build_comparison_artifact(
                trial=build_promotable_trial_artifact(
                    settings=settings,
                    canary_percentage=20.0,
                )
            ),
        )
    )
    assert compared.lifecycle_state is ForgePromotionLifecycleState.COMPARED
    assert compared.comparison is not None
    assert compared.comparison.recommendation_disposition is (
        OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    )
    assert compared.comparison.rank == 1

    promoted = service.promote_default(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_artifact_id,
            reason="operator reviewed favorable canary evidence",
        )
    )
    assert promoted.lifecycle_state is ForgePromotionLifecycleState.PROMOTED_DEFAULT
    assert promoted.rollout_mode is PolicyRolloutMode.ACTIVE_GUARDED
    assert promoted.promotion_disposition is (
        OptimizationPromotionDisposition.PROMOTED_DEFAULT
    )
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.ACTIVE_GUARDED

    reset = service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_artifact_id,
            reason="restore baseline",
        )
    )
    assert reset.lifecycle_state is ForgePromotionLifecycleState.ROLLED_BACK
    assert reset.applied is False
    assert reset.active_config_profile_id == _baseline_config_profile_id(settings)
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


def test_forge_promotion_invalid_lifecycle_transitions_are_rejected() -> None:
    settings = Settings()
    service = ForgePromotionService(
        settings=settings,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        max_canary_percentage=15.0,
        requires_operator_review=True,
        policy_rollout=PolicyRolloutService(PolicyRolloutSettings()),
        spillover=RemoteSpilloverControlService(HybridExecutionSettings()),
    )
    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    assert proposed.rollout_artifact_id is not None
    rollout_artifact_id = proposed.rollout_artifact_id

    with pytest.raises(ValueError, match="expected approved"):
        service.apply(
            ForgePromotionApplyRequest(
                rollout_artifact_id=rollout_artifact_id,
            )
        )

    with pytest.raises(ValueError, match="expected compared"):
        service.promote_default(
            ForgePromotionDecisionRequest(
                rollout_artifact_id=rollout_artifact_id,
            )
        )

    approved = service.approve(
        ForgePromotionDecisionRequest(rollout_artifact_id=rollout_artifact_id)
    )
    assert approved.lifecycle_state is ForgePromotionLifecycleState.APPROVED

    with pytest.raises(ValueError, match="expected canary_active"):
        service.compare(
            ForgePromotionCompareRequest(
                rollout_artifact_id=rollout_artifact_id,
                comparison_artifact=build_comparison_artifact(
                    trial=build_promotable_trial_artifact(settings=settings)
                ),
            )
        )


def test_forge_promotion_rejects_pending_rollout_without_runtime_mutation() -> None:
    settings = Settings()
    policy_rollout = PolicyRolloutService(PolicyRolloutSettings())
    service = ForgePromotionService(
        settings=settings,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        max_canary_percentage=15.0,
        requires_operator_review=True,
        policy_rollout=policy_rollout,
        spillover=RemoteSpilloverControlService(HybridExecutionSettings()),
    )
    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    assert proposed.rollout_artifact_id is not None
    rollout_artifact_id = proposed.rollout_artifact_id
    approved = service.approve(
        ForgePromotionDecisionRequest(rollout_artifact_id=rollout_artifact_id)
    )
    assert approved.lifecycle_state is ForgePromotionLifecycleState.APPROVED

    rejected = service.reject(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_artifact_id,
            reason="operator declined rollout",
        )
    )
    assert rejected.lifecycle_state is ForgePromotionLifecycleState.REJECTED
    assert rejected.active_config_profile_id == _baseline_config_profile_id(settings)
    assert rejected.applied is False
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


def test_forge_promotion_rejects_mismatched_comparison_artifact() -> None:
    settings = Settings()
    service = ForgePromotionService(
        settings=settings,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        max_canary_percentage=15.0,
        requires_operator_review=True,
        policy_rollout=PolicyRolloutService(PolicyRolloutSettings()),
        spillover=RemoteSpilloverControlService(HybridExecutionSettings()),
    )
    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    assert proposed.rollout_artifact_id is not None
    rollout_artifact_id = proposed.rollout_artifact_id
    service.approve(
        ForgePromotionDecisionRequest(rollout_artifact_id=rollout_artifact_id)
    )
    service.apply(
        ForgePromotionApplyRequest(rollout_artifact_id=rollout_artifact_id)
    )

    with pytest.raises(ValueError, match="does not contain the active candidate configuration"):
        service.compare(
            ForgePromotionCompareRequest(
                rollout_artifact_id=rollout_artifact_id,
                comparison_artifact=build_comparison_artifact(
                    trial=build_promotable_trial_artifact(
                        settings=settings,
                        config_profile_id="phase9-other-profile",
                    )
                ),
            )
        )


def test_forge_promotion_rejects_unsupported_runtime_knob_change() -> None:
    settings = Settings()
    service = ForgePromotionService(
        settings=settings,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        max_canary_percentage=15.0,
        requires_operator_review=True,
        policy_rollout=PolicyRolloutService(PolicyRolloutSettings()),
        spillover=RemoteSpilloverControlService(HybridExecutionSettings()),
    )
    unsupported_trial = build_promotable_trial_artifact(
        settings=settings,
        knob_changes=[
            OptimizationKnobChange(
                knob_id="shadow_sampling_rate",
                config_path="phase4.shadow_routing.default_sampling_rate",
                baseline_value=0.0,
                candidate_value=0.2,
            )
        ],
    )

    with pytest.raises(ValueError, match="cannot be applied safely at runtime"):
        service.propose(ForgePromotionProposeRequest(trial_artifact=unsupported_trial))


def build_promotable_trial_artifact(
    *,
    settings: Settings,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    knob_changes: list[OptimizationKnobChange] | None = None,
    canary_percentage: float = 10.0,
) -> OptimizationTrialArtifact:
    trial_identity = OptimizationTrialIdentity(
        trial_id=f"trial-{config_profile_id}",
        candidate_id=f"routing-policy:{routing_policy.value}",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )
    candidate_configuration = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=f"candidate-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        candidate=trial_identity,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        config_profile_id=config_profile_id,
        knob_changes=knob_changes
        or [
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value=RoutingPolicy.BALANCED.value,
                candidate_value=routing_policy.value,
            )
        ],
    )
    return OptimizationTrialArtifact(
        trial_artifact_id=f"trial-artifact-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-phase9-baseline",
        candidate_configuration=candidate_configuration,
        trial_identity=trial_identity,
        recommendation_summary=OptimizationRecommendationSummary(
            recommendation_summary_id=f"recommendation-{config_profile_id}",
            disposition=OptimizationRecommendationDisposition.PROMOTE_CANDIDATE,
            recommendation_label=OptimizationRecommendationLabel.PROMOTION_ELIGIBLE,
            confidence=RecommendationConfidence.MEDIUM,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=config_profile_id,
            evidence_kinds=[
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.REPLAYED,
            ],
            reason_codes=[
                OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED
            ],
            rationale=["candidate improved the primary objective"],
        ),
        promotion_decision=OptimizationPromotionDecision(
            promotion_decision_id=f"promotion-{config_profile_id}",
            disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=config_profile_id,
            rollout_mode=PolicyRolloutMode.CANARY,
            canary_percentage=canary_percentage,
        ),
    )


def build_comparison_artifact(
    *,
    trial: OptimizationTrialArtifact,
) -> OptimizationCampaignComparisonArtifact:
    recommendation = trial.recommendation_summary
    assert recommendation is not None
    return OptimizationCampaignComparisonArtifact(
        comparison_artifact_id=f"comparison-{trial.campaign_id}",
        campaign_id=trial.campaign_id,
        baseline_candidate_configuration_id=trial.baseline_candidate_configuration_id,
        candidate_comparisons=[
            OptimizationCandidateComparisonArtifact(
                candidate_configuration_id=(
                    trial.candidate_configuration.candidate_configuration_id
                ),
                trial_artifact_id=trial.trial_artifact_id,
                config_profile_id=trial.candidate_configuration.config_profile_id,
                rank=1,
                pareto_optimal=True,
                dominated=False,
                recommendation_summary=recommendation,
                notes=["artifact-backed canary comparison favored the candidate"],
            )
        ],
        notes=["comparison stayed explicit about observed and replay-backed evidence"],
    )


def _baseline_config_profile_id(settings: Settings) -> str:
    return build_baseline_optimization_config_profile(settings).config_profile_id


# ---------------------------------------------------------------------------
# Service factory and pipeline helpers for lifecycle tests
# ---------------------------------------------------------------------------


def _create_service(
    *,
    settings: Settings | None = None,
    max_canary_percentage: float = 15.0,
) -> tuple[ForgePromotionService, PolicyRolloutService, RemoteSpilloverControlService]:
    """Construct a fresh ForgePromotionService with its dependencies."""
    if settings is None:
        settings = Settings()
    policy_rollout = PolicyRolloutService(PolicyRolloutSettings())
    spillover = RemoteSpilloverControlService(HybridExecutionSettings())
    service = ForgePromotionService(
        settings=settings,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        max_canary_percentage=max_canary_percentage,
        requires_operator_review=True,
        policy_rollout=policy_rollout,
        spillover=spillover,
    )
    return service, policy_rollout, spillover


def _propose_approve_apply(
    service: ForgePromotionService,
    *,
    settings: Settings,
    trial: OptimizationTrialArtifact | None = None,
    canary_percentage: float = 10.0,
) -> str:
    """Propose -> approve -> apply helper.  Returns rollout_artifact_id."""
    if trial is None:
        trial = build_promotable_trial_artifact(
            settings=settings, canary_percentage=canary_percentage
        )
    proposed = service.propose(ForgePromotionProposeRequest(trial_artifact=trial))
    rollout_id = proposed.rollout_artifact_id
    assert rollout_id is not None
    service.approve(ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id))
    service.apply(ForgePromotionApplyRequest(rollout_artifact_id=rollout_id))
    return rollout_id


def _propose_approve_apply_compare(
    service: ForgePromotionService,
    *,
    settings: Settings,
    trial: OptimizationTrialArtifact | None = None,
) -> str:
    """Propose -> approve -> apply -> compare helper.  Returns rollout_artifact_id."""
    if trial is None:
        trial = build_promotable_trial_artifact(settings=settings)
    rollout_id = _propose_approve_apply(service, settings=settings, trial=trial)
    service.compare(
        ForgePromotionCompareRequest(
            rollout_artifact_id=rollout_id,
            comparison_artifact=build_comparison_artifact(trial=trial),
        )
    )
    return rollout_id


def _build_campaign_artifact(
    *,
    settings: Settings,
    trial: OptimizationTrialArtifact,
) -> OptimizationCampaignArtifact:
    """Build a minimal campaign artifact wrapping a promotable trial."""
    baseline = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id="candidate-phase9-baseline",
        campaign_id=trial.campaign_id,
        candidate=OptimizationTrialIdentity(
            trial_id="trial-baseline",
            candidate_id="routing-policy:balanced",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id=_baseline_config_profile_id(settings),
            routing_policy=RoutingPolicy.BALANCED,
        ),
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        config_profile_id=_baseline_config_profile_id(settings),
    )
    return OptimizationCampaignArtifact(
        campaign_artifact_id="campaign-artifact-001",
        campaign=OptimizationCampaignMetadata(
            campaign_id=trial.campaign_id,
            optimization_profile_id="phase9-stage-a-baseline",
            objective=CounterfactualObjective.BALANCED,
            evidence_sources=[ForgeEvidenceSourceKind.OBSERVED_RUNTIME],
        ),
        baseline_candidate_configuration=baseline,
        candidate_configurations=[trial.candidate_configuration],
        trials=[trial],
    )


# ---------------------------------------------------------------------------
# Reject from every active lifecycle state
# ---------------------------------------------------------------------------


def test_reject_from_canary_active_restores_baseline_runtime() -> None:
    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply(service, settings=settings)
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.CANARY

    rejected = service.reject(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="operator rejected canary",
        )
    )
    assert rejected.lifecycle_state is ForgePromotionLifecycleState.REJECTED
    assert rejected.applied is False
    assert rejected.active_config_profile_id == _baseline_config_profile_id(settings)
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


def test_reject_from_compared_restores_baseline_runtime() -> None:
    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply_compare(service, settings=settings)
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.CANARY

    rejected = service.reject(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="comparison was unfavorable",
        )
    )
    assert rejected.lifecycle_state is ForgePromotionLifecycleState.REJECTED
    assert rejected.applied is False
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


def test_reject_from_promoted_default_restores_baseline_runtime() -> None:
    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply_compare(service, settings=settings)
    service.promote_default(
        ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id)
    )
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.ACTIVE_GUARDED

    rejected = service.reject(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="reverting promoted default",
        )
    )
    assert rejected.lifecycle_state is ForgePromotionLifecycleState.REJECTED
    assert rejected.applied is False
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


# ---------------------------------------------------------------------------
# Reset / rollback from active states
# ---------------------------------------------------------------------------


def test_reset_from_canary_active_restores_baseline() -> None:
    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply(service, settings=settings)

    reset = service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="canary showed degradation",
        )
    )
    assert reset.lifecycle_state is ForgePromotionLifecycleState.ROLLED_BACK
    assert reset.applied is False
    assert reset.active_config_profile_id == _baseline_config_profile_id(settings)
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


def test_reset_from_compared_restores_baseline() -> None:
    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply_compare(service, settings=settings)

    reset = service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="comparison did not justify promotion",
        )
    )
    assert reset.lifecycle_state is ForgePromotionLifecycleState.ROLLED_BACK
    assert reset.applied is False
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


# ---------------------------------------------------------------------------
# Re-proposal after finalization
# ---------------------------------------------------------------------------


def test_re_proposal_after_rejection() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    first_rollout_id = proposed.rollout_artifact_id
    assert first_rollout_id is not None
    service.reject(
        ForgePromotionDecisionRequest(rollout_artifact_id=first_rollout_id)
    )

    proposed2 = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    assert proposed2.lifecycle_state is ForgePromotionLifecycleState.PROPOSED
    assert proposed2.rollout_artifact_id is not None


def test_re_proposal_after_rollback() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply(service, settings=settings)
    service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id, reason="rolling back"
        )
    )

    proposed2 = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    assert proposed2.lifecycle_state is ForgePromotionLifecycleState.PROPOSED
    assert proposed2.rollout_artifact_id is not None


# ---------------------------------------------------------------------------
# Blocked proposal and artifact-identity errors
# ---------------------------------------------------------------------------


def test_propose_blocked_while_active_rollout_exists() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )

    with pytest.raises(ValueError, match="must be finalized"):
        service.propose(
            ForgePromotionProposeRequest(
                trial_artifact=build_promotable_trial_artifact(settings=settings),
            )
        )


def test_reject_already_rejected_raises() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    rollout_id = proposed.rollout_artifact_id
    assert rollout_id is not None
    service.reject(ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id))

    with pytest.raises(ValueError, match="already rejected"):
        service.reject(
            ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id)
        )


def test_reject_already_rolled_back_raises() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    rollout_id = _propose_approve_apply(service, settings=settings)
    service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id, reason="rollback"
        )
    )

    with pytest.raises(ValueError, match="already rolled_back"):
        service.reject(
            ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id)
        )


def test_mismatched_rollout_artifact_id_rejected() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )

    with pytest.raises(ValueError, match="does not match"):
        service.approve(
            ForgePromotionDecisionRequest(rollout_artifact_id="wrong-artifact-id")
        )


# ---------------------------------------------------------------------------
# Canary percentage bounding
# ---------------------------------------------------------------------------


def test_canary_percentage_bounded_by_max() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(
        settings=settings, max_canary_percentage=10.0
    )

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(
                settings=settings, canary_percentage=50.0
            ),
        )
    )
    assert proposed.canary_percentage == 10.0


def test_canary_percentage_requested_override_capped() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(
                settings=settings, canary_percentage=5.0
            ),
        )
    )
    rollout_id = proposed.rollout_artifact_id
    assert rollout_id is not None
    service.approve(ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id))

    canary = service.apply(
        ForgePromotionApplyRequest(
            rollout_artifact_id=rollout_id,
            canary_percentage=50.0,
        )
    )
    assert canary.canary_percentage == 15.0


# ---------------------------------------------------------------------------
# Comparison evidence and semantics
# ---------------------------------------------------------------------------


def test_comparison_evidence_propagated_correctly() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    trial = build_promotable_trial_artifact(settings=settings)
    rollout_id = _propose_approve_apply(service, settings=settings, trial=trial)

    compared = service.compare(
        ForgePromotionCompareRequest(
            rollout_artifact_id=rollout_id,
            comparison_artifact=build_comparison_artifact(trial=trial),
        )
    )

    comparison = compared.comparison
    assert comparison is not None
    assert comparison.campaign_id == "campaign-phase9-001"
    assert (
        comparison.recommendation_disposition
        is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    )
    assert (
        comparison.recommendation_label
        is OptimizationRecommendationLabel.PROMOTION_ELIGIBLE
    )
    assert comparison.rank == 1
    assert comparison.pareto_optimal is True
    assert comparison.dominated is False
    assert len(comparison.evidence_kinds) > 0
    assert OptimizationArtifactEvidenceKind.OBSERVED in comparison.evidence_kinds
    assert OptimizationArtifactEvidenceKind.REPLAYED in comparison.evidence_kinds


def test_comparison_preserves_pareto_and_rank_metadata() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    trial = build_promotable_trial_artifact(settings=settings)
    rollout_id = _propose_approve_apply(service, settings=settings, trial=trial)

    comparison_artifact = build_comparison_artifact(trial=trial)
    compared = service.compare(
        ForgePromotionCompareRequest(
            rollout_artifact_id=rollout_id,
            comparison_artifact=comparison_artifact,
        )
    )

    comparison = compared.comparison
    assert comparison is not None
    assert comparison.comparison_artifact_id == comparison_artifact.comparison_artifact_id
    assert comparison.rank == 1
    assert comparison.pareto_optimal is True
    assert comparison.dominated is False
    assert comparison.rationale == ["candidate improved the primary objective"]


# ---------------------------------------------------------------------------
# Hybrid knob change application and rollback
# ---------------------------------------------------------------------------


def test_hybrid_knob_changes_applied_through_canary() -> None:
    settings = Settings()
    service, _policy_rollout, spillover = _create_service(settings=settings)

    trial = build_promotable_trial_artifact(
        settings=settings,
        knob_changes=[
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value=RoutingPolicy.BALANCED.value,
                candidate_value=RoutingPolicy.LOCAL_PREFERRED.value,
            ),
            OptimizationKnobChange(
                knob_id="hybrid_max_remote_share_percent",
                config_path="phase7.hybrid_execution.max_remote_share_percent",
                baseline_value=0.0,
                candidate_value=25.0,
            ),
        ],
    )

    _propose_approve_apply(service, settings=settings, trial=trial)

    canary = service.inspect_state()
    assert canary.applied is True
    applied_knob_ids = {k.knob_id for k in canary.applied_knob_changes}
    assert "default_routing_policy" in applied_knob_ids
    assert "hybrid_max_remote_share_percent" in applied_knob_ids

    hybrid_state = spillover.inspect_state()
    assert hybrid_state.max_remote_share_percent == 25.0


def test_hybrid_knob_changes_rolled_back() -> None:
    settings = Settings()
    service, policy_rollout, spillover = _create_service(settings=settings)

    baseline_max_remote = spillover.inspect_state().max_remote_share_percent

    trial = build_promotable_trial_artifact(
        settings=settings,
        knob_changes=[
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value=RoutingPolicy.BALANCED.value,
                candidate_value=RoutingPolicy.LOCAL_PREFERRED.value,
            ),
            OptimizationKnobChange(
                knob_id="hybrid_max_remote_share_percent",
                config_path="phase7.hybrid_execution.max_remote_share_percent",
                baseline_value=baseline_max_remote,
                candidate_value=30.0,
            ),
        ],
    )

    rollout_id = _propose_approve_apply(service, settings=settings, trial=trial)
    assert spillover.inspect_state().max_remote_share_percent == 30.0

    service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="rollback hybrid changes",
        )
    )

    assert spillover.inspect_state().max_remote_share_percent == baseline_max_remote
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED


# ---------------------------------------------------------------------------
# Lifecycle audit trail
# ---------------------------------------------------------------------------


def test_lifecycle_events_form_complete_audit_trail() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    rollout_id = proposed.rollout_artifact_id
    assert rollout_id is not None

    service.approve(ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id))
    service.apply(ForgePromotionApplyRequest(rollout_artifact_id=rollout_id))
    service.compare(
        ForgePromotionCompareRequest(
            rollout_artifact_id=rollout_id,
            comparison_artifact=build_comparison_artifact(
                trial=build_promotable_trial_artifact(settings=settings),
            ),
        )
    )
    final = service.promote_default(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="evidence supports promotion",
        )
    )

    events = final.lifecycle_events
    assert len(events) == 5
    expected_states = [
        ForgePromotionLifecycleState.PROPOSED,
        ForgePromotionLifecycleState.APPROVED,
        ForgePromotionLifecycleState.CANARY_ACTIVE,
        ForgePromotionLifecycleState.COMPARED,
        ForgePromotionLifecycleState.PROMOTED_DEFAULT,
    ]
    actual_states = [event.lifecycle_state for event in events]
    assert actual_states == expected_states

    # Every event should have a unique ID.
    event_ids = [event.event_id for event in events]
    assert len(set(event_ids)) == len(event_ids)

    # Events should be chronologically ordered.
    for i in range(1, len(events)):
        assert events[i].recorded_at >= events[i - 1].recorded_at


def test_reject_lifecycle_events_include_reason() -> None:
    settings = Settings()
    service, _policy_rollout, _spillover = _create_service(settings=settings)

    proposed = service.propose(
        ForgePromotionProposeRequest(
            trial_artifact=build_promotable_trial_artifact(settings=settings),
        )
    )
    rollout_id = proposed.rollout_artifact_id
    assert rollout_id is not None

    rejected = service.reject(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="not ready for production",
        )
    )

    events = rejected.lifecycle_events
    assert len(events) == 2  # PROPOSED + REJECTED
    reject_event = events[-1]
    assert reject_event.lifecycle_state is ForgePromotionLifecycleState.REJECTED
    assert (
        reject_event.promotion_disposition
        is OptimizationPromotionDisposition.REJECTED
    )
    assert any("not ready for production" in note for note in reject_event.notes)


# ---------------------------------------------------------------------------
# End-to-end: recommendation -> profile -> safe rollout -> rollback
# ---------------------------------------------------------------------------


def test_end_to_end_recommendation_to_rollout_and_rollback() -> None:
    """Full pipeline from recommendation artifacts through promotion lifecycle."""
    from switchyard.optimization_profiles import promote_recommendation_to_profile

    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    trial = build_promotable_trial_artifact(settings=settings)
    campaign = _build_campaign_artifact(settings=settings, trial=trial)
    comparison = build_comparison_artifact(trial=trial)

    # Step 1: Convert recommendation to explicit profile.
    profile = promote_recommendation_to_profile(
        settings=settings,
        campaign_artifact=campaign,
        comparison_artifact=comparison,
        candidate_configuration_id=trial.candidate_configuration.candidate_configuration_id,
    )
    assert profile.config_profile_id == "phase9-local-preferred"

    # Step 2: Propose through the promotion service.
    proposed = service.propose(
        ForgePromotionProposeRequest(trial_artifact=trial)
    )
    assert proposed.lifecycle_state is ForgePromotionLifecycleState.PROPOSED
    assert proposed.config_profile is not None
    assert proposed.config_profile.config_profile_id == profile.config_profile_id
    rollout_id = proposed.rollout_artifact_id
    assert rollout_id is not None

    # Step 3: Approve and apply canary.
    service.approve(ForgePromotionDecisionRequest(rollout_artifact_id=rollout_id))
    canary = service.apply(
        ForgePromotionApplyRequest(rollout_artifact_id=rollout_id)
    )
    assert canary.lifecycle_state is ForgePromotionLifecycleState.CANARY_ACTIVE
    assert canary.applied is True
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.CANARY

    # Step 4: Attach comparison evidence.
    compared = service.compare(
        ForgePromotionCompareRequest(
            rollout_artifact_id=rollout_id,
            comparison_artifact=comparison,
        )
    )
    assert compared.lifecycle_state is ForgePromotionLifecycleState.COMPARED
    assert compared.comparison is not None

    # Step 5: Safe rollback.
    reset = service.reset(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="testing safe rollback from compared state",
        )
    )
    assert reset.lifecycle_state is ForgePromotionLifecycleState.ROLLED_BACK
    assert reset.applied is False
    assert reset.active_config_profile_id == _baseline_config_profile_id(settings)
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.DISABLED

    # Verify the full audit trail.
    events = reset.lifecycle_events
    expected_states = [
        ForgePromotionLifecycleState.PROPOSED,
        ForgePromotionLifecycleState.APPROVED,
        ForgePromotionLifecycleState.CANARY_ACTIVE,
        ForgePromotionLifecycleState.COMPARED,
        ForgePromotionLifecycleState.ROLLED_BACK,
    ]
    actual_states = [event.lifecycle_state for event in events]
    assert actual_states == expected_states


def test_end_to_end_promotion_to_default_with_full_audit() -> None:
    """Full lifecycle through promote-default with serialization round-trip."""
    settings = Settings()
    service, policy_rollout, _spillover = _create_service(settings=settings)

    trial = build_promotable_trial_artifact(settings=settings)
    rollout_id = _propose_approve_apply_compare(
        service, settings=settings, trial=trial
    )

    promoted = service.promote_default(
        ForgePromotionDecisionRequest(
            rollout_artifact_id=rollout_id,
            reason="canary evidence was favorable",
        )
    )
    assert promoted.lifecycle_state is ForgePromotionLifecycleState.PROMOTED_DEFAULT
    assert promoted.rollout_mode is PolicyRolloutMode.ACTIVE_GUARDED
    assert (
        promoted.promotion_disposition
        is OptimizationPromotionDisposition.PROMOTED_DEFAULT
    )
    assert policy_rollout.inspect_state().mode is PolicyRolloutMode.ACTIVE_GUARDED

    # Verify serialization round-trip of the promotion summary.
    from switchyard.schemas.forge import ForgePromotionRuntimeSummary

    round_tripped = ForgePromotionRuntimeSummary.model_validate_json(
        promoted.model_dump_json()
    )
    assert (
        round_tripped.lifecycle_state is ForgePromotionLifecycleState.PROMOTED_DEFAULT
    )
    assert len(round_tripped.lifecycle_events) == 5
