from __future__ import annotations

import pytest

from switchyard.config import HybridExecutionSettings, PolicyRolloutSettings, Settings
from switchyard.control.forge_promotion import ForgePromotionService
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.spillover import RemoteSpilloverControlService
from switchyard.optimization import build_baseline_optimization_config_profile
from switchyard.schemas.benchmark import RecommendationConfidence
from switchyard.schemas.forge import (
    ForgePromotionApplyRequest,
    ForgePromotionCompareRequest,
    ForgePromotionDecisionRequest,
    ForgePromotionLifecycleState,
    ForgePromotionProposeRequest,
)
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    OptimizationArtifactEvidenceKind,
    OptimizationCampaignComparisonArtifact,
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
