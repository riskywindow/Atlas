"""Tests for explicit optimization-profile operations.

Covers: profile creation from recommendations, application-boundary validation,
scope compatibility, profile diff computation, provenance tracking, and edge cases.
"""

from __future__ import annotations

import pytest

from switchyard.config import Settings
from switchyard.optimization import (
    build_baseline_optimization_config_profile,
    build_candidate_optimization_config_profile,
    build_optimization_profile,
)
from switchyard.optimization_profiles import (
    ProfileApplicationBoundary,
    check_profile_scope_compatibility,
    check_profile_scope_compatibility_for_knob,
    compute_profile_diff,
    promote_recommendation_to_profile,
    validate_profile_application_boundary,
)
from switchyard.schemas.benchmark import (
    CounterfactualObjective,
    RecommendationConfidence,
    WorkloadScenarioFamily,
)
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactSourceType,
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationCampaignMetadata,
    OptimizationCandidateComparisonArtifact,
    OptimizationCandidateConfigurationArtifact,
    OptimizationConfigProfileRole,
    OptimizationEvidenceRecord,
    OptimizationGoal,
    OptimizationKnobChange,
    OptimizationObjectiveDelta,
    OptimizationObjectiveMetric,
    OptimizationPromotionDecision,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationScope,
    OptimizationScopeKind,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import PolicyRolloutMode, RoutingPolicy

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return Settings()


def _baseline_config_profile_id(settings: Settings) -> str:
    return build_baseline_optimization_config_profile(settings).config_profile_id


def _trial_identity(
    *,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
) -> OptimizationTrialIdentity:
    return OptimizationTrialIdentity(
        trial_id=f"trial-{config_profile_id}",
        candidate_id=f"routing-policy:{routing_policy.value}",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )


def _candidate_configuration(
    *,
    settings: Settings,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    knob_changes: list[OptimizationKnobChange] | None = None,
) -> OptimizationCandidateConfigurationArtifact:
    identity = _trial_identity(
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )
    return OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=f"candidate-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        candidate=identity,
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


def _recommendation_summary(
    *,
    candidate_configuration_id: str,
    config_profile_id: str,
) -> OptimizationRecommendationSummary:
    return OptimizationRecommendationSummary(
        recommendation_summary_id=f"rec-{config_profile_id}",
        disposition=OptimizationRecommendationDisposition.PROMOTE_CANDIDATE,
        recommendation_label=OptimizationRecommendationLabel.PROMOTION_ELIGIBLE,
        confidence=RecommendationConfidence.MEDIUM,
        candidate_configuration_id=candidate_configuration_id,
        config_profile_id=config_profile_id,
        evidence_kinds=[
            OptimizationArtifactEvidenceKind.OBSERVED,
            OptimizationArtifactEvidenceKind.REPLAYED,
        ],
        reason_codes=[
            OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED,
        ],
        rationale=["candidate improved the primary objective"],
    )


def _promotion_decision(
    *,
    candidate_configuration_id: str,
    config_profile_id: str,
) -> OptimizationPromotionDecision:
    return OptimizationPromotionDecision(
        promotion_decision_id=f"promo-{config_profile_id}",
        disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
        candidate_configuration_id=candidate_configuration_id,
        config_profile_id=config_profile_id,
        rollout_mode=PolicyRolloutMode.CANARY,
        canary_percentage=10.0,
    )


def _trial_artifact(
    *,
    settings: Settings,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    knob_changes: list[OptimizationKnobChange] | None = None,
) -> OptimizationTrialArtifact:
    candidate = _candidate_configuration(
        settings=settings,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
        knob_changes=knob_changes,
    )
    return OptimizationTrialArtifact(
        trial_artifact_id=f"trial-artifact-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-phase9-baseline",
        candidate_configuration=candidate,
        trial_identity=candidate.candidate,
        evidence_records=[
            OptimizationEvidenceRecord(
                evidence_id="evidence-observed-1",
                evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
                source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
                source_artifact_id="benchmark-run-001",
            ),
        ],
        recommendation_summary=_recommendation_summary(
            candidate_configuration_id=candidate.candidate_configuration_id,
            config_profile_id=config_profile_id,
        ),
        promotion_decision=_promotion_decision(
            candidate_configuration_id=candidate.candidate_configuration_id,
            config_profile_id=config_profile_id,
        ),
    )


def _baseline_candidate_configuration(
    settings: Settings,
) -> OptimizationCandidateConfigurationArtifact:
    baseline_id = _baseline_config_profile_id(settings)
    return OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id="candidate-phase9-baseline",
        campaign_id="campaign-phase9-001",
        candidate=OptimizationTrialIdentity(
            trial_id="trial-baseline",
            candidate_id="routing-policy:balanced",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id=baseline_id,
            routing_policy=RoutingPolicy.BALANCED,
        ),
        baseline_config_profile_id=baseline_id,
        config_profile_id=baseline_id,
    )


def _campaign_artifact(
    settings: Settings,
    *,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    knob_changes: list[OptimizationKnobChange] | None = None,
) -> OptimizationCampaignArtifact:
    baseline = _baseline_candidate_configuration(settings)
    candidate = _candidate_configuration(
        settings=settings,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
        knob_changes=knob_changes,
    )
    trial = _trial_artifact(
        settings=settings,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
        knob_changes=knob_changes,
    )
    return OptimizationCampaignArtifact(
        campaign_artifact_id="campaign-artifact-001",
        campaign=OptimizationCampaignMetadata(
            campaign_id="campaign-phase9-001",
            optimization_profile_id="phase9-stage-a-baseline",
            objective=CounterfactualObjective.BALANCED,
            evidence_sources=[ForgeEvidenceSourceKind.OBSERVED_RUNTIME],
        ),
        baseline_candidate_configuration=baseline,
        candidate_configurations=[candidate],
        trials=[trial],
    )


def _comparison_artifact(
    *,
    settings: Settings,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    knob_changes: list[OptimizationKnobChange] | None = None,
) -> OptimizationCampaignComparisonArtifact:
    candidate = _candidate_configuration(
        settings=settings,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
        knob_changes=knob_changes,
    )
    return OptimizationCampaignComparisonArtifact(
        comparison_artifact_id="comparison-001",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-phase9-baseline",
        candidate_comparisons=[
            OptimizationCandidateComparisonArtifact(
                candidate_configuration_id=candidate.candidate_configuration_id,
                trial_artifact_id=f"trial-artifact-{config_profile_id}",
                config_profile_id=config_profile_id,
                rank=1,
                pareto_optimal=True,
                dominated=False,
                objective_deltas=[
                    OptimizationObjectiveDelta(
                        objective_id="latency-primary",
                        metric=OptimizationObjectiveMetric.LATENCY_MS,
                        goal=OptimizationGoal.MINIMIZE,
                        baseline_value=100.0,
                        candidate_value=90.0,
                        absolute_delta=-10.0,
                        improved=True,
                    )
                ],
                recommendation_summary=_recommendation_summary(
                    candidate_configuration_id=candidate.candidate_configuration_id,
                    config_profile_id=config_profile_id,
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# promote_recommendation_to_profile
# ---------------------------------------------------------------------------


class TestPromoteRecommendationToProfile:
    def test_creates_profile_from_recommended_candidate(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        profile = promote_recommendation_to_profile(
            settings=settings,
            campaign_artifact=campaign,
            comparison_artifact=comparison,
            candidate_configuration_id="candidate-phase9-local-preferred",
        )

        assert profile.config_profile_id == "phase9-local-preferred"
        assert profile.profile_role is OptimizationConfigProfileRole.PROMOTED
        assert profile.routing_policy is RoutingPolicy.LOCAL_PREFERRED
        assert profile.provenance.campaign_artifact_id == "campaign-artifact-001"
        assert (
            profile.provenance.trial_artifact_id
            == "trial-artifact-phase9-local-preferred"
        )
        assert (
            profile.provenance.candidate_configuration_id
            == "candidate-phase9-local-preferred"
        )

    def test_carries_provenance_back_to_campaign(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        profile = promote_recommendation_to_profile(
            settings=settings,
            campaign_artifact=campaign,
            comparison_artifact=comparison,
            candidate_configuration_id="candidate-phase9-local-preferred",
        )

        provenance = profile.provenance
        assert provenance.campaign_id == "campaign-phase9-001"
        assert provenance.campaign_artifact_id == "campaign-artifact-001"
        assert provenance.candidate_kind is ForgeCandidateKind.ROUTING_POLICY
        assert provenance.recommendation_summary_id == "rec-phase9-local-preferred"
        assert (
            provenance.recommendation_disposition
            is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
        )

    def test_respects_profile_version(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        profile = promote_recommendation_to_profile(
            settings=settings,
            campaign_artifact=campaign,
            comparison_artifact=comparison,
            candidate_configuration_id="candidate-phase9-local-preferred",
            profile_version=5,
        )

        assert profile.profile_version == 5

    def test_rejects_missing_candidate_in_comparison(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        with pytest.raises(ValueError, match="not present in the comparison"):
            promote_recommendation_to_profile(
                settings=settings,
                campaign_artifact=campaign,
                comparison_artifact=comparison,
                candidate_configuration_id="nonexistent-candidate",
            )

    def test_rejects_missing_candidate_in_campaign_trials(self) -> None:
        settings = _settings()
        # Create a campaign with no trials.
        baseline = _baseline_candidate_configuration(settings)
        candidate = _candidate_configuration(settings=settings)
        campaign = OptimizationCampaignArtifact(
            campaign_artifact_id="campaign-artifact-001",
            campaign=OptimizationCampaignMetadata(
                campaign_id="campaign-phase9-001",
                optimization_profile_id="phase9-stage-a-baseline",
            ),
            baseline_candidate_configuration=baseline,
            candidate_configurations=[candidate],
            trials=[],
        )
        comparison = _comparison_artifact(settings=settings)

        with pytest.raises(ValueError, match="not present in the campaign artifact"):
            promote_recommendation_to_profile(
                settings=settings,
                campaign_artifact=campaign,
                comparison_artifact=comparison,
                candidate_configuration_id="candidate-phase9-local-preferred",
            )

    def test_rejects_campaign_id_mismatch(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)
        # Mutate the comparison to have a different campaign_id.
        wrong_comparison = comparison.model_copy(
            update={"campaign_id": "wrong-campaign"},
            deep=True,
        )

        with pytest.raises(ValueError, match="does not match"):
            promote_recommendation_to_profile(
                settings=settings,
                campaign_artifact=campaign,
                comparison_artifact=wrong_comparison,
                candidate_configuration_id="candidate-phase9-local-preferred",
            )


# ---------------------------------------------------------------------------
# validate_profile_application_boundary
# ---------------------------------------------------------------------------


class TestValidateProfileApplicationBoundary:
    def test_baseline_profile_is_within_boundary(self) -> None:
        settings = _settings()
        baseline = build_baseline_optimization_config_profile(settings)

        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=baseline,
        )

        assert boundary.within_boundary is True
        assert boundary.total_changes == 0
        assert boundary.undeclared_knob_ids == []

    def test_declared_tunable_knob_is_within_boundary(self) -> None:
        settings = _settings()
        candidate = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=20.0,
                )
            ],
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=profile,
        )

        assert boundary.within_boundary is True
        assert boundary.total_changes == 1
        assert "hybrid_max_remote_share_percent" in boundary.tunable_knob_ids
        assert "hybrid_max_remote_share_percent" in boundary.runtime_mutable_knob_ids
        assert boundary.all_runtime_mutable is True

    def test_undeclared_knob_breaks_boundary(self) -> None:
        settings = _settings()
        candidate = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="nonexistent_knob",
                    config_path="nonexistent.path",
                    baseline_value=0,
                    candidate_value=1,
                )
            ],
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=profile,
        )

        assert boundary.within_boundary is False
        assert "nonexistent_knob" in boundary.undeclared_knob_ids
        # The knob entry should show it's not declared tunable.
        knob_entry = next(
            (k for k in boundary.knobs if k.knob_id == "nonexistent_knob"), None
        )
        assert knob_entry is not None
        assert knob_entry.declared_tunable is False
        assert knob_entry.allowed is False

    def test_immutable_knob_marks_not_all_runtime_mutable(self) -> None:
        settings = _settings()
        candidate = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="default_routing_policy",
                    config_path="default_routing_policy",
                    baseline_value=RoutingPolicy.BALANCED.value,
                    candidate_value=RoutingPolicy.LOCAL_PREFERRED.value,
                )
            ],
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=profile,
        )

        # default_routing_policy is mutable_at_runtime=False in the surface.
        assert boundary.within_boundary is True
        assert "default_routing_policy" in boundary.immutable_knob_ids
        assert boundary.all_runtime_mutable is False

    def test_mixed_knobs_produces_correct_classification(self) -> None:
        settings = _settings()
        candidate = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="default_routing_policy",
                    config_path="default_routing_policy",
                    baseline_value=RoutingPolicy.BALANCED.value,
                    candidate_value=RoutingPolicy.LOCAL_PREFERRED.value,
                ),
                OptimizationKnobChange(
                    knob_id="hybrid_spillover_enabled",
                    config_path="phase7.hybrid_execution.spillover_enabled",
                    baseline_value=False,
                    candidate_value=True,
                ),
            ],
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=profile,
        )

        assert boundary.within_boundary is True
        assert boundary.total_changes == 2
        assert "default_routing_policy" in boundary.immutable_knob_ids
        assert "hybrid_spillover_enabled" in boundary.runtime_mutable_knob_ids
        assert boundary.all_runtime_mutable is False
        assert boundary.scope_compatible is True

    def test_boundary_round_trips_through_json(self) -> None:
        settings = _settings()
        baseline = build_baseline_optimization_config_profile(settings)
        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=baseline,
        )

        round_tripped = ProfileApplicationBoundary.model_validate_json(
            boundary.model_dump_json()
        )
        assert round_tripped == boundary


# ---------------------------------------------------------------------------
# compute_profile_diff
# ---------------------------------------------------------------------------


class TestComputeProfileDiff:
    def test_identical_profiles_produce_empty_diff(self) -> None:
        settings = _settings()
        baseline = build_baseline_optimization_config_profile(settings)

        diff = compute_profile_diff(baseline=baseline, candidate=baseline)

        assert diff.changed_knob_ids == []
        assert diff.changed_groups == []
        assert diff.mutable_runtime_knob_ids == []
        assert diff.immutable_knob_ids == []
        assert any("equivalent" in note for note in diff.notes)

    def test_diff_detects_changed_knobs(self) -> None:
        settings = _settings()
        baseline = build_baseline_optimization_config_profile(settings)
        candidate_config = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=20.0,
                )
            ],
        )
        candidate = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate_config,
        )

        diff = compute_profile_diff(baseline=baseline, candidate=candidate)

        assert "hybrid_max_remote_share_percent" in diff.changed_knob_ids
        assert diff.mutable_runtime_knob_ids == ["hybrid_max_remote_share_percent"]
        assert diff.immutable_knob_ids == []

    def test_diff_between_two_candidates(self) -> None:
        settings = _settings()
        candidate_a_config = _candidate_configuration(
            settings=settings,
            config_profile_id="phase9-candidate-a",
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=20.0,
                )
            ],
        )
        candidate_b_config = _candidate_configuration(
            settings=settings,
            config_profile_id="phase9-candidate-b",
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=30.0,
                )
            ],
        )
        candidate_a = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate_a_config,
        )
        candidate_b = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate_b_config,
        )

        diff = compute_profile_diff(baseline=candidate_a, candidate=candidate_b)

        assert "hybrid_max_remote_share_percent" in diff.changed_knob_ids
        assert diff.baseline_config_profile_id == "phase9-candidate-a"
        assert diff.config_profile_id == "phase9-candidate-b"

    def test_diff_with_disjoint_knob_sets(self) -> None:
        settings = _settings()
        candidate_a_config = _candidate_configuration(
            settings=settings,
            config_profile_id="phase9-candidate-a",
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_spillover_enabled",
                    config_path="phase7.hybrid_execution.spillover_enabled",
                    baseline_value=False,
                    candidate_value=True,
                )
            ],
        )
        candidate_b_config = _candidate_configuration(
            settings=settings,
            config_profile_id="phase9-candidate-b",
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=30.0,
                )
            ],
        )
        candidate_a = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate_a_config,
        )
        candidate_b = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate_b_config,
        )

        diff = compute_profile_diff(baseline=candidate_a, candidate=candidate_b)

        # Both knobs should appear as changed since one is in A but not B and vice versa.
        assert "hybrid_spillover_enabled" in diff.changed_knob_ids
        assert "hybrid_max_remote_share_percent" in diff.changed_knob_ids


# ---------------------------------------------------------------------------
# check_profile_scope_compatibility
# ---------------------------------------------------------------------------


class TestCheckProfileScopeCompatibility:
    def test_global_profile_covers_global_target(self) -> None:
        settings = _settings()
        profile = build_baseline_optimization_config_profile(settings)

        result = check_profile_scope_compatibility(
            config_profile=profile,
            target_scope=OptimizationScope(kind=OptimizationScopeKind.GLOBAL),
        )

        assert result is True

    def test_global_profile_covers_specific_target(self) -> None:
        settings = _settings()
        profile = build_baseline_optimization_config_profile(settings)

        result = check_profile_scope_compatibility(
            config_profile=profile,
            target_scope=OptimizationScope(
                kind=OptimizationScopeKind.SERVING_TARGET,
                target="chat-shared",
            ),
        )

        assert result is True

    def test_scoped_profile_does_not_cover_different_target(self) -> None:
        settings = _settings()
        candidate = _candidate_configuration(settings=settings)
        # Override the trial identity to have a specific scope.
        scoped_identity = OptimizationTrialIdentity(
            trial_id="trial-scoped",
            candidate_id="routing-policy:balanced-scoped",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="phase9-scoped",
            routing_policy=RoutingPolicy.BALANCED,
            applies_to=[
                OptimizationScope(
                    kind=OptimizationScopeKind.SCENARIO_FAMILY,
                    target=WorkloadScenarioFamily.REPEATED_PREFIX.value,
                )
            ],
        )
        scoped_candidate = candidate.model_copy(
            update={
                "candidate": scoped_identity,
                "config_profile_id": "phase9-scoped",
            },
            deep=True,
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=scoped_candidate,
        )

        # Should be compatible with the matching scenario family.
        assert (
            check_profile_scope_compatibility(
                config_profile=profile,
                target_scope=OptimizationScope(
                    kind=OptimizationScopeKind.SCENARIO_FAMILY,
                    target=WorkloadScenarioFamily.REPEATED_PREFIX.value,
                ),
            )
            is True
        )

        # Should NOT be compatible with a different scope.
        assert (
            check_profile_scope_compatibility(
                config_profile=profile,
                target_scope=OptimizationScope(
                    kind=OptimizationScopeKind.WORKER_CLASS,
                    target="host_native",
                ),
            )
            is False
        )

    def test_scoped_profile_does_not_cover_global_target(self) -> None:
        settings = _settings()
        scoped_identity = OptimizationTrialIdentity(
            trial_id="trial-scoped",
            candidate_id="routing-policy:balanced-scoped",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="phase9-scoped",
            routing_policy=RoutingPolicy.BALANCED,
            applies_to=[
                OptimizationScope(
                    kind=OptimizationScopeKind.SCENARIO_FAMILY,
                    target=WorkloadScenarioFamily.REPEATED_PREFIX.value,
                )
            ],
        )
        candidate = OptimizationCandidateConfigurationArtifact(
            candidate_configuration_id="candidate-scoped",
            campaign_id="campaign-phase9-001",
            candidate=scoped_identity,
            baseline_config_profile_id=_baseline_config_profile_id(settings),
            config_profile_id="phase9-scoped",
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        assert (
            check_profile_scope_compatibility(
                config_profile=profile,
                target_scope=OptimizationScope(kind=OptimizationScopeKind.GLOBAL),
            )
            is False
        )


# ---------------------------------------------------------------------------
# check_profile_scope_compatibility_for_knob
# ---------------------------------------------------------------------------


class TestCheckProfileScopeCompatibilityForKnob:
    def test_knob_present_in_profile_is_compatible(self) -> None:
        settings = _settings()
        candidate = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=20.0,
                )
            ],
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        assert (
            check_profile_scope_compatibility_for_knob(
                config_profile=profile,
                knob_id="hybrid_max_remote_share_percent",
                target_scope=OptimizationScope(kind=OptimizationScopeKind.GLOBAL),
            )
            is True
        )

    def test_absent_knob_is_not_compatible(self) -> None:
        settings = _settings()
        profile = build_baseline_optimization_config_profile(settings)

        assert (
            check_profile_scope_compatibility_for_knob(
                config_profile=profile,
                knob_id="nonexistent_knob",
                target_scope=OptimizationScope(kind=OptimizationScopeKind.GLOBAL),
            )
            is False
        )

    def test_checks_against_optimization_surface(self) -> None:
        settings = _settings()
        surface = build_optimization_profile(settings)
        candidate = _candidate_configuration(
            settings=settings,
            knob_changes=[
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=0.0,
                    candidate_value=20.0,
                )
            ],
        )
        profile = build_candidate_optimization_config_profile(
            settings=settings,
            candidate_configuration=candidate,
        )

        # Global knob checked against a worker_class scope: should work because
        # the knob's surface scope is GLOBAL which covers any target.
        assert (
            check_profile_scope_compatibility_for_knob(
                config_profile=profile,
                knob_id="hybrid_max_remote_share_percent",
                target_scope=OptimizationScope(
                    kind=OptimizationScopeKind.WORKER_CLASS,
                    target="host_native",
                ),
                optimization_surface=surface,
            )
            is True
        )


# ---------------------------------------------------------------------------
# End-to-end: recommendation → profile → boundary
# ---------------------------------------------------------------------------


class TestEndToEndRecommendationToProfile:
    def test_full_promotion_pipeline(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        # Step 1: Convert recommendation to profile.
        profile = promote_recommendation_to_profile(
            settings=settings,
            campaign_artifact=campaign,
            comparison_artifact=comparison,
            candidate_configuration_id="candidate-phase9-local-preferred",
        )

        # Step 2: Validate the boundary.
        boundary = validate_profile_application_boundary(
            settings=settings,
            config_profile=profile,
        )
        assert boundary.within_boundary is True

        # Step 3: Diff against baseline.
        baseline = build_baseline_optimization_config_profile(settings)
        diff = compute_profile_diff(baseline=baseline, candidate=profile)
        assert "default_routing_policy" in diff.changed_knob_ids

        # Step 4: Check scope.
        assert (
            check_profile_scope_compatibility(
                config_profile=profile,
                target_scope=OptimizationScope(kind=OptimizationScopeKind.GLOBAL),
            )
            is True
        )

    def test_promoted_profile_carries_evidence_kinds_in_provenance(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        profile = promote_recommendation_to_profile(
            settings=settings,
            campaign_artifact=campaign,
            comparison_artifact=comparison,
            candidate_configuration_id="candidate-phase9-local-preferred",
        )

        assert len(profile.provenance.evidence_kinds) > 0
        assert OptimizationArtifactEvidenceKind.OBSERVED in profile.provenance.evidence_kinds

    def test_promoted_profile_serializes_and_round_trips(self) -> None:
        settings = _settings()
        campaign = _campaign_artifact(settings)
        comparison = _comparison_artifact(settings=settings)

        profile = promote_recommendation_to_profile(
            settings=settings,
            campaign_artifact=campaign,
            comparison_artifact=comparison,
            candidate_configuration_id="candidate-phase9-local-preferred",
        )

        from switchyard.schemas.optimization import OptimizationConfigProfile

        round_tripped = OptimizationConfigProfile.model_validate_json(
            profile.model_dump_json()
        )
        assert round_tripped == profile
        assert round_tripped.provenance.campaign_artifact_id == "campaign-artifact-001"
