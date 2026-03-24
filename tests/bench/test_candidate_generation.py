from __future__ import annotations

from datetime import UTC, datetime

from switchyard.bench.candidate_generation import generate_forge_stage_a_candidates
from switchyard.config import Settings
from switchyard.optimization import build_optimization_profile
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    OptimizationCandidateEligibilityStatus,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationStrategy,
)
from switchyard.schemas.routing import RoutingPolicy


def test_candidate_generation_is_deterministic_when_seeded() -> None:
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_SLO,
        RoutingPolicy.LOCAL_PREFERRED,
    )
    config = OptimizationCandidateGenerationConfig(
        strategies=[
            OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
            OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH,
            OptimizationCandidateGenerationStrategy.RANDOM_SEARCH,
        ],
        allowed_knob_ids=[
            "default_routing_policy",
            "policy_rollout_mode",
            "policy_rollout_canary_percentage",
            "shadow_sampling_rate",
        ],
        seed=17,
        max_random_candidates=4,
    )

    first = generate_forge_stage_a_candidates(
        settings=settings,
        config=config,
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
    )
    second = generate_forge_stage_a_candidates(
        settings=settings,
        config=config,
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
    )

    assert [
        (
            candidate.trial.candidate_id,
            candidate.generation.strategy,
            [(change.knob_id, change.candidate_value) for change in candidate.knob_changes],
        )
        for candidate in first.eligible_candidates
    ] == [
        (
            candidate.trial.candidate_id,
            candidate.generation.strategy,
            [(change.knob_id, change.candidate_value) for change in candidate.knob_changes],
        )
        for candidate in second.eligible_candidates
    ]
    assert [
        (
            candidate.trial.candidate_id,
            candidate.eligibility.status,
            list(candidate.eligibility.rejection_reasons),
        )
        for candidate in first.rejected_candidates
    ] == [
        (
            candidate.trial.candidate_id,
            candidate.eligibility.status,
            list(candidate.eligibility.rejection_reasons),
        )
        for candidate in second.rejected_candidates
    ]


def test_candidate_generation_prunes_hard_constraint_violations() -> None:
    settings = Settings()
    profile = build_optimization_profile(settings)
    for constraint in profile.constraints:
        if constraint.constraint_id == "max_shadow_sampling_rate":
            constraint.threshold_value = 0.05
            break
    else:
        raise AssertionError("expected max_shadow_sampling_rate constraint")

    result = generate_forge_stage_a_candidates(
        settings=settings,
        profile=profile,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["shadow_sampling_rate"],
            seed=11,
        ),
    )

    pruned = [
        candidate
        for candidate in result.rejected_candidates
        if candidate.eligibility.status is OptimizationCandidateEligibilityStatus.PRUNED
    ]

    assert pruned
    assert "max_shadow_sampling_rate" in pruned[0].eligibility.blocking_constraint_ids


def test_candidate_generation_rejects_invalid_combinations_early() -> None:
    settings = Settings()

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=[
                "policy_rollout_mode",
                "policy_rollout_canary_percentage",
            ],
            seed=3,
        ),
    )

    rejected = [
        candidate
        for candidate in result.rejected_candidates
        if candidate.eligibility.status is OptimizationCandidateEligibilityStatus.REJECTED
    ]

    assert rejected
    assert any(
        "canary rollout mode requires a positive canary percentage"
        in candidate.eligibility.rejection_reasons
        for candidate in rejected
    )


def test_candidate_generation_records_metadata_for_eligible_candidates() -> None:
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LOCAL_PREFERRED,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["default_routing_policy"],
            seed=29,
        ),
    )

    assert len(result.eligible_candidates) == 1
    candidate = result.eligible_candidates[0]

    assert candidate.trial.candidate_kind is ForgeCandidateKind.ROUTING_POLICY
    assert candidate.generation.strategy is (
        OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME
    )
    assert candidate.generation.seed == 29
    assert candidate.generation.varied_knob_ids == ["default_routing_policy"]
    assert candidate.eligibility.status is OptimizationCandidateEligibilityStatus.ELIGIBLE
    assert candidate.eligibility.eligible is True
    assert candidate.knob_changes[0].knob_id == "default_routing_policy"
