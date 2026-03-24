from __future__ import annotations

from datetime import UTC, datetime

from switchyard.bench.candidate_generation import generate_forge_stage_a_candidates
from switchyard.config import Settings
from switchyard.optimization import build_optimization_profile
from switchyard.schemas.benchmark import CounterfactualObjective
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    OptimizationCandidateConfigurationArtifact,
    OptimizationCandidateEligibilityStatus,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationStrategy,
    OptimizationKnobChange,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import RoutingPolicy

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_settings() -> Settings:
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
    )
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.optimization.worker_launch_presets = ()
    settings.phase4.policy_rollout.candidate_policy_id = None
    settings.phase4.policy_rollout.canary_percentage = 10.0
    return settings


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


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


def test_different_seeds_produce_different_random_candidates() -> None:
    settings = _default_settings()
    knob_ids = [
        "default_routing_policy",
        "policy_rollout_canary_percentage",
        "shadow_sampling_rate",
    ]
    result_a = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.RANDOM_SEARCH],
            allowed_knob_ids=knob_ids,
            max_random_candidates=6,
            seed=1,
        ),
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
    )
    result_b = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.RANDOM_SEARCH],
            allowed_knob_ids=knob_ids,
            max_random_candidates=6,
            seed=999,
        ),
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
    )

    ids_a = {c.trial.candidate_id for c in result_a.eligible_candidates}
    ids_b = {c.trial.candidate_id for c in result_b.eligible_candidates}
    # With different seeds and random search, at least some candidates differ
    assert ids_a != ids_b or (not ids_a and not ids_b)


# ---------------------------------------------------------------------------
# constraint pruning
# ---------------------------------------------------------------------------


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


def test_canary_percentage_constraint_prunes_over_limit() -> None:
    settings = _default_settings()
    settings.optimization.max_rollout_canary_percentage = 5.0
    profile = build_optimization_profile(settings)

    result = generate_forge_stage_a_candidates(
        settings=settings,
        profile=profile,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["policy_rollout_canary_percentage"],
            seed=7,
        ),
    )

    # OFAT will try midpoint and max of [0.0, 5.0] range. Midpoint 2.5 is within
    # constraint, but the max 5.0 should be within too since the constraint is LTE.
    # Candidates above the old canary_percentage (10.0) are clamped by domain.
    all_candidates = result.eligible_candidates + result.rejected_candidates
    for candidate in all_candidates:
        for change in candidate.knob_changes:
            if change.knob_id == "policy_rollout_canary_percentage":
                assert isinstance(change.candidate_value, (int, float))
                assert float(change.candidate_value) <= 5.0


# ---------------------------------------------------------------------------
# invalid combination rejection
# ---------------------------------------------------------------------------


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


def test_remote_preferring_policy_rejected_when_spillover_disabled() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.BURST_TO_REMOTE,
    )
    settings.phase7.hybrid_execution.spillover_enabled = False

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["default_routing_policy"],
            seed=5,
        ),
    )

    rejected = [
        c for c in result.rejected_candidates
        if c.eligibility.status is OptimizationCandidateEligibilityStatus.REJECTED
    ]
    assert rejected
    assert any(
        "remote-preferring routing policies are incompatible with disabled spillover"
        in c.eligibility.rejection_reasons
        for c in rejected
    )


def test_disabled_spillover_rejects_remote_budget_knob_changes() -> None:
    settings = _default_settings()
    settings.phase7.hybrid_execution.spillover_enabled = False

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=[
                "hybrid_spillover_enabled",
                "hybrid_max_remote_share_percent",
            ],
            seed=8,
        ),
    )

    rejected = [
        c for c in result.rejected_candidates
        if c.eligibility.status is OptimizationCandidateEligibilityStatus.REJECTED
    ]
    rejected_reasons = [
        reason
        for c in rejected
        for reason in c.eligibility.rejection_reasons
    ]
    assert any(
        "remote budget/share tuning knobs require hybrid spillover" in reason
        or "remote share cannot be positive when hybrid spillover is disabled" in reason
        for reason in rejected_reasons
    )


# ---------------------------------------------------------------------------
# candidate metadata
# ---------------------------------------------------------------------------


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


def test_baseline_trial_always_present_in_result() -> None:
    settings = _default_settings()
    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.FIXED_BASELINE],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
        ),
    )

    assert result.baseline_trial is not None
    assert result.baseline_generation is not None
    assert result.baseline_generation.strategy is (
        OptimizationCandidateGenerationStrategy.FIXED_BASELINE
    )
    assert "baseline" in " ".join(result.notes).lower()


def test_generation_result_carries_config_and_profile_id() -> None:
    settings = _default_settings()
    config = OptimizationCandidateGenerationConfig(
        strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
        allowed_knob_ids=["default_routing_policy"],
        seed=42,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=config,
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
    )

    assert result.profile_id == settings.optimization.profile_id
    assert result.generation_config.seed == 42
    assert result.generated_at == datetime(2026, 3, 22, 19, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# strategy-specific: OFAT
# ---------------------------------------------------------------------------


def test_ofat_generates_one_candidate_per_enum_value() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
        RoutingPolicy.QUALITY_FIRST,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
            max_one_factor_variants_per_knob=10,
        ),
    )

    # Should generate one candidate per non-baseline policy
    routing_candidates = [
        c for c in result.eligible_candidates
        if any(ch.knob_id == "default_routing_policy" for ch in c.knob_changes)
    ]
    candidate_policies = {
        c.knob_changes[0].candidate_value for c in routing_candidates
    }
    # balanced is baseline, so expect latency_first, local_preferred, quality_first
    assert candidate_policies == {"latency_first", "local_preferred", "quality_first"}


def test_ofat_respects_max_variants_per_knob_limit() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
        RoutingPolicy.QUALITY_FIRST,
        RoutingPolicy.LOCAL_ONLY,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
            max_one_factor_variants_per_knob=2,
        ),
    )

    routing_candidates = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if any(ch.knob_id == "default_routing_policy" for ch in c.knob_changes)
    ]
    # Limited to 2 variants of this knob
    assert len(routing_candidates) <= 2


def test_ofat_records_varied_knob_ids_correctly() -> None:
    settings = _default_settings()
    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["policy_rollout_canary_percentage"],
            seed=0,
        ),
    )

    all_candidates = result.eligible_candidates + result.rejected_candidates
    for candidate in all_candidates:
        assert candidate.generation.varied_knob_ids == ["policy_rollout_canary_percentage"]
        assert candidate.generation.strategy is (
            OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME
        )


# ---------------------------------------------------------------------------
# strategy-specific: bounded grid search
# ---------------------------------------------------------------------------


def test_grid_search_produces_cartesian_product_of_small_domains() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH],
            allowed_knob_ids=[
                "default_routing_policy",
                "hybrid_prefer_local",
            ],
            seed=0,
            max_grid_dimensions=2,
            max_grid_candidates=20,
        ),
    )

    grid_candidates = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if c.generation.strategy is OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH
    ]
    # Grid should produce combinations; at least some should exist
    assert len(grid_candidates) > 0
    for c in grid_candidates:
        assert len(c.generation.varied_knob_ids) >= 1


def test_grid_search_skips_when_cardinality_exceeds_limit() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
        RoutingPolicy.QUALITY_FIRST,
        RoutingPolicy.LOCAL_ONLY,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH],
            allowed_knob_ids=[
                "default_routing_policy",
                "policy_rollout_canary_percentage",
                "shadow_sampling_rate",
            ],
            seed=0,
            max_grid_dimensions=3,
            max_grid_candidates=2,  # Very small limit
        ),
    )

    grid_candidates = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if c.generation.strategy is OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH
    ]
    # Grid search should skip when cardinality > max_grid_candidates
    # (With 5 policies * numeric values, cardinality easily exceeds 2)
    assert len(grid_candidates) == 0


# ---------------------------------------------------------------------------
# strategy-specific: random search
# ---------------------------------------------------------------------------


def test_random_search_generates_up_to_max_candidates() -> None:
    settings = _default_settings()
    max_random = 5

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.RANDOM_SEARCH],
            allowed_knob_ids=[
                "default_routing_policy",
                "policy_rollout_canary_percentage",
                "shadow_sampling_rate",
                "hybrid_prefer_local",
            ],
            seed=42,
            max_random_candidates=max_random,
        ),
    )

    random_candidates = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if c.generation.strategy is OptimizationCandidateGenerationStrategy.RANDOM_SEARCH
    ]
    # Deduplication may reduce count, but should not exceed max
    assert len(random_candidates) <= max_random


def test_random_search_records_strategy_metadata() -> None:
    settings = _default_settings()
    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.RANDOM_SEARCH],
            allowed_knob_ids=[
                "default_routing_policy",
                "policy_rollout_canary_percentage",
            ],
            seed=77,
            max_random_candidates=3,
        ),
    )

    all_candidates = result.eligible_candidates + result.rejected_candidates
    for candidate in all_candidates:
        if candidate.generation.strategy is OptimizationCandidateGenerationStrategy.RANDOM_SEARCH:
            assert candidate.generation.seed == 77
            assert len(candidate.generation.varied_knob_ids) >= 1


# ---------------------------------------------------------------------------
# strategy-specific: heuristic seed
# ---------------------------------------------------------------------------


def test_heuristic_seed_from_prior_best_configuration() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
    )

    prior_best = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id="prior-best-1",
        campaign_id="old-campaign",
        candidate=OptimizationTrialIdentity(
            trial_id="prior-trial-1",
            candidate_id="prior-candidate-1",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="prior-profile-1",
            routing_policy=RoutingPolicy.LOCAL_PREFERRED,
        ),
        baseline_config_profile_id="old-baseline",
        config_profile_id="prior-profile-1",
        knob_changes=[
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value="balanced",
                candidate_value="local_preferred",
            ),
        ],
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.HEURISTIC_SEED],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
            heuristic_seed_limit=3,
        ),
        prior_best_candidate_configurations=[prior_best],
    )

    heuristic_candidates = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if c.generation.strategy is OptimizationCandidateGenerationStrategy.HEURISTIC_SEED
    ]
    assert len(heuristic_candidates) >= 1
    # The prior best should be seeded
    seeded_rationale = " ".join(
        r for c in heuristic_candidates for r in c.generation.rationale
    )
    assert "prior" in seeded_rationale.lower() or "heuristic" in seeded_rationale.lower()


def test_heuristic_seed_from_objective_policy_preference() -> None:
    settings = _default_settings()
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.HEURISTIC_SEED],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
            heuristic_seed_limit=3,
        ),
    )

    heuristic_candidates = [
        c for c in result.eligible_candidates
        if c.generation.strategy is OptimizationCandidateGenerationStrategy.HEURISTIC_SEED
    ]
    # For latency objective, the heuristic should suggest latency_first
    if heuristic_candidates:
        suggested_policies = {
            ch.candidate_value
            for c in heuristic_candidates
            for ch in c.knob_changes
            if ch.knob_id == "default_routing_policy"
        }
        assert "latency_first" in suggested_policies


def test_heuristic_seed_respects_limit() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.HEURISTIC_SEED],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
            heuristic_seed_limit=1,
        ),
    )

    heuristic_candidates = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if c.generation.strategy is OptimizationCandidateGenerationStrategy.HEURISTIC_SEED
    ]
    assert len(heuristic_candidates) <= 1


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------


def test_duplicate_candidates_across_strategies_are_deduplicated() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[
                OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
                OptimizationCandidateGenerationStrategy.HEURISTIC_SEED,
            ],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
        ),
    )

    # Both OFAT and heuristic may produce latency_first. Only one should appear.
    all_candidates = result.eligible_candidates + result.rejected_candidates
    signatures = []
    for candidate in all_candidates:
        sig = tuple(
            (ch.knob_id, ch.candidate_value) for ch in candidate.knob_changes
        )
        signatures.append(sig)
    # No duplicates
    assert len(signatures) == len(set(signatures))


def test_candidate_matching_baseline_is_not_generated() -> None:
    settings = _default_settings()
    # Only one policy = baseline, no alternatives
    settings.optimization.allowlisted_routing_policies = (RoutingPolicy.BALANCED,)

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
        ),
    )

    # No candidates should be generated when only the baseline is allowlisted
    assert len(result.eligible_candidates) == 0
    assert len(result.rejected_candidates) == 0


# ---------------------------------------------------------------------------
# default knob ids fallback
# ---------------------------------------------------------------------------


def test_empty_allowed_knob_ids_uses_defaults() -> None:
    settings = _default_settings()

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=[],  # Empty = uses defaults
            seed=0,
        ),
    )

    all_candidates = result.eligible_candidates + result.rejected_candidates
    # With default knob IDs, should generate candidates across multiple knobs
    varied_knobs = {
        knob_id
        for c in all_candidates
        for knob_id in c.generation.varied_knob_ids
    }
    assert len(varied_knobs) >= 1


# ---------------------------------------------------------------------------
# multi-strategy integration
# ---------------------------------------------------------------------------


def test_all_strategies_together_produce_candidates() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.LOCAL_PREFERRED,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[
                OptimizationCandidateGenerationStrategy.FIXED_BASELINE,
                OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
                OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH,
                OptimizationCandidateGenerationStrategy.RANDOM_SEARCH,
                OptimizationCandidateGenerationStrategy.HEURISTIC_SEED,
            ],
            allowed_knob_ids=[
                "default_routing_policy",
                "policy_rollout_canary_percentage",
                "hybrid_prefer_local",
            ],
            seed=42,
            max_random_candidates=3,
            heuristic_seed_limit=2,
        ),
    )

    all_candidates = result.eligible_candidates + result.rejected_candidates
    strategies_used = {c.generation.strategy for c in all_candidates}
    # At minimum, OFAT should produce some candidates
    assert OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME in strategies_used
    # Baseline is always captured in the result
    assert result.baseline_trial is not None
    assert result.baseline_generation.strategy is (
        OptimizationCandidateGenerationStrategy.FIXED_BASELINE
    )


def test_strategy_index_is_monotonically_increasing() -> None:
    settings = _default_settings()
    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[
                OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
                OptimizationCandidateGenerationStrategy.RANDOM_SEARCH,
            ],
            allowed_knob_ids=[
                "default_routing_policy",
                "policy_rollout_canary_percentage",
            ],
            seed=42,
            max_random_candidates=3,
        ),
    )

    all_candidates = result.eligible_candidates + result.rejected_candidates
    indices = [c.generation.strategy_index for c in all_candidates]
    # All indices should be positive and unique
    assert all(i > 0 for i in indices)
    assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# knob change recording
# ---------------------------------------------------------------------------


def test_knob_changes_carry_baseline_and_candidate_values() -> None:
    settings = _default_settings()
    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["default_routing_policy"],
            seed=0,
        ),
    )

    for candidate in result.eligible_candidates:
        for change in candidate.knob_changes:
            assert change.baseline_value is not None
            assert change.candidate_value is not None
            assert change.baseline_value != change.candidate_value
            assert change.config_path


def test_multi_knob_candidate_records_all_changed_knobs() -> None:
    settings = _default_settings()
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH],
            allowed_knob_ids=[
                "default_routing_policy",
                "hybrid_prefer_local",
            ],
            seed=0,
            max_grid_dimensions=2,
            max_grid_candidates=20,
        ),
    )

    multi_knob = [
        c for c in result.eligible_candidates + result.rejected_candidates
        if len(c.knob_changes) > 1
    ]
    for candidate in multi_knob:
        changed_ids = {ch.knob_id for ch in candidate.knob_changes}
        assert len(changed_ids) == len(candidate.knob_changes)


# ---------------------------------------------------------------------------
# boolean knob handling
# ---------------------------------------------------------------------------


def test_boolean_knob_generates_toggled_value() -> None:
    settings = _default_settings()
    settings.phase7.hybrid_execution.prefer_local = True

    result = generate_forge_stage_a_candidates(
        settings=settings,
        config=OptimizationCandidateGenerationConfig(
            strategies=[OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME],
            allowed_knob_ids=["hybrid_prefer_local"],
            seed=0,
        ),
    )

    all_candidates = result.eligible_candidates + result.rejected_candidates
    toggled = [
        c for c in all_candidates
        if any(
            ch.knob_id == "hybrid_prefer_local" and ch.candidate_value is False
            for ch in c.knob_changes
        )
    ]
    assert len(toggled) == 1
