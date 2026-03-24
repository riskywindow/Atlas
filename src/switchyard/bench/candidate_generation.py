"""Explicit Forge Stage A candidate-generation helpers."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import product

from switchyard.config import Settings
from switchyard.optimization import build_optimization_profile
from switchyard.schemas.benchmark import BenchmarkRunArtifact, CounterfactualObjective
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    OptimizationCandidateEligibilityRecord,
    OptimizationCandidateEligibilityStatus,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationMetadata,
    OptimizationCandidateGenerationResult,
    OptimizationComparisonOperator,
    OptimizationConstraint,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationGeneratedCandidate,
    OptimizationKnobChange,
    OptimizationKnobSurface,
    OptimizationKnobType,
    OptimizationProfile,
    OptimizationTrialIdentity,
)
from switchyard.schemas.optimization import (
    OptimizationCandidateGenerationStrategy as Strategy,
)
from switchyard.schemas.routing import PolicyRolloutMode, RoutingPolicy

KnobValue = bool | int | float | str | list[str] | None

_DEFAULT_GENERATION_KNOB_IDS = (
    "default_routing_policy",
    "policy_rollout_mode",
    "policy_rollout_canary_percentage",
    "shadow_sampling_rate",
    "hybrid_prefer_local",
    "hybrid_spillover_enabled",
    "hybrid_max_remote_share_percent",
)

_REMOTE_PREFERRING_POLICIES = {
    RoutingPolicy.BURST_TO_REMOTE,
    RoutingPolicy.LATENCY_SLO,
    RoutingPolicy.QUALITY_ON_DEMAND,
    RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY,
}


def generate_forge_stage_a_candidates(
    *,
    settings: Settings,
    profile: OptimizationProfile | None = None,
    config: OptimizationCandidateGenerationConfig | None = None,
    history_artifacts: Sequence[BenchmarkRunArtifact] | None = None,
    prior_best_candidate_configurations: Sequence[object] | None = None,
    timestamp: datetime | None = None,
) -> OptimizationCandidateGenerationResult:
    """Generate explicit, pruned Forge Stage A candidates from the typed knob surface."""

    resolved_profile = profile or build_optimization_profile(settings)
    if resolved_profile.baseline_trial is None:
        msg = "optimization profile must expose baseline_trial"
        raise ValueError(msg)
    generation_config = config or OptimizationCandidateGenerationConfig()
    run_timestamp = timestamp or datetime.now(UTC)
    knobs_by_id = _searchable_knobs(
        resolved_profile.knobs,
        allowed_knob_ids=generation_config.allowed_knob_ids,
    )
    baseline_values = {
        knob_id: knob.current_value for knob_id, knob in knobs_by_id.items()
    }

    eligible_candidates: list[OptimizationGeneratedCandidate] = []
    rejected_candidates: list[OptimizationGeneratedCandidate] = []
    seen_signatures: set[str] = set()
    rng = random.Random(generation_config.seed)
    strategy_index = 0

    if Strategy.ONE_FACTOR_AT_A_TIME in generation_config.strategies:
        for knob in knobs_by_id.values():
            for value in _sample_ofat_values(
                knob=knob,
                limit=generation_config.max_one_factor_variants_per_knob,
            ):
                strategy_index += 1
                _maybe_register_candidate(
                    baseline_trial=resolved_profile.baseline_trial,
                    profile=resolved_profile,
                    baseline_values=baseline_values,
                    knobs_by_id=knobs_by_id,
                    candidate_values={knob.knob_id: value},
                    generation=OptimizationCandidateGenerationMetadata(
                        strategy=Strategy.ONE_FACTOR_AT_A_TIME,
                        strategy_index=strategy_index,
                        seed=generation_config.seed,
                        varied_knob_ids=[knob.knob_id],
                        rationale=[
                            "single-knob ablation from the active baseline",
                            f"varied knob={knob.knob_id}",
                        ],
                    ),
                    seen_signatures=seen_signatures,
                    eligible_candidates=eligible_candidates,
                    rejected_candidates=rejected_candidates,
                    settings=settings,
                )

    if Strategy.BOUNDED_GRID_SEARCH in generation_config.strategies:
        grid_knobs = _grid_knobs(
            knobs_by_id=knobs_by_id,
            max_dimensions=generation_config.max_grid_dimensions,
        )
        grid_values = {
            knob.knob_id: _grid_values(knob)
            for knob in grid_knobs
        }
        grid_cardinality = 1
        for values in grid_values.values():
            grid_cardinality *= len(values)
        if 1 < grid_cardinality <= generation_config.max_grid_candidates:
            ordered_knob_ids = [knob.knob_id for knob in grid_knobs]
            for combo in product(*(grid_values[knob_id] for knob_id in ordered_knob_ids)):
                strategy_index += 1
                _maybe_register_candidate(
                    baseline_trial=resolved_profile.baseline_trial,
                    profile=resolved_profile,
                    baseline_values=baseline_values,
                    knobs_by_id=knobs_by_id,
                    candidate_values=dict(zip(ordered_knob_ids, combo, strict=True)),
                    generation=OptimizationCandidateGenerationMetadata(
                        strategy=Strategy.BOUNDED_GRID_SEARCH,
                        strategy_index=strategy_index,
                        seed=generation_config.seed,
                        varied_knob_ids=ordered_knob_ids,
                        rationale=[
                            "small Cartesian search over compact domains",
                            f"grid knobs={', '.join(ordered_knob_ids)}",
                        ],
                    ),
                    seen_signatures=seen_signatures,
                    eligible_candidates=eligible_candidates,
                    rejected_candidates=rejected_candidates,
                    settings=settings,
                )

    if (
        Strategy.RANDOM_SEARCH in generation_config.strategies
        and generation_config.max_random_candidates > 0
    ):
        random_knobs = _random_search_knobs(knobs_by_id)
        if random_knobs:
            for _ in range(generation_config.max_random_candidates):
                chosen_knobs = rng.sample(
                    random_knobs,
                    k=min(len(random_knobs), rng.randint(1, min(3, len(random_knobs)))),
                )
                sampled_values = {
                    knob.knob_id: _random_value(knob, rng=rng)
                    for knob in chosen_knobs
                }
                strategy_index += 1
                _maybe_register_candidate(
                    baseline_trial=resolved_profile.baseline_trial,
                    profile=resolved_profile,
                    baseline_values=baseline_values,
                    knobs_by_id=knobs_by_id,
                    candidate_values=sampled_values,
                    generation=OptimizationCandidateGenerationMetadata(
                        strategy=Strategy.RANDOM_SEARCH,
                        strategy_index=strategy_index,
                        seed=generation_config.seed,
                        varied_knob_ids=sorted(sampled_values),
                        rationale=[
                            "deterministic random search over bounded candidate knobs",
                        ],
                    ),
                    seen_signatures=seen_signatures,
                    eligible_candidates=eligible_candidates,
                    rejected_candidates=rejected_candidates,
                    settings=settings,
                )

    if (
        Strategy.HEURISTIC_SEED in generation_config.strategies
        and generation_config.heuristic_seed_limit > 0
    ):
        heuristic_states = _heuristic_seed_states(
            settings=settings,
            profile=resolved_profile,
            baseline_values=baseline_values,
            history_artifacts=list(history_artifacts or []),
            prior_best_candidate_configurations=list(
                prior_best_candidate_configurations or []
            ),
            limit=generation_config.heuristic_seed_limit,
        )
        for state, rationale in heuristic_states:
            strategy_index += 1
            _maybe_register_candidate(
                baseline_trial=resolved_profile.baseline_trial,
                profile=resolved_profile,
                baseline_values=baseline_values,
                knobs_by_id=knobs_by_id,
                candidate_values=state,
                generation=OptimizationCandidateGenerationMetadata(
                    strategy=Strategy.HEURISTIC_SEED,
                    strategy_index=strategy_index,
                    seed=generation_config.seed,
                    varied_knob_ids=sorted(state),
                    rationale=rationale,
                ),
                seen_signatures=seen_signatures,
                eligible_candidates=eligible_candidates,
                rejected_candidates=rejected_candidates,
                settings=settings,
            )

    notes = [
        "candidate generation is explicit, deterministic when seeded, and debuggable",
        "hard-constraint pruning is applied before offline execution when dimensions are known",
    ]
    if Strategy.FIXED_BASELINE in generation_config.strategies:
        notes.append("baseline configuration remains the fixed comparison point for all candidates")

    return OptimizationCandidateGenerationResult(
        profile_id=resolved_profile.profile_id,
        generated_at=run_timestamp,
        baseline_trial=resolved_profile.baseline_trial,
        baseline_generation=OptimizationCandidateGenerationMetadata(
            strategy=Strategy.FIXED_BASELINE,
            strategy_index=0,
            seed=generation_config.seed,
            rationale=["current active config profile captured as the fixed baseline"],
        ),
        generation_config=generation_config,
        eligible_candidates=eligible_candidates,
        rejected_candidates=rejected_candidates,
        notes=notes,
    )


def _searchable_knobs(
    knobs: Sequence[OptimizationKnobSurface],
    *,
    allowed_knob_ids: Sequence[str],
) -> dict[str, OptimizationKnobSurface]:
    allowlist = set(allowed_knob_ids or _DEFAULT_GENERATION_KNOB_IDS)
    return {
        knob.knob_id: knob
        for knob in knobs
        if knob.knob_id in allowlist
    }


def _sample_ofat_values(
    *,
    knob: OptimizationKnobSurface,
    limit: int,
) -> list[KnobValue]:
    sampled = _sampled_values(knob)
    return sampled[:limit]


def _grid_knobs(
    *,
    knobs_by_id: dict[str, OptimizationKnobSurface],
    max_dimensions: int,
) -> list[OptimizationKnobSurface]:
    candidates = [
        knob
        for knob in knobs_by_id.values()
        if 1 < len(_grid_values(knob)) <= 3
    ]
    return sorted(candidates, key=lambda knob: knob.knob_id)[:max_dimensions]


def _grid_values(knob: OptimizationKnobSurface) -> list[KnobValue]:
    sampled = [knob.current_value, *_sampled_values(knob)]
    return _unique_values(sampled)


def _random_search_knobs(
    knobs_by_id: dict[str, OptimizationKnobSurface],
) -> list[OptimizationKnobSurface]:
    return sorted(
        [
            knob
            for knob in knobs_by_id.values()
            if knob.knob_type in {
                OptimizationKnobType.ENUM,
                OptimizationKnobType.INTEGER,
                OptimizationKnobType.FLOAT,
                OptimizationKnobType.BOOLEAN,
            }
        ],
        key=lambda knob: knob.knob_id,
    )


def _random_value(
    knob: OptimizationKnobSurface,
    *,
    rng: random.Random,
) -> KnobValue:
    if knob.knob_type is OptimizationKnobType.BOOLEAN:
        return rng.choice([False, True])
    if knob.knob_type is OptimizationKnobType.ENUM:
        return rng.choice(list(knob.allowed_values))
    if knob.knob_type is OptimizationKnobType.INTEGER:
        if knob.min_value is None or knob.max_value is None:
            return knob.current_value
        return rng.randint(int(knob.min_value), int(knob.max_value))
    if knob.knob_type is OptimizationKnobType.FLOAT:
        if knob.min_value is None or knob.max_value is None:
            return knob.current_value
        return round(rng.uniform(float(knob.min_value), float(knob.max_value)), 3)
    return knob.current_value


def _sampled_values(knob: OptimizationKnobSurface) -> list[KnobValue]:
    current = knob.current_value
    if knob.knob_type is OptimizationKnobType.BOOLEAN:
        return [not bool(current)]
    if knob.knob_type is OptimizationKnobType.ENUM:
        return [value for value in knob.allowed_values if value != current]
    if knob.knob_type is OptimizationKnobType.INTEGER:
        if knob.min_value is None or knob.max_value is None:
            return []
        min_value = int(knob.min_value)
        max_value = int(knob.max_value)
        midpoint = min_value + ((max_value - min_value) // 2)
        return [
            value
            for value in _unique_values([min_value, midpoint, max_value])
            if value != current
        ]
    if knob.knob_type is OptimizationKnobType.FLOAT:
        if knob.min_value is None or knob.max_value is None:
            return []
        min_value_float = float(knob.min_value)
        max_value_float = float(knob.max_value)
        midpoint_float = round((min_value_float + max_value_float) / 2.0, 3)
        return [
            value
            for value in _unique_values(
                [min_value_float, midpoint_float, max_value_float]
            )
            if value != current
        ]
    return []


def _heuristic_seed_states(
    *,
    settings: Settings,
    profile: OptimizationProfile,
    baseline_values: dict[str, KnobValue],
    history_artifacts: Sequence[BenchmarkRunArtifact],
    prior_best_candidate_configurations: Sequence[object],
    limit: int,
) -> list[tuple[dict[str, KnobValue], list[str]]]:
    states: list[tuple[dict[str, KnobValue], list[str]]] = []
    for candidate in prior_best_candidate_configurations:
        knob_changes = getattr(candidate, "knob_changes", None)
        candidate_id = getattr(candidate, "candidate_configuration_id", "prior")
        if not isinstance(knob_changes, list):
            continue
        state: dict[str, KnobValue] = {}
        for change in knob_changes:
            knob_id = getattr(change, "knob_id", None)
            candidate_value = getattr(change, "candidate_value", None)
            if isinstance(knob_id, str) and knob_id in baseline_values:
                state[knob_id] = candidate_value
        if state:
            states.append(
                (
                    state,
                    [
                        f"seeded from prior candidate={candidate_id}",
                        "reuses previously known config deltas as an explicit starting point",
                    ],
                )
            )
        if len(states) >= limit:
            return states

    policy = _heuristic_policy_for_history(
        objective=settings.optimization.objective,
        allowlisted=profile.allowlisted_routing_policies,
        history_artifacts=history_artifacts,
    )
    if policy is not None and policy != profile.active_routing_policy:
        states.append(
            (
                {"default_routing_policy": policy.value},
                [
                    f"seeded from heuristic policy preference={policy.value}",
                    "heuristic seed uses the campaign objective and observed history posture",
                ],
            )
        )
    return states[:limit]


def _heuristic_policy_for_history(
    *,
    objective: CounterfactualObjective,
    allowlisted: Sequence[RoutingPolicy],
    history_artifacts: Sequence[BenchmarkRunArtifact],
) -> RoutingPolicy | None:
    allowlisted_set = set(allowlisted)
    if objective is CounterfactualObjective.LATENCY:
        if _history_shows_remote_capacity(history_artifacts):
            for policy in (
                RoutingPolicy.BURST_TO_REMOTE,
                RoutingPolicy.LATENCY_SLO,
                RoutingPolicy.LATENCY_FIRST,
            ):
                if policy in allowlisted_set:
                    return policy
        for policy in (RoutingPolicy.LATENCY_SLO, RoutingPolicy.LATENCY_FIRST):
            if policy in allowlisted_set:
                return policy
    if objective is CounterfactualObjective.RELIABILITY:
        for policy in (RoutingPolicy.QUALITY_ON_DEMAND, RoutingPolicy.QUALITY_FIRST):
            if policy in allowlisted_set:
                return policy
    if objective is CounterfactualObjective.THROUGHPUT:
        for policy in (RoutingPolicy.BURST_TO_REMOTE, RoutingPolicy.LATENCY_SLO):
            if policy in allowlisted_set:
                return policy
    for policy in (RoutingPolicy.LOCAL_PREFERRED, RoutingPolicy.BALANCED):
        if policy in allowlisted_set:
            return policy
    return None


def _history_shows_remote_capacity(
    artifacts: Sequence[BenchmarkRunArtifact],
) -> bool:
    for artifact in artifacts:
        hybrid_summary = artifact.summary.hybrid_summary
        if hybrid_summary is None:
            continue
        if hybrid_summary.remote_only_count > 0 or hybrid_summary.hybrid_spillover_count > 0:
            return True
    return False


def _maybe_register_candidate(
    *,
    baseline_trial: OptimizationTrialIdentity,
    profile: OptimizationProfile,
    baseline_values: dict[str, KnobValue],
    knobs_by_id: dict[str, OptimizationKnobSurface],
    candidate_values: dict[str, KnobValue],
    generation: OptimizationCandidateGenerationMetadata,
    seen_signatures: set[str],
    eligible_candidates: list[OptimizationGeneratedCandidate],
    rejected_candidates: list[OptimizationGeneratedCandidate],
    settings: Settings,
) -> None:
    resolved_values = dict(baseline_values)
    resolved_values.update(candidate_values)
    knob_changes = _knob_changes(
        baseline_values=baseline_values,
        candidate_values=resolved_values,
        knobs_by_id=knobs_by_id,
    )
    if not knob_changes:
        return
    signature = _candidate_signature(knob_changes)
    if signature in seen_signatures:
        return
    trial = _trial_identity_for_candidate(
        profile=profile,
        knob_changes=knob_changes,
        generation=generation,
    )
    eligibility = _assess_candidate(
        settings=settings,
        profile=profile,
        resolved_values=resolved_values,
        knob_changes=knob_changes,
    )
    candidate = OptimizationGeneratedCandidate(
        trial=trial,
        knob_changes=knob_changes,
        generation=generation,
        eligibility=eligibility,
        notes=[],
    )
    seen_signatures.add(signature)
    if eligibility.eligible:
        eligible_candidates.append(candidate)
    else:
        rejected_candidates.append(candidate)


def _knob_changes(
    *,
    baseline_values: dict[str, KnobValue],
    candidate_values: dict[str, KnobValue],
    knobs_by_id: dict[str, OptimizationKnobSurface],
) -> list[OptimizationKnobChange]:
    changes: list[OptimizationKnobChange] = []
    for knob_id, knob in sorted(knobs_by_id.items()):
        baseline_value = baseline_values[knob_id]
        candidate_value = candidate_values[knob_id]
        if baseline_value == candidate_value:
            continue
        changes.append(
            OptimizationKnobChange(
                knob_id=knob_id,
                config_path=knob.config_path,
                applies_to=list(knob.applies_to),
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                notes=[f"generated from knob surface group={knob.group.value}"],
            )
        )
    return changes


def _trial_identity_for_candidate(
    *,
    profile: OptimizationProfile,
    knob_changes: Sequence[OptimizationKnobChange],
    generation: OptimizationCandidateGenerationMetadata,
) -> OptimizationTrialIdentity:
    change_map = {change.knob_id: change for change in knob_changes}
    if set(change_map) == {"default_routing_policy"}:
        candidate_value = change_map["default_routing_policy"].candidate_value
        if not isinstance(candidate_value, str):
            msg = "default_routing_policy candidates require a string value"
            raise ValueError(msg)
        candidate_policy = RoutingPolicy(candidate_value)
        return OptimizationTrialIdentity(
            trial_id=_bounded_id(profile.profile_id, suffix=f"trial-{candidate_policy.value}"),
            candidate_id=f"routing-policy:{candidate_policy.value}",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id=_bounded_id(
                profile.profile_id,
                suffix=f"profile-{candidate_policy.value}",
            ),
            routing_policy=candidate_policy,
            notes=[
                f"generated_by={generation.strategy.value}",
                "single routing-policy change remains executable by the offline runner",
            ],
        )
    digest = _candidate_signature(knob_changes)[:12]
    return OptimizationTrialIdentity(
        trial_id=_bounded_id(profile.profile_id, suffix=f"generated-{digest}"),
        candidate_id=f"generated-config:{digest}",
        candidate_kind=ForgeCandidateKind.CONFIG_PROFILE,
        config_profile_id=_bounded_id(profile.profile_id, suffix=f"config-{digest}"),
        notes=[
            f"generated_by={generation.strategy.value}",
            "config-profile candidate requires explicit execution support to move beyond lineage",
        ],
    )


def _assess_candidate(
    *,
    settings: Settings,
    profile: OptimizationProfile,
    resolved_values: dict[str, KnobValue],
    knob_changes: Sequence[OptimizationKnobChange],
) -> OptimizationCandidateEligibilityRecord:
    domain_reasons = _domain_rejection_reasons(profile=profile, resolved_values=resolved_values)
    if domain_reasons:
        return OptimizationCandidateEligibilityRecord(
            status=OptimizationCandidateEligibilityStatus.REJECTED,
            rejection_reasons=domain_reasons,
            notes=["candidate violated the exported knob domain"],
        )
    incompatibility_reasons = _incompatibility_reasons(resolved_values, knob_changes)
    if incompatibility_reasons:
        return OptimizationCandidateEligibilityRecord(
            status=OptimizationCandidateEligibilityStatus.REJECTED,
            rejection_reasons=incompatibility_reasons,
            notes=["candidate was rejected before execution as impossible or unsafe"],
        )
    blocking_constraint_ids = _blocking_hard_constraint_ids(
        settings=settings,
        constraints=profile.constraints,
        resolved_values=resolved_values,
    )
    if blocking_constraint_ids:
        return OptimizationCandidateEligibilityRecord(
            status=OptimizationCandidateEligibilityStatus.PRUNED,
            blocking_constraint_ids=blocking_constraint_ids,
            rejection_reasons=["hard constraints prune this candidate before execution"],
            notes=["pre-execution pruning used only explicitly modeled hard constraints"],
        )
    return OptimizationCandidateEligibilityRecord(
        status=OptimizationCandidateEligibilityStatus.ELIGIBLE,
        notes=["candidate passed explicit domain, incompatibility, and hard-constraint checks"],
    )


def _domain_rejection_reasons(
    *,
    profile: OptimizationProfile,
    resolved_values: dict[str, KnobValue],
) -> list[str]:
    knob_map = {knob.knob_id: knob for knob in profile.knobs}
    reasons: list[str] = []
    for knob_id, value in resolved_values.items():
        knob = knob_map.get(knob_id)
        if knob is None:
            continue
        if value == knob.current_value:
            continue
        if knob.knob_type is OptimizationKnobType.BOOLEAN and not isinstance(value, bool):
            reasons.append(f"{knob_id} requires a boolean value")
        elif knob.knob_type is OptimizationKnobType.ENUM and value not in knob.allowed_values:
            reasons.append(f"{knob_id} value {value!r} is outside the allowed enum domain")
        elif knob.knob_type is OptimizationKnobType.INTEGER:
            if value is not None and not isinstance(value, int):
                reasons.append(f"{knob_id} requires an integer value")
            elif isinstance(value, int):
                if knob.min_value is not None and value < int(knob.min_value):
                    reasons.append(f"{knob_id} is below the allowed minimum")
                if knob.max_value is not None and value > int(knob.max_value):
                    reasons.append(f"{knob_id} is above the allowed maximum")
        elif knob.knob_type is OptimizationKnobType.FLOAT:
            if value is not None and not isinstance(value, (int, float)):
                reasons.append(f"{knob_id} requires a numeric value")
            elif isinstance(value, (int, float)):
                numeric = float(value)
                if knob.min_value is not None and numeric < float(knob.min_value):
                    reasons.append(f"{knob_id} is below the allowed minimum")
                if knob.max_value is not None and numeric > float(knob.max_value):
                    reasons.append(f"{knob_id} is above the allowed maximum")
        elif knob.knob_type is OptimizationKnobType.STRING_LIST and not (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            reasons.append(f"{knob_id} requires a list of strings")
    return reasons


def _incompatibility_reasons(
    resolved_values: dict[str, KnobValue],
    knob_changes: Sequence[OptimizationKnobChange],
) -> list[str]:
    reasons: list[str] = []
    changed_knob_ids = {change.knob_id for change in knob_changes}
    rollout_mode_raw = resolved_values.get("policy_rollout_mode")
    canary_raw = resolved_values.get("policy_rollout_canary_percentage")
    shadow_raw = resolved_values.get("shadow_sampling_rate")
    spillover_raw = resolved_values.get("hybrid_spillover_enabled")
    remote_share_raw = resolved_values.get("hybrid_max_remote_share_percent")
    routing_policy_raw = resolved_values.get("default_routing_policy")

    rollout_mode = (
        PolicyRolloutMode(rollout_mode_raw)
        if isinstance(rollout_mode_raw, str)
        else None
    )
    routing_policy = (
        RoutingPolicy(routing_policy_raw)
        if isinstance(routing_policy_raw, str)
        else None
    )
    canary_percentage = (
        float(canary_raw) if isinstance(canary_raw, (int, float)) else 0.0
    )
    shadow_sampling_rate = (
        float(shadow_raw) if isinstance(shadow_raw, (int, float)) else 0.0
    )
    spillover_enabled = bool(spillover_raw) if isinstance(spillover_raw, bool) else False
    remote_share = (
        float(remote_share_raw)
        if isinstance(remote_share_raw, (int, float))
        else 0.0
    )

    if rollout_mode is PolicyRolloutMode.CANARY and canary_percentage <= 0.0:
        reasons.append("canary rollout mode requires a positive canary percentage")
    if rollout_mode is PolicyRolloutMode.SHADOW_ONLY and shadow_sampling_rate <= 0.0:
        reasons.append("shadow-only rollout requires a positive shadow sampling rate")
    if not spillover_enabled and remote_share > 0.0:
        reasons.append("remote share cannot be positive when hybrid spillover is disabled")
    if (
        not spillover_enabled
        and routing_policy in _REMOTE_PREFERRING_POLICIES
    ):
        reasons.append(
            "remote-preferring routing policies are incompatible with disabled spillover"
        )
    if (
        not spillover_enabled
        and changed_knob_ids
        & {
            "hybrid_max_remote_share_percent",
            "hybrid_remote_request_budget_per_minute",
            "hybrid_remote_concurrency_cap",
        }
    ):
        reasons.append(
            "remote budget/share tuning knobs require hybrid spillover to stay enabled"
        )
    return reasons


def _blocking_hard_constraint_ids(
    *,
    settings: Settings,
    constraints: Sequence[OptimizationConstraint],
    resolved_values: dict[str, KnobValue],
) -> list[str]:
    blocking: list[str] = []
    for constraint in constraints:
        if constraint.strength is not OptimizationConstraintStrength.HARD:
            continue
        evaluated_value = _pre_execution_constraint_value(
            settings=settings,
            constraint=constraint,
            resolved_values=resolved_values,
        )
        if evaluated_value is None:
            continue
        if not _constraint_satisfied(
            operator=constraint.operator,
            evaluated_value=evaluated_value,
            threshold_value=constraint.threshold_value,
        ):
            blocking.append(constraint.constraint_id)
    return blocking


def _pre_execution_constraint_value(
    *,
    settings: Settings,
    constraint: OptimizationConstraint,
    resolved_values: dict[str, KnobValue],
) -> bool | int | float | str | None:
    if constraint.dimension is OptimizationConstraintDimension.CANARY_PERCENTAGE:
        value = resolved_values.get("policy_rollout_canary_percentage")
        return value if isinstance(value, (int, float)) else None
    if constraint.dimension is OptimizationConstraintDimension.SHADOW_SAMPLING_RATE:
        value = resolved_values.get("shadow_sampling_rate")
        return value if isinstance(value, (int, float)) else None
    if constraint.dimension is OptimizationConstraintDimension.REMOTE_SHARE_PERCENT:
        value = resolved_values.get("hybrid_max_remote_share_percent")
        return value if isinstance(value, (int, float)) else None
    if constraint.dimension is OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE:
        value = resolved_values.get(
            "hybrid_remote_request_budget_per_minute",
            settings.phase7.hybrid_execution.remote_request_budget_per_minute,
        )
        return value if isinstance(value, int) or value is None else None
    if constraint.dimension is OptimizationConstraintDimension.REMOTE_CONCURRENCY_CAP:
        value = resolved_values.get(
            "hybrid_remote_concurrency_cap",
            settings.phase7.hybrid_execution.remote_concurrency_cap,
        )
        return value if isinstance(value, int) or value is None else None
    if constraint.dimension is OptimizationConstraintDimension.OPERATOR_REVIEW_REQUIRED:
        return settings.optimization.promotion_requires_operator_review
    if constraint.dimension is OptimizationConstraintDimension.LOCAL_PREFERENCE_ENABLED:
        value = resolved_values.get("hybrid_prefer_local")
        return value if isinstance(value, bool) else None
    return None


def _constraint_satisfied(
    *,
    operator: OptimizationComparisonOperator,
    evaluated_value: bool | int | float | str,
    threshold_value: bool | int | float | str,
) -> bool:
    if (
        isinstance(evaluated_value, bool)
        and isinstance(threshold_value, bool)
        and operator is OptimizationComparisonOperator.EQ
    ):
        return evaluated_value == threshold_value
    if (
        isinstance(evaluated_value, (int, float))
        and not isinstance(evaluated_value, bool)
        and isinstance(threshold_value, (int, float))
        and not isinstance(threshold_value, bool)
    ):
        if operator is OptimizationComparisonOperator.LTE:
            return float(evaluated_value) <= float(threshold_value)
        if operator is OptimizationComparisonOperator.GTE:
            return float(evaluated_value) >= float(threshold_value)
        return float(evaluated_value) == float(threshold_value)
    if (
        isinstance(evaluated_value, str)
        and isinstance(threshold_value, str)
        and operator is OptimizationComparisonOperator.EQ
    ):
        return evaluated_value == threshold_value
    return True


def _candidate_signature(knob_changes: Sequence[OptimizationKnobChange]) -> str:
    parts = [
        f"{change.knob_id}={_normalized_value(change.candidate_value)}"
        for change in sorted(knob_changes, key=lambda change: change.knob_id)
    ]
    raw = "|".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return digest


def _normalized_value(value: KnobValue) -> str:
    if isinstance(value, list):
        return ",".join(value)
    return str(value)


def _unique_values(values: Sequence[KnobValue]) -> list[KnobValue]:
    seen: set[str] = set()
    ordered: list[KnobValue] = []
    for value in values:
        key = _normalized_value(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _bounded_id(prefix: str, *, suffix: str) -> str:
    candidate = f"{prefix}-{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    trimmed = candidate[: 117 - len(suffix)].rstrip("-")
    return f"{trimmed}-{suffix}-{digest}"
