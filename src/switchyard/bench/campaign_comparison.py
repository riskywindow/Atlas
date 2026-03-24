"""Explainable offline comparison helpers for Forge Stage A campaigns."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    CounterfactualSimulationRecord,
    RecommendationConfidence,
    SimulationSourceKind,
    WorkloadScenarioFamily,
)
from switchyard.schemas.optimization import (
    OptimizationArtifactEvidenceKind,
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationCandidateComparisonArtifact,
    OptimizationConstraintStrength,
    OptimizationEvidenceMixSummary,
    OptimizationGoal,
    OptimizationObjectiveDelta,
    OptimizationObjectiveMetric,
    OptimizationParetoSummary,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationTrialArtifact,
    OptimizationWorkloadImpactSummary,
    OptimizationWorkloadMetricDelta,
)

_NORMALIZED_TIE_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class _ProjectedOutcome:
    latency_ms: float | None
    error_rate: float | None
    tokens_per_second: float | None


@dataclass(frozen=True, slots=True)
class _CandidateContext:
    trial: OptimizationTrialArtifact
    objective_deltas: list[OptimizationObjectiveDelta]
    workload_impacts: list[OptimizationWorkloadImpactSummary]
    evidence_mix: OptimizationEvidenceMixSummary
    hard_constraint_violations: list[str]
    soft_constraint_violations: list[str]
    satisfied_constraint_ids: list[str]
    improved_objective_ids: list[str]
    regressed_objective_ids: list[str]
    benefited_workload_families: list[str]
    regressed_workload_families: list[str]
    normalized_tradeoff_score: float


@dataclass(frozen=True, slots=True)
class _RankEntry:
    rank: int
    tied_ids: list[str]


def compare_optimization_campaign(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    simulation_comparison: CounterfactualSimulationComparisonArtifact | None,
    evaluation_artifacts: Sequence[BenchmarkRunArtifact] = (),
    timestamp: datetime | None = None,
) -> OptimizationCampaignComparisonArtifact:
    """Compare executed campaign candidates against the baseline truthfully."""

    if not campaign_artifact.trials:
        return OptimizationCampaignComparisonArtifact(
            comparison_artifact_id=_bounded_id(
                campaign_artifact.campaign.campaign_id,
                suffix="comparison",
            ),
            timestamp=timestamp or datetime.now(UTC),
            campaign_id=campaign_artifact.campaign.campaign_id,
            baseline_candidate_configuration_id=(
                campaign_artifact.baseline_candidate_configuration.candidate_configuration_id
            ),
            notes=["campaign did not contain any executed candidate trials"],
        )

    evaluation_by_policy_id = _evaluation_by_policy_id(simulation_comparison)
    baseline_policy = campaign_artifact.baseline_candidate_configuration.candidate.routing_policy
    baseline_evaluation = None
    if baseline_policy is not None:
        baseline_evaluation = evaluation_by_policy_id.get(baseline_policy.value)
    if baseline_evaluation is None and simulation_comparison is not None:
        baseline_evaluation = simulation_comparison.evaluations[0]
    benchmark_record_index = _benchmark_record_index(evaluation_artifacts)
    contexts = [
        _candidate_context(
            trial=trial,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=_candidate_evaluation(
                trial=trial,
                evaluation_by_policy_id=evaluation_by_policy_id,
            ),
            benchmark_record_index=benchmark_record_index,
        )
        for trial in campaign_artifact.trials
    ]
    domination_map = _domination_map(contexts)
    rank_map = _rank_map(contexts, domination_map=domination_map)

    candidate_comparisons: list[OptimizationCandidateComparisonArtifact] = []
    for context in contexts:
        candidate_id = context.trial.candidate_configuration.candidate_configuration_id
        dominated_by = domination_map[candidate_id]
        pareto_optimal = not dominated_by and not context.hard_constraint_violations
        rank_entry = rank_map[candidate_id]
        tied_ids = rank_entry.tied_ids
        rank = rank_entry.rank
        recommendation_summary = _recommendation_summary(
            campaign_artifact=campaign_artifact,
            context=context,
            dominated_by=dominated_by,
            pareto_optimal=pareto_optimal,
            tied_candidate_ids=tied_ids,
        )
        candidate_comparisons.append(
            OptimizationCandidateComparisonArtifact(
                candidate_configuration_id=candidate_id,
                trial_artifact_id=context.trial.trial_artifact_id,
                config_profile_id=context.trial.candidate_configuration.config_profile_id,
                rank=rank,
                tied_candidate_configuration_ids=tied_ids,
                objective_deltas=context.objective_deltas,
                normalized_tradeoff_score=round(
                    context.normalized_tradeoff_score,
                    6,
                ),
                hard_constraint_violations=context.hard_constraint_violations,
                soft_constraint_violations=context.soft_constraint_violations,
                pareto_optimal=pareto_optimal,
                dominated=bool(dominated_by),
                dominated_by_candidate_configuration_ids=dominated_by,
                workload_impacts=context.workload_impacts,
                recommendation_summary=recommendation_summary,
                notes=_comparison_notes(
                    context=context,
                    pareto_optimal=pareto_optimal,
                    dominated_by=dominated_by,
                ),
            )
        )

    candidate_comparisons.sort(
        key=lambda comparison: (
            comparison.rank,
            comparison.candidate_configuration_id,
        )
    )
    frontier_ids = [
        comparison.candidate_configuration_id
        for comparison in candidate_comparisons
        if comparison.pareto_optimal
    ]
    dominated_ids = [
        comparison.candidate_configuration_id
        for comparison in candidate_comparisons
        if comparison.dominated
    ]
    notes = [
        "comparison ranks candidates conservatively and keeps multi-objective tradeoffs explicit",
        "Pareto posture only considers hard-feasible candidates with measurable objectives",
    ]
    if baseline_evaluation is None:
        notes.append(
            "baseline simulation evaluation was unavailable; metric deltas may be "
            "incomplete"
        )
    return OptimizationCampaignComparisonArtifact(
        comparison_artifact_id=_bounded_id(
            campaign_artifact.campaign.campaign_id,
            suffix="comparison",
        ),
        timestamp=timestamp or datetime.now(UTC),
        campaign_id=campaign_artifact.campaign.campaign_id,
        baseline_candidate_configuration_id=(
            campaign_artifact.baseline_candidate_configuration.candidate_configuration_id
        ),
        candidate_comparisons=candidate_comparisons,
        pareto_summary=OptimizationParetoSummary(
            frontier_candidate_configuration_ids=frontier_ids,
            dominated_candidate_configuration_ids=dominated_ids,
            notes=[
                "frontier candidates are not globally ranked as a single winner "
                "when tradeoffs remain"
            ],
        ),
        notes=notes,
    )


def _candidate_context(
    *,
    trial: OptimizationTrialArtifact,
    baseline_evaluation: CounterfactualSimulationArtifact | None,
    candidate_evaluation: CounterfactualSimulationArtifact | None,
    benchmark_record_index: Mapping[tuple[str, str], BenchmarkRequestRecord],
) -> _CandidateContext:
    objective_deltas = _objective_deltas(
        trial=trial,
        baseline_evaluation=baseline_evaluation,
        candidate_evaluation=candidate_evaluation,
    )
    hard_constraint_violations = [
        assessment.constraint_id
        for assessment in trial.constraint_assessments
        if assessment.strength is OptimizationConstraintStrength.HARD
        and assessment.satisfied is False
    ]
    soft_constraint_violations = [
        assessment.constraint_id
        for assessment in trial.constraint_assessments
        if assessment.strength is OptimizationConstraintStrength.SOFT
        and assessment.satisfied is False
    ]
    satisfied_constraint_ids = [
        assessment.constraint_id
        for assessment in trial.constraint_assessments
        if assessment.satisfied is True
    ]
    workload_impacts = _workload_impacts(
        objective_metrics=[
            objective.metric
            for objective in trial.candidate_configuration.objectives_in_scope
        ],
        baseline_evaluation=baseline_evaluation,
        candidate_evaluation=candidate_evaluation,
        benchmark_record_index=benchmark_record_index,
    )
    evidence_mix = _evidence_mix(
        trial=trial,
        candidate_evaluation=candidate_evaluation,
    )
    improved_objective_ids = [
        delta.objective_id for delta in objective_deltas if delta.improved is True
    ]
    regressed_objective_ids = [
        delta.objective_id for delta in objective_deltas if delta.improved is False
    ]
    benefited_workload_families = [
        impact.workload_family for impact in workload_impacts if impact.improved_metrics
    ]
    regressed_workload_families = [
        impact.workload_family for impact in workload_impacts if impact.regressed_metrics
    ]
    return _CandidateContext(
        trial=trial,
        objective_deltas=objective_deltas,
        workload_impacts=workload_impacts,
        evidence_mix=evidence_mix,
        hard_constraint_violations=hard_constraint_violations,
        soft_constraint_violations=soft_constraint_violations,
        satisfied_constraint_ids=satisfied_constraint_ids,
        improved_objective_ids=improved_objective_ids,
        regressed_objective_ids=regressed_objective_ids,
        benefited_workload_families=benefited_workload_families,
        regressed_workload_families=regressed_workload_families,
        normalized_tradeoff_score=_normalized_tradeoff_score(
            trial=trial,
            objective_deltas=objective_deltas,
        ),
    )


def _objective_deltas(
    *,
    trial: OptimizationTrialArtifact,
    baseline_evaluation: CounterfactualSimulationArtifact | None,
    candidate_evaluation: CounterfactualSimulationArtifact | None,
) -> list[OptimizationObjectiveDelta]:
    deltas: list[OptimizationObjectiveDelta] = []
    assessment_by_id = {
        assessment.objective_id: assessment for assessment in trial.objective_assessments
    }
    for objective in trial.candidate_configuration.objectives_in_scope:
        assessment = assessment_by_id.get(objective.objective_id)
        candidate_value = (
            None if assessment is None else assessment.measured_value
        )
        if candidate_value is None:
            candidate_value = _metric_value(
                metric=objective.metric,
                evaluation=candidate_evaluation,
                evidence_records=trial.evidence_records,
            )
        baseline_value = _metric_value(
            metric=objective.metric,
            evaluation=baseline_evaluation,
            evidence_records=trial.evidence_records,
        )
        absolute_delta = (
            None
            if baseline_value is None or candidate_value is None
            else candidate_value - baseline_value
        )
        relative_delta = _relative_delta(
            baseline_value=baseline_value,
            absolute_delta=absolute_delta,
        )
        normalized_tradeoff = _normalized_tradeoff(
            goal=objective.goal,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
        )
        notes: list[str] = []
        if baseline_value is None:
            notes.append("baseline value was unavailable for this objective")
        if candidate_value is None:
            notes.append("candidate value was unavailable for this objective")
        deltas.append(
            OptimizationObjectiveDelta(
                objective_id=objective.objective_id,
                metric=objective.metric,
                goal=objective.goal,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                absolute_delta=absolute_delta,
                relative_delta=relative_delta,
                normalized_tradeoff=normalized_tradeoff,
                improved=(
                    None
                    if normalized_tradeoff is None
                    else normalized_tradeoff > _NORMALIZED_TIE_TOLERANCE
                ),
                notes=notes,
            )
        )
    return deltas


def _workload_impacts(
    *,
    objective_metrics: Sequence[OptimizationObjectiveMetric],
    baseline_evaluation: CounterfactualSimulationArtifact | None,
    candidate_evaluation: CounterfactualSimulationArtifact | None,
    benchmark_record_index: Mapping[tuple[str, str], BenchmarkRequestRecord],
) -> list[OptimizationWorkloadImpactSummary]:
    if baseline_evaluation is None or candidate_evaluation is None:
        return []
    baseline_records = {
        _record_key(record): record for record in baseline_evaluation.records
    }
    candidate_records = {
        _record_key(record): record for record in candidate_evaluation.records
    }
    accumulators: dict[str, dict[OptimizationObjectiveMetric, dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: {"baseline": [], "candidate": []}))
    )
    request_counts: dict[str, int] = defaultdict(int)

    for key, baseline_record in baseline_records.items():
        candidate_record = candidate_records.get(key)
        if candidate_record is None:
            continue
        workload_family = _workload_family(
            record=baseline_record,
            benchmark_record_index=benchmark_record_index,
        )
        request_counts[workload_family] += 1
        baseline_outcome = _selected_outcome(
            record=baseline_record,
            benchmark_record_index=benchmark_record_index,
        )
        candidate_outcome = _selected_outcome(
            record=candidate_record,
            benchmark_record_index=benchmark_record_index,
        )
        for metric in objective_metrics:
            baseline_value = _outcome_metric(baseline_outcome, metric)
            candidate_value = _outcome_metric(candidate_outcome, metric)
            if baseline_value is None or candidate_value is None:
                continue
            accumulators[workload_family][metric]["baseline"].append(baseline_value)
            accumulators[workload_family][metric]["candidate"].append(candidate_value)

    workload_impacts: list[OptimizationWorkloadImpactSummary] = []
    for workload_family, metrics in sorted(accumulators.items()):
        metric_deltas: list[OptimizationWorkloadMetricDelta] = []
        improved_metrics: list[OptimizationObjectiveMetric] = []
        regressed_metrics: list[OptimizationObjectiveMetric] = []
        notes: list[str] = []
        for metric, values in sorted(metrics.items(), key=lambda item: item[0].value):
            baseline_value = _average(values["baseline"])
            candidate_value = _average(values["candidate"])
            absolute_delta = (
                None
                if baseline_value is None or candidate_value is None
                else candidate_value - baseline_value
            )
            improved = _metric_improved(
                metric=metric,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
            )
            metric_deltas.append(
                OptimizationWorkloadMetricDelta(
                    metric=metric,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    absolute_delta=absolute_delta,
                    improved=improved,
                )
            )
            if improved is True:
                improved_metrics.append(metric)
            elif improved is False:
                regressed_metrics.append(metric)
        if improved_metrics and regressed_metrics:
            notes.append("workload family shows mixed multi-objective tradeoffs")
        workload_impacts.append(
            OptimizationWorkloadImpactSummary(
                workload_family=workload_family,
                request_count=request_counts[workload_family],
                metric_deltas=metric_deltas,
                improved_metrics=improved_metrics,
                regressed_metrics=regressed_metrics,
                notes=notes,
            )
        )
    return workload_impacts


def _evidence_mix(
    *,
    trial: OptimizationTrialArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact | None,
) -> OptimizationEvidenceMixSummary:
    total_request_count = (
        0 if candidate_evaluation is None else candidate_evaluation.summary.request_count
    )
    direct_observation_count = (
        0
        if candidate_evaluation is None
        else candidate_evaluation.summary.direct_observation_count
    )
    estimated_request_count = (
        0
        if candidate_evaluation is None
        else (
            candidate_evaluation.summary.predictor_estimate_count
            + candidate_evaluation.summary.low_confidence_count
        )
    )
    unsupported_request_count = (
        0 if candidate_evaluation is None else candidate_evaluation.summary.unsupported_count
    )
    replay_backed_request_count = total_request_count
    simulated_request_count = total_request_count
    notes = [
        "replay-backed and simulated counts reflect offline evaluation coverage, "
        "not live rollout traffic"
    ]
    if any(
        record.evidence_kind is OptimizationArtifactEvidenceKind.ESTIMATED
        for record in trial.evidence_records
    ):
        notes.append("estimated evidence remained explicit in the comparison output")
    return OptimizationEvidenceMixSummary(
        total_request_count=total_request_count,
        replay_backed_request_count=replay_backed_request_count,
        simulated_request_count=simulated_request_count,
        direct_observation_count=direct_observation_count,
        estimated_request_count=estimated_request_count,
        unsupported_request_count=unsupported_request_count,
        observed_share=_share(direct_observation_count, total_request_count),
        replayed_share=_share(replay_backed_request_count, total_request_count),
        simulated_share=_share(simulated_request_count, total_request_count),
        estimated_share=_share(estimated_request_count, total_request_count),
        notes=notes,
    )


def _domination_map(
    contexts: Sequence[_CandidateContext],
) -> dict[str, list[str]]:
    feasible_contexts = [
        context for context in contexts if not context.hard_constraint_violations
    ]
    domination_map: dict[str, list[str]] = {
        context.trial.candidate_configuration.candidate_configuration_id: []
        for context in contexts
    }
    for candidate in feasible_contexts:
        for peer in feasible_contexts:
            if candidate is peer:
                continue
            if _dominates(candidate, peer):
                domination_map[peer.trial.candidate_configuration.candidate_configuration_id].append(
                    candidate.trial.candidate_configuration.candidate_configuration_id
                )
    for _candidate_id, dominated_by in domination_map.items():
        dominated_by.sort()
    return domination_map


def _rank_map(
    contexts: Sequence[_CandidateContext],
    *,
    domination_map: Mapping[str, Sequence[str]],
) -> dict[str, _RankEntry]:
    def sort_key(context: _CandidateContext) -> tuple[float, ...]:
        candidate_id = context.trial.candidate_configuration.candidate_configuration_id
        label_score = _label_score(
            context=context,
            dominated=bool(domination_map[candidate_id]),
        )
        pareto_score = (
            1.0
            if not domination_map[candidate_id] and not context.hard_constraint_violations
            else 0.0
        )
        observed_share = context.evidence_mix.observed_share or 0.0
        estimated_share = context.evidence_mix.estimated_share or 0.0
        return (
            -label_score,
            -pareto_score,
            -round(context.normalized_tradeoff_score, 6),
            -float(len(context.improved_objective_ids)),
            float(len(context.regressed_objective_ids)),
            float(len(context.soft_constraint_violations)),
            -round(observed_share, 6),
            round(estimated_share, 6),
        )

    ranked_contexts = sorted(
        contexts,
        key=lambda context: (
            *sort_key(context),
            context.trial.candidate_configuration.candidate_configuration_id,
        ),
    )
    result: dict[str, _RankEntry] = {}
    previous_signature: tuple[float, ...] | None = None
    current_rank = 0
    for position, context in enumerate(ranked_contexts, start=1):
        signature = sort_key(context)
        if previous_signature is None or signature != previous_signature:
            current_rank = position
            previous_signature = signature
        candidate_id = context.trial.candidate_configuration.candidate_configuration_id
        tied_ids = [
            peer.trial.candidate_configuration.candidate_configuration_id
            for peer in ranked_contexts
            if peer is not context and sort_key(peer) == signature
        ]
        result[candidate_id] = _RankEntry(rank=current_rank, tied_ids=tied_ids)
    return result


def _recommendation_summary(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    context: _CandidateContext,
    dominated_by: Sequence[str],
    pareto_optimal: bool,
    tied_candidate_ids: Sequence[str],
) -> OptimizationRecommendationSummary:
    prior_summary = context.trial.recommendation_summary
    confidence = (
        RecommendationConfidence.INSUFFICIENT
        if prior_summary is None
        else prior_summary.confidence
    )
    evidence_kinds = sorted(
        {
            record.evidence_kind for record in context.trial.evidence_records
        },
        key=lambda item: item.value,
    )
    primary_objective_id = (
        context.trial.candidate_configuration.objectives_in_scope[0].objective_id
        if context.trial.candidate_configuration.objectives_in_scope
        else None
    )
    primary_improved = primary_objective_id in context.improved_objective_ids
    primary_regressed = primary_objective_id in context.regressed_objective_ids
    reason_codes: list[OptimizationRecommendationReasonCode] = []
    rationale: list[str] = []
    if primary_improved:
        reason_codes.append(
            OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED
        )
        rationale.append("primary objective improved versus the baseline")
    if primary_regressed:
        reason_codes.append(
            OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_REGRESSED
        )
        rationale.append("primary objective regressed versus the baseline")
    if any(
        objective_id != primary_objective_id
        for objective_id in context.improved_objective_ids
    ):
        reason_codes.append(
            OptimizationRecommendationReasonCode.SECONDARY_OBJECTIVE_IMPROVED
        )
    if any(
        objective_id != primary_objective_id
        for objective_id in context.regressed_objective_ids
    ):
        reason_codes.append(
            OptimizationRecommendationReasonCode.SECONDARY_OBJECTIVE_REGRESSED
        )
    if context.hard_constraint_violations:
        reason_codes.append(
            OptimizationRecommendationReasonCode.HARD_CONSTRAINT_VIOLATED
        )
        rationale.append(
            "hard constraints were violated under offline comparison"
        )
    if context.soft_constraint_violations:
        reason_codes.append(
            OptimizationRecommendationReasonCode.SOFT_CONSTRAINT_VIOLATED
        )
        rationale.append(
            "one or more advisory constraints were violated and require review"
        )
    if pareto_optimal:
        reason_codes.append(OptimizationRecommendationReasonCode.NON_DOMINATED)
        rationale.append("candidate remains on the Pareto frontier")
    if dominated_by:
        reason_codes.append(OptimizationRecommendationReasonCode.DOMINATED)
        rationale.append(
            "another feasible candidate matched or beat this candidate across comparable objectives"
        )
    if context.benefited_workload_families:
        reason_codes.append(
            OptimizationRecommendationReasonCode.WORKLOAD_FAMILY_BENEFIT
        )
    if context.regressed_workload_families:
        reason_codes.append(
            OptimizationRecommendationReasonCode.WORKLOAD_FAMILY_REGRESSION
        )
    if context.benefited_workload_families and context.regressed_workload_families:
        reason_codes.append(
            OptimizationRecommendationReasonCode.MIXED_WORKLOAD_TRADEOFF
        )
        rationale.append("workload-family impacts were mixed rather than uniformly better")
    if (context.evidence_mix.observed_share or 0.0) > 0.0:
        reason_codes.append(
            OptimizationRecommendationReasonCode.OBSERVED_EVIDENCE_PRESENT
        )
    else:
        reason_codes.append(
            OptimizationRecommendationReasonCode.OBSERVED_EVIDENCE_MISSING
        )
        rationale.append("offline result lacked direct observed outcomes for the recommended path")
    if (context.evidence_mix.estimated_share or 0.0) > 0.0:
        reason_codes.append(
            OptimizationRecommendationReasonCode.ESTIMATED_EVIDENCE_PRESENT
        )
    if tied_candidate_ids:
        reason_codes.append(OptimizationRecommendationReasonCode.TIED_WITH_PEER)
        rationale.append("ranking tied with another candidate after normalization")
    if not context.improved_objective_ids and not context.regressed_objective_ids:
        reason_codes.append(
            OptimizationRecommendationReasonCode.NO_MEANINGFUL_DELTA
        )
        rationale.append("measured objectives did not show a meaningful delta")

    label = OptimizationRecommendationLabel.REVIEW_ONLY
    disposition = OptimizationRecommendationDisposition.NO_CHANGE
    if context.hard_constraint_violations or dominated_by:
        label = OptimizationRecommendationLabel.REJECTED
        disposition = OptimizationRecommendationDisposition.KEEP_BASELINE
    elif not context.improved_objective_ids and not context.benefited_workload_families:
        label = OptimizationRecommendationLabel.REVIEW_ONLY
        disposition = OptimizationRecommendationDisposition.NO_CHANGE
    elif (
        pareto_optimal
        and not context.regressed_objective_ids
        and not context.soft_constraint_violations
        and not context.regressed_workload_families
        and (context.evidence_mix.observed_share or 0.0) > 0.0
    ):
        label = OptimizationRecommendationLabel.PROMOTION_ELIGIBLE
        disposition = OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    elif (context.evidence_mix.observed_share or 0.0) == 0.0:
        label = OptimizationRecommendationLabel.REVIEW_ONLY
        disposition = OptimizationRecommendationDisposition.NEED_MORE_EVIDENCE

    if campaign_artifact.campaign.promotion_requires_operator_review:
        reason_codes.append(
            OptimizationRecommendationReasonCode.PROMOTION_REQUIRES_REVIEW
        )
        rationale.append("promotion remains bounded by explicit operator review")

    return OptimizationRecommendationSummary(
        recommendation_summary_id=_bounded_id(
            context.trial.trial_identity.trial_id,
            suffix="recommendation",
        ),
        disposition=disposition,
        recommendation_label=label,
        confidence=confidence,
        candidate_configuration_id=(
            context.trial.candidate_configuration.candidate_configuration_id
        ),
        config_profile_id=context.trial.candidate_configuration.config_profile_id,
        evidence_kinds=evidence_kinds,
        reason_codes=reason_codes,
        improved_objective_ids=context.improved_objective_ids,
        regressed_objective_ids=context.regressed_objective_ids,
        satisfied_constraint_ids=context.satisfied_constraint_ids,
        violated_constraint_ids=[
            *context.hard_constraint_violations,
            *context.soft_constraint_violations,
        ],
        benefited_workload_families=context.benefited_workload_families,
        regressed_workload_families=context.regressed_workload_families,
        evidence_mix=context.evidence_mix,
        rationale=rationale,
    )


def _comparison_notes(
    *,
    context: _CandidateContext,
    pareto_optimal: bool,
    dominated_by: Sequence[str],
) -> list[str]:
    notes = [
        f"normalized_tradeoff_score={context.normalized_tradeoff_score:.6f}"
    ]
    if pareto_optimal:
        notes.append("candidate is not dominated by another hard-feasible candidate")
    if dominated_by:
        notes.append(
            "dominated_by="
            + ",".join(dominated_by)
        )
    return notes


def _normalized_tradeoff_score(
    *,
    trial: OptimizationTrialArtifact,
    objective_deltas: Sequence[OptimizationObjectiveDelta],
) -> float:
    weight_by_objective_id = {
        objective.objective_id: objective.weight
        for objective in trial.candidate_configuration.objectives_in_scope
    }
    return sum(
        (delta.normalized_tradeoff or 0.0) * weight_by_objective_id.get(delta.objective_id, 1.0)
        for delta in objective_deltas
    )


def _evaluation_by_policy_id(
    comparison: CounterfactualSimulationComparisonArtifact | None,
) -> dict[str, CounterfactualSimulationArtifact]:
    if comparison is None:
        return {}
    return {
        evaluation.policy.policy_id: evaluation for evaluation in comparison.evaluations
    }


def _candidate_evaluation(
    *,
    trial: OptimizationTrialArtifact,
    evaluation_by_policy_id: Mapping[str, CounterfactualSimulationArtifact],
) -> CounterfactualSimulationArtifact | None:
    policy = trial.trial_identity.routing_policy
    if policy is None:
        return None
    return evaluation_by_policy_id.get(policy.value)


def _benchmark_record_index(
    artifacts: Sequence[BenchmarkRunArtifact],
) -> dict[tuple[str, str], BenchmarkRequestRecord]:
    return {
        (artifact.run_id, record.request_id): record
        for artifact in artifacts
        for record in artifact.records
    }


def _record_key(record: CounterfactualSimulationRecord) -> tuple[str, str, str]:
    return (
        record.source_kind.value,
        record.source_run_id or "",
        record.source_record_id or "",
    )


def _workload_family(
    *,
    record: CounterfactualSimulationRecord,
    benchmark_record_index: Mapping[tuple[str, str], BenchmarkRequestRecord],
) -> str:
    if (
        record.source_kind is SimulationSourceKind.BENCHMARK_RUN
        and record.source_run_id is not None
        and record.source_record_id is not None
    ):
        source_record = benchmark_record_index.get(
            (record.source_run_id, record.source_record_id)
        )
        if source_record is not None and source_record.scenario_family is not None:
            return source_record.scenario_family.value
    if record.source_kind is SimulationSourceKind.CAPTURED_TRACE:
        return "trace_replay"
    return WorkloadScenarioFamily.MIXED.value


def _selected_outcome(
    *,
    record: CounterfactualSimulationRecord,
    benchmark_record_index: Mapping[tuple[str, str], BenchmarkRequestRecord],
) -> _ProjectedOutcome:
    selected_candidate = next(
        candidate
        for candidate in record.candidate_scores
        if candidate.backend_name == record.recommendation.recommended_backend
    )
    benchmark_record = (
        None
        if record.source_kind is not SimulationSourceKind.BENCHMARK_RUN
        or record.source_run_id is None
        or record.source_record_id is None
        else benchmark_record_index.get((record.source_run_id, record.source_record_id))
    )
    latency_ms = selected_candidate.observed_latency_ms
    error_rate = (
        None
        if selected_candidate.observed_success is None
        else 0.0 if selected_candidate.observed_success else 1.0
    )
    tokens_per_second = (
        None if benchmark_record is None else benchmark_record.tokens_per_second
    )
    if selected_candidate.estimate is not None:
        latency_ms = (
            latency_ms
            if latency_ms is not None
            else selected_candidate.estimate.expected_latency_ms
        )
        error_rate = (
            error_rate
            if error_rate is not None
            else selected_candidate.estimate.expected_error_rate
        )
        tokens_per_second = (
            tokens_per_second
            if tokens_per_second is not None
            else selected_candidate.estimate.expected_tokens_per_second
        )
    return _ProjectedOutcome(
        latency_ms=latency_ms,
        error_rate=error_rate,
        tokens_per_second=tokens_per_second,
    )


def _outcome_metric(
    outcome: _ProjectedOutcome,
    metric: OptimizationObjectiveMetric,
) -> float | None:
    if metric is OptimizationObjectiveMetric.LATENCY_MS:
        return outcome.latency_ms
    if metric is OptimizationObjectiveMetric.ERROR_RATE:
        return outcome.error_rate
    if metric is OptimizationObjectiveMetric.TOKENS_PER_SECOND:
        return outcome.tokens_per_second
    return None


def _metric_value(
    *,
    metric: OptimizationObjectiveMetric,
    evaluation: CounterfactualSimulationArtifact | None,
    evidence_records: Sequence[object],
) -> float | None:
    del evidence_records
    if evaluation is None:
        return None
    if metric is OptimizationObjectiveMetric.LATENCY_MS:
        return evaluation.summary.projected_avg_latency_ms
    if metric is OptimizationObjectiveMetric.ERROR_RATE:
        return evaluation.summary.projected_error_rate
    if metric is OptimizationObjectiveMetric.TOKENS_PER_SECOND:
        return evaluation.summary.projected_avg_tokens_per_second
    return None


def _relative_delta(
    *,
    baseline_value: float | None,
    absolute_delta: float | None,
) -> float | None:
    if (
        baseline_value is None
        or absolute_delta is None
        or abs(baseline_value) <= _NORMALIZED_TIE_TOLERANCE
    ):
        return None
    return absolute_delta / abs(baseline_value)


def _normalized_tradeoff(
    *,
    goal: OptimizationGoal,
    baseline_value: float | None,
    candidate_value: float | None,
) -> float | None:
    if baseline_value is None or candidate_value is None:
        return None
    denominator = max(abs(baseline_value), 1.0)
    if goal in {OptimizationGoal.MINIMIZE, OptimizationGoal.AT_MOST}:
        return (baseline_value - candidate_value) / denominator
    return (candidate_value - baseline_value) / denominator


def _metric_improved(
    *,
    metric: OptimizationObjectiveMetric,
    baseline_value: float | None,
    candidate_value: float | None,
) -> bool | None:
    if baseline_value is None or candidate_value is None:
        return None
    if metric in {
        OptimizationObjectiveMetric.LATENCY_MS,
        OptimizationObjectiveMetric.ERROR_RATE,
    }:
        if abs(candidate_value - baseline_value) <= _NORMALIZED_TIE_TOLERANCE:
            return None
        return candidate_value < baseline_value
    if metric is OptimizationObjectiveMetric.TOKENS_PER_SECOND:
        if abs(candidate_value - baseline_value) <= _NORMALIZED_TIE_TOLERANCE:
            return None
        return candidate_value > baseline_value
    return None


def _dominates(candidate: _CandidateContext, peer: _CandidateContext) -> bool:
    comparable_pairs = [
        (left, right)
        for left in candidate.objective_deltas
        for right in peer.objective_deltas
        if left.objective_id == right.objective_id
        and left.candidate_value is not None
        and right.candidate_value is not None
    ]
    if not comparable_pairs:
        return False
    strictly_better = False
    for left, right in comparable_pairs:
        assert left.candidate_value is not None
        assert right.candidate_value is not None
        if not _value_at_least_as_good(
            goal=left.goal,
            left_value=left.candidate_value,
            right_value=right.candidate_value,
        ):
            return False
        if _value_strictly_better(
            goal=left.goal,
            left_value=left.candidate_value,
            right_value=right.candidate_value,
        ):
            strictly_better = True
    return strictly_better


def _value_at_least_as_good(
    *,
    goal: OptimizationGoal,
    left_value: float,
    right_value: float,
) -> bool:
    if goal in {OptimizationGoal.MINIMIZE, OptimizationGoal.AT_MOST}:
        return left_value <= right_value + _NORMALIZED_TIE_TOLERANCE
    return left_value + _NORMALIZED_TIE_TOLERANCE >= right_value


def _value_strictly_better(
    *,
    goal: OptimizationGoal,
    left_value: float,
    right_value: float,
) -> bool:
    if goal in {OptimizationGoal.MINIMIZE, OptimizationGoal.AT_MOST}:
        return left_value < right_value - _NORMALIZED_TIE_TOLERANCE
    return left_value > right_value + _NORMALIZED_TIE_TOLERANCE


def _label_score(
    *,
    context: _CandidateContext,
    dominated: bool,
) -> float:
    if context.hard_constraint_violations or dominated:
        return 0.0
    if (
        context.improved_objective_ids
        and not context.regressed_objective_ids
        and not context.soft_constraint_violations
        and not context.regressed_workload_families
        and (context.evidence_mix.observed_share or 0.0) > 0.0
    ):
        return 2.0
    return 1.0


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _share(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _bounded_id(seed: str, *, suffix: str) -> str:
    candidate = f"{seed}-{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    trimmed = candidate[: 117 - len(suffix)].rstrip("-")
    return f"{trimmed}-{suffix}-{digest}"
