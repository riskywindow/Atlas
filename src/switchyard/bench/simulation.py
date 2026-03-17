"""Offline policy simulation and counterfactual comparison helpers."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import cast

from switchyard.bench.history import TransparentHistoricalRoutePredictor
from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    CandidateRouteEstimateContext,
    CounterfactualCandidateScore,
    CounterfactualObjective,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationRecord,
    CounterfactualSimulationSummary,
    ExplainablePolicySpec,
    HistoricalRouteEstimate,
    PolicyRecommendation,
)


def simulate_policy_counterfactual(
    *,
    evaluation_artifacts: list[BenchmarkRunArtifact],
    history_artifacts: list[BenchmarkRunArtifact] | None = None,
    policy: ExplainablePolicySpec,
    timestamp: datetime | None = None,
) -> CounterfactualSimulationArtifact:
    """Simulate an explainable policy against authoritative benchmark artifacts."""

    evaluation_records = [
        (artifact.run_id, record)
        for artifact in evaluation_artifacts
        for record in artifact.records
    ]
    historical_artifacts = history_artifacts or evaluation_artifacts
    predictor = TransparentHistoricalRoutePredictor(
        historical_artifacts,
        min_samples=policy.min_evidence_count,
    )
    simulation_records = [
        _simulate_record(
            source_run_id=source_run_id,
            record=record,
            predictor=predictor,
            policy=policy,
        )
        for source_run_id, record in evaluation_records
    ]
    return CounterfactualSimulationArtifact(
        simulation_id=_build_simulation_id(
            timestamp=timestamp or datetime.now(UTC),
            policy=policy,
        ),
        timestamp=timestamp or datetime.now(UTC),
        source_run_ids=[artifact.run_id for artifact in evaluation_artifacts],
        historical_source_run_ids=[artifact.run_id for artifact in historical_artifacts],
        policy=policy,
        summary=_summarize_simulation(simulation_records),
        records=simulation_records,
        metadata={
            "evaluation_run_count": str(len(evaluation_artifacts)),
            "history_run_count": str(len(historical_artifacts)),
        },
    )


def recommend_policy_from_simulation(
    artifact: CounterfactualSimulationArtifact,
) -> str:
    """Return a compact operator-facing recommendation from a simulation artifact."""

    summary = artifact.summary
    changed_rate = (
        0.0 if summary.request_count == 0 else summary.changed_count / summary.request_count
    )
    if summary.insufficient_data_count == summary.request_count and summary.request_count > 0:
        return (
            "Keep the baseline policy. The simulation did not have enough historical evidence "
            "to justify route changes."
        )
    if summary.guardrail_block_count > 0 and changed_rate < 0.1:
        return (
            "Keep the policy in shadow or recommendation mode. Guardrails blocked most route "
            "changes under the available evidence."
        )
    if changed_rate >= 0.25:
        return (
            "Consider a bounded rollout. The simulated policy materially changes routing while "
            "still respecting the configured guardrails."
        )
    return (
        "Start in recommendation mode. The simulated policy produces explainable changes, "
        "but the impact looks incremental rather than broad."
    )


def _simulate_record(
    *,
    source_run_id: str,
    record: BenchmarkRequestRecord,
    predictor: TransparentHistoricalRoutePredictor,
    policy: ExplainablePolicySpec,
) -> CounterfactualSimulationRecord:
    candidate_names = _candidate_names_for_record(record)
    scored_candidates = [
        _score_candidate(
            record=record,
            backend_name=backend_name,
            predictor=predictor,
            policy=policy,
        )
        for backend_name in candidate_names
    ]
    observed_backend_score = next(
        (
            candidate
            for candidate in scored_candidates
            if candidate.backend_name == record.backend_name
        ),
        None,
    )
    best_scored_candidate = max(
        [candidate for candidate in scored_candidates if candidate.score is not None],
        key=lambda candidate: cast(float, candidate.score),
        default=None,
    )
    eligible_candidates = [candidate for candidate in scored_candidates if candidate.eligible]
    best_candidate = max(
        eligible_candidates,
        key=lambda candidate: cast(float, candidate.score),
        default=None,
    )

    recommendation_backend = record.backend_name
    recommendation_rationale = [
        f"mode={policy.mode.value}",
        f"objective={policy.objective.value}",
    ]
    guardrail_blocked = False
    insufficient_data = False

    if best_candidate is None:
        insufficient_data = True
        recommendation_rationale.append(
            "kept observed backend because no candidate had sufficient eligible evidence"
        )
        if (
            best_scored_candidate is not None
            and best_scored_candidate.backend_name != record.backend_name
            and not best_scored_candidate.eligible
        ):
            guardrail_blocked = True
            recommendation_rationale.append(
                "highest-scoring candidate was blocked by configured guardrails"
            )
    elif best_candidate.backend_name == record.backend_name:
        recommendation_rationale.append("observed backend already scored best under the policy")
        if (
            best_scored_candidate is not None
            and best_scored_candidate.backend_name != record.backend_name
            and not best_scored_candidate.eligible
        ):
            guardrail_blocked = True
            recommendation_rationale.append(
                "configured guardrails rejected a faster counterfactual candidate"
            )
    else:
        allowed, reason = _guardrails_allow_change(
            observed=observed_backend_score,
            proposed=best_candidate,
            policy=policy,
        )
        if allowed:
            recommendation_backend = best_candidate.backend_name
            recommendation_rationale.extend(
                [
                    f"recommended backend={best_candidate.backend_name}",
                    *best_candidate.rationale,
                ]
            )
        else:
            guardrail_blocked = True
            insufficient_data = "insufficient" in reason
            recommendation_rationale.append(reason)

    return CounterfactualSimulationRecord(
        request_id=record.request_id,
        source_run_id=source_run_id,
        workload_item_id=record.workload_item_id,
        source_trace_record_id=record.source_trace_record_id,
        model_alias=record.model_alias,
        request_class=record.request_class,
        request_features=record.request_features,
        observed_backend=record.backend_name,
        observed_latency_ms=record.latency_ms,
        observed_success=record.success,
        candidate_scores=scored_candidates,
        recommendation=PolicyRecommendation(
            observed_backend=record.backend_name,
            recommended_backend=recommendation_backend,
            recommendation_changed=recommendation_backend != record.backend_name,
            guardrail_blocked=guardrail_blocked,
            insufficient_data=insufficient_data,
            rationale=recommendation_rationale,
        ),
    )


def _score_candidate(
    *,
    record: BenchmarkRequestRecord,
    backend_name: str,
    predictor: TransparentHistoricalRoutePredictor,
    policy: ExplainablePolicySpec,
) -> CounterfactualCandidateScore:
    context = _estimate_context_for_record(record=record, backend_name=backend_name)
    estimate = predictor.estimate(context)
    eligible, rejection_reason = _estimate_is_eligible(estimate=estimate, policy=policy)
    rationale = [f"policy_id={policy.policy_id}", f"backend={backend_name}", *estimate.rationale]
    score = _score_estimate(estimate=estimate, objective=policy.objective)
    if not eligible:
        return CounterfactualCandidateScore(
            backend_name=backend_name,
            score=round(score, 6),
            eligible=False,
            rejection_reason=rejection_reason,
            estimate=estimate,
            rationale=[*rationale, *([] if rejection_reason is None else [rejection_reason])],
        )
    rationale.extend(
        [
            f"objective={policy.objective.value}",
            f"score={score:.6f}",
        ]
    )
    return CounterfactualCandidateScore(
        backend_name=backend_name,
        score=round(score, 6),
        eligible=True,
        estimate=estimate,
        rationale=rationale,
    )


def _estimate_context_for_record(
    *,
    record: BenchmarkRequestRecord,
    backend_name: str,
) -> CandidateRouteEstimateContext:
    request_features = record.request_features
    signal = record.route_decision.prefix_locality_signal if record.route_decision else None
    workload_tags = [] if request_features is None else list(request_features.workload_tags)
    return CandidateRouteEstimateContext(
        model_alias=record.model_alias or record.model_identifier or "unknown",
        backend_name=backend_name,
        backend_type=record.backend_type,
        policy_id=(
            record.policy_reference.policy_id
            if record.policy_reference is not None
            else (
                None if record.routing_policy is None else record.routing_policy.value
            )
        ),
        request_class=record.request_class,
        tenant_id=record.tenant_id,
        input_length_bucket=(
            None if request_features is None else request_features.input_length_bucket
        ),
        history_depth_bucket=(
            None if request_features is None else request_features.history_depth_bucket
        ),
        workload_tags=workload_tags,
        prefix_hotness=None if signal is None else signal.hotness,
        cache_opportunity=None if signal is None else signal.cache_opportunity,
        locality_benefit=(
            None if signal is None else signal.likely_benefits_from_locality
        ),
    )


def _candidate_names_for_record(record: BenchmarkRequestRecord) -> list[str]:
    if record.route_decision is not None and record.route_decision.considered_backends:
        return list(dict.fromkeys(record.route_decision.considered_backends))
    return [record.backend_name]


def _estimate_is_eligible(
    *,
    estimate: HistoricalRouteEstimate,
    policy: ExplainablePolicySpec,
) -> tuple[bool, str | None]:
    if policy.guardrails.require_sufficient_data and not estimate.sufficient_data:
        return False, "insufficient historical evidence for guarded recommendation"
    if (
        policy.guardrails.max_predicted_error_rate is not None
        and estimate.expected_error_rate is not None
        and estimate.expected_error_rate > policy.guardrails.max_predicted_error_rate
    ):
        return False, "predicted error rate exceeds configured guardrail"
    return True, None


def _score_estimate(
    *,
    estimate: HistoricalRouteEstimate,
    objective: CounterfactualObjective,
) -> float:
    latency = estimate.expected_latency_ms or 1000.0
    ttft = estimate.expected_ttft_ms or 0.0
    error_rate = estimate.expected_error_rate or 0.0
    throughput = estimate.expected_tokens_per_second or 0.0
    queue_delay = estimate.expected_queue_delay_ms or 0.0
    if objective is CounterfactualObjective.LATENCY:
        return -(latency + queue_delay + (ttft / 2.0) + (error_rate * 500.0))
    if objective is CounterfactualObjective.THROUGHPUT:
        return throughput - (error_rate * 200.0) - (latency / 20.0)
    if objective is CounterfactualObjective.RELIABILITY:
        return -(error_rate * 1000.0) - (latency / 50.0)
    return (
        throughput
        - latency
        - queue_delay
        - (ttft / 2.0)
        - (error_rate * 400.0)
    )


def _guardrails_allow_change(
    *,
    observed: CounterfactualCandidateScore | None,
    proposed: CounterfactualCandidateScore,
    policy: ExplainablePolicySpec,
) -> tuple[bool, str]:
    if not proposed.eligible:
        return False, proposed.rejection_reason or "proposed backend was not eligible"
    if observed is None:
        if policy.guardrails.require_observed_backend_evidence:
            return False, "observed backend estimate was unavailable under the configured guardrail"
        return True, "observed backend estimate unavailable; allowing best eligible recommendation"
    if observed.estimate is None or proposed.estimate is None:
        return False, "candidate estimates were incomplete"
    if (
        policy.guardrails.require_observed_backend_evidence
        and not observed.estimate.sufficient_data
    ):
        return False, "observed backend estimate lacked sufficient evidence"
    max_regression = policy.guardrails.max_predicted_latency_regression_ms
    if (
        max_regression is not None
        and observed.estimate.expected_latency_ms is not None
        and proposed.estimate.expected_latency_ms is not None
        and proposed.estimate.expected_latency_ms
        > observed.estimate.expected_latency_ms + max_regression
    ):
        return False, "predicted latency regression exceeds configured guardrail"
    return True, "proposed backend satisfied guardrails"


def _summarize_simulation(
    records: list[CounterfactualSimulationRecord],
) -> CounterfactualSimulationSummary:
    observed_counts: Counter[str] = Counter()
    recommended_counts: Counter[str] = Counter()
    projected_latencies: list[float] = []
    projected_error_rates: list[float] = []
    projected_throughput: list[float] = []

    changed_count = 0
    insufficient_data_count = 0
    guardrail_block_count = 0

    for record in records:
        observed_counts[record.observed_backend] += 1
        recommended_counts[record.recommendation.recommended_backend] += 1
        if record.recommendation.recommendation_changed:
            changed_count += 1
        if record.recommendation.insufficient_data:
            insufficient_data_count += 1
        if record.recommendation.guardrail_blocked:
            guardrail_block_count += 1
        selected_candidate = next(
            candidate
            for candidate in record.candidate_scores
            if candidate.backend_name == record.recommendation.recommended_backend
        )
        if (
            selected_candidate.estimate is not None
            and selected_candidate.estimate.expected_latency_ms is not None
        ):
            projected_latencies.append(selected_candidate.estimate.expected_latency_ms)
        if (
            selected_candidate.estimate is not None
            and selected_candidate.estimate.expected_error_rate is not None
        ):
            projected_error_rates.append(selected_candidate.estimate.expected_error_rate)
        if (
            selected_candidate.estimate is not None
            and selected_candidate.estimate.expected_tokens_per_second is not None
        ):
            projected_throughput.append(
                selected_candidate.estimate.expected_tokens_per_second
            )

    request_count = len(records)
    return CounterfactualSimulationSummary(
        request_count=request_count,
        changed_count=changed_count,
        unchanged_count=request_count - changed_count,
        insufficient_data_count=insufficient_data_count,
        guardrail_block_count=guardrail_block_count,
        observed_backend_counts=dict(sorted(observed_counts.items())),
        recommended_backend_counts=dict(sorted(recommended_counts.items())),
        projected_avg_latency_ms=_average(projected_latencies),
        projected_error_rate=_average(projected_error_rates),
        projected_avg_tokens_per_second=_average(projected_throughput),
    )


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _build_simulation_id(
    *,
    timestamp: datetime,
    policy: ExplainablePolicySpec,
) -> str:
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_simulate_{policy.policy_id}"
