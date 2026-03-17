"""Offline policy simulation and comparison helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast

from switchyard.bench.history import TransparentHistoricalRoutePredictor
from switchyard.schemas.backend import BackendInstance
from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    CandidateRouteEstimateContext,
    CapturedTraceRecord,
    CounterfactualCandidateScore,
    CounterfactualObjective,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    CounterfactualSimulationRecord,
    CounterfactualSimulationSummary,
    DeployedTopologyEndpoint,
    ExplainablePolicySpec,
    HistoricalRouteEstimate,
    PolicyRecommendation,
    SimulationBucketDimension,
    SimulationBucketSummary,
    SimulationEvidenceKind,
    SimulationSourceKind,
)
from switchyard.schemas.routing import (
    RequestClass,
    RequestFeatureVector,
    RoutingPolicy,
    TopologySnapshotReference,
)


@dataclass(frozen=True, slots=True)
class SimulationCase:
    """Normalized offline evaluation case derived from a run artifact or trace."""

    source_kind: SimulationSourceKind
    source_run_id: str | None
    source_trace_record_id: str | None
    source_record_id: str
    request_id: str
    workload_item_id: str | None
    model_alias: str | None
    tenant_id: str
    request_class: RequestClass
    request_features: RequestFeatureVector | None
    topology_reference: TopologySnapshotReference | None
    candidate_names: list[str]
    observed_backend: str
    observed_backend_instance_id: str | None
    observed_latency_ms: float | None
    observed_success: bool | None
    observed_ttft_ms: float | None
    observed_tokens_per_second: float | None
    observed_queue_delay_ms: float | None
    routing_policy_id: str | None
    backend_type: str | None


@dataclass(slots=True)
class BucketAccumulator:
    count: int = 0
    changed: int = 0
    direct: int = 0
    predictor: int = 0
    low: int = 0
    unsupported: int = 0
    latencies: list[float] = field(default_factory=list)


def compatibility_policy_spec(policy: RoutingPolicy) -> ExplainablePolicySpec:
    """Return an explicit offline policy spec for a fixed routing mode."""

    objective = CounterfactualObjective.BALANCED
    rationale = ["compatibility policy evaluated offline with historical evidence"]
    if policy is RoutingPolicy.LATENCY_FIRST:
        objective = CounterfactualObjective.LATENCY
        rationale.append("approximates fixed latency-first routing with latency-weighted evidence")
    elif policy is RoutingPolicy.QUALITY_FIRST:
        objective = CounterfactualObjective.RELIABILITY
        rationale.append(
            "approximates fixed quality-first routing with reliability-weighted evidence"
        )
    elif policy is RoutingPolicy.LOCAL_ONLY:
        objective = CounterfactualObjective.BALANCED
        rationale.append(
            "local-only compatibility remains approximate offline unless locality or topology "
            "evidence is present in the source artifacts"
        )
    else:
        rationale.append("balanced compatibility uses mixed latency, throughput, and error signals")
    return ExplainablePolicySpec(
        policy_id=policy.value,
        policy_version="phase6.v1",
        objective=objective,
        rationale=rationale,
    )


def simulate_policy_counterfactual(
    *,
    evaluation_artifacts: list[BenchmarkRunArtifact],
    history_artifacts: list[BenchmarkRunArtifact] | None = None,
    policy: ExplainablePolicySpec,
    trace_records: list[CapturedTraceRecord] | None = None,
    history_trace_records: list[CapturedTraceRecord] | None = None,
    timestamp: datetime | None = None,
) -> CounterfactualSimulationArtifact:
    """Simulate one explainable policy against benchmark artifacts and/or traces."""

    comparison = compare_candidate_policies_offline(
        policies=[policy],
        evaluation_artifacts=evaluation_artifacts,
        evaluation_trace_records=trace_records,
        history_artifacts=history_artifacts,
        history_trace_records=history_trace_records,
        timestamp=timestamp,
    )
    return comparison.evaluations[0]


def compare_candidate_policies_offline(
    *,
    policies: Sequence[ExplainablePolicySpec],
    evaluation_artifacts: Sequence[BenchmarkRunArtifact] | None = None,
    evaluation_trace_records: Sequence[CapturedTraceRecord] | None = None,
    history_artifacts: Sequence[BenchmarkRunArtifact] | None = None,
    history_trace_records: Sequence[CapturedTraceRecord] | None = None,
    timestamp: datetime | None = None,
) -> CounterfactualSimulationComparisonArtifact:
    """Compare several candidate policies against the same authoritative offline inputs."""

    run_timestamp = timestamp or datetime.now(UTC)
    resolved_evaluation_artifacts = list(evaluation_artifacts or [])
    resolved_evaluation_traces = list(evaluation_trace_records or [])
    resolved_history_artifacts = (
        list(history_artifacts)
        if history_artifacts is not None
        else resolved_evaluation_artifacts
    )
    resolved_history_traces = (
        list(history_trace_records)
        if history_trace_records is not None
        else resolved_evaluation_traces
    )
    cases = _build_simulation_cases(
        artifacts=resolved_evaluation_artifacts,
        traces=resolved_evaluation_traces,
    )
    history_records = _build_history_records(
        artifacts=resolved_history_artifacts,
        traces=resolved_history_traces,
    )
    predictor = TransparentHistoricalRoutePredictor(
        history_records,
        min_samples=max(policy.min_evidence_count for policy in policies),
    )
    topology_references, deployed_topology, worker_inventory = _collect_topology_context(
        [*resolved_evaluation_artifacts, *resolved_history_artifacts]
    )
    evaluations = [
        _simulate_cases(
            cases=cases,
            predictor=predictor,
            policy=policy,
            evaluation_artifacts=resolved_evaluation_artifacts,
            evaluation_traces=resolved_evaluation_traces,
            history_artifacts=resolved_history_artifacts,
            history_traces=resolved_history_traces,
            topology_references=topology_references,
            deployed_topology=deployed_topology,
            worker_inventory=worker_inventory,
            timestamp=run_timestamp,
        )
        for policy in policies
    ]
    return CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id=_build_comparison_id(run_timestamp),
        timestamp=run_timestamp,
        source_run_ids=[artifact.run_id for artifact in resolved_evaluation_artifacts],
        source_trace_ids=[trace.record_id for trace in resolved_evaluation_traces],
        historical_source_run_ids=[artifact.run_id for artifact in resolved_history_artifacts],
        historical_source_trace_ids=[trace.record_id for trace in resolved_history_traces],
        policies=list(policies),
        evaluations=evaluations,
        topology_references=topology_references,
        deployed_topology=deployed_topology,
        worker_instance_inventory=worker_inventory,
        limitation_notes=_comparison_limitations(evaluations),
        metadata={
            "evaluation_case_count": str(len(cases)),
            "history_record_count": str(len(history_records)),
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
    if summary.unsupported_count == summary.request_count and summary.request_count > 0:
        return (
            "Keep the baseline policy. The source artifacts did not provide enough candidate "
            "evidence to support this offline comparison."
        )
    if summary.low_confidence_count > 0 and changed_rate >= 0.25:
        return (
            "Keep the policy in recommendation mode. The offline comparison shows material "
            "changes, but too many of them rely on low-confidence estimates."
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


def select_best_policy_from_comparison(
    artifact: CounterfactualSimulationComparisonArtifact,
) -> str:
    """Return the policy id with the best supported projected outcome."""

    def ranking_key(
        evaluation: CounterfactualSimulationArtifact,
    ) -> tuple[float, float, float]:
        summary = evaluation.summary
        supported = summary.request_count - summary.unsupported_count
        return (
            float(supported),
            -(summary.projected_avg_latency_ms or float("inf")),
            -(summary.projected_error_rate or 1.0),
        )

    return max(artifact.evaluations, key=ranking_key).policy.policy_id


def _simulate_cases(
    *,
    cases: Sequence[SimulationCase],
    predictor: TransparentHistoricalRoutePredictor,
    policy: ExplainablePolicySpec,
    evaluation_artifacts: Sequence[BenchmarkRunArtifact],
    evaluation_traces: Sequence[CapturedTraceRecord],
    history_artifacts: Sequence[BenchmarkRunArtifact],
    history_traces: Sequence[CapturedTraceRecord],
    topology_references: list[TopologySnapshotReference],
    deployed_topology: list[DeployedTopologyEndpoint],
    worker_inventory: list[BackendInstance],
    timestamp: datetime,
) -> CounterfactualSimulationArtifact:
    simulation_records = [
        _simulate_case(case=case, predictor=predictor, policy=policy) for case in cases
    ]
    summary = _summarize_simulation(simulation_records)
    return CounterfactualSimulationArtifact(
        simulation_id=_build_simulation_id(timestamp=timestamp, policy=policy),
        timestamp=timestamp,
        source_run_ids=[artifact.run_id for artifact in evaluation_artifacts],
        source_trace_ids=[trace.record_id for trace in evaluation_traces],
        historical_source_run_ids=[artifact.run_id for artifact in history_artifacts],
        historical_source_trace_ids=[trace.record_id for trace in history_traces],
        policy=policy,
        summary=summary,
        records=simulation_records,
        topology_references=topology_references,
        deployed_topology=deployed_topology,
        worker_instance_inventory=worker_inventory,
        metadata={
            "policy_recommendation": recommend_policy_from_simulation(
                CounterfactualSimulationArtifact(
                    simulation_id=_build_simulation_id(timestamp=timestamp, policy=policy),
                    timestamp=timestamp,
                    policy=policy,
                    summary=summary,
                    records=simulation_records,
                )
            )
        },
    )


def _build_simulation_cases(
    *,
    artifacts: Sequence[BenchmarkRunArtifact],
    traces: Sequence[CapturedTraceRecord],
) -> list[SimulationCase]:
    cases = [case for artifact in artifacts for case in _cases_from_artifact(artifact)]
    cases.extend(_case_from_trace(trace) for trace in traces)
    return cases


def _build_history_records(
    *,
    artifacts: Sequence[BenchmarkRunArtifact],
    traces: Sequence[CapturedTraceRecord],
) -> list[BenchmarkRequestRecord]:
    records = [record for artifact in artifacts for record in artifact.records]
    records.extend(
        record
        for trace in traces
        if (record := _history_record_from_trace(trace)) is not None
    )
    return records


def _cases_from_artifact(artifact: BenchmarkRunArtifact) -> list[SimulationCase]:
    return [_case_from_record(record, source_run_id=artifact.run_id) for record in artifact.records]


def _case_from_record(record: BenchmarkRequestRecord, *, source_run_id: str) -> SimulationCase:
    candidate_names = _candidate_names_for_record(record)
    return SimulationCase(
        source_kind=SimulationSourceKind.BENCHMARK_RUN,
        source_run_id=source_run_id,
        source_trace_record_id=record.source_trace_record_id,
        source_record_id=record.request_id,
        request_id=record.request_id,
        workload_item_id=record.workload_item_id,
        model_alias=record.model_alias,
        tenant_id=record.tenant_id,
        request_class=record.request_class,
        request_features=record.request_features,
        topology_reference=record.topology_reference,
        candidate_names=candidate_names,
        observed_backend=record.backend_name,
        observed_backend_instance_id=record.backend_instance_id,
        observed_latency_ms=record.latency_ms,
        observed_success=record.success,
        observed_ttft_ms=record.ttft_ms,
        observed_tokens_per_second=record.tokens_per_second,
        observed_queue_delay_ms=record.queue_delay_ms,
        routing_policy_id=_policy_id_for_record(record),
        backend_type=record.backend_type,
    )


def _case_from_trace(trace: CapturedTraceRecord) -> SimulationCase:
    candidate_names = (
        list(dict.fromkeys(trace.route_decision.considered_backends))
        if trace.route_decision is not None and trace.route_decision.considered_backends
        else [trace.chosen_backend or "unknown"]
    )
    observed_success = None
    if trace.status_code is not None:
        observed_success = trace.status_code < 400
    return SimulationCase(
        source_kind=SimulationSourceKind.CAPTURED_TRACE,
        source_run_id=None,
        source_trace_record_id=trace.record_id,
        source_record_id=trace.record_id,
        request_id=trace.request_id,
        workload_item_id=None,
        model_alias=trace.logical_alias,
        tenant_id=trace.tenant_id,
        request_class=trace.request_class,
        request_features=trace.request_features,
        topology_reference=trace.topology_reference,
        candidate_names=candidate_names,
        observed_backend=trace.chosen_backend or "unknown",
        observed_backend_instance_id=None,
        observed_latency_ms=trace.latency_ms,
        observed_success=observed_success,
        observed_ttft_ms=trace.ttft_ms,
        observed_tokens_per_second=None,
        observed_queue_delay_ms=None,
        routing_policy_id=(
            None if trace.policy_reference is None else trace.policy_reference.policy_id
        ),
        backend_type=None,
    )


def _history_record_from_trace(trace: CapturedTraceRecord) -> BenchmarkRequestRecord | None:
    if trace.chosen_backend is None or trace.latency_ms is None or trace.status_code is None:
        return None
    return BenchmarkRequestRecord(
        request_id=trace.request_id,
        source_trace_record_id=trace.record_id,
        tenant_id=trace.tenant_id,
        request_class=trace.request_class,
        session_id=trace.session_id,
        request_features=trace.request_features,
        policy_reference=trace.policy_reference,
        topology_reference=trace.topology_reference,
        backend_name=trace.chosen_backend,
        model_alias=trace.logical_alias,
        started_at=trace.request_timestamp,
        completed_at=trace.request_timestamp,
        latency_ms=trace.latency_ms,
        ttft_ms=trace.ttft_ms,
        output_tokens=trace.output_tokens,
        queue_delay_ms=(
            None
            if trace.control_plane_metadata is None
            or trace.control_plane_metadata.execution_observation is None
            else trace.control_plane_metadata.execution_observation.queue_delay_ms
        ),
        routing_policy=(
            None
            if trace.route_decision is None
            else trace.route_decision.policy
        ),
        route_decision=trace.route_decision,
        success=trace.status_code < 400,
        status_code=trace.status_code,
        error=trace.error,
        error_category=trace.error_category,
    )


def _simulate_case(
    *,
    case: SimulationCase,
    predictor: TransparentHistoricalRoutePredictor,
    policy: ExplainablePolicySpec,
) -> CounterfactualSimulationRecord:
    scored_candidates = [
        _score_candidate(
            case=case,
            backend_name=backend_name,
            predictor=predictor,
            policy=policy,
        )
        for backend_name in case.candidate_names
    ]
    observed_backend_score = next(
        (
            candidate
            for candidate in scored_candidates
            if candidate.backend_name == case.observed_backend
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

    recommendation_backend = case.observed_backend
    evidence_kind = SimulationEvidenceKind.UNSUPPORTED
    recommendation_rationale = [
        f"mode={policy.mode.value}",
        f"objective={policy.objective.value}",
    ]
    guardrail_blocked = False
    insufficient_data = False

    if best_candidate is None:
        insufficient_data = True
        recommendation_rationale.append(
            "kept observed backend because no candidate had sufficient supported evidence"
        )
        if observed_backend_score is not None:
            evidence_kind = observed_backend_score.evidence_kind
        if (
            best_scored_candidate is not None
            and best_scored_candidate.backend_name != case.observed_backend
            and not best_scored_candidate.eligible
        ):
            guardrail_blocked = True
            recommendation_rationale.append(
                "highest-scoring candidate was blocked by guardrails or low-confidence evidence"
            )
    elif best_candidate.backend_name == case.observed_backend:
        evidence_kind = best_candidate.evidence_kind
        recommendation_rationale.append("observed backend already scored best under the policy")
        if (
            best_scored_candidate is not None
            and best_scored_candidate.backend_name != case.observed_backend
            and not best_scored_candidate.eligible
        ):
            guardrail_blocked = True
            recommendation_rationale.append(
                "configured guardrails rejected a counterfactual candidate with weaker evidence"
            )
    else:
        allowed, reason = _guardrails_allow_change(
            observed=observed_backend_score,
            proposed=best_candidate,
            policy=policy,
        )
        if allowed:
            recommendation_backend = best_candidate.backend_name
            evidence_kind = best_candidate.evidence_kind
            recommendation_rationale.extend(
                [
                    f"recommended backend={best_candidate.backend_name}",
                    *best_candidate.rationale,
                ]
            )
        else:
            guardrail_blocked = True
            insufficient_data = "insufficient" in reason or "unsupported" in reason
            evidence_kind = (
                observed_backend_score.evidence_kind
                if observed_backend_score is not None
                else SimulationEvidenceKind.UNSUPPORTED
            )
            recommendation_rationale.append(reason)

    return CounterfactualSimulationRecord(
        request_id=case.request_id,
        source_run_id=case.source_run_id,
        source_kind=case.source_kind,
        source_record_id=case.source_record_id,
        workload_item_id=case.workload_item_id,
        source_trace_record_id=case.source_trace_record_id,
        model_alias=case.model_alias,
        tenant_id=case.tenant_id,
        request_class=case.request_class,
        request_features=case.request_features,
        observed_backend=case.observed_backend,
        observed_backend_instance_id=case.observed_backend_instance_id,
        observed_latency_ms=case.observed_latency_ms,
        observed_success=case.observed_success,
        topology_reference=case.topology_reference,
        candidate_scores=scored_candidates,
        recommendation=PolicyRecommendation(
            observed_backend=case.observed_backend,
            recommended_backend=recommendation_backend,
            recommendation_changed=recommendation_backend != case.observed_backend,
            guardrail_blocked=guardrail_blocked,
            insufficient_data=insufficient_data,
            evidence_kind=evidence_kind,
            rationale=recommendation_rationale,
        ),
    )


def _score_candidate(
    *,
    case: SimulationCase,
    backend_name: str,
    predictor: TransparentHistoricalRoutePredictor,
    policy: ExplainablePolicySpec,
) -> CounterfactualCandidateScore:
    estimate = predictor.estimate(_estimate_context_for_case(case=case, backend_name=backend_name))
    if backend_name == case.observed_backend and case.observed_latency_ms is not None:
        observed_score = _score_metrics(
            latency=case.observed_latency_ms,
            ttft=case.observed_ttft_ms,
            error_rate=(0.0 if case.observed_success is True else 1.0),
            throughput=case.observed_tokens_per_second,
            queue_delay=case.observed_queue_delay_ms,
            objective=policy.objective,
        )
        rationale = [
            f"policy_id={policy.policy_id}",
            f"backend={backend_name}",
            "used direct observed outcome from the authoritative source artifact",
        ]
        return CounterfactualCandidateScore(
            backend_name=backend_name,
            score=round(observed_score, 6),
            eligible=True,
            evidence_kind=SimulationEvidenceKind.DIRECT_OBSERVATION,
            evidence_count=1,
            estimate=estimate,
            directly_observed=True,
            observed_latency_ms=case.observed_latency_ms,
            observed_success=case.observed_success,
            backend_instance_id=case.observed_backend_instance_id,
            confidence_note="directly observed historical execution",
            rationale=rationale,
        )

    evidence_kind = _evidence_kind_for_estimate(estimate)
    eligible, rejection_reason = _estimate_is_eligible(
        estimate=estimate,
        policy=policy,
        evidence_kind=evidence_kind,
    )
    score: float | None = (
        None
        if evidence_kind is SimulationEvidenceKind.UNSUPPORTED
        else _score_estimate(estimate=estimate, objective=policy.objective)
    )
    rationale = [f"policy_id={policy.policy_id}", f"backend={backend_name}", *estimate.rationale]
    confidence_note = _confidence_note(estimate=estimate, evidence_kind=evidence_kind)
    if score is not None:
        rationale.extend(
            [
                f"objective={policy.objective.value}",
                f"score={score:.6f}",
            ]
        )
    if rejection_reason is not None:
        rationale.append(rejection_reason)
    return CounterfactualCandidateScore(
        backend_name=backend_name,
        score=None if score is None else round(score, 6),
        eligible=eligible,
        evidence_kind=evidence_kind,
        evidence_count=estimate.evidence_count,
        rejection_reason=rejection_reason,
        estimate=estimate,
        backend_instance_id=estimate.context.backend_instance_id,
        confidence_note=confidence_note,
        rationale=rationale,
    )


def _estimate_context_for_case(
    *,
    case: SimulationCase,
    backend_name: str,
) -> CandidateRouteEstimateContext:
    request_features = case.request_features
    return CandidateRouteEstimateContext(
        model_alias=case.model_alias or "unknown",
        backend_name=backend_name,
        backend_type=case.backend_type,
        backend_instance_id=(
            case.observed_backend_instance_id if backend_name == case.observed_backend else None
        ),
        policy_id=case.routing_policy_id,
        request_class=case.request_class,
        tenant_id=case.tenant_id,
        input_length_bucket=(
            None if request_features is None else request_features.input_length_bucket
        ),
        history_depth_bucket=(
            None if request_features is None else request_features.history_depth_bucket
        ),
        workload_tags=(
            [] if request_features is None else list(request_features.workload_tags)
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
    evidence_kind: SimulationEvidenceKind,
) -> tuple[bool, str | None]:
    if evidence_kind is SimulationEvidenceKind.UNSUPPORTED:
        return False, "no historical evidence supports this candidate"
    if policy.guardrails.require_sufficient_data and (
        evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE
    ):
        return False, "insufficient historical evidence for guarded recommendation"
    if (
        policy.guardrails.max_predicted_error_rate is not None
        and estimate.expected_error_rate is not None
        and estimate.expected_error_rate > policy.guardrails.max_predicted_error_rate
    ):
        return False, "predicted error rate exceeds configured guardrail"
    return True, None


def _evidence_kind_for_estimate(estimate: HistoricalRouteEstimate) -> SimulationEvidenceKind:
    if estimate.sufficient_data:
        return SimulationEvidenceKind.PREDICTOR_ESTIMATE
    if estimate.evidence_count > 0:
        return SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE
    return SimulationEvidenceKind.UNSUPPORTED


def _confidence_note(
    *,
    estimate: HistoricalRouteEstimate,
    evidence_kind: SimulationEvidenceKind,
) -> str:
    if evidence_kind is SimulationEvidenceKind.PREDICTOR_ESTIMATE:
        return f"predictor estimate from {estimate.evidence_count} historical samples"
    if evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE:
        return (
            f"low-confidence estimate from only {estimate.evidence_count} historical samples"
        )
    return "unsupported candidate because no suitable historical slice was found"


def _score_estimate(
    *,
    estimate: HistoricalRouteEstimate,
    objective: CounterfactualObjective,
) -> float:
    return _score_metrics(
        latency=estimate.expected_latency_ms,
        ttft=estimate.expected_ttft_ms,
        error_rate=estimate.expected_error_rate,
        throughput=estimate.expected_tokens_per_second,
        queue_delay=estimate.expected_queue_delay_ms,
        objective=objective,
    )


def _score_metrics(
    *,
    latency: float | None,
    ttft: float | None,
    error_rate: float | None,
    throughput: float | None,
    queue_delay: float | None,
    objective: CounterfactualObjective,
) -> float:
    resolved_latency = latency or 1000.0
    resolved_ttft = ttft or 0.0
    resolved_error_rate = error_rate or 0.0
    resolved_throughput = throughput or 0.0
    resolved_queue_delay = queue_delay or 0.0
    if objective is CounterfactualObjective.LATENCY:
        return -(
            resolved_latency
            + resolved_queue_delay
            + (resolved_ttft / 2.0)
            + (resolved_error_rate * 500.0)
        )
    if objective is CounterfactualObjective.THROUGHPUT:
        return (
            resolved_throughput
            - (resolved_error_rate * 200.0)
            - (resolved_latency / 20.0)
        )
    if objective is CounterfactualObjective.RELIABILITY:
        return -(resolved_error_rate * 1000.0) - (resolved_latency / 50.0)
    return (
        resolved_throughput
        - resolved_latency
        - resolved_queue_delay
        - (resolved_ttft / 2.0)
        - (resolved_error_rate * 400.0)
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
    if proposed.evidence_kind is SimulationEvidenceKind.UNSUPPORTED:
        return False, "proposed backend is unsupported by the available evidence"
    if policy.guardrails.require_observed_backend_evidence and (
        observed.evidence_kind in {
            SimulationEvidenceKind.UNSUPPORTED,
            SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE,
        }
    ):
        return False, "observed backend estimate lacked sufficient evidence"
    max_regression = policy.guardrails.max_predicted_latency_regression_ms
    if (
        max_regression is not None
        and observed.estimate is not None
        and observed.estimate.expected_latency_ms is not None
        and proposed.estimate is not None
        and proposed.estimate.expected_latency_ms is not None
        and proposed.estimate.expected_latency_ms
        > observed.estimate.expected_latency_ms + max_regression
    ):
        return False, "predicted latency regression exceeds configured guardrail"
    return True, "proposed backend satisfied guardrails"


def _summarize_simulation(
    records: Sequence[CounterfactualSimulationRecord],
) -> CounterfactualSimulationSummary:
    observed_counts: Counter[str] = Counter()
    recommended_counts: Counter[str] = Counter()
    projected_latencies: list[float] = []
    projected_error_rates: list[float] = []
    projected_throughput: list[float] = []

    changed_count = 0
    insufficient_data_count = 0
    guardrail_block_count = 0
    direct_observation_count = 0
    predictor_estimate_count = 0
    low_confidence_count = 0
    unsupported_count = 0
    bucket_accumulators: dict[
        tuple[SimulationBucketDimension, str],
        BucketAccumulator,
    ] = defaultdict(BucketAccumulator)
    limitation_notes: list[str] = []

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
        if selected_candidate.evidence_kind is SimulationEvidenceKind.DIRECT_OBSERVATION:
            direct_observation_count += 1
        elif selected_candidate.evidence_kind is SimulationEvidenceKind.PREDICTOR_ESTIMATE:
            predictor_estimate_count += 1
        elif selected_candidate.evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE:
            low_confidence_count += 1
        else:
            unsupported_count += 1

        if (
            selected_candidate.directly_observed
            and selected_candidate.observed_latency_ms is not None
        ):
            projected_latencies.append(selected_candidate.observed_latency_ms)
            projected_error_rates.append(0.0 if selected_candidate.observed_success else 1.0)
        elif (
            selected_candidate.estimate is not None
            and selected_candidate.estimate.expected_latency_ms is not None
        ):
            projected_latencies.append(selected_candidate.estimate.expected_latency_ms)
            if selected_candidate.estimate.expected_error_rate is not None:
                projected_error_rates.append(selected_candidate.estimate.expected_error_rate)
            if selected_candidate.estimate.expected_tokens_per_second is not None:
                projected_throughput.append(
                    selected_candidate.estimate.expected_tokens_per_second
                )

        for dimension, bucket_key in _bucket_values_for_record(record):
            bucket = bucket_accumulators[(dimension, bucket_key)]
            bucket.count += 1
            if record.recommendation.recommendation_changed:
                bucket.changed += 1
            if selected_candidate.evidence_kind is SimulationEvidenceKind.DIRECT_OBSERVATION:
                bucket.direct += 1
            elif selected_candidate.evidence_kind is SimulationEvidenceKind.PREDICTOR_ESTIMATE:
                bucket.predictor += 1
            elif (
                selected_candidate.evidence_kind
                is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE
            ):
                bucket.low += 1
            else:
                bucket.unsupported += 1
            if (
                selected_candidate.directly_observed
                and selected_candidate.observed_latency_ms is not None
            ):
                bucket.latencies.append(selected_candidate.observed_latency_ms)
            elif (
                selected_candidate.estimate is not None
                and selected_candidate.estimate.expected_latency_ms is not None
            ):
                bucket.latencies.append(selected_candidate.estimate.expected_latency_ms)

    request_count = len(records)
    if low_confidence_count > 0:
        limitation_notes.append(
            f"{low_confidence_count} requests relied on low-confidence predictor estimates"
        )
    if unsupported_count > 0:
        limitation_notes.append(
            f"{unsupported_count} requests remained unsupported by the available evidence"
        )
    bucket_summaries = [
        SimulationBucketSummary(
            dimension=dimension,
            bucket_key=bucket_key,
            request_count=values.count,
            changed_count=values.changed,
            direct_observation_count=values.direct,
            predictor_estimate_count=values.predictor,
            low_confidence_count=values.low,
            unsupported_count=values.unsupported,
            avg_projected_latency_ms=_average(values.latencies),
        )
        for (dimension, bucket_key), values in sorted(
            bucket_accumulators.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        )
    ]
    return CounterfactualSimulationSummary(
        request_count=request_count,
        changed_count=changed_count,
        unchanged_count=request_count - changed_count,
        direct_observation_count=direct_observation_count,
        predictor_estimate_count=predictor_estimate_count,
        low_confidence_count=low_confidence_count,
        unsupported_count=unsupported_count,
        insufficient_data_count=insufficient_data_count,
        guardrail_block_count=guardrail_block_count,
        observed_backend_counts=dict(sorted(observed_counts.items())),
        recommended_backend_counts=dict(sorted(recommended_counts.items())),
        projected_avg_latency_ms=_average(projected_latencies),
        projected_error_rate=_average(projected_error_rates),
        projected_avg_tokens_per_second=_average(projected_throughput),
        bucket_summaries=bucket_summaries,
        limitation_notes=limitation_notes,
    )


def _bucket_values_for_record(
    record: CounterfactualSimulationRecord,
) -> list[tuple[SimulationBucketDimension, str]]:
    buckets = [
        (
            SimulationBucketDimension.MODEL_ALIAS,
            record.model_alias or "unknown",
        ),
        (
            SimulationBucketDimension.TENANT_ID,
            record.tenant_id,
        ),
    ]
    if record.request_features is not None:
        buckets.append(
            (
                SimulationBucketDimension.INPUT_LENGTH_BUCKET,
                record.request_features.input_length_bucket.value,
            )
        )
    if record.observed_backend_instance_id is not None:
        buckets.append(
            (
                SimulationBucketDimension.BACKEND_INSTANCE_ID,
                record.observed_backend_instance_id,
            )
        )
    return buckets


def _collect_topology_context(
    artifacts: Sequence[BenchmarkRunArtifact],
) -> tuple[
    list[TopologySnapshotReference],
    list[DeployedTopologyEndpoint],
    list[BackendInstance],
]:
    topology_references: dict[str, TopologySnapshotReference] = {}
    deployed_topology: dict[str, DeployedTopologyEndpoint] = {}
    worker_inventory: dict[str, BackendInstance] = {}
    for artifact in artifacts:
        if artifact.environment.topology_reference is not None:
            topology_references[
                artifact.environment.topology_reference.topology_snapshot_id
            ] = artifact.environment.topology_reference
        for record in artifact.records:
            if record.topology_reference is not None:
                topology_references[
                    record.topology_reference.topology_snapshot_id
                ] = record.topology_reference
        for endpoint in artifact.environment.deployed_topology:
            deployed_topology[endpoint.endpoint_id] = endpoint
        for instance in artifact.environment.worker_instance_inventory:
            worker_inventory[instance.instance_id] = instance
    return (
        [topology_references[key] for key in sorted(topology_references)],
        [deployed_topology[key] for key in sorted(deployed_topology)],
        [worker_inventory[key] for key in sorted(worker_inventory)],
    )


def _comparison_limitations(
    evaluations: Sequence[CounterfactualSimulationArtifact],
) -> list[str]:
    notes: list[str] = []
    if any(evaluation.summary.unsupported_count > 0 for evaluation in evaluations):
        notes.append(
            "Some simulated requests remained unsupported because the source artifacts "
            "did not expose enough historical evidence or candidate detail."
        )
    if any(evaluation.summary.low_confidence_count > 0 for evaluation in evaluations):
        notes.append(
            "Some policy outcomes rely on low-confidence predictor estimates rather than "
            "directly observed historical outcomes."
        )
    elif any(
        candidate.evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE
        for evaluation in evaluations
        for record in evaluation.records
        for candidate in record.candidate_scores
    ):
        notes.append(
            "Some candidate backends were only comparable through low-confidence "
            "predictor estimates."
        )
    return notes


def _average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _policy_id_for_record(record: BenchmarkRequestRecord) -> str | None:
    if record.policy_reference is not None:
        return record.policy_reference.policy_id
    if record.routing_policy is not None:
        return record.routing_policy.value
    return None


def _build_simulation_id(
    *,
    timestamp: datetime,
    policy: ExplainablePolicySpec,
) -> str:
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_simulate_{policy.policy_id}"


def _build_comparison_id(timestamp: datetime) -> str:
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_simulate_compare"
