"""Evidence-based routing recommendation helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    CounterfactualSimulationRecord,
    PolicyRecommendationReportArtifact,
    RecommendationConfidence,
    RecommendationDisposition,
    RecommendationEvidenceWindow,
    RecommendationScopeKind,
    RoutingPolicyGuidance,
    SimulationEvidenceKind,
)
from switchyard.schemas.routing import RequestClass, WorkloadTag

LoadedArtifact = (
    BenchmarkRunArtifact
    | CounterfactualSimulationArtifact
    | CounterfactualSimulationComparisonArtifact
)

_MIN_SCOPE_SAMPLE_SIZE = 5
_MIN_REPEATED_PREFIX_SAMPLE_SIZE = 3
_MIN_LATENCY_IMPROVEMENT_MS = 5.0


@dataclass(frozen=True, slots=True)
class PolicyScopeStats:
    policy_id: str
    sample_size: int
    supported_records: int
    low_confidence: int
    guardrail_blocks: int
    changed: int
    avg_latency: float | None
    counterexamples: list[str]


def build_policy_recommendation_report(
    artifacts: Sequence[LoadedArtifact],
    *,
    report_id: str,
    timestamp: datetime,
) -> PolicyRecommendationReportArtifact:
    """Build a conservative recommendation report from authoritative artifacts."""

    benchmark_artifacts = [
        artifact for artifact in artifacts if isinstance(artifact, BenchmarkRunArtifact)
    ]
    simulation_artifacts = [
        artifact for artifact in artifacts if isinstance(artifact, CounterfactualSimulationArtifact)
    ]
    comparison_artifacts = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, CounterfactualSimulationComparisonArtifact)
    ]
    recommendations: list[RoutingPolicyGuidance] = []
    recommendations.extend(
        _simulation_scope_recommendations(comparison_artifacts=comparison_artifacts)
    )
    recommendations.extend(
        _simulation_scope_recommendations_from_single_runs(
            simulation_artifacts=simulation_artifacts
        )
    )
    recommendations.extend(
        _repeated_prefix_backend_recommendations(benchmark_artifacts=benchmark_artifacts)
    )
    limitations = _collect_limitations(
        benchmark_artifacts=benchmark_artifacts,
        simulation_artifacts=simulation_artifacts,
        comparison_artifacts=comparison_artifacts,
        recommendations=recommendations,
    )
    return PolicyRecommendationReportArtifact(
        recommendation_report_id=report_id,
        timestamp=timestamp,
        evidence_window=_build_evidence_window(
            benchmark_artifacts=benchmark_artifacts,
            simulation_artifacts=simulation_artifacts,
            comparison_artifacts=comparison_artifacts,
        ),
        recommendations=recommendations,
        notable_limitations=limitations,
        metadata={
            "benchmark_artifact_count": str(len(benchmark_artifacts)),
            "simulation_artifact_count": str(len(simulation_artifacts)),
            "comparison_artifact_count": str(len(comparison_artifacts)),
        },
    )


def render_policy_recommendation_report_markdown(
    artifact: PolicyRecommendationReportArtifact,
) -> str:
    """Render a compact markdown report for policy recommendations."""

    lines = [
        f"# Switchyard Policy Recommendation Report: {artifact.recommendation_report_id}",
        "",
        "## Evidence Window",
        f"- Source runs: `{', '.join(artifact.evidence_window.source_run_ids) or 'none'}`",
        f"- Source traces: `{', '.join(artifact.evidence_window.source_trace_ids) or 'none'}`",
        (
            "- Historical runs: "
            f"`{', '.join(artifact.evidence_window.historical_source_run_ids) or 'none'}`"
        ),
        (
            "- Historical traces: "
            f"`{', '.join(artifact.evidence_window.historical_source_trace_ids) or 'none'}`"
        ),
        (
            "- Window: "
            f"`{artifact.evidence_window.window_started_at}` to "
            f"`{artifact.evidence_window.window_ended_at}`"
        ),
        "",
        "## Recommendations",
    ]
    if not artifact.recommendations:
        lines.append("- No recommendation. Available artifacts did not provide enough evidence.")
    for recommendation in artifact.recommendations:
        lines.extend(
            [
                (
                    f"- Scope `{recommendation.scope_kind.value}:{recommendation.scope_key}`: "
                    f"`{recommendation.recommendation.value}`"
                ),
                f"  Sample size: `{recommendation.sample_size}`",
                f"  Confidence: `{recommendation.confidence.value}`",
                (
                    "  Workload buckets: "
                    f"`{', '.join(recommendation.workload_buckets) or 'unavailable'}`"
                ),
            ]
        )
        if recommendation.recommended_policy_id is not None:
            lines.append(f"  Recommended policy: `{recommendation.recommended_policy_id}`")
        if recommendation.recommended_target is not None:
            lines.append(
                "  Recommended target: "
                f"`{recommendation.recommended_target_type}:{recommendation.recommended_target}`"
            )
        if recommendation.evidence_summary:
            lines.append(f"  Evidence: {'; '.join(recommendation.evidence_summary)}")
        if recommendation.caveats:
            lines.append(f"  Caveats: {'; '.join(recommendation.caveats)}")
        if recommendation.confidence_notes:
            lines.append(
                f"  Confidence notes: {'; '.join(recommendation.confidence_notes)}"
            )
        if recommendation.notable_regressions:
            lines.append(
                f"  Notable regressions: {'; '.join(recommendation.notable_regressions)}"
            )
        if recommendation.counterexamples:
            lines.append(f"  Counterexamples: {'; '.join(recommendation.counterexamples)}")
    if artifact.notable_limitations:
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in artifact.notable_limitations)
    return "\n".join(lines)


def _simulation_scope_recommendations(
    *,
    comparison_artifacts: Sequence[CounterfactualSimulationComparisonArtifact],
) -> list[RoutingPolicyGuidance]:
    recommendations: list[RoutingPolicyGuidance] = []
    for artifact in comparison_artifacts:
        recommendations.extend(
            _scope_recommendations_for_comparison(
                artifact=artifact,
                scope_kind=RecommendationScopeKind.MODEL_ALIAS,
            )
        )
        recommendations.extend(
            _scope_recommendations_for_comparison(
                artifact=artifact,
                scope_kind=RecommendationScopeKind.REQUEST_CLASS,
            )
        )
    return recommendations


def _simulation_scope_recommendations_from_single_runs(
    *,
    simulation_artifacts: Sequence[CounterfactualSimulationArtifact],
) -> list[RoutingPolicyGuidance]:
    recommendations: list[RoutingPolicyGuidance] = []
    for artifact in simulation_artifacts:
        recommendations.append(
            _single_simulation_guidance(artifact=artifact)
        )
    return recommendations


def _scope_recommendations_for_comparison(
    *,
    artifact: CounterfactualSimulationComparisonArtifact,
    scope_kind: RecommendationScopeKind,
) -> list[RoutingPolicyGuidance]:
    grouped: dict[str, list[CounterfactualSimulationArtifact]] = defaultdict(list)
    scope_to_records: dict[tuple[str, str], list[CounterfactualSimulationRecord]] = {}
    for evaluation in artifact.evaluations:
        groups = _group_simulation_records(evaluation=evaluation, scope_kind=scope_kind)
        for scope_key, records in groups.items():
            grouped[scope_key].append(evaluation)
            scope_to_records[(scope_key, evaluation.policy.policy_id)] = records
    recommendations: list[RoutingPolicyGuidance] = []
    for scope_key, evaluations in grouped.items():
        policy_stats = []
        workload_buckets: set[str] = set()
        for evaluation in evaluations:
            records = scope_to_records[(scope_key, evaluation.policy.policy_id)]
            policy_stats.append(_policy_scope_stats(evaluation.policy.policy_id, records))
            workload_buckets.update(_workload_buckets(records))
        recommendations.append(
            _policy_guidance_from_stats(
                scope_kind=scope_kind,
                scope_key=scope_key,
                policy_stats=policy_stats,
                workload_buckets=sorted(workload_buckets),
            )
        )
    return recommendations


def _group_simulation_records(
    *,
    evaluation: CounterfactualSimulationArtifact,
    scope_kind: RecommendationScopeKind,
) -> dict[str, list[CounterfactualSimulationRecord]]:
    grouped: dict[str, list[CounterfactualSimulationRecord]] = defaultdict(list)
    for record in evaluation.records:
        if scope_kind is RecommendationScopeKind.MODEL_ALIAS:
            scope_key = record.model_alias or "unknown"
        else:
            scope_key = record.request_class.value
        grouped[scope_key].append(record)
    return dict(grouped)


def _policy_scope_stats(
    policy_id: str,
    records: Sequence[CounterfactualSimulationRecord],
) -> PolicyScopeStats:
    sample_size = len(records)
    supported_records = 0
    low_confidence = 0
    guardrail_blocks = 0
    changed = 0
    projected_latencies: list[float] = []
    counterexamples: list[str] = []
    for record in records:
        recommendation = record.recommendation
        if recommendation.guardrail_blocked:
            guardrail_blocks += 1
        if recommendation.recommendation_changed:
            changed += 1
        if recommendation.evidence_kind in {
            SimulationEvidenceKind.DIRECT_OBSERVATION,
            SimulationEvidenceKind.PREDICTOR_ESTIMATE,
        }:
            supported_records += 1
        if recommendation.evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE:
            low_confidence += 1
        best_candidate = next(
            (
                candidate
                for candidate in record.candidate_scores
                if candidate.backend_name == recommendation.recommended_backend
            ),
            None,
        )
        if best_candidate is not None and best_candidate.estimate is not None:
            latency = best_candidate.estimate.expected_latency_ms
            if latency is not None:
                projected_latencies.append(latency)
        if recommendation.guardrail_blocked or recommendation.insufficient_data:
            counterexamples.append(record.request_id)
    avg_latency = (
        None if not projected_latencies else sum(projected_latencies) / len(projected_latencies)
    )
    return PolicyScopeStats(
        policy_id=policy_id,
        sample_size=sample_size,
        supported_records=supported_records,
        low_confidence=low_confidence,
        guardrail_blocks=guardrail_blocks,
        changed=changed,
        avg_latency=avg_latency,
        counterexamples=counterexamples[:3],
    )


def _policy_guidance_from_stats(
    *,
    scope_kind: RecommendationScopeKind,
    scope_key: str,
    policy_stats: Sequence[PolicyScopeStats],
    workload_buckets: Sequence[str],
) -> RoutingPolicyGuidance:
    total_sample_size = sum(item.sample_size for item in policy_stats)
    if total_sample_size < _MIN_SCOPE_SAMPLE_SIZE:
        return RoutingPolicyGuidance(
            scope_kind=scope_kind,
            scope_key=scope_key,
            recommendation=RecommendationDisposition.INSUFFICIENT_EVIDENCE,
            confidence=RecommendationConfidence.INSUFFICIENT,
            sample_size=total_sample_size,
            workload_buckets=list(workload_buckets),
            evidence_summary=["simulation sample size was too small for a scoped recommendation"],
            caveats=["keep the baseline policy until more authoritative evidence accumulates"],
            confidence_notes=[f"minimum scoped sample size is {_MIN_SCOPE_SAMPLE_SIZE}"],
        )
    ranked = sorted(
        policy_stats,
        key=lambda item: (
            -item.supported_records,
            float("inf") if item.avg_latency is None else item.avg_latency,
            item.guardrail_blocks,
        ),
    )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    confidence = _policy_confidence(best=best, sample_size=total_sample_size)
    evidence_summary = [
        f"supported requests={best.supported_records}/{best.sample_size}",
        f"changed requests={best.changed}",
        f"guardrail blocks={best.guardrail_blocks}",
    ]
    if best.avg_latency is not None:
        evidence_summary.append(f"projected avg latency={best.avg_latency:.1f}ms")
    if best.guardrail_blocks > max(1, best.sample_size // 3):
        return RoutingPolicyGuidance(
            scope_kind=scope_kind,
            scope_key=scope_key,
            recommendation=RecommendationDisposition.KEEP_SHADOW_ONLY,
            confidence=confidence,
            sample_size=total_sample_size,
            workload_buckets=list(workload_buckets),
            recommended_policy_id=best.policy_id,
            evidence_summary=evidence_summary,
            caveats=["guardrail blocks remain high for this scope"],
            confidence_notes=["keep the candidate policy in shadow or report-only mode"],
            notable_regressions=_runner_up_note(runner_up),
            counterexamples=list(best.counterexamples),
        )
    if runner_up is not None and (
        best.avg_latency is None
        or runner_up.avg_latency is None
        or (runner_up.avg_latency - best.avg_latency) < _MIN_LATENCY_IMPROVEMENT_MS
    ):
        return RoutingPolicyGuidance(
            scope_kind=scope_kind,
            scope_key=scope_key,
            recommendation=RecommendationDisposition.NO_CHANGE,
            confidence=RecommendationConfidence.LOW,
            sample_size=total_sample_size,
            workload_buckets=list(workload_buckets),
            evidence_summary=evidence_summary,
            caveats=["top policies were too close to justify a scoped change"],
            confidence_notes=["latency margin did not clear the recommendation threshold"],
            notable_regressions=_runner_up_note(runner_up),
            counterexamples=list(best.counterexamples),
        )
    return RoutingPolicyGuidance(
        scope_kind=scope_kind,
        scope_key=scope_key,
        recommendation=RecommendationDisposition.PREFER_POLICY,
        confidence=confidence,
        sample_size=total_sample_size,
        workload_buckets=list(workload_buckets),
        recommended_policy_id=best.policy_id,
        evidence_summary=evidence_summary,
        caveats=[],
        confidence_notes=["recommendation is derived from authoritative offline simulation"],
        notable_regressions=_runner_up_note(runner_up),
        counterexamples=list(best.counterexamples),
    )


def _single_simulation_guidance(
    *,
    artifact: CounterfactualSimulationArtifact,
) -> RoutingPolicyGuidance:
    workload_buckets = sorted(
        {
            record.request_features.input_length_bucket.value
            for record in artifact.records
            if record.request_features is not None
        }
    )
    recommendation = RecommendationDisposition.PREFER_POLICY
    confidence = RecommendationConfidence.MEDIUM
    caveats: list[str] = []
    if artifact.summary.insufficient_data_count > 0:
        recommendation = RecommendationDisposition.KEEP_SHADOW_ONLY
        confidence = RecommendationConfidence.LOW
        caveats.append("some requests remained below the evidence threshold")
    if artifact.summary.request_count < _MIN_SCOPE_SAMPLE_SIZE:
        recommendation = RecommendationDisposition.INSUFFICIENT_EVIDENCE
        confidence = RecommendationConfidence.INSUFFICIENT
        caveats.append("too few requests were simulated to recommend activation")
    return RoutingPolicyGuidance(
        scope_kind=RecommendationScopeKind.MODEL_ALIAS,
        scope_key=artifact.records[0].model_alias or "unknown" if artifact.records else "unknown",
        recommendation=recommendation,
        confidence=confidence,
        sample_size=artifact.summary.request_count,
        workload_buckets=workload_buckets,
        recommended_policy_id=artifact.policy.policy_id,
        evidence_summary=[
            f"request count={artifact.summary.request_count}",
            f"changed requests={artifact.summary.changed_count}",
            f"guardrail blocks={artifact.summary.guardrail_block_count}",
        ],
        caveats=caveats,
        confidence_notes=list(artifact.summary.limitation_notes[:2]),
    )


def _repeated_prefix_backend_recommendations(
    *,
    benchmark_artifacts: Sequence[BenchmarkRunArtifact],
) -> list[RoutingPolicyGuidance]:
    repeated_prefix_records = [
        record
        for artifact in benchmark_artifacts
        for record in artifact.records
        if record.request_features is not None
        and WorkloadTag.REPEATED_PREFIX in record.request_features.workload_tags
    ]
    if not repeated_prefix_records:
        return []
    by_backend: dict[str, list[BenchmarkRequestRecord]] = defaultdict(list)
    by_instance: dict[str, list[BenchmarkRequestRecord]] = defaultdict(list)
    for record in repeated_prefix_records:
        by_backend[record.backend_name].append(record)
        if record.backend_instance_id is not None:
            by_instance[record.backend_instance_id].append(record)
    recommendations = []
    recommendations.extend(
        _backend_strength_guidance(
            grouped_records=by_backend,
            scope_kind=RecommendationScopeKind.REPEATED_PREFIX_BACKEND,
            target_type="backend",
        )
    )
    recommendations.extend(
        _backend_strength_guidance(
            grouped_records=by_instance,
            scope_kind=RecommendationScopeKind.REPEATED_PREFIX_INSTANCE,
            target_type="instance",
        )
    )
    return recommendations


def _backend_strength_guidance(
    *,
    grouped_records: dict[str, list[BenchmarkRequestRecord]],
    scope_kind: RecommendationScopeKind,
    target_type: str,
) -> list[RoutingPolicyGuidance]:
    if not grouped_records:
        return []
    stats = []
    for key, records in grouped_records.items():
        sample_size = len(records)
        success_rate = sum(1 for record in records if record.success) / sample_size
        avg_latency = sum(record.latency_ms for record in records) / sample_size
        stats.append((key, sample_size, success_rate, avg_latency))
    stats.sort(key=lambda item: (-item[1], -item[2], item[3], item[0]))
    strong = stats[0]
    weak = sorted(stats, key=lambda item: (item[2], -item[3], -item[1], item[0]))[0]
    recommendations: list[RoutingPolicyGuidance] = []
    if strong[1] >= _MIN_REPEATED_PREFIX_SAMPLE_SIZE:
        recommendations.append(
            RoutingPolicyGuidance(
                scope_kind=scope_kind,
                scope_key="repeated_prefix",
                recommendation=RecommendationDisposition.PREFER_BACKEND,
                confidence=_repeated_prefix_confidence(strong[1], strong[2]),
                sample_size=strong[1],
                workload_buckets=["repeated_prefix"],
                recommended_target=strong[0],
                recommended_target_type=target_type,
                evidence_summary=[
                    f"success rate={strong[2]:.2f}",
                    f"avg latency={strong[3]:.1f}ms",
                ],
                confidence_notes=[
                    "recommendation is based on authoritative repeated-prefix benchmark records"
                ],
            )
        )
    else:
        recommendations.append(
            RoutingPolicyGuidance(
                scope_kind=scope_kind,
                scope_key="repeated_prefix",
                recommendation=RecommendationDisposition.INSUFFICIENT_EVIDENCE,
                confidence=RecommendationConfidence.INSUFFICIENT,
                sample_size=strong[1],
                workload_buckets=["repeated_prefix"],
                evidence_summary=["not enough repeated-prefix samples to identify a strong target"],
                confidence_notes=[
                    f"minimum repeated-prefix sample size is {_MIN_REPEATED_PREFIX_SAMPLE_SIZE}"
                ],
            )
        )
    if weak[1] >= _MIN_REPEATED_PREFIX_SAMPLE_SIZE and len(stats) > 1:
        recommendations.append(
            RoutingPolicyGuidance(
                scope_kind=scope_kind,
                scope_key="repeated_prefix",
                recommendation=RecommendationDisposition.AVOID_BACKEND,
                confidence=_repeated_prefix_confidence(weak[1], 1.0 - weak[2]),
                sample_size=weak[1],
                workload_buckets=["repeated_prefix"],
                recommended_target=weak[0],
                recommended_target_type=target_type,
                evidence_summary=[
                    f"success rate={weak[2]:.2f}",
                    f"avg latency={weak[3]:.1f}ms",
                ],
                caveats=["treat this as a repeated-prefix weakness signal, not a global ban"],
            )
        )
    return recommendations


def _build_evidence_window(
    *,
    benchmark_artifacts: Sequence[BenchmarkRunArtifact],
    simulation_artifacts: Sequence[CounterfactualSimulationArtifact],
    comparison_artifacts: Sequence[CounterfactualSimulationComparisonArtifact],
) -> RecommendationEvidenceWindow:
    timestamps = [artifact.timestamp for artifact in benchmark_artifacts]
    timestamps.extend(artifact.timestamp for artifact in simulation_artifacts)
    timestamps.extend(artifact.timestamp for artifact in comparison_artifacts)
    source_run_ids = sorted(
        {
            run_id
            for artifact in comparison_artifacts
            for run_id in artifact.source_run_ids
        }
        | {artifact.run_id for artifact in benchmark_artifacts}
        | {
            run_id
            for artifact in simulation_artifacts
            for run_id in artifact.source_run_ids
        }
    )
    source_trace_ids = sorted(
        {
            trace_id
            for artifact in comparison_artifacts
            for trace_id in artifact.source_trace_ids
        }
        | {
            trace_id
            for artifact in simulation_artifacts
            for trace_id in artifact.source_trace_ids
        }
    )
    historical_run_ids = sorted(
        {
            run_id
            for artifact in comparison_artifacts
            for run_id in artifact.historical_source_run_ids
        }
        | {
            run_id
            for artifact in simulation_artifacts
            for run_id in artifact.historical_source_run_ids
        }
    )
    historical_trace_ids = sorted(
        {
            trace_id
            for artifact in comparison_artifacts
            for trace_id in artifact.historical_source_trace_ids
        }
        | {
            trace_id
            for artifact in simulation_artifacts
            for trace_id in artifact.historical_source_trace_ids
        }
    )
    return RecommendationEvidenceWindow(
        source_run_ids=source_run_ids,
        source_trace_ids=source_trace_ids,
        historical_source_run_ids=historical_run_ids,
        historical_source_trace_ids=historical_trace_ids,
        window_started_at=None if not timestamps else min(timestamps),
        window_ended_at=None if not timestamps else max(timestamps),
    )


def _collect_limitations(
    *,
    benchmark_artifacts: Sequence[BenchmarkRunArtifact],
    simulation_artifacts: Sequence[CounterfactualSimulationArtifact],
    comparison_artifacts: Sequence[CounterfactualSimulationComparisonArtifact],
    recommendations: Sequence[RoutingPolicyGuidance],
) -> list[str]:
    limitations = [
        (
            "recommendations are derived from benchmark and offline simulation "
            "artifacts, not live runtime logs"
        ),
        (
            "offline comparisons are not counterfactual ground truth and remain "
            "bounded by observed evidence quality"
        ),
    ]
    if not comparison_artifacts and not simulation_artifacts:
        limitations.append("no simulation artifacts were provided, so policy guidance is limited")
    if not benchmark_artifacts:
        limitations.append(
            "no benchmark run artifacts were provided, so repeated-prefix backend "
            "guidance is unavailable"
        )
    if all(
        recommendation.recommendation
        in {
            RecommendationDisposition.NO_CHANGE,
            RecommendationDisposition.INSUFFICIENT_EVIDENCE,
            RecommendationDisposition.KEEP_SHADOW_ONLY,
        }
        for recommendation in recommendations
    ):
        limitations.append("available evidence did not justify an active policy promotion")
    return limitations


def _policy_confidence(*, best: PolicyScopeStats, sample_size: int) -> RecommendationConfidence:
    supported = best.supported_records
    low_confidence = best.low_confidence
    if sample_size < _MIN_SCOPE_SAMPLE_SIZE or supported < _MIN_SCOPE_SAMPLE_SIZE:
        return RecommendationConfidence.INSUFFICIENT
    if low_confidence > max(1, sample_size // 3):
        return RecommendationConfidence.LOW
    if sample_size >= 20:
        return RecommendationConfidence.HIGH
    return RecommendationConfidence.MEDIUM


def _repeated_prefix_confidence(
    sample_size: int,
    signal_strength: float,
) -> RecommendationConfidence:
    if sample_size < _MIN_REPEATED_PREFIX_SAMPLE_SIZE:
        return RecommendationConfidence.INSUFFICIENT
    if sample_size >= 8 and signal_strength >= 0.75:
        return RecommendationConfidence.HIGH
    if signal_strength >= 0.5:
        return RecommendationConfidence.MEDIUM
    return RecommendationConfidence.LOW


def _runner_up_note(runner_up: PolicyScopeStats | None) -> list[str]:
    if runner_up is None:
        return []
    details = [f"runner-up policy={runner_up.policy_id}"]
    if runner_up.avg_latency is not None:
        details.append(f"avg latency={runner_up.avg_latency:.1f}ms")
    details.append(f"guardrail blocks={runner_up.guardrail_blocks}")
    return ["; ".join(details)]


def _workload_buckets(records: Iterable[CounterfactualSimulationRecord]) -> list[str]:
    buckets = {
        record.request_features.input_length_bucket.value
        for record in records
        if record.request_features is not None
    }
    request_classes = {
        record.request_class.value
        for record in records
        if record.request_class is not RequestClass.STANDARD
    }
    return sorted(buckets | request_classes)
