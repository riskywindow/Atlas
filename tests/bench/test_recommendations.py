from __future__ import annotations

from datetime import UTC, datetime, timedelta

from switchyard.bench.recommendations import (
    build_policy_recommendation_report,
    render_policy_recommendation_report_markdown,
)
from switchyard.schemas.benchmark import (
    BenchmarkEnvironmentMetadata,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
    CandidateRouteEstimateContext,
    CounterfactualCandidateScore,
    CounterfactualObjective,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    CounterfactualSimulationRecord,
    CounterfactualSimulationSummary,
    ExplainablePolicySpec,
    HistoricalRouteEstimate,
    PolicyRecommendation,
    SimulationEvidenceKind,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import (
    InputLengthBucket,
    RequestClass,
    RequestFeatureVector,
    RoutingPolicy,
    WorkloadShape,
    WorkloadTag,
)


def _request_features(*, repeated_prefix: bool) -> RequestFeatureVector:
    tags = [WorkloadTag.SHORT_CHAT]
    if repeated_prefix:
        tags.append(WorkloadTag.REPEATED_PREFIX)
    return RequestFeatureVector(
        message_count=1,
        user_message_count=1,
        prompt_character_count=64,
        prompt_token_estimate=12,
        max_output_tokens=32,
        expected_total_tokens=44,
        input_length_bucket=InputLengthBucket.SHORT,
        workload_tags=tags,
        repeated_prefix_candidate=repeated_prefix,
        prefix_character_count=16 if repeated_prefix else 0,
        prefix_fingerprint="feedfacecafebeef" if repeated_prefix else None,
        locality_key="00112233445566778899",
    )


def _simulation_record(
    *,
    request_id: str,
    policy_backend: str,
    projected_latency_ms: float,
    request_class: RequestClass = RequestClass.STANDARD,
    evidence_kind: SimulationEvidenceKind = SimulationEvidenceKind.PREDICTOR_ESTIMATE,
    guardrail_blocked: bool = False,
    insufficient_data: bool = False,
) -> CounterfactualSimulationRecord:
    estimate = HistoricalRouteEstimate(
        context=CandidateRouteEstimateContext(
            model_alias="chat-shared",
            backend_name=policy_backend,
            request_class=request_class,
            input_length_bucket=InputLengthBucket.SHORT,
            workload_tags=[WorkloadTag.SHORT_CHAT],
        ),
        evidence_count=12,
        sufficient_data=not insufficient_data,
        expected_latency_ms=projected_latency_ms,
        expected_error_rate=0.01,
        rationale=["test estimate"],
    )
    return CounterfactualSimulationRecord(
        request_id=request_id,
        model_alias="chat-shared",
        request_class=request_class,
        request_features=_request_features(repeated_prefix=False),
        observed_backend="mock-observed",
        observed_latency_ms=45.0,
        observed_success=True,
        candidate_scores=[
            CounterfactualCandidateScore(
                backend_name=policy_backend,
                score=-projected_latency_ms,
                eligible=True,
                evidence_kind=evidence_kind,
                evidence_count=12,
                estimate=estimate,
                confidence_note="sufficient historical evidence",
                rationale=["candidate selected in test fixture"],
            )
        ],
        recommendation=PolicyRecommendation(
            observed_backend="mock-observed",
            recommended_backend=policy_backend,
            recommendation_changed=policy_backend != "mock-observed",
            guardrail_blocked=guardrail_blocked,
            insufficient_data=insufficient_data,
            evidence_kind=evidence_kind,
            rationale=["fixture recommendation"],
        ),
    )


def _simulation_artifact(
    *,
    policy_id: str,
    latencies: list[float],
    request_class: RequestClass = RequestClass.STANDARD,
    guardrail_blocked: bool = False,
) -> CounterfactualSimulationArtifact:
    records = [
        _simulation_record(
            request_id=f"{policy_id}-{index}",
            policy_backend=f"{policy_id}-backend",
            projected_latency_ms=latency,
            request_class=request_class,
            guardrail_blocked=guardrail_blocked,
        )
        for index, latency in enumerate(latencies)
    ]
    return CounterfactualSimulationArtifact(
        simulation_id=f"{policy_id}-simulation",
        source_run_ids=["run-a"],
        policy=ExplainablePolicySpec(
            policy_id=policy_id,
            objective=CounterfactualObjective.LATENCY,
        ),
        summary=CounterfactualSimulationSummary(
            request_count=len(records),
            changed_count=len(records),
            unchanged_count=0,
            direct_observation_count=0,
            predictor_estimate_count=len(records),
            low_confidence_count=0,
            unsupported_count=0,
            insufficient_data_count=0,
            guardrail_block_count=len(records) if guardrail_blocked else 0,
            projected_avg_latency_ms=sum(latencies) / len(latencies),
            observed_backend_counts={"mock-observed": len(records)},
            recommended_backend_counts={f"{policy_id}-backend": len(records)},
        ),
        records=records,
    )


def _benchmark_artifact() -> BenchmarkRunArtifact:
    started_at = datetime(2026, 3, 17, tzinfo=UTC)
    records = []
    for index in range(4):
        records.append(
            BenchmarkRequestRecord(
                request_id=f"prefix-strong-{index}",
                backend_name="mlx-hot",
                backend_type="mock",
                model_alias="chat-shared",
                model_identifier="chat-shared",
                backend_instance_id="mlx-hot-01",
                started_at=started_at,
                completed_at=started_at + timedelta(milliseconds=10),
                latency_ms=10.0,
                output_tokens=16,
                tokens_per_second=160.0,
                request_features=_request_features(repeated_prefix=True),
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=12, completion_tokens=16, total_tokens=28),
            )
        )
    for index in range(3):
        records.append(
            BenchmarkRequestRecord(
                request_id=f"prefix-weak-{index}",
                backend_name="vllm-cold",
                backend_type="mock",
                model_alias="chat-shared",
                model_identifier="chat-shared",
                backend_instance_id="vllm-cold-01",
                started_at=started_at,
                completed_at=started_at + timedelta(milliseconds=45),
                latency_ms=45.0,
                output_tokens=16,
                tokens_per_second=35.0,
                request_features=_request_features(repeated_prefix=True),
                success=index != 2,
                status_code=200 if index != 2 else 503,
                usage=UsageStats(prompt_tokens=12, completion_tokens=16, total_tokens=28),
                error=None if index != 2 else "failed",
                error_category=None if index != 2 else "runtime_error",
            )
        )
    return BenchmarkRunArtifact(
        run_id="bench-prefix",
        timestamp=started_at,
        scenario=BenchmarkScenario(
            name="repeated-prefix",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=len(records),
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mlx-hot", "vllm-cold"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=len(records),
        summary=BenchmarkSummary(
            request_count=len(records),
            success_count=6,
            failure_count=1,
            avg_latency_ms=25.0,
            p50_latency_ms=10.0,
            p95_latency_ms=45.0,
            avg_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=112,
            avg_output_tokens=16.0,
            avg_tokens_per_second=90.0,
            p95_tokens_per_second=160.0,
            fallback_count=0,
            chosen_backend_counts={"mlx-hot": 4, "vllm-cold": 3},
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="synthetic"),
        records=records,
    )


def test_build_policy_recommendation_report_prefers_policy_and_repeated_prefix_targets() -> None:
    fast = _simulation_artifact(policy_id="adaptive-fast", latencies=[15.0] * 6)
    safe = _simulation_artifact(policy_id="adaptive-safe", latencies=[35.0] * 6)
    comparison = CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id="comparison-guidance",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
        source_run_ids=["run-a"],
        policies=[fast.policy, safe.policy],
        evaluations=[fast, safe],
    )

    report = build_policy_recommendation_report(
        [comparison, _benchmark_artifact()],
        report_id="policy-guidance",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    alias_recommendation = next(
        item for item in report.recommendations if item.scope_kind.value == "model_alias"
    )
    assert alias_recommendation.recommended_policy_id == "adaptive-fast"
    assert alias_recommendation.recommendation.value == "prefer_policy"
    repeated_prefix_prefer = next(
        item
        for item in report.recommendations
        if item.recommendation.value == "prefer_backend"
    )
    assert repeated_prefix_prefer.recommended_target == "mlx-hot"
    repeated_prefix_avoid = next(
        item
        for item in report.recommendations
        if item.recommendation.value == "avoid_backend"
    )
    assert repeated_prefix_avoid.recommended_target == "vllm-cold"

    markdown = render_policy_recommendation_report_markdown(report)
    assert "# Switchyard Policy Recommendation Report:" in markdown
    assert "## Evidence Window" in markdown
    assert "adaptive-fast" in markdown


def test_build_policy_recommendation_report_handles_sparse_data_honestly() -> None:
    sparse = _simulation_artifact(policy_id="adaptive-fast", latencies=[15.0, 16.0])
    comparison = CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id="comparison-sparse",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
        source_run_ids=["run-sparse"],
        policies=[sparse.policy],
        evaluations=[sparse],
    )

    report = build_policy_recommendation_report(
        [comparison],
        report_id="policy-guidance-sparse",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    recommendation = report.recommendations[0]
    assert recommendation.recommendation.value == "insufficient_evidence"
    assert recommendation.confidence.value == "insufficient"
    assert recommendation.sample_size == 2


def test_build_policy_recommendation_report_keeps_guarded_policy_shadow_only_when_blocks_are_high(
) -> None:
    guarded = _simulation_artifact(
        policy_id="adaptive-guarded",
        latencies=[14.0] * 6,
        guardrail_blocked=True,
    )
    slower = _simulation_artifact(policy_id="balanced-offline", latencies=[30.0] * 6)
    comparison = CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id="comparison-guarded",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
        source_run_ids=["run-guarded"],
        policies=[guarded.policy, slower.policy],
        evaluations=[guarded, slower],
    )

    report = build_policy_recommendation_report(
        [comparison],
        report_id="policy-guidance-guarded",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    recommendation = report.recommendations[0]
    assert recommendation.recommendation.value == "keep_shadow_only"
    assert recommendation.recommended_policy_id == "adaptive-guarded"
    assert "guardrail" in recommendation.caveats[0]
