from __future__ import annotations

from datetime import UTC, datetime, timedelta

from switchyard.bench.simulation import (
    compare_candidate_policies_offline,
    recommend_policy_from_simulation,
    simulate_policy_counterfactual,
)
from switchyard.schemas.benchmark import (
    AdaptivePolicyGuardrails,
    AdaptivePolicyMode,
    BenchmarkEnvironmentMetadata,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
    CapturedTraceRecord,
    CounterfactualObjective,
    ExecutionTarget,
    ExecutionTargetType,
    ExplainablePolicySpec,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import (
    InputLengthBucket,
    PolicyReference,
    PrefixHotness,
    PrefixLocalitySignal,
    RequestFeatureVector,
    RouteDecision,
    RoutingPolicy,
    WorkloadShape,
    WorkloadTag,
)


def _build_record(
    *,
    request_id: str,
    backend_name: str,
    latency_ms: float,
    success: bool = True,
    considered_backends: list[str] | None = None,
    tokens_per_second: float = 120.0,
) -> BenchmarkRequestRecord:
    started_at = datetime(2026, 3, 17, tzinfo=UTC)
    features = RequestFeatureVector(
        message_count=1,
        user_message_count=1,
        prompt_character_count=128,
        prompt_token_estimate=24,
        max_output_tokens=64,
        expected_total_tokens=88,
        input_length_bucket=InputLengthBucket.SHORT,
        workload_tags=[WorkloadTag.REPEATED_PREFIX],
        repeated_prefix_candidate=True,
        prefix_character_count=48,
        prefix_fingerprint="feedfacecafebeef",
        locality_key="00112233445566778899",
    )
    signal = PrefixLocalitySignal(
        serving_target="chat-shared",
        locality_key="00112233445566778899",
        prefix_fingerprint="feedfacecafebeef",
        repeated_prefix_detected=True,
        recent_request_count=4,
        hotness=PrefixHotness.HOT,
        cache_opportunity=True,
        likely_benefits_from_locality=True,
        preferred_backend="mock-fast",
        preferred_backend_request_count=3,
        candidate_local_backend="mock-fast",
        candidate_local_backend_request_count=3,
        recent_backend_counts={"mock-fast": 3},
    )
    return BenchmarkRequestRecord(
        request_id=request_id,
        backend_name=backend_name,
        backend_type="mock",
        model_alias="chat-shared",
        model_identifier="chat-shared",
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=int(latency_ms)),
        latency_ms=latency_ms,
        ttft_ms=latency_ms / 2.0,
        output_tokens=24,
        tokens_per_second=tokens_per_second,
        route_decision=RouteDecision(
            backend_name=backend_name,
            serving_target="chat-shared",
            policy=RoutingPolicy.BALANCED,
            request_id=request_id,
            workload_shape=WorkloadShape.INTERACTIVE,
            rationale=["observed route"],
            considered_backends=considered_backends or [backend_name],
            request_features=features,
            prefix_locality_signal=signal,
        ),
        success=success,
        status_code=200 if success else 503,
        usage=UsageStats(prompt_tokens=24, completion_tokens=24, total_tokens=48),
        error=None if success else "failed",
        error_category=None if success else "runtime_error",
    )


def _build_artifact(*, run_id: str, records: list[BenchmarkRequestRecord]) -> BenchmarkRunArtifact:
    return BenchmarkRunArtifact(
        run_id=run_id,
        scenario=BenchmarkScenario(
            name="simulation",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=len(records),
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=sorted({record.backend_name for record in records}),
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=len(records),
        summary=BenchmarkSummary(
            request_count=len(records),
            success_count=sum(1 for record in records if record.success),
            failure_count=sum(1 for record in records if not record.success),
            avg_latency_ms=sum(record.latency_ms for record in records) / len(records),
            p50_latency_ms=records[0].latency_ms,
            p95_latency_ms=max(record.latency_ms for record in records),
            avg_ttft_ms=records[0].ttft_ms,
            p50_ttft_ms=records[0].ttft_ms,
            p95_ttft_ms=max(record.ttft_ms or 0.0 for record in records),
            total_output_tokens=sum(record.output_tokens or 0 for record in records),
            avg_output_tokens=sum(record.output_tokens or 0 for record in records) / len(records),
            avg_tokens_per_second=120.0,
            p95_tokens_per_second=120.0,
            fallback_count=0,
            chosen_backend_counts={
                backend: sum(1 for record in records if record.backend_name == backend)
                for backend in sorted({record.backend_name for record in records})
            },
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="synthetic"),
        records=records,
    )


def _build_trace(
    *,
    record_id: str,
    request_id: str,
    chosen_backend: str,
    latency_ms: float | None,
    status_code: int | None,
    considered_backends: list[str] | None = None,
) -> CapturedTraceRecord:
    features = RequestFeatureVector(
        message_count=1,
        user_message_count=1,
        prompt_character_count=96,
        prompt_token_estimate=18,
        max_output_tokens=64,
        expected_total_tokens=82,
        input_length_bucket=InputLengthBucket.SHORT,
        repeated_prefix_candidate=True,
        prefix_character_count=32,
        prefix_fingerprint="deadbeefcafefeed",
        locality_key="11223344556677889900",
    )
    return CapturedTraceRecord(
        record_id=record_id,
        request_id=request_id,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.LOGICAL_ALIAS,
            model_alias="chat-shared",
        ),
        logical_alias="chat-shared",
        request_features=features,
        policy_reference=PolicyReference(policy_id="balanced", policy_version="phase6.v1"),
        route_decision=RouteDecision(
            backend_name=chosen_backend,
            serving_target="chat-shared",
            policy=RoutingPolicy.BALANCED,
            request_id=request_id,
            workload_shape=WorkloadShape.INTERACTIVE,
            rationale=["trace route"],
            considered_backends=considered_backends or [chosen_backend],
            request_features=features,
        ),
        chosen_backend=chosen_backend,
        latency_ms=latency_ms,
        status_code=status_code,
        error=None if status_code is None or status_code < 400 else "trace failure",
    )


def test_simulate_policy_counterfactual_recommends_faster_backend() -> None:
    history_artifact = _build_artifact(
        run_id="history-run",
        records=[
            _build_record(
                request_id="hist-fast-1",
                backend_name="mock-fast",
                latency_ms=12.0,
                considered_backends=["mock-slow", "mock-fast"],
            ),
            _build_record(
                request_id="hist-fast-2",
                backend_name="mock-fast",
                latency_ms=14.0,
                considered_backends=["mock-slow", "mock-fast"],
            ),
            _build_record(
                request_id="hist-fast-3",
                backend_name="mock-fast",
                latency_ms=16.0,
                considered_backends=["mock-slow", "mock-fast"],
            ),
            _build_record(
                request_id="hist-slow-1",
                backend_name="mock-slow",
                latency_ms=70.0,
                considered_backends=["mock-slow", "mock-fast"],
            ),
            _build_record(
                request_id="hist-slow-2",
                backend_name="mock-slow",
                latency_ms=80.0,
                considered_backends=["mock-slow", "mock-fast"],
            ),
            _build_record(
                request_id="hist-slow-3",
                backend_name="mock-slow",
                latency_ms=90.0,
                considered_backends=["mock-slow", "mock-fast"],
            ),
        ],
    )
    evaluation_artifact = _build_artifact(
        run_id="eval-run",
        records=[
            _build_record(
                request_id="eval-1",
                backend_name="mock-slow",
                latency_ms=85.0,
                considered_backends=["mock-slow", "mock-fast"],
            )
        ],
    )
    policy = ExplainablePolicySpec(
        policy_id="adaptive-latency-v1",
        objective=CounterfactualObjective.LATENCY,
        mode=AdaptivePolicyMode.RECOMMEND,
        min_evidence_count=3,
        guardrails=AdaptivePolicyGuardrails(require_sufficient_data=True),
    )

    artifact = simulate_policy_counterfactual(
        evaluation_artifacts=[evaluation_artifact],
        history_artifacts=[history_artifact],
        policy=policy,
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    assert artifact.summary.request_count == 1
    assert artifact.summary.changed_count == 1
    assert artifact.records[0].recommendation.recommended_backend == "mock-fast"
    assert artifact.records[0].candidate_scores[0].estimate is not None
    assert recommend_policy_from_simulation(artifact) == (
        "Consider a bounded rollout. The simulated policy materially changes routing while "
        "still respecting the configured guardrails."
    )


def test_simulate_policy_counterfactual_respects_error_guardrail() -> None:
    history_artifact = _build_artifact(
        run_id="history-guardrail",
        records=[
            _build_record(
                request_id="hist-safe-1",
                backend_name="mock-safe",
                latency_ms=40.0,
                considered_backends=["mock-safe", "mock-risky"],
                tokens_per_second=100.0,
            ),
            _build_record(
                request_id="hist-safe-2",
                backend_name="mock-safe",
                latency_ms=42.0,
                considered_backends=["mock-safe", "mock-risky"],
                tokens_per_second=100.0,
            ),
            _build_record(
                request_id="hist-safe-3",
                backend_name="mock-safe",
                latency_ms=44.0,
                considered_backends=["mock-safe", "mock-risky"],
                tokens_per_second=100.0,
            ),
            _build_record(
                request_id="hist-risky-1",
                backend_name="mock-risky",
                latency_ms=10.0,
                success=False,
                considered_backends=["mock-safe", "mock-risky"],
                tokens_per_second=500.0,
            ),
            _build_record(
                request_id="hist-risky-2",
                backend_name="mock-risky",
                latency_ms=12.0,
                success=False,
                considered_backends=["mock-safe", "mock-risky"],
                tokens_per_second=500.0,
            ),
            _build_record(
                request_id="hist-risky-3",
                backend_name="mock-risky",
                latency_ms=14.0,
                success=True,
                considered_backends=["mock-safe", "mock-risky"],
                tokens_per_second=500.0,
            ),
        ],
    )
    evaluation_artifact = _build_artifact(
        run_id="eval-guardrail",
        records=[
            _build_record(
                request_id="eval-safe",
                backend_name="mock-safe",
                latency_ms=45.0,
                considered_backends=["mock-safe", "mock-risky"],
            )
        ],
    )
    policy = ExplainablePolicySpec(
        policy_id="adaptive-latency-guarded",
        objective=CounterfactualObjective.THROUGHPUT,
        min_evidence_count=3,
        guardrails=AdaptivePolicyGuardrails(
            require_sufficient_data=True,
            max_predicted_error_rate=0.2,
        ),
    )

    artifact = simulate_policy_counterfactual(
        evaluation_artifacts=[evaluation_artifact],
        history_artifacts=[history_artifact],
        policy=policy,
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    recommendation = artifact.records[0].recommendation
    assert recommendation.recommended_backend == "mock-safe"
    assert recommendation.guardrail_blocked is True
    assert artifact.summary.guardrail_block_count == 1


def test_compare_candidate_policies_offline_marks_low_confidence_and_unsupported() -> None:
    history_artifact = _build_artifact(
        run_id="history-low-confidence",
        records=[
            _build_record(
                request_id="hist-a-1",
                backend_name="mock-a",
                latency_ms=20.0,
                considered_backends=["mock-a", "mock-b", "mock-c"],
            ),
            _build_record(
                request_id="hist-b-1",
                backend_name="mock-b",
                latency_ms=10.0,
                considered_backends=["mock-a", "mock-b", "mock-c"],
                tokens_per_second=300.0,
            ),
        ],
    )
    evaluation_trace = _build_trace(
        record_id="trace-1",
        request_id="trace-req-1",
        chosen_backend="mock-a",
        latency_ms=25.0,
        status_code=200,
        considered_backends=["mock-a", "mock-b", "mock-c"],
    )
    policies = [
        ExplainablePolicySpec(
            policy_id="guarded-balanced",
            objective=CounterfactualObjective.BALANCED,
            min_evidence_count=3,
            guardrails=AdaptivePolicyGuardrails(require_sufficient_data=True),
        ),
        ExplainablePolicySpec(
            policy_id="explore-throughput",
            objective=CounterfactualObjective.THROUGHPUT,
            min_evidence_count=3,
            guardrails=AdaptivePolicyGuardrails(require_sufficient_data=False),
        ),
    ]

    comparison = compare_candidate_policies_offline(
        policies=policies,
        evaluation_trace_records=[evaluation_trace],
        history_artifacts=[history_artifact],
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    assert comparison.source_trace_ids == ["trace-1"]
    assert len(comparison.evaluations) == 2
    guarded = comparison.evaluations[0]
    exploring = comparison.evaluations[1]
    assert guarded.summary.unsupported_count == 0
    assert guarded.summary.low_confidence_count == 0
    assert exploring.records[0].candidate_scores[1].evidence_kind.value == "low_confidence_estimate"
    assert "low-confidence predictor estimates" in comparison.limitation_notes[0]
    assert exploring.records[0].candidate_scores[-1].evidence_kind.value == "unsupported"


def test_compare_candidate_policies_offline_keeps_trace_limitations_visible() -> None:
    trace = _build_trace(
        record_id="trace-unsupported",
        request_id="trace-unsupported-req",
        chosen_backend="mock-a",
        latency_ms=None,
        status_code=None,
        considered_backends=["mock-a", "mock-b"],
    )
    policy = ExplainablePolicySpec(
        policy_id="unsupported-check",
        objective=CounterfactualObjective.BALANCED,
        min_evidence_count=3,
        guardrails=AdaptivePolicyGuardrails(require_sufficient_data=True),
    )

    comparison = compare_candidate_policies_offline(
        policies=[policy],
        evaluation_trace_records=[trace],
        history_artifacts=[],
        history_trace_records=[],
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    evaluation = comparison.evaluations[0]
    assert evaluation.summary.unsupported_count == 1
    assert evaluation.summary.limitation_notes == [
        "1 requests remained unsupported by the available evidence"
    ]
