from datetime import UTC, datetime, timedelta

from switchyard.bench.history import (
    TransparentHistoricalRoutePredictor,
    summarize_historical_artifacts,
    summarize_historical_records,
)
from switchyard.schemas.benchmark import (
    BenchmarkEnvironmentMetadata,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
    CandidateRouteEstimateContext,
    HistoricalDimension,
    HistoricalSummaryQuery,
    WorkloadGenerationConfig,
    WorkloadPattern,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import (
    InputLengthBucket,
    PrefixHotness,
    PrefixLocalitySignal,
    RequestClass,
    RequestFeatureVector,
    RouteDecision,
    RoutingPolicy,
    WorkloadShape,
    WorkloadTag,
)


def build_record(
    *,
    request_id: str,
    backend_name: str,
    latency_ms: float,
    success: bool = True,
    status_code: int = 200,
    ttft_ms: float | None = None,
    tokens_per_second: float | None = None,
    queue_delay_ms: float | None = None,
    workload_tags: list[WorkloadTag] | None = None,
    prefix_hotness: PrefixHotness | None = None,
    cache_opportunity: bool | None = None,
    locality_benefit: bool | None = None,
    backend_type: str = "mock",
    model_alias: str = "chat-shared",
    request_class: RequestClass = RequestClass.STANDARD,
    policy: RoutingPolicy = RoutingPolicy.BALANCED,
    input_length_bucket: InputLengthBucket = InputLengthBucket.SHORT,
    started_offset_ms: int = 0,
) -> BenchmarkRequestRecord:
    started_at = datetime(2026, 3, 17, tzinfo=UTC) + timedelta(milliseconds=started_offset_ms)
    features = RequestFeatureVector(
        message_count=1,
        user_message_count=1,
        prompt_character_count=64,
        prompt_token_estimate=10,
        max_output_tokens=128,
        expected_total_tokens=138,
        input_length_bucket=input_length_bucket,
        workload_tags=workload_tags or [],
        repeated_prefix_candidate=cache_opportunity is True,
        prefix_character_count=32 if prefix_hotness is not None else 0,
        prefix_fingerprint="feedfacecafebeef" if prefix_hotness is not None else None,
        locality_key="00112233445566778899",
        request_class=request_class,
    )
    signal = None
    if prefix_hotness is not None:
        signal = PrefixLocalitySignal(
            serving_target=model_alias,
            locality_key="00112233445566778899",
            prefix_fingerprint="feedfacecafebeef",
            repeated_prefix_detected=prefix_hotness is not PrefixHotness.COLD,
            recent_request_count=3 if prefix_hotness is PrefixHotness.HOT else 1,
            hotness=prefix_hotness,
            cache_opportunity=bool(cache_opportunity),
            likely_benefits_from_locality=bool(locality_benefit),
            preferred_backend=backend_name,
            preferred_backend_request_count=2,
            candidate_local_backend=backend_name,
            candidate_local_backend_request_count=2,
            recent_backend_counts={backend_name: 2},
        )
    route_decision = RouteDecision(
        backend_name=backend_name,
        serving_target=model_alias,
        policy=policy,
        request_id=request_id,
        workload_shape=WorkloadShape.INTERACTIVE,
        rationale=["historical sample"],
        considered_backends=[backend_name],
        request_features=features,
        prefix_locality_signal=signal,
    )
    return BenchmarkRequestRecord(
        request_id=request_id,
        backend_name=backend_name,
        backend_type=backend_type,
        model_alias=model_alias,
        model_identifier=model_alias,
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=int(latency_ms)),
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        output_tokens=12 if success else 0,
        tokens_per_second=tokens_per_second,
        queue_delay_ms=queue_delay_ms,
        route_decision=route_decision,
        success=success,
        status_code=status_code,
        usage=UsageStats(prompt_tokens=10, completion_tokens=12, total_tokens=22),
        error=None if success else "failed",
        error_category=None if success else "runtime_error",
    )


def test_historical_summary_groups_records_by_backend_name() -> None:
    records = [
        build_record(
            request_id="req-1",
            backend_name="mock-a",
            latency_ms=10.0,
            ttft_ms=4.0,
            tokens_per_second=120.0,
            queue_delay_ms=2.0,
            workload_tags=[WorkloadTag.SHORT_CHAT],
            prefix_hotness=PrefixHotness.WARM,
            cache_opportunity=True,
            locality_benefit=True,
            started_offset_ms=0,
        ),
        build_record(
            request_id="req-2",
            backend_name="mock-a",
            latency_ms=20.0,
            ttft_ms=5.0,
            tokens_per_second=100.0,
            queue_delay_ms=3.0,
            workload_tags=[WorkloadTag.SHORT_CHAT],
            prefix_hotness=PrefixHotness.WARM,
            cache_opportunity=True,
            locality_benefit=False,
            started_offset_ms=10,
        ),
        build_record(
            request_id="req-3",
            backend_name="mock-b",
            latency_ms=300.0,
            success=False,
            status_code=503,
            workload_tags=[WorkloadTag.LONG_CONTEXT],
            started_offset_ms=20,
        ),
    ]

    index = summarize_historical_records(
        records,
        query=HistoricalSummaryQuery(group_by=[HistoricalDimension.BACKEND_NAME]),
    )

    assert index.matched_record_count == 3
    assert [summary.key.backend_name for summary in index.summaries] == ["mock-a", "mock-b"]
    mock_a = index.summaries[0]
    assert mock_a.request_count == 2
    assert mock_a.error_rate == 0.0
    assert mock_a.latency_ms.avg == 15.0
    assert mock_a.latency_ms.ewma == 13.0
    assert mock_a.cache_opportunity_rate == 1.0
    assert mock_a.locality_benefit_rate == 0.5


def test_historical_summary_expands_workload_tags_and_buckets_metrics() -> None:
    records = [
        build_record(
            request_id="req-tags",
            backend_name="mock-a",
            latency_ms=5.0,
            workload_tags=[WorkloadTag.SHORT_CHAT, WorkloadTag.REPEATED_PREFIX],
            prefix_hotness=PrefixHotness.HOT,
            cache_opportunity=True,
            locality_benefit=True,
        ),
        build_record(
            request_id="req-tags-2",
            backend_name="mock-a",
            latency_ms=300.0,
            workload_tags=[WorkloadTag.REPEATED_PREFIX],
            prefix_hotness=PrefixHotness.HOT,
            cache_opportunity=True,
            locality_benefit=True,
            started_offset_ms=10,
        ),
    ]

    index = summarize_historical_records(
        records,
        query=HistoricalSummaryQuery(group_by=[HistoricalDimension.WORKLOAD_TAG]),
    )

    tags = {summary.key.workload_tag: summary for summary in index.summaries}
    assert tags[WorkloadTag.SHORT_CHAT].request_count == 1
    assert tags[WorkloadTag.REPEATED_PREFIX].request_count == 2
    latency_buckets = {
        bucket.bucket_label: bucket.count
        for bucket in tags[WorkloadTag.REPEATED_PREFIX].latency_ms.buckets
    }
    assert latency_buckets["0-10"] == 1
    assert latency_buckets["250-500"] == 1


def test_historical_artifact_summary_uses_authoritative_records() -> None:
    records = [build_record(request_id="req-1", backend_name="mock-a", latency_ms=10.0)]
    artifact = BenchmarkRunArtifact(
        run_id="run-history-1",
        scenario=BenchmarkScenario(
            name="history",
            model="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=1,
            workload_generation=WorkloadGenerationConfig(pattern=WorkloadPattern.UNIFORM),
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-a"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=10.0,
            p50_latency_ms=10.0,
            p95_latency_ms=10.0,
            avg_ttft_ms=None,
            p50_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=12,
            avg_output_tokens=12.0,
            avg_tokens_per_second=None,
            p95_tokens_per_second=None,
            fallback_count=0,
            chosen_backend_counts={"mock-a": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="synthetic"),
        records=records,
    )

    index = summarize_historical_artifacts(
        [artifact],
        query=HistoricalSummaryQuery(backend_name="mock-a"),
    )

    assert index.source_record_count == 1
    assert index.matched_record_count == 1
    assert index.summaries[0].key.backend_name == "mock-a"


def test_transparent_predictor_reports_insufficient_data_for_sparse_history() -> None:
    predictor = TransparentHistoricalRoutePredictor(
        [build_record(request_id="req-1", backend_name="mock-a", latency_ms=10.0)],
        min_samples=2,
    )

    estimate = predictor.estimate(
        CandidateRouteEstimateContext(
            model_alias="chat-shared",
            backend_name="mock-a",
            policy_id="balanced",
            input_length_bucket=InputLengthBucket.SHORT,
        )
    )

    assert estimate.sufficient_data is False
    assert estimate.evidence_count == 1
    assert estimate.insufficiency_reason is not None


def test_transparent_predictor_uses_best_available_historical_slice() -> None:
    records = [
        build_record(
            request_id="req-1",
            backend_name="mock-a",
            latency_ms=10.0,
            ttft_ms=4.0,
            tokens_per_second=100.0,
            queue_delay_ms=1.0,
            prefix_hotness=PrefixHotness.WARM,
            cache_opportunity=True,
            locality_benefit=True,
            started_offset_ms=0,
        ),
        build_record(
            request_id="req-2",
            backend_name="mock-a",
            latency_ms=20.0,
            ttft_ms=6.0,
            tokens_per_second=90.0,
            queue_delay_ms=2.0,
            prefix_hotness=PrefixHotness.WARM,
            cache_opportunity=True,
            locality_benefit=True,
            started_offset_ms=10,
        ),
        build_record(
            request_id="req-3",
            backend_name="mock-a",
            latency_ms=30.0,
            ttft_ms=8.0,
            tokens_per_second=80.0,
            queue_delay_ms=3.0,
            prefix_hotness=PrefixHotness.WARM,
            cache_opportunity=True,
            locality_benefit=True,
            started_offset_ms=20,
        ),
    ]
    predictor = TransparentHistoricalRoutePredictor(records, min_samples=3)

    estimate = predictor.estimate(
        CandidateRouteEstimateContext(
            model_alias="chat-shared",
            backend_name="mock-a",
            policy_id="balanced",
            request_class=RequestClass.STANDARD,
            input_length_bucket=InputLengthBucket.SHORT,
            workload_tags=[WorkloadTag.REPEATED_PREFIX],
            prefix_hotness=PrefixHotness.WARM,
            cache_opportunity=True,
            locality_benefit=True,
        )
    )

    assert estimate.sufficient_data is True
    assert estimate.evidence_count == 3
    assert estimate.evidence_key is not None
    assert estimate.evidence_key.backend_name == "mock-a"
    assert estimate.expected_error_rate == 0.0
    assert estimate.expected_latency_ms == 18.1
    assert estimate.expected_ttft_ms == 5.62
