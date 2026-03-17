"""Historical performance summaries and transparent route estimation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import product
from typing import Protocol

from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    CandidateRouteEstimateContext,
    HistoricalDimension,
    HistoricalMetricSummary,
    HistoricalPerformanceIndex,
    HistoricalPerformanceSummary,
    HistoricalRouteEstimate,
    HistoricalSummaryKey,
    HistoricalSummaryQuery,
    PerformanceBucketSummary,
)

_LATENCY_BUCKETS_MS = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
_THROUGHPUT_BUCKETS = [10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
_EWMA_ALPHA = 0.3


class HistoricalRoutePredictor(Protocol):
    """Protocol for transparent history-backed route estimators."""

    def estimate(self, context: CandidateRouteEstimateContext) -> HistoricalRouteEstimate: ...


def summarize_historical_records(
    records: Sequence[BenchmarkRequestRecord],
    *,
    query: HistoricalSummaryQuery | None = None,
) -> HistoricalPerformanceIndex:
    """Aggregate historical request outcomes using authoritative record fields."""

    resolved_query = query or HistoricalSummaryQuery()
    matched_records = [
        record for record in records if _record_matches_query(record, query=resolved_query)
    ]
    grouped: dict[str, tuple[HistoricalSummaryKey, list[BenchmarkRequestRecord]]] = {}

    if resolved_query.group_by:
        for record in matched_records:
            for key in _group_keys_for_record(record, dimensions=resolved_query.group_by):
                bucket = grouped.setdefault(key.model_dump_json(), (key, []))[1]
                bucket.append(record)
    elif matched_records:
        key = _summary_key_from_query(resolved_query)
        grouped[key.model_dump_json()] = (key, list(matched_records))

    summaries = [
        _build_summary(key=key, records=group_records)
        for key, group_records in (
            grouped[group_key] for group_key in sorted(grouped)
        )
    ]
    return HistoricalPerformanceIndex(
        query=resolved_query,
        source_record_count=len(records),
        matched_record_count=len(matched_records),
        summaries=summaries,
    )


def summarize_historical_artifacts(
    artifacts: Sequence[BenchmarkRunArtifact],
    *,
    query: HistoricalSummaryQuery | None = None,
) -> HistoricalPerformanceIndex:
    """Aggregate historical evidence directly from benchmark artifacts."""

    return summarize_historical_records(
        [record for artifact in artifacts for record in artifact.records],
        query=query,
    )


class TransparentHistoricalRoutePredictor:
    """Simple estimator that falls back through explicit evidence slices."""

    def __init__(
        self,
        records: Sequence[BenchmarkRequestRecord] | Sequence[BenchmarkRunArtifact],
        *,
        min_samples: int = 3,
    ) -> None:
        items = list(records)
        if items and isinstance(items[0], BenchmarkRunArtifact):
            artifacts = [item for item in items if isinstance(item, BenchmarkRunArtifact)]
            self._records = [record for artifact in artifacts for record in artifact.records]
        else:
            self._records = [item for item in items if isinstance(item, BenchmarkRequestRecord)]
        self._min_samples = min_samples

    def estimate(self, context: CandidateRouteEstimateContext) -> HistoricalRouteEstimate:
        """Estimate candidate outcomes using the strongest sufficient slice available."""

        best_partial_summary: HistoricalPerformanceSummary | None = None
        for query, rationale in _predictor_queries(context):
            index = summarize_historical_records(self._records, query=query)
            if not index.summaries:
                continue
            summary = index.summaries[0]
            if summary.request_count < self._min_samples:
                if (
                    best_partial_summary is None
                    or summary.request_count > best_partial_summary.request_count
                ):
                    best_partial_summary = summary
                continue
            return HistoricalRouteEstimate(
                context=context,
                evidence_key=summary.key,
                evidence_count=summary.request_count,
                sufficient_data=True,
                expected_error_rate=summary.error_rate,
                expected_latency_ms=summary.latency_ms.ewma or summary.latency_ms.avg,
                expected_ttft_ms=summary.ttft_ms.ewma or summary.ttft_ms.avg,
                expected_tokens_per_second=(
                    summary.tokens_per_second.ewma or summary.tokens_per_second.avg
                ),
                expected_queue_delay_ms=summary.queue_delay_ms.ewma or summary.queue_delay_ms.avg,
                rationale=[
                    rationale,
                    f"matched {summary.request_count} historical requests",
                    "estimates use EWMA when available, otherwise arithmetic mean",
                ],
            )

        return HistoricalRouteEstimate(
            context=context,
            evidence_key=None if best_partial_summary is None else best_partial_summary.key,
            evidence_count=(
                0 if best_partial_summary is None else best_partial_summary.request_count
            ),
            sufficient_data=False,
            insufficiency_reason=(
                "insufficient historical evidence for this candidate under the configured "
                f"minimum sample count ({self._min_samples})"
            ),
            rationale=["no matching historical slice met the minimum evidence threshold"],
        )


def _predictor_queries(
    context: CandidateRouteEstimateContext,
) -> Iterable[tuple[HistoricalSummaryQuery, str]]:
    primary_workload_tag = (
        sorted(context.workload_tags, key=lambda tag: tag.value)[0]
        if context.workload_tags
        else None
    )
    yield (
        HistoricalSummaryQuery(
            model_alias=context.model_alias,
            backend_name=context.backend_name,
            policy_id=context.policy_id,
            request_class=context.request_class,
            input_length_bucket=context.input_length_bucket,
            workload_tag=primary_workload_tag,
            prefix_hotness=context.prefix_hotness,
            cache_opportunity=context.cache_opportunity,
            locality_benefit=context.locality_benefit,
        ),
        "used the most specific historical slice for backend, policy, request class, "
        "size, and locality context",
    )
    yield (
        HistoricalSummaryQuery(
            model_alias=context.model_alias,
            backend_name=context.backend_name,
            policy_id=context.policy_id,
            request_class=context.request_class,
            input_length_bucket=context.input_length_bucket,
            workload_tag=primary_workload_tag,
        ),
        "fell back to backend, policy, request class, and input-size evidence",
    )
    yield (
        HistoricalSummaryQuery(
            model_alias=context.model_alias,
            backend_name=context.backend_name,
            policy_id=context.policy_id,
        ),
        "fell back to backend and policy evidence for this alias",
    )
    yield (
        HistoricalSummaryQuery(
            model_alias=context.model_alias,
            backend_name=context.backend_name,
        ),
        "fell back to backend evidence for this alias",
    )
    yield (
        HistoricalSummaryQuery(backend_name=context.backend_name),
        "fell back to backend-only evidence",
    )


def _record_matches_query(
    record: BenchmarkRequestRecord,
    *,
    query: HistoricalSummaryQuery,
) -> bool:
    request_features = record.request_features
    signal = record.route_decision.prefix_locality_signal if record.route_decision else None
    policy_id = _policy_id_for_record(record)

    if query.model_alias is not None and record.model_alias != query.model_alias:
        return False
    if query.backend_type is not None and record.backend_type != query.backend_type:
        return False
    if query.backend_name is not None and record.backend_name != query.backend_name:
        return False
    if (
        query.backend_instance_id is not None
        and record.backend_instance_id != query.backend_instance_id
    ):
        return False
    if query.policy_id is not None and policy_id != query.policy_id:
        return False
    if query.request_class is not None and record.request_class is not query.request_class:
        return False
    if query.tenant_id is not None and record.tenant_id != query.tenant_id:
        return False
    if query.input_length_bucket is not None and (
        request_features is None
        or request_features.input_length_bucket is not query.input_length_bucket
    ):
        return False
    if query.history_depth_bucket is not None and (
        request_features is None
        or request_features.history_depth_bucket is not query.history_depth_bucket
    ):
        return False
    if query.workload_tag is not None and (
        request_features is None or query.workload_tag not in request_features.workload_tags
    ):
        return False
    if query.prefix_hotness is not None and (
        signal is None or signal.hotness is not query.prefix_hotness
    ):
        return False
    if query.cache_opportunity is not None and (
        signal is None or signal.cache_opportunity is not query.cache_opportunity
    ):
        return False
    if query.locality_benefit is not None and (
        signal is None or signal.likely_benefits_from_locality is not query.locality_benefit
    ):
        return False
    return True


def _group_keys_for_record(
    record: BenchmarkRequestRecord,
    *,
    dimensions: list[HistoricalDimension],
) -> list[HistoricalSummaryKey]:
    request_features = record.request_features
    signal = record.route_decision.prefix_locality_signal if record.route_decision else None
    policy_id = _policy_id_for_record(record)
    values_by_dimension: dict[HistoricalDimension, list[object | None]] = {
        HistoricalDimension.MODEL_ALIAS: [record.model_alias],
        HistoricalDimension.BACKEND_TYPE: [record.backend_type],
        HistoricalDimension.BACKEND_NAME: [record.backend_name],
        HistoricalDimension.BACKEND_INSTANCE_ID: [record.backend_instance_id],
        HistoricalDimension.POLICY_ID: [policy_id],
        HistoricalDimension.REQUEST_CLASS: [record.request_class],
        HistoricalDimension.TENANT_ID: [record.tenant_id],
        HistoricalDimension.INPUT_LENGTH_BUCKET: [
            None if request_features is None else request_features.input_length_bucket
        ],
        HistoricalDimension.HISTORY_DEPTH_BUCKET: [
            None if request_features is None else request_features.history_depth_bucket
        ],
        HistoricalDimension.WORKLOAD_TAG: (
            list(request_features.workload_tags)
            if request_features is not None and request_features.workload_tags
            else [None]
        ),
        HistoricalDimension.PREFIX_HOTNESS: [None if signal is None else signal.hotness],
        HistoricalDimension.CACHE_OPPORTUNITY: [
            None if signal is None else signal.cache_opportunity
        ],
        HistoricalDimension.LOCALITY_BENEFIT: [
            None if signal is None else signal.likely_benefits_from_locality
        ],
    }
    keys: list[HistoricalSummaryKey] = []
    for combination in product(*(values_by_dimension[dimension] for dimension in dimensions)):
        key = HistoricalSummaryKey(dimensions=dimensions)
        for dimension, value in zip(dimensions, combination, strict=True):
            _assign_dimension(key, dimension=dimension, value=value)
        keys.append(key)
    return keys


def _summary_key_from_query(query: HistoricalSummaryQuery) -> HistoricalSummaryKey:
    key = HistoricalSummaryKey(dimensions=query.group_by)
    if not query.group_by:
        for dimension in HistoricalDimension:
            _assign_dimension(
                key,
                dimension=dimension,
                value=getattr(query, dimension.value),
            )
    return key


def _assign_dimension(
    key: HistoricalSummaryKey,
    *,
    dimension: HistoricalDimension,
    value: object | None,
) -> None:
    setattr(key, dimension.value, value)


def _build_summary(
    *,
    key: HistoricalSummaryKey,
    records: list[BenchmarkRequestRecord],
) -> HistoricalPerformanceSummary:
    ordered_records = sorted(records, key=lambda record: record.started_at)
    success_count = sum(1 for record in ordered_records if record.success)
    failure_count = len(ordered_records) - success_count
    signals = [
        record.route_decision.prefix_locality_signal
        for record in ordered_records
        if (
            record.route_decision is not None
            and record.route_decision.prefix_locality_signal is not None
        )
    ]
    return HistoricalPerformanceSummary(
        key=key,
        request_count=len(ordered_records),
        success_count=success_count,
        failure_count=failure_count,
        error_rate=_safe_rate(failure_count, len(ordered_records)),
        fallback_rate=_safe_rate(
            sum(1 for record in ordered_records if record.fallback_used),
            len(ordered_records),
        ),
        cache_opportunity_rate=(
            None
            if not signals
            else _safe_rate(sum(1 for signal in signals if signal.cache_opportunity), len(signals))
        ),
        locality_benefit_rate=(
            None
            if not signals
            else _safe_rate(
                sum(1 for signal in signals if signal.likely_benefits_from_locality),
                len(signals),
            )
        ),
        error_category_counts=dict(
            sorted(
                Counter(
                    (record.error_category or "unknown")
                    for record in ordered_records
                    if not record.success
                ).items()
            )
        ),
        latency_ms=_metric_summary(
            [record.latency_ms for record in ordered_records],
            bucket_edges=_LATENCY_BUCKETS_MS,
        ),
        ttft_ms=_metric_summary(
            [record.ttft_ms for record in ordered_records if record.ttft_ms is not None],
            bucket_edges=_LATENCY_BUCKETS_MS,
        ),
        tokens_per_second=_metric_summary(
            [
                record.tokens_per_second
                for record in ordered_records
                if record.tokens_per_second is not None
            ],
            bucket_edges=_THROUGHPUT_BUCKETS,
        ),
        queue_delay_ms=_metric_summary(
            [
                record.queue_delay_ms
                for record in ordered_records
                if record.queue_delay_ms is not None
            ],
            bucket_edges=_LATENCY_BUCKETS_MS,
        ),
    )


def _metric_summary(
    values: list[float | int],
    *,
    bucket_edges: list[float],
) -> HistoricalMetricSummary:
    numeric = [float(value) for value in values]
    if not numeric:
        return HistoricalMetricSummary()
    return HistoricalMetricSummary(
        observation_count=len(numeric),
        avg=round(sum(numeric) / len(numeric), 6),
        ewma=round(_ewma(numeric), 6),
        p50=round(_percentile(numeric, 50), 6),
        p95=round(_percentile(numeric, 95), 6),
        buckets=_bucket_summary(numeric, bucket_edges=bucket_edges),
    )


def _bucket_summary(
    values: list[float],
    *,
    bucket_edges: list[float],
) -> list[PerformanceBucketSummary]:
    lower = 0.0
    buckets: list[PerformanceBucketSummary] = []
    remaining = list(values)
    for upper in bucket_edges:
        count = sum(1 for value in remaining if lower <= value < upper)
        buckets.append(
            PerformanceBucketSummary(
                bucket_label=f"{int(lower)}-{int(upper)}",
                lower_bound=lower,
                upper_bound=upper,
                count=count,
            )
        )
        lower = upper
    tail_count = sum(1 for value in remaining if value >= bucket_edges[-1])
    buckets.append(
        PerformanceBucketSummary(
            bucket_label=f"{int(bucket_edges[-1])}+",
            lower_bound=bucket_edges[-1],
            upper_bound=None,
            count=tail_count,
        )
    )
    return buckets


def _policy_id_for_record(record: BenchmarkRequestRecord) -> str | None:
    if record.policy_reference is not None:
        return record.policy_reference.policy_id
    if record.routing_policy is not None:
        return record.routing_policy.value
    return None


def _ewma(values: list[float]) -> float:
    estimate = values[0]
    for value in values[1:]:
        estimate = (_EWMA_ALPHA * value) + ((1 - _EWMA_ALPHA) * estimate)
    return estimate


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)
