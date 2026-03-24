"""Benchmark artifact and replay helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_output
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel

from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.adapters.registry import AdapterRegistry
from switchyard.bench.recommendations import render_policy_recommendation_report_markdown
from switchyard.config import Settings
from switchyard.optimization import attach_benchmark_config_snapshot
from switchyard.router.service import RouterService
from switchyard.schemas.admin import RuntimeInspectionResponse
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendImageMetadata,
    BackendInstance,
    BackendInstanceSource,
    BackendLoadState,
    BackendNetworkEndpoint,
    BackendRegistrationMetadata,
    BackendType,
    CloudPlacementMetadata,
    CostBudgetProfile,
    DeploymentProfile,
    DeviceClass,
    ExecutionModeLabel,
    NetworkCharacteristics,
    NetworkProfile,
    ReadinessHints,
    TrustMetadata,
    WorkerAuthState,
    WorkerLocalityClass,
    WorkerRegistrationState,
    WorkerTransportType,
    WorkerTrustState,
)
from switchyard.schemas.benchmark import (
    BenchmarkArtifactSchemaVersion,
    BenchmarkComparabilityAssessment,
    BenchmarkComparisonArtifact,
    BenchmarkComparisonDeltaSummary,
    BenchmarkComparisonSideSummary,
    BenchmarkDeploymentTarget,
    BenchmarkEnvironmentMetadata,
    BenchmarkEvidenceClass,
    BenchmarkEvidenceSummary,
    BenchmarkPolicyComparison,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkRunConfig,
    BenchmarkRunKind,
    BenchmarkScenario,
    BenchmarkSummary,
    BenchmarkTargetComparisonArtifact,
    BenchmarkWarmupConfig,
    CacheObservation,
    CapturedTraceRecord,
    CloudCostEvidence,
    CloudEvidenceSource,
    CloudPlacementEvidence,
    CloudWorkerRuntimeEvidence,
    ComparisonSourceKind,
    ControlPlaneReportMetadata,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    DeployedTopologyEndpoint,
    ExecutionTarget,
    ExecutionTargetType,
    FamilyBenchmarkSummary,
    HybridBenchmarkSummary,
    HybridComparisonOutcome,
    HybridComparisonSummary,
    HybridConditionProfile,
    HybridConditionSource,
    HybridExecutionContext,
    HybridExecutionPath,
    PolicyRecommendationReportArtifact,
    RecommendationConfidence,
    RemoteBudgetOutcome,
    RemoteTemperature,
    ReplayMode,
    ReplayPlan,
    ReplayRequest,
    ScenarioDelta,
    SimulationEvidenceKind,
    WorkloadGenerationConfig,
    WorkloadItem,
    WorkloadPattern,
    WorkloadScenario,
    WorkloadScenarioFamily,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, UsageStats
from switchyard.schemas.forge import (
    ForgeCampaignInspectionResponse,
    ForgePromotionRuntimeSummary,
    ForgeTrialInspectionSummary,
)
from switchyard.schemas.optimization import (
    OptimizationCampaignArtifact,
    OptimizationCandidateConfigurationArtifact,
    OptimizationConstraintDimension,
    OptimizationTrialArtifact,
)
from switchyard.schemas.routing import (
    AdmissionDecision,
    AffinityDisposition,
    CircuitBreakerPhase,
    CircuitBreakerState,
    RequestClass,
    RequestContext,
    RolloutDisposition,
    RouteDecision,
    RouteExplanation,
    RoutingPolicy,
    ShadowDisposition,
    WorkloadShape,
)
from switchyard.telemetry import compute_tokens_per_second, estimate_token_count


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Result of a benchmark run including the artifact and written path."""

    artifact: BenchmarkRunArtifact
    output_path: Path


@dataclass(frozen=True, slots=True)
class PrometheusSample:
    """A single parsed Prometheus sample."""

    name: str
    labels: dict[str, str]
    value: float


def build_synthetic_scenario(
    *,
    request_count: int,
    model: str = "mock-chat",
    policy: RoutingPolicy = RoutingPolicy.BALANCED,
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE,
    workload_generation: WorkloadGenerationConfig | None = None,
) -> BenchmarkScenario:
    """Construct a small synthetic Phase 0 benchmark scenario."""

    return BenchmarkScenario(
        name="synthetic_phase0",
        model=model,
        policy=policy,
        workload_shape=workload_shape,
        request_count=request_count,
        input_messages_per_request=1,
        prompt_template="Synthetic benchmark request {index}",
        workload_generation=workload_generation or WorkloadGenerationConfig(),
    )


def build_gateway_scenario(
    *,
    model: str,
    request_count: int = 3,
    policy: RoutingPolicy = RoutingPolicy.BALANCED,
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE,
) -> BenchmarkScenario:
    """Construct a conservative Phase 1 gateway benchmark scenario."""

    return BenchmarkScenario(
        name="phase1_gateway_light",
        model=model,
        policy=policy,
        workload_shape=workload_shape,
        request_count=request_count,
        input_messages_per_request=1,
        stream=True,
        max_output_tokens=64,
        temperature=0.2,
        top_p=1.0,
        prompt_template=(
            "In two short sentences, explain why typed observability matters for "
            "a local inference gateway. Variant {index}."
        ),
    )


def build_default_registry() -> AdapterRegistry:
    """Create a small default registry for local benchmark runs."""

    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-local-fast",
            simulated_latency_ms=5.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                concurrency_limit=1,
                quality_tier=3,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} prompt={user_message}"
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-local-premium",
            simulated_latency_ms=40.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                concurrency_limit=1,
                quality_tier=5,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} prompt={user_message}"
            ),
        )
    )
    return registry


async def compare_synthetic_policies(
    *,
    request_count: int,
    policies: list[RoutingPolicy] | None = None,
    pinned_backends: list[str] | None = None,
    model: str = "mock-chat",
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE,
    workload_generation: WorkloadGenerationConfig | None = None,
    registry: AdapterRegistry | None = None,
    timestamp: datetime | None = None,
) -> BenchmarkComparisonArtifact:
    """Run a comparable synthetic benchmark across several routing policies."""

    resolved_policies = policies or list(RoutingPolicy)
    run_timestamp = timestamp or datetime.now(UTC)
    results: list[BenchmarkPolicyComparison] = []

    resolved_pins = [None, *(pinned_backends or [])]
    for policy in resolved_policies:
        for internal_backend_pin in resolved_pins:
            artifact = await run_synthetic_benchmark(
                scenario=build_synthetic_scenario(
                    request_count=request_count,
                    model=model,
                    policy=policy,
                    workload_shape=workload_shape,
                    workload_generation=workload_generation,
                ),
                registry=registry,
                timestamp=run_timestamp,
                internal_backend_pin=internal_backend_pin,
            )
            results.append(
                BenchmarkPolicyComparison(
                    comparison_label=_comparison_label(
                        policy=policy,
                        internal_backend_pin=internal_backend_pin,
                    ),
                    policy=policy,
                    internal_backend_pin=internal_backend_pin,
                    run_id=artifact.run_id,
                    backends_involved=artifact.backends_involved,
                    summary=artifact.summary,
                )
            )

    unpinned_results = [result for result in results if result.internal_backend_pin is None]
    policy_candidates = unpinned_results or results
    best_by_latency = min(
        policy_candidates,
        key=lambda result: result.summary.avg_latency_ms,
    ).policy
    throughput_candidates = [
        result for result in policy_candidates if result.summary.avg_tokens_per_second is not None
    ]
    best_by_throughput = None
    if throughput_candidates:
        best_by_throughput = max(
            throughput_candidates,
            key=lambda result: result.summary.avg_tokens_per_second or 0.0,
        ).policy

    best_result_by_latency = min(
        results,
        key=lambda result: result.summary.avg_latency_ms,
    ).comparison_label
    throughput_result_candidates = [
        result for result in results if result.summary.avg_tokens_per_second is not None
    ]
    best_result_by_throughput = None
    if throughput_result_candidates:
        best_result_by_throughput = max(
            throughput_result_candidates,
            key=lambda result: result.summary.avg_tokens_per_second or 0.0,
        ).comparison_label

    return BenchmarkComparisonArtifact(
        run_id=f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}_comparison",
        timestamp=run_timestamp,
        scenario_name="synthetic_policy_comparison",
        model=model,
        workload_shape=workload_shape,
        request_count=request_count,
        results=results,
        best_policy_by_latency=best_by_latency,
        best_policy_by_throughput=best_by_throughput,
        best_result_by_latency=best_result_by_latency,
        best_result_by_throughput=best_result_by_throughput,
    )


async def run_synthetic_benchmark(
    *,
    scenario: BenchmarkScenario,
    registry: AdapterRegistry | None = None,
    settings: Settings | None = None,
    timestamp: datetime | None = None,
    internal_backend_pin: str | None = None,
) -> BenchmarkRunArtifact:
    """Run a small benchmark directly against the router and backend layer."""

    resolved_registry = registry or build_default_registry()
    router = RouterService(resolved_registry)
    run_timestamp = timestamp or datetime.now(UTC)
    records: list[BenchmarkRequestRecord] = []

    for index in range(scenario.request_count):
        request_id = f"{scenario.name}_{index:04d}"
        request = _build_request_for_scenario(scenario=scenario, index=index)
        context = RequestContext(
            request_id=request_id,
            policy=scenario.policy,
            workload_shape=scenario.workload_shape,
            internal_backend_pin=internal_backend_pin,
        )
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        try:
            decision = await router.route(request, context)
            adapter = resolved_registry.get(decision.backend_name)
            capabilities = await adapter.capabilities()
            response = await adapter.generate(request, context)
        except Exception as exc:
            completed_at = datetime.now(UTC)
            latency_ms = (perf_counter() - started_perf) * 1000
            records.append(
                BenchmarkRequestRecord(
                    request_id=request_id,
                    backend_name="unrouted",
                    backend_type="unknown",
                    model_alias=scenario.model,
                    routing_policy=scenario.policy,
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_ms=round(latency_ms, 3),
                    success=False,
                    status_code=503,
                    scenario_family=scenario.family,
                    error=str(exc),
                )
            )
            continue

        completed_at = datetime.now(UTC)
        latency_ms = (perf_counter() - started_perf) * 1000
        output_tokens = response.usage.completion_tokens
        records.append(
            BenchmarkRequestRecord(
                request_id=request_id,
                backend_name=decision.backend_name,
                backend_type=_infer_backend_type(response.backend_name),
                model_alias=scenario.model,
                model_identifier=scenario.model,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=round(latency_ms, 3),
                output_tokens=output_tokens,
                tokens_per_second=compute_tokens_per_second(
                    output_tokens=output_tokens,
                    total_latency_ms=latency_ms,
                ),
                scenario_family=scenario.family,
                routing_policy=scenario.policy,
                route_decision=decision,
                route_candidate_count=len(decision.considered_backends),
                fallback_used=False,
                route_reason=_compact_route_reason(decision.explanation),
                route_explanation=decision.explanation,
                cache_observation=_cache_observation_from_capabilities(capabilities),
                success=True,
                status_code=200,
                usage=response.usage,
            )
        )

    return _build_artifact(
        scenario=scenario,
        records=records,
        run_timestamp=run_timestamp,
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="synthetic",
            stream=scenario.stream,
            timeout_seconds=30.0,
            metadata={
                "runner": "router+adapter",
                "internal_backend_pin": internal_backend_pin or "",
            },
        ),
        run_id_suffix=None if internal_backend_pin is None else f"pinned_{internal_backend_pin}",
        run_config=BenchmarkRunConfig(
            benchmark_mode="synthetic",
            concurrency=1,
            timeout_seconds=30.0,
        ),
        settings=settings,
    )


async def run_gateway_benchmark(
    *,
    scenario: BenchmarkScenario,
    gateway_base_url: str,
    metrics_path: str | None = "/metrics",
    timeout_seconds: float = 60.0,
    deployment_target: BenchmarkDeploymentTarget | None = None,
    deployment_profile: DeploymentProfile | None = None,
    config_profile_name: str | None = None,
    control_plane_image: BackendImageMetadata | None = None,
    runtime_inspection_path: str | None = "/admin/runtime",
    settings: Settings | None = None,
    timestamp: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkRunArtifact:
    """Run a small benchmark against the local gateway over HTTP."""

    run_timestamp = timestamp or datetime.now(UTC)
    owns_client = client is None
    benchmark_client = client or httpx.AsyncClient(
        base_url=gateway_base_url,
        timeout=httpx.Timeout(timeout_seconds),
    )
    environment = await _build_http_benchmark_environment(
        benchmark_mode="gateway",
        benchmark_client=benchmark_client,
        gateway_base_url=gateway_base_url,
        metrics_path=metrics_path,
        stream=scenario.stream,
        timeout_seconds=timeout_seconds,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        control_plane_image=control_plane_image,
        runtime_inspection_path=runtime_inspection_path,
        metadata={"runner": "httpx"},
    )
    metrics_before = await _get_metrics_snapshot(
        benchmark_client=benchmark_client,
        metrics_path=metrics_path,
    )
    records: list[BenchmarkRequestRecord] = []
    metrics_after: list[PrometheusSample] | None = None
    try:
        for index in range(scenario.request_count):
            request_id = f"{scenario.name}_{index:04d}"
            started_at = datetime.now(UTC)
            started_perf = perf_counter()
            payload = _build_request_for_scenario(scenario=scenario, index=index).model_dump(
                mode="json",
                exclude_none=True,
            )
            headers = {
                "x-request-id": request_id,
                "x-switchyard-routing-policy": scenario.policy.value,
                "x-switchyard-workload-shape": scenario.workload_shape.value,
            }
            if scenario.stream:
                record = await _run_streaming_gateway_request(
                    benchmark_client=benchmark_client,
                    request_id=request_id,
                    payload=payload,
                    headers=headers,
                    model_alias=scenario.model,
                    workload_item_id=None,
                    scenario_family=scenario.family,
                    request_metadata=None,
                    started_at=started_at,
                    started_perf=started_perf,
                )
            else:
                record = await _run_non_streaming_gateway_request(
                    benchmark_client=benchmark_client,
                    request_id=request_id,
                    payload=payload,
                    headers=headers,
                    model_alias=scenario.model,
                    workload_item_id=None,
                    scenario_family=scenario.family,
                    request_metadata=None,
                    started_at=started_at,
                    started_perf=started_perf,
                )
            records.append(record)
    finally:
        metrics_after = await _get_metrics_snapshot(
            benchmark_client=benchmark_client,
            metrics_path=metrics_path,
        )
        if owns_client:
            await benchmark_client.aclose()
    _merge_server_metrics_into_records(
        records=records,
        before=metrics_before,
        after=metrics_after,
    )
    return _build_artifact(
        scenario=scenario,
        records=records,
        run_timestamp=run_timestamp,
        environment=environment.model_copy(
            update={
                "metadata": {
                    **environment.metadata,
                    "metrics_enabled": str(metrics_after is not None),
                }
            }
        ),
        run_config=BenchmarkRunConfig(
            benchmark_mode="gateway",
            concurrency=1,
            timeout_seconds=timeout_seconds,
            canary_percentage=(
                0.0 if settings is None else settings.phase4.canary_routing.default_percentage
            ),
            shadow_sampling_rate=(
                0.0 if settings is None else settings.phase4.shadow_routing.default_sampling_rate
            ),
            session_affinity_ttl_seconds=(
                None if settings is None else settings.phase4.session_affinity.ttl_seconds
            ),
        ),
        settings=settings,
    )


async def run_workload_manifest_benchmark(
    *,
    scenario: WorkloadScenario,
    gateway_base_url: str,
    execution_target: ExecutionTarget | None = None,
    warmup: BenchmarkWarmupConfig | None = None,
    metrics_path: str | None = "/metrics",
    timeout_seconds: float = 60.0,
    deployment_target: BenchmarkDeploymentTarget | None = None,
    deployment_profile: DeploymentProfile | None = None,
    config_profile_name: str | None = None,
    control_plane_image: BackendImageMetadata | None = None,
    runtime_inspection_path: str | None = "/admin/runtime",
    settings: Settings | None = None,
    timestamp: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkRunArtifact:
    """Run a workload manifest against the existing gateway path."""

    run_timestamp = timestamp or datetime.now(UTC)
    resolved_target = execution_target or ExecutionTarget(
        target_type=ExecutionTargetType.ROUTING_POLICY,
        model_alias=scenario.model_alias or scenario.model,
        routing_policy=scenario.policy,
    )
    execution_scenario = BenchmarkScenario.model_validate(
        scenario.model_copy(
            update={
                "model": resolved_target.model_alias,
                "model_alias": resolved_target.model_alias,
                "policy": resolved_target.routing_policy or scenario.policy,
            }
        ).model_dump(mode="python")
    )
    resolved_warmup = _resolved_workload_warmup(
        scenario=execution_scenario,
        warmup=warmup,
    )
    owns_client = client is None
    benchmark_client = client or httpx.AsyncClient(
        base_url=gateway_base_url,
        timeout=httpx.Timeout(timeout_seconds),
    )
    environment = await _build_http_benchmark_environment(
        benchmark_mode="workload_manifest",
        benchmark_client=benchmark_client,
        gateway_base_url=gateway_base_url,
        metrics_path=metrics_path,
        stream=execution_scenario.stream,
        timeout_seconds=timeout_seconds,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        control_plane_image=control_plane_image,
        runtime_inspection_path=runtime_inspection_path,
        metadata={
            "runner": "httpx_workload_manifest",
            "warmup_request_count": str(resolved_warmup.request_count),
        },
    )
    try:
        await _run_workload_warmup(
            benchmark_client=benchmark_client,
            scenario=execution_scenario,
            warmup=resolved_warmup,
            execution_target=resolved_target,
        )
        metrics_before = await _get_metrics_snapshot(
            benchmark_client=benchmark_client,
            metrics_path=metrics_path,
        )
        records = await _execute_workload_items(
            benchmark_client=benchmark_client,
            scenario=execution_scenario,
            execution_target=resolved_target,
        )
        metrics_after = await _get_metrics_snapshot(
            benchmark_client=benchmark_client,
            metrics_path=metrics_path,
        )
    finally:
        if owns_client:
            await benchmark_client.aclose()
    _merge_server_metrics_into_records(
        records=records,
        before=metrics_before,
        after=metrics_after,
    )
    return _build_artifact(
        scenario=execution_scenario,
        records=records,
        run_timestamp=run_timestamp,
        environment=environment.model_copy(
            update={
                "metadata": {
                    **environment.metadata,
                    "metrics_enabled": str(metrics_after is not None),
                }
            }
        ),
        run_id_suffix=_workload_run_suffix(resolved_target),
        execution_target=resolved_target,
        run_config=BenchmarkRunConfig(
            benchmark_mode="workload_manifest",
            execution_target=resolved_target,
            concurrency=1,
            warmup=resolved_warmup,
            timeout_seconds=timeout_seconds,
            canary_percentage=(
                0.0 if settings is None else settings.phase4.canary_routing.default_percentage
            ),
            shadow_sampling_rate=(
                0.0 if settings is None else settings.phase4.shadow_routing.default_sampling_rate
            ),
            session_affinity_ttl_seconds=(
                None if settings is None else settings.phase4.session_affinity.ttl_seconds
            ),
        ),
        settings=settings,
    )


def load_captured_traces(trace_path: Path) -> list[CapturedTraceRecord]:
    """Load captured traces from a JSONL file."""

    traces: list[CapturedTraceRecord] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        raw_line = line.strip()
        if not raw_line:
            continue
        traces.append(CapturedTraceRecord.model_validate_json(raw_line))
    return traces


def validate_replayable_traces(
    traces: list[CapturedTraceRecord],
    *,
    model_alias: str | None = None,
) -> None:
    """Fail fast when a trace set cannot produce replayable requests."""

    for trace in traces:
        resolved_model_alias = (
            model_alias or trace.logical_alias or trace.execution_target.model_alias
        )
        _build_request_from_trace(
            trace=trace,
            model_alias=resolved_model_alias,
        )


def build_replay_plan(
    *,
    traces: list[CapturedTraceRecord],
    execution_target: ExecutionTarget,
    replay_mode: ReplayMode,
    concurrency: int,
    warmup: BenchmarkWarmupConfig | None = None,
    source_run_id: str,
    settings: Settings | None = None,
) -> ReplayPlan:
    """Build a typed replay plan from captured traces."""

    planned_requests = _planned_replay_requests(traces=traces, replay_mode=replay_mode)
    resolved_concurrency = concurrency if replay_mode is ReplayMode.FIXED_CONCURRENCY else 1
    replay_plan = ReplayPlan(
        plan_id=f"replay_{source_run_id}",
        source_run_id=source_run_id,
        source_schema_version=BenchmarkArtifactSchemaVersion.V2,
        execution_target=execution_target,
        replay_mode=replay_mode,
        concurrency=resolved_concurrency,
        warmup=warmup or BenchmarkWarmupConfig(),
        requests=planned_requests,
        metadata={
            "time_strategy": "ready_for_time_scaled_replay",
            "original_timestamp_count": str(
                sum(1 for trace in traces if trace.request_timestamp is not None)
            ),
        },
    )
    if settings is None:
        return replay_plan
    resolved_run_config = attach_benchmark_config_snapshot(
        settings=settings,
        run_config=BenchmarkRunConfig(
            benchmark_mode="trace_replay",
            execution_target=execution_target,
            concurrency=resolved_concurrency,
            warmup=warmup or BenchmarkWarmupConfig(),
            replay_mode=replay_mode,
            timeout_seconds=30.0,
            canary_percentage=settings.phase4.canary_routing.default_percentage,
            shadow_sampling_rate=settings.phase4.shadow_routing.default_sampling_rate,
            session_affinity_ttl_seconds=settings.phase4.session_affinity.ttl_seconds,
        ),
    )
    return replay_plan.model_copy(
        update={
            "config_fingerprint": resolved_run_config.config_fingerprint,
            "immutable_config": resolved_run_config.immutable_config,
        },
        deep=True,
    )


async def run_trace_replay_benchmark(
    *,
    traces: list[CapturedTraceRecord],
    gateway_base_url: str,
    execution_target: ExecutionTarget,
    replay_mode: ReplayMode = ReplayMode.SEQUENTIAL,
    concurrency: int = 1,
    warmup: BenchmarkWarmupConfig | None = None,
    source_run_id: str = "captured_traces",
    metrics_path: str | None = "/metrics",
    timeout_seconds: float = 60.0,
    deployment_target: BenchmarkDeploymentTarget | None = None,
    deployment_profile: DeploymentProfile | None = None,
    config_profile_name: str | None = None,
    control_plane_image: BackendImageMetadata | None = None,
    runtime_inspection_path: str | None = "/admin/runtime",
    settings: Settings | None = None,
    timestamp: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkRunArtifact:
    """Replay captured traces through the existing gateway path."""

    if not traces:
        msg = "trace replay requires at least one captured trace record"
        raise ValueError(msg)
    validate_replayable_traces(
        traces,
        model_alias=execution_target.model_alias,
    )

    run_timestamp = timestamp or datetime.now(UTC)
    replay_plan = build_replay_plan(
        traces=traces,
        execution_target=execution_target,
        replay_mode=replay_mode,
        concurrency=concurrency,
        warmup=warmup,
        source_run_id=source_run_id,
        settings=settings,
    )
    scenario = _build_replay_scenario(
        traces=traces,
        execution_target=execution_target,
        source_run_id=source_run_id,
    )
    owns_client = client is None
    benchmark_client = client or httpx.AsyncClient(
        base_url=gateway_base_url,
        timeout=httpx.Timeout(timeout_seconds),
    )
    environment = await _build_http_benchmark_environment(
        benchmark_mode="trace_replay",
        benchmark_client=benchmark_client,
        gateway_base_url=gateway_base_url,
        metrics_path=metrics_path,
        stream=any(request.stream for request in replay_plan.requests),
        timeout_seconds=timeout_seconds,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        control_plane_image=control_plane_image,
        runtime_inspection_path=runtime_inspection_path,
        metadata={
            "runner": "httpx_trace_replay",
            "source_run_id": source_run_id,
            "replay_mode": replay_mode.value,
        },
    )
    try:
        await _run_trace_replay_warmup(
            benchmark_client=benchmark_client,
            traces=traces,
            replay_plan=replay_plan,
            scenario=scenario,
        )
        metrics_before = await _get_metrics_snapshot(
            benchmark_client=benchmark_client,
            metrics_path=metrics_path,
        )
        records = await _execute_trace_replay(
            benchmark_client=benchmark_client,
            traces=traces,
            replay_plan=replay_plan,
            scenario=scenario,
        )
        metrics_after = await _get_metrics_snapshot(
            benchmark_client=benchmark_client,
            metrics_path=metrics_path,
        )
    finally:
        if owns_client:
            await benchmark_client.aclose()
    _merge_server_metrics_into_records(
        records=records,
        before=metrics_before,
        after=metrics_after,
    )
    return _build_artifact(
        scenario=scenario,
        records=records,
        run_timestamp=run_timestamp,
        environment=environment.model_copy(
            update={
                "metadata": {
                    **environment.metadata,
                    "metrics_enabled": str(metrics_after is not None),
                }
            }
        ),
        run_id_suffix=_trace_replay_run_suffix(execution_target, replay_mode),
        execution_target=execution_target,
        run_config=BenchmarkRunConfig(
            benchmark_mode="trace_replay",
            execution_target=execution_target,
            concurrency=replay_plan.concurrency,
            warmup=replay_plan.warmup,
            replay_mode=replay_mode,
            timeout_seconds=timeout_seconds,
            canary_percentage=(
                0.0 if settings is None else settings.phase4.canary_routing.default_percentage
            ),
            shadow_sampling_rate=(
                0.0 if settings is None else settings.phase4.shadow_routing.default_sampling_rate
            ),
            session_affinity_ttl_seconds=(
                None if settings is None else settings.phase4.session_affinity.ttl_seconds
            ),
            config_fingerprint=replay_plan.config_fingerprint,
            immutable_config=replay_plan.immutable_config,
            metadata={"source_run_id": source_run_id},
        ),
        settings=settings,
    )


async def compare_workload_execution_targets(
    *,
    scenario: WorkloadScenario,
    gateway_base_url: str,
    left_target: ExecutionTarget,
    right_target: ExecutionTarget,
    warmup: BenchmarkWarmupConfig | None = None,
    metrics_path: str | None = "/metrics",
    timeout_seconds: float = 60.0,
    timestamp: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkTargetComparisonArtifact:
    """Run the same workload manifest against two execution targets and compare results."""

    run_timestamp = timestamp or datetime.now(UTC)
    left_artifact = await run_workload_manifest_benchmark(
        scenario=scenario,
        gateway_base_url=gateway_base_url,
        execution_target=left_target,
        warmup=warmup,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        timestamp=run_timestamp,
        client=client,
    )
    right_artifact = await run_workload_manifest_benchmark(
        scenario=scenario,
        gateway_base_url=gateway_base_url,
        execution_target=right_target,
        warmup=warmup,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        timestamp=run_timestamp,
        client=client,
    )
    return compare_benchmark_runs(
        left_artifact=left_artifact,
        right_artifact=right_artifact,
        source_kind=ComparisonSourceKind.WORKLOAD_MANIFEST,
        source_name=scenario.name,
    )


async def compare_trace_execution_targets(
    *,
    traces: list[CapturedTraceRecord],
    gateway_base_url: str,
    left_target: ExecutionTarget,
    right_target: ExecutionTarget,
    replay_mode: ReplayMode = ReplayMode.SEQUENTIAL,
    concurrency: int = 1,
    warmup: BenchmarkWarmupConfig | None = None,
    source_run_id: str = "captured_traces",
    metrics_path: str | None = "/metrics",
    timeout_seconds: float = 60.0,
    timestamp: datetime | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkTargetComparisonArtifact:
    """Replay the same trace set against two execution targets and compare results."""

    run_timestamp = timestamp or datetime.now(UTC)
    left_artifact = await run_trace_replay_benchmark(
        traces=traces,
        gateway_base_url=gateway_base_url,
        execution_target=left_target,
        replay_mode=replay_mode,
        concurrency=concurrency,
        warmup=warmup,
        source_run_id=source_run_id,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        timestamp=run_timestamp,
        client=client,
    )
    right_artifact = await run_trace_replay_benchmark(
        traces=traces,
        gateway_base_url=gateway_base_url,
        execution_target=right_target,
        replay_mode=replay_mode,
        concurrency=concurrency,
        warmup=warmup,
        source_run_id=source_run_id,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        timestamp=run_timestamp,
        client=client,
    )
    return compare_benchmark_runs(
        left_artifact=left_artifact,
        right_artifact=right_artifact,
        source_kind=ComparisonSourceKind.TRACE_SET,
        source_name=source_run_id,
    )


def compare_benchmark_runs(
    *,
    left_artifact: BenchmarkRunArtifact,
    right_artifact: BenchmarkRunArtifact,
    source_kind: ComparisonSourceKind,
    source_name: str,
) -> BenchmarkTargetComparisonArtifact:
    """Build a deterministic side-by-side comparison from two benchmark artifacts."""

    left_summary = _comparison_side_summary(left_artifact)
    right_summary = _comparison_side_summary(right_artifact)
    return BenchmarkTargetComparisonArtifact(
        comparison_id=_comparison_id(left_artifact=left_artifact, right_artifact=right_artifact),
        timestamp=max(left_artifact.timestamp, right_artifact.timestamp),
        source_kind=source_kind,
        source_name=source_name,
        request_count=min(left_artifact.request_count, right_artifact.request_count),
        left=left_summary,
        right=right_summary,
        delta=BenchmarkComparisonDeltaSummary(
            request_count_delta=right_artifact.request_count - left_artifact.request_count,
            success_rate_delta=round(
                right_summary.success_rate - left_summary.success_rate,
                6,
            ),
            error_rate_delta=round(right_summary.error_rate - left_summary.error_rate, 6),
            fallback_rate_delta=round(
                right_summary.fallback_rate - left_summary.fallback_rate,
                6,
            ),
            p50_latency_delta_ms=round(
                right_summary.p50_latency_ms - left_summary.p50_latency_ms,
                3,
            ),
            p95_latency_delta_ms=round(
                right_summary.p95_latency_ms - left_summary.p95_latency_ms,
                3,
            ),
            p50_ttft_delta_ms=_optional_delta(
                right_summary.p50_ttft_ms,
                left_summary.p50_ttft_ms,
            ),
            p95_ttft_delta_ms=_optional_delta(
                right_summary.p95_ttft_ms,
                left_summary.p95_ttft_ms,
            ),
            avg_tokens_per_second_delta=_optional_delta(
                right_summary.avg_tokens_per_second,
                left_summary.avg_tokens_per_second,
            ),
            p95_tokens_per_second_delta=_optional_delta(
                right_summary.p95_tokens_per_second,
                left_summary.p95_tokens_per_second,
            ),
            route_distribution_delta=_distribution_delta(
                left_summary.route_distribution,
                right_summary.route_distribution,
            ),
            backend_distribution_delta=_distribution_delta(
                left_summary.backend_distribution,
                right_summary.backend_distribution,
            ),
            hybrid_summary=_hybrid_comparison_summary(
                left_artifact.records,
                right_artifact.records,
            ),
            notable_scenario_deltas=_scenario_deltas(
                left_artifact.records,
                right_artifact.records,
            ),
        ),
        comparability_assessment=_comparability_assessment(
            left_artifact=left_artifact,
            right_artifact=right_artifact,
        ),
    )


def summarize_records(records: list[BenchmarkRequestRecord]) -> BenchmarkSummary:
    """Summarize per-request benchmark records."""

    for record in records:
        record.cloud_worker_evidence = _cloud_worker_evidence_for_record(record)
        record.evidence_class = _classify_record_evidence(record)
        record.confidence_notes = _confidence_notes_for_record(record)
        record.comparability_limitations = _record_comparability_limitations(record)

    latencies = [record.latency_ms for record in records]
    ttfts = [record.ttft_ms for record in records if record.ttft_ms is not None]
    output_tokens = [_output_tokens_for_record(record) for record in records]
    tps_values = [
        record.tokens_per_second for record in records if record.tokens_per_second is not None
    ]
    success_count = sum(1 for record in records if record.success)
    failure_count = len(records) - success_count
    chosen_backend_counts: dict[str, int] = {}
    for record in records:
        chosen_backend_counts[record.backend_name] = (
            chosen_backend_counts.get(record.backend_name, 0) + 1
        )
    family_summaries = _summarize_records_by_family(records)
    evidence_summary = _summarize_record_evidence(records)

    return BenchmarkSummary(
        request_count=len(records),
        success_count=success_count,
        failure_count=failure_count,
        avg_latency_ms=_average(latencies),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        avg_ttft_ms=None if not ttfts else _average(ttfts),
        p50_ttft_ms=None if not ttfts else _percentile(ttfts, 50),
        p95_ttft_ms=None if not ttfts else _percentile(ttfts, 95),
        total_output_tokens=sum(output_tokens),
        avg_output_tokens=_average([float(tokens) for tokens in output_tokens]),
        avg_tokens_per_second=None if not tps_values else _average(tps_values),
        p95_tokens_per_second=None if not tps_values else _percentile(tps_values, 95),
        fallback_count=sum(1 for record in records if record.fallback_used),
        chosen_backend_counts=dict(sorted(chosen_backend_counts.items())),
        hybrid_summary=_summarize_hybrid_records(records),
        evidence_summary=evidence_summary,
        family_summaries=family_summaries,
    )


def _summarize_record_evidence(
    records: list[BenchmarkRequestRecord],
) -> BenchmarkEvidenceSummary:
    class_counts: Counter[BenchmarkEvidenceClass] = Counter(
        record.evidence_class for record in records
    )
    provider_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    runtime_counts: Counter[str] = Counter()
    runtime_version_counts: Counter[str] = Counter()
    worker_identity_counts: Counter[str] = Counter()
    observed_budget_outcomes: Counter[str] = Counter()
    queue_delays: list[float] = []
    remote_latencies: list[float] = []
    confidence_notes: list[str] = []
    comparability_limitations: list[str] = []
    observed_error_count = 0

    for record in records:
        for note in record.confidence_notes:
            if note not in confidence_notes:
                confidence_notes.append(note)
        for limitation in record.comparability_limitations:
            if limitation not in comparability_limitations:
                comparability_limitations.append(limitation)
        evidence = record.cloud_worker_evidence
        if evidence is None:
            continue
        if evidence.provider is not None:
            provider_counts[evidence.provider] += 1
        if evidence.region is not None:
            region_counts[evidence.region] += 1
        if evidence.runtime is not None:
            runtime_label = evidence.runtime.runtime_label or evidence.runtime.runtime_family
            runtime_counts[runtime_label] += 1
            if evidence.runtime.runtime_version is not None:
                runtime_version_counts[f"{runtime_label}:{evidence.runtime.runtime_version}"] += 1
        identity = (
            evidence.backend_name
            if evidence.backend_instance_id is None
            else f"{evidence.backend_name}/{evidence.backend_instance_id}"
        )
        worker_identity_counts[identity] += 1
        if evidence.observed_budget_outcome is not None:
            observed_budget_outcomes[evidence.observed_budget_outcome.value] += 1
        if evidence.observed_queue_delay_ms is not None:
            queue_delays.append(evidence.observed_queue_delay_ms)
        if evidence.observed_latency_ms is not None:
            remote_latencies.append(evidence.observed_latency_ms)
        if evidence.observed_error_category is not None:
            observed_error_count += 1

    observed_count = class_counts.get(BenchmarkEvidenceClass.OBSERVED, 0)
    mock_count = class_counts.get(BenchmarkEvidenceClass.MOCK, 0)
    configured_count = class_counts.get(BenchmarkEvidenceClass.CONFIGURED, 0)
    estimated_count = class_counts.get(BenchmarkEvidenceClass.ESTIMATED, 0)
    mixed_count = class_counts.get(BenchmarkEvidenceClass.MIXED, 0)
    unsupported_count = class_counts.get(BenchmarkEvidenceClass.UNSUPPORTED, 0)
    if mixed_count > 0:
        comparability_limitations.append(
            "run mixes observed cloud evidence with configured, mock, or estimated evidence"
        )
    if observed_count == 0 and any(
        count > 0 for count in (mock_count, configured_count, estimated_count)
    ):
        comparability_limitations.append(
            "run does not contain direct observed cloud-worker execution"
        )
    run_kind = _run_kind_for_records(records, class_counts)
    return BenchmarkEvidenceSummary(
        run_kind=run_kind,
        sample_size=len(records),
        evidence_class_counts=dict(sorted(class_counts.items(), key=lambda item: item[0].value)),
        observed_cloud_request_count=observed_count,
        mock_request_count=mock_count,
        configured_cloud_request_count=configured_count,
        estimated_request_count=estimated_count,
        unsupported_request_count=unsupported_count,
        mixed_request_count=mixed_count,
        cloud_provider_counts=dict(sorted(provider_counts.items())),
        cloud_region_counts=dict(sorted(region_counts.items())),
        cloud_runtime_counts=dict(sorted(runtime_counts.items())),
        cloud_runtime_version_counts=dict(sorted(runtime_version_counts.items())),
        cloud_worker_identity_counts=dict(sorted(worker_identity_counts.items())),
        observed_error_count=observed_error_count,
        observed_budget_outcome_counts=dict(sorted(observed_budget_outcomes.items())),
        avg_observed_queue_delay_ms=(None if not queue_delays else _average(queue_delays)),
        avg_observed_remote_latency_ms=(
            None if not remote_latencies else _average(remote_latencies)
        ),
        confidence_notes=confidence_notes,
        comparability_limitations=list(dict.fromkeys(comparability_limitations)),
    )


def _run_kind_for_records(
    records: list[BenchmarkRequestRecord],
    class_counts: Counter[BenchmarkEvidenceClass],
) -> BenchmarkRunKind:
    if records and all(_is_local_only_record(record) for record in records):
        return BenchmarkRunKind.LOCAL_ONLY
    observed_count = class_counts.get(BenchmarkEvidenceClass.OBSERVED, 0)
    mock_count = class_counts.get(BenchmarkEvidenceClass.MOCK, 0)
    configured_count = class_counts.get(BenchmarkEvidenceClass.CONFIGURED, 0)
    estimated_count = class_counts.get(BenchmarkEvidenceClass.ESTIMATED, 0)
    mixed_count = class_counts.get(BenchmarkEvidenceClass.MIXED, 0)
    if observed_count > 0 and any(
        count > 0 for count in (mock_count, configured_count, estimated_count, mixed_count)
    ):
        return BenchmarkRunKind.MIXED_EVIDENCE
    if mixed_count > 0:
        return BenchmarkRunKind.MIXED_EVIDENCE
    if observed_count > 0:
        return BenchmarkRunKind.OBSERVED_CLOUD
    if mock_count > 0 and configured_count == 0 and estimated_count == 0:
        return BenchmarkRunKind.MOCK_REMOTE_HYBRID
    if configured_count > 0 and estimated_count == 0:
        return BenchmarkRunKind.CONFIGURED_CLOUD
    if estimated_count > 0 and configured_count == 0:
        return BenchmarkRunKind.ESTIMATED_CLOUD
    if configured_count > 0 or estimated_count > 0:
        return BenchmarkRunKind.MIXED_EVIDENCE
    return BenchmarkRunKind.UNSUPPORTED


def _classify_record_evidence(record: BenchmarkRequestRecord) -> BenchmarkEvidenceClass:
    classes = _record_evidence_classes(record)
    if not classes:
        return BenchmarkEvidenceClass.UNSUPPORTED
    if len(classes) > 1:
        return BenchmarkEvidenceClass.MIXED
    return next(iter(classes))


def _record_evidence_classes(
    record: BenchmarkRequestRecord,
) -> set[BenchmarkEvidenceClass]:
    classes: set[BenchmarkEvidenceClass] = set()
    if _record_uses_mock_evidence(record):
        classes.add(BenchmarkEvidenceClass.MOCK)
    if _record_has_observed_cloud_evidence(record):
        classes.add(BenchmarkEvidenceClass.OBSERVED)
    if _record_has_estimated_evidence(record):
        classes.add(BenchmarkEvidenceClass.ESTIMATED)
    if _record_has_configured_cloud_evidence(record):
        classes.add(BenchmarkEvidenceClass.CONFIGURED)
    if not classes:
        classes.add(BenchmarkEvidenceClass.UNSUPPORTED)
    return classes


def _record_uses_mock_evidence(record: BenchmarkRequestRecord) -> bool:
    if (record.backend_type or "").lower() == "mock" or record.backend_name.startswith("mock"):
        return True
    context = record.hybrid_context
    return bool(context is not None and context.injected_condition is not None)


def _record_has_observed_cloud_evidence(record: BenchmarkRequestRecord) -> bool:
    context = record.hybrid_context
    if context is not None:
        if (
            context.observed_placement_evidence is not None
            and context.observed_placement_evidence.source is CloudEvidenceSource.OBSERVED_RUNTIME
        ):
            return True
        if (
            context.observed_cost_evidence is not None
            and context.observed_cost_evidence.source is CloudEvidenceSource.OBSERVED_RUNTIME
        ):
            return True
    observed_instance = _observed_backend_instance(record.route_decision)
    if observed_instance is None:
        return False
    return bool(
        observed_instance.runtime is not None
        or observed_instance.placement.provider is not None
        or observed_instance.placement.region is not None
        or observed_instance.placement.zone is not None
        or observed_instance.execution_mode
        in {ExecutionModeLabel.REMOTE_WORKER, ExecutionModeLabel.EXTERNAL_SERVICE}
    )


def _record_has_estimated_evidence(record: BenchmarkRequestRecord) -> bool:
    context = record.hybrid_context
    return bool(context is not None and context.predictor_condition is not None)


def _record_has_configured_cloud_evidence(record: BenchmarkRequestRecord) -> bool:
    if _record_has_observed_cloud_evidence(record):
        return False
    context = record.hybrid_context
    if context is not None:
        if (
            context.observed_placement_evidence is not None
            and context.observed_placement_evidence.source
            is CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE
        ):
            return True
        if (
            context.observed_cost_evidence is not None
            and context.observed_cost_evidence.source
            is CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE
        ):
            return True
    route_decision = record.route_decision
    if route_decision is None or route_decision.selected_deployment is None:
        return False
    deployment = route_decision.selected_deployment
    if deployment.runtime is not None or any(
        instance.runtime is not None for instance in deployment.instances
    ):
        return True
    if (
        deployment.placement.provider is not None
        or deployment.placement.region is not None
        or deployment.placement.zone is not None
    ):
        return True
    if (
        deployment.cost_profile.relative_cost_index is not None
        or deployment.cost_profile.budget_bucket is not None
        or deployment.cost_profile.currency is not None
    ):
        return True
    return False


def _cloud_worker_evidence_for_record(
    record: BenchmarkRequestRecord,
) -> CloudWorkerRuntimeEvidence | None:
    route_decision = record.route_decision
    deployment = None if route_decision is None else route_decision.selected_deployment
    observed_instance = _observed_backend_instance(route_decision)
    context = record.hybrid_context
    has_context_cloud_metadata = bool(
        context is not None
        and (
            context.observed_placement_evidence is not None
            or context.observed_cost_evidence is not None
        )
    )
    remote_backend_name = record.backend_name.startswith("remote-worker:")
    if observed_instance is None and deployment is None and context is None:
        return None
    if (
        observed_instance is None
        and deployment is None
        and not has_context_cloud_metadata
        and not remote_backend_name
    ):
        return None
    runtime = (
        observed_instance.runtime.model_copy(deep=True)
        if observed_instance is not None and observed_instance.runtime is not None
        else deployment.runtime.model_copy(deep=True)
        if deployment is not None and deployment.runtime is not None
        else None
    )
    provider = (
        observed_instance.placement.provider
        if observed_instance is not None and observed_instance.placement.provider is not None
        else deployment.placement.provider
        if deployment is not None
        else context.observed_placement_evidence.provider
        if context is not None and context.observed_placement_evidence is not None
        else None
    )
    region = (
        observed_instance.placement.region
        if observed_instance is not None and observed_instance.placement.region is not None
        else deployment.placement.region
        if deployment is not None
        else context.observed_placement_evidence.region
        if context is not None and context.observed_placement_evidence is not None
        else None
    )
    zone = (
        observed_instance.placement.zone
        if observed_instance is not None and observed_instance.placement.zone is not None
        else deployment.placement.zone
        if deployment is not None
        else context.observed_placement_evidence.zone
        if context is not None and context.observed_placement_evidence is not None
        else None
    )
    notes: list[str] = []
    if observed_instance is not None:
        notes.append("cloud worker identity derived from observed backend instance metadata")
    elif deployment is not None:
        notes.append("cloud worker identity derived from configured deployment metadata")
    return CloudWorkerRuntimeEvidence(
        backend_name=record.backend_name,
        backend_instance_id=record.backend_instance_id,
        runtime=runtime,
        provider=provider,
        region=region,
        zone=zone,
        observed_queue_delay_ms=record.queue_delay_ms,
        observed_latency_ms=record.latency_ms,
        observed_ttft_ms=record.ttft_ms,
        observed_status_code=record.status_code,
        observed_error_category=record.error_category,
        observed_budget_outcome=None if context is None else context.observed_budget_outcome,
        placement_evidence_source=(
            None
            if context is None or context.observed_placement_evidence is None
            else context.observed_placement_evidence.source
        ),
        cost_evidence_source=(
            None
            if context is None or context.observed_cost_evidence is None
            else context.observed_cost_evidence.source
        ),
        notes=notes,
    )


def _confidence_notes_for_record(record: BenchmarkRequestRecord) -> list[str]:
    notes: list[str] = []
    context = record.hybrid_context
    if context is None:
        return notes
    if (
        context.predictor_condition is not None
        and context.predictor_condition.confidence is not None
    ):
        notes.append(
            f"predictor estimate confidence={context.predictor_condition.confidence.value}"
        )
    if context.injected_condition is not None and context.injected_condition.confidence is not None:
        notes.append(f"injected condition confidence={context.injected_condition.confidence.value}")
    if context.observed_execution_path is HybridExecutionPath.UNKNOWN:
        notes.append("no observed execution path was captured")
    return notes


def _record_comparability_limitations(record: BenchmarkRequestRecord) -> list[str]:
    limitations: list[str] = []
    if record.evidence_class is BenchmarkEvidenceClass.MOCK:
        limitations.append("record relies on mock backend or injected mock cloud conditions")
    elif record.evidence_class is BenchmarkEvidenceClass.CONFIGURED:
        limitations.append("record relies on configured cloud metadata without observed execution")
    elif record.evidence_class is BenchmarkEvidenceClass.ESTIMATED:
        limitations.append("record relies on predictor-based cloud estimates")
    elif record.evidence_class is BenchmarkEvidenceClass.MIXED:
        limitations.append("record mixes observed cloud evidence with non-observed evidence")
    elif record.evidence_class is BenchmarkEvidenceClass.UNSUPPORTED and _is_local_only_record(
        record
    ):
        limitations.append("record stayed local-only and carries no cloud-worker evidence")
    return limitations


def _is_local_only_record(record: BenchmarkRequestRecord) -> bool:
    context = record.hybrid_context
    if context is not None and context.observed_execution_path is not HybridExecutionPath.UNKNOWN:
        return context.observed_execution_path is HybridExecutionPath.LOCAL_ONLY
    return not record.backend_name.startswith("remote-worker:")


def _summarize_records_by_family(
    records: list[BenchmarkRequestRecord],
) -> dict[WorkloadScenarioFamily, FamilyBenchmarkSummary]:
    grouped: dict[WorkloadScenarioFamily, list[BenchmarkRequestRecord]] = {}
    for record in records:
        if record.scenario_family is None:
            continue
        grouped.setdefault(record.scenario_family, []).append(record)
    return {
        family: _family_summary(family=family, records=family_records)
        for family, family_records in sorted(grouped.items(), key=lambda item: item[0].value)
    }


def _family_summary(
    *,
    family: WorkloadScenarioFamily,
    records: list[BenchmarkRequestRecord],
) -> FamilyBenchmarkSummary:
    summary = summarize_records_without_families(records)
    return FamilyBenchmarkSummary(
        family=family,
        request_count=summary.request_count,
        success_count=summary.success_count,
        failure_count=summary.failure_count,
        avg_latency_ms=summary.avg_latency_ms,
        p50_latency_ms=summary.p50_latency_ms,
        p95_latency_ms=summary.p95_latency_ms,
        avg_ttft_ms=summary.avg_ttft_ms,
        p50_ttft_ms=summary.p50_ttft_ms,
        p95_ttft_ms=summary.p95_ttft_ms,
        total_output_tokens=summary.total_output_tokens,
        avg_output_tokens=summary.avg_output_tokens,
        avg_tokens_per_second=summary.avg_tokens_per_second,
        p95_tokens_per_second=summary.p95_tokens_per_second,
        fallback_count=summary.fallback_count,
        chosen_backend_counts=summary.chosen_backend_counts,
    )


def summarize_records_without_families(records: list[BenchmarkRequestRecord]) -> BenchmarkSummary:
    """Summarize records without recursively deriving family breakdowns."""

    latencies = [record.latency_ms for record in records]
    ttfts = [record.ttft_ms for record in records if record.ttft_ms is not None]
    output_tokens = [_output_tokens_for_record(record) for record in records]
    tps_values = [
        record.tokens_per_second for record in records if record.tokens_per_second is not None
    ]
    success_count = sum(1 for record in records if record.success)
    failure_count = len(records) - success_count
    chosen_backend_counts: dict[str, int] = {}
    for record in records:
        chosen_backend_counts[record.backend_name] = (
            chosen_backend_counts.get(record.backend_name, 0) + 1
        )

    return BenchmarkSummary(
        request_count=len(records),
        success_count=success_count,
        failure_count=failure_count,
        avg_latency_ms=_average(latencies),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        avg_ttft_ms=None if not ttfts else _average(ttfts),
        p50_ttft_ms=None if not ttfts else _percentile(ttfts, 50),
        p95_ttft_ms=None if not ttfts else _percentile(ttfts, 95),
        total_output_tokens=sum(output_tokens),
        avg_output_tokens=_average([float(tokens) for tokens in output_tokens]),
        avg_tokens_per_second=None if not tps_values else _average(tps_values),
        p95_tokens_per_second=None if not tps_values else _percentile(tps_values, 95),
        fallback_count=sum(1 for record in records if record.fallback_used),
        chosen_backend_counts=dict(sorted(chosen_backend_counts.items())),
        hybrid_summary=_summarize_hybrid_records(records),
    )


def _cache_observation_from_capabilities(
    capabilities: BackendCapabilities | None,
) -> CacheObservation | None:
    if capabilities is None:
        return CacheObservation()
    cache = capabilities.cache_capabilities
    return CacheObservation(
        supports_prefix_cache=cache.supports_prefix_cache,
        supports_prompt_cache_read=cache.supports_prompt_cache_read,
        supports_prompt_cache_write=cache.supports_prompt_cache_write,
        supports_kv_cache_reuse=cache.supports_kv_cache_reuse,
    )


def write_artifact(artifact: BenchmarkRunArtifact, output_path: Path) -> Path:
    """Write a benchmark artifact to disk in a stable JSON format."""

    return write_json_model(artifact, output_path)


def write_markdown_report(markdown: str, output_path: Path) -> Path:
    """Write a Markdown benchmark report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return output_path


def load_benchmark_artifact_model(
    artifact_path: Path,
) -> (
    BenchmarkRunArtifact
    | BenchmarkComparisonArtifact
    | BenchmarkTargetComparisonArtifact
    | CounterfactualSimulationArtifact
    | CounterfactualSimulationComparisonArtifact
    | PolicyRecommendationReportArtifact
    | OptimizationCandidateConfigurationArtifact
    | OptimizationTrialArtifact
    | OptimizationCampaignArtifact
):
    """Load a benchmark or comparison artifact from disk."""

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if "campaign_artifact_id" in payload:
        return OptimizationCampaignArtifact.model_validate(payload)
    if "trial_artifact_id" in payload:
        return OptimizationTrialArtifact.model_validate(payload)
    if "candidate_configuration_id" in payload and "candidate" in payload:
        return OptimizationCandidateConfigurationArtifact.model_validate(payload)
    if "recommendation_report_id" in payload:
        return PolicyRecommendationReportArtifact.model_validate(payload)
    if "simulation_comparison_id" in payload:
        return CounterfactualSimulationComparisonArtifact.model_validate(payload)
    if "simulation_id" in payload:
        return CounterfactualSimulationArtifact.model_validate(payload)
    if "comparison_id" in payload:
        return BenchmarkTargetComparisonArtifact.model_validate(payload)
    if "scenario_name" in payload and "results" in payload:
        return BenchmarkComparisonArtifact.model_validate(payload)
    return BenchmarkRunArtifact.model_validate(payload)


def default_markdown_report_path(base_dir: Path, run_id: str) -> Path:
    """Return the default output path for a Markdown report."""

    return base_dir / f"{run_id}.md"


def default_generated_report_path(
    artifact_paths: list[Path],
    output_dir: Path | None = None,
) -> Path:
    """Return the default path for a generated markdown report."""

    if output_dir is not None:
        if len(artifact_paths) == 1:
            return output_dir / f"{artifact_paths[0].stem}.md"
        return output_dir / "benchmark-report.md"
    base_dir = artifact_paths[0].parent
    if len(artifact_paths) == 1:
        return base_dir / f"{artifact_paths[0].stem}.md"
    return base_dir / "benchmark-report.md"


def render_run_report_markdown(artifact: BenchmarkRunArtifact) -> str:
    """Render a compact Markdown report for a single benchmark artifact."""

    workload = artifact.scenario.workload_generation
    worker_inventory_summary = _worker_inventory_summary(
        artifact.environment.worker_instance_inventory
    )
    route_distribution = _route_distribution(artifact.records)
    error_categories = _error_category_counts(artifact.records)
    admission_outcomes = _admission_outcome_counts(artifact.records)
    queue_wait_summary = _queue_wait_summary(artifact.records)
    breaker_phases = _breaker_phase_counts(artifact.records)
    breaker_reasons = _breaker_reason_counts(artifact.records)
    affinity_dispositions = _affinity_disposition_counts(artifact.records)
    rollout_dispositions = _rollout_disposition_counts(artifact.records)
    canary_policies = _canary_policy_counts(artifact.records)
    shadow_dispositions = _shadow_disposition_counts(artifact.records)
    shadow_targets = _shadow_target_counts(artifact.records)
    control_plane_notes = _control_plane_note_counts(artifact.records)
    execution_target = (
        artifact.execution_target.target_type.value
        if artifact.execution_target
        else "routing_policy"
    )
    p95_ttft = (
        artifact.summary.p95_ttft_ms if artifact.summary.p95_ttft_ms is not None else "unavailable"
    )
    deployment_target = (
        artifact.environment.deployment_target.value
        if artifact.environment.deployment_target is not None
        else "unspecified"
    )
    deployment_profile = (
        artifact.environment.deployment_profile.value
        if artifact.environment.deployment_profile is not None
        else "unspecified"
    )
    config_profile_name = artifact.environment.config_profile_name or "unspecified"
    topology_capture_source = artifact.environment.topology_capture_source or "not_captured"
    evidence_summary = _resolved_evidence_summary_for_artifact(artifact)
    run_kind = evidence_summary.run_kind if evidence_summary is not None else artifact.run_kind
    lines = [
        f"# Switchyard Benchmark Report: {artifact.run_id}",
        "",
        "## Run Metadata",
        f"- Schema version: `{artifact.schema_version.value}`",
        f"- Timestamp: `{artifact.timestamp.isoformat()}`",
        f"- Git revision: `{artifact.git_revision or 'unavailable'}`",
        f"- Model alias: `{artifact.model_alias or artifact.scenario.model}`",
        f"- Run kind: `{run_kind.value}`",
        "",
        "## Environment",
        f"- Benchmark mode: `{artifact.environment.benchmark_mode}`",
        f"- Platform: `{artifact.environment.platform}`",
        f"- Machine: `{artifact.environment.machine}`",
        f"- Python: `{artifact.environment.python_version}`",
        f"- Gateway URL: `{artifact.environment.gateway_base_url or 'local-adapter-path'}`",
        f"- Deployment target: `{deployment_target}`",
        f"- Deployment profile: `{deployment_profile}`",
        f"- Config profile: `{config_profile_name}`",
        f"- Worker instances captured: `{len(artifact.environment.worker_instance_inventory)}`",
        (
            "- Captured locality mix: "
            f"`{worker_inventory_summary['local']} local / "
            f"{worker_inventory_summary['remote']} remote / "
            f"{worker_inventory_summary['external']} external`"
        ),
        f"- Topology capture source: `{topology_capture_source}`",
        "",
        "## Benchmark Configuration",
        f"- Execution target: `{execution_target}`",
        f"- Policy: `{artifact.policy.value}`",
        f"- Concurrency: `{artifact.run_config.concurrency}`",
        f"- Warmup requests: `{artifact.run_config.warmup.request_count}`",
        f"- Timeout seconds: `{artifact.run_config.timeout_seconds}`",
        "",
        "## Scenario Mix",
        f"- Scenario: `{artifact.scenario.name}`",
        f"- Workload shape: `{artifact.scenario.workload_shape.value}`",
        f"- Workload pattern: `{workload.pattern.value}`",
        f"- Workload seed: `{workload.seed}`",
        f"- Request count: `{artifact.request_count}`",
        "",
        "## Aggregate Metrics",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Successes | `{artifact.summary.success_count}` |",
        f"| Failures | `{artifact.summary.failure_count}` |",
        f"| Average latency | `{artifact.summary.avg_latency_ms:.3f} ms` |",
        f"| P50 latency | `{artifact.summary.p50_latency_ms:.3f} ms` |",
        f"| P95 latency | `{artifact.summary.p95_latency_ms:.3f} ms` |",
        f"| P95 TTFT | `{p95_ttft}` |",
        f"| Total output tokens | `{artifact.summary.total_output_tokens}` |",
    ]
    if artifact.environment.control_plane_image is not None:
        image_tag = artifact.environment.control_plane_image.image_tag or "unavailable"
        git_sha = artifact.environment.control_plane_image.git_sha or "unavailable"
        lines[15:15] = [
            f"- Control-plane image tag: `{image_tag}`",
            f"- Control-plane git revision: `{git_sha}`",
        ]
    if artifact.summary.avg_tokens_per_second is not None:
        lines.append(f"| Average tokens/sec | `{artifact.summary.avg_tokens_per_second:.3f}` |")
    if evidence_summary is not None:
        evidence_classes = _format_empty_distribution(
            _enum_key_distribution(evidence_summary.evidence_class_counts),
            empty_label="none",
        )
        cloud_identities = _format_empty_distribution(
            evidence_summary.cloud_worker_identity_counts,
            empty_label="none captured",
        )
        cloud_runtimes = _format_empty_distribution(
            evidence_summary.cloud_runtime_counts,
            empty_label="none captured",
        )
        cloud_runtime_versions = _format_empty_distribution(
            evidence_summary.cloud_runtime_version_counts,
            empty_label="none captured",
        )
        provider_tags = _format_empty_distribution(
            evidence_summary.cloud_provider_counts,
            empty_label="none",
        )
        region_tags = _format_empty_distribution(
            evidence_summary.cloud_region_counts,
            empty_label="none",
        )
        budget_outcomes = _format_empty_distribution(
            evidence_summary.observed_budget_outcome_counts,
            empty_label="none captured",
        )
        lines.extend(
            [
                "",
                "## Evidence Posture",
                f"- Evidence classes: {evidence_classes}",
                (
                    "- Cloud request mix: "
                    f"observed=`{evidence_summary.observed_cloud_request_count}` "
                    f"mock=`{evidence_summary.mock_request_count}` "
                    f"configured=`{evidence_summary.configured_cloud_request_count}` "
                    f"estimated=`{evidence_summary.estimated_request_count}` "
                    f"mixed=`{evidence_summary.mixed_request_count}`"
                ),
                f"- Cloud worker identities: {cloud_identities}",
                f"- Cloud runtimes: {cloud_runtimes}",
                f"- Cloud runtime versions: {cloud_runtime_versions}",
                (f"- Cloud placement tags: providers={provider_tags} regions={region_tags}"),
                (
                    "- Observed runtime signals: "
                    f"avg_queue_delay_ms=`{evidence_summary.avg_observed_queue_delay_ms}` "
                    f"avg_remote_latency_ms=`{evidence_summary.avg_observed_remote_latency_ms}` "
                    f"observed_errors=`{evidence_summary.observed_error_count}`"
                ),
                f"- Observed budget outcomes: {budget_outcomes}",
            ]
        )
        if evidence_summary.confidence_notes:
            lines.append(f"- Confidence notes: {'; '.join(evidence_summary.confidence_notes)}")
        if artifact.comparability_limitations:
            lines.append(
                f"- Comparability limitations: {'; '.join(artifact.comparability_limitations)}"
            )
    if artifact.summary.hybrid_summary is not None:
        hybrid_summary = artifact.summary.hybrid_summary
        lines.extend(
            [
                "",
                "## Hybrid Evidence",
                (
                    "- Observed paths: "
                    f"`{hybrid_summary.local_only_count}` local / "
                    f"`{hybrid_summary.hybrid_spillover_count}` hybrid spillover / "
                    f"`{hybrid_summary.remote_only_count}` remote only / "
                    f"`{hybrid_summary.remote_blocked_count}` remote blocked"
                ),
                (
                    "- Evidence sources: "
                    f"`{hybrid_summary.observed_runtime_count}` observed / "
                    f"`{hybrid_summary.injected_condition_count}` injected mock / "
                    f"`{hybrid_summary.predictor_estimate_count}` predictor"
                ),
                (
                    "- Remote temperature: "
                    f"`{hybrid_summary.remote_cold_count}` cold / "
                    f"`{hybrid_summary.remote_warm_count}` warm"
                ),
                (
                    "- Budget posture: "
                    f"`{hybrid_summary.budget_exhausted_count}` exhausted / "
                    f"`{hybrid_summary.budget_disabled_count}` disabled"
                ),
                (
                    "- Cloud evidence: "
                    f"placement observed=`{hybrid_summary.observed_placement_evidence_count}` "
                    f"estimated=`{hybrid_summary.estimated_placement_evidence_count}` "
                    f"cost observed=`{hybrid_summary.observed_cost_evidence_count}` "
                    f"estimated=`{hybrid_summary.estimated_cost_evidence_count}`"
                ),
                (
                    "- Network penalty (ms): "
                    f"observed=`{hybrid_summary.avg_observed_network_penalty_ms}` "
                    f"injected=`{hybrid_summary.avg_injected_network_penalty_ms}` "
                    f"predicted=`{hybrid_summary.avg_predicted_network_penalty_ms}`"
                ),
                (
                    "- Modeled cost: "
                    f"total=`{hybrid_summary.total_modeled_cost}` "
                    f"avg=`{hybrid_summary.avg_modeled_cost}`"
                ),
                (
                    "- Uncertainty: "
                    f"`{hybrid_summary.low_confidence_count}` low-confidence / "
                    f"`{hybrid_summary.unsupported_count}` unsupported"
                ),
            ]
        )
        if hybrid_summary.notes:
            lines.append(f"- Notes: {'; '.join(hybrid_summary.notes)}")
    if artifact.environment.hybrid_execution is not None:
        hybrid = artifact.environment.hybrid_execution
        remote_budget_remaining = (
            "unbounded"
            if hybrid.remote_budget_requests_remaining is None
            else str(hybrid.remote_budget_requests_remaining)
        )
        lines.extend(
            [
                "",
                "## Hybrid Execution",
                f"- Enabled: `{hybrid.enabled}`",
                f"- Prefer local: `{hybrid.prefer_local}`",
                f"- Spillover enabled: `{hybrid.spillover_enabled}`",
                (
                    "- Remote budget usage: "
                    f"`{hybrid.remote_budget_requests_used}` used / "
                    f"`{remote_budget_remaining}` remaining"
                ),
                (
                    "- Remote health posture: "
                    f"`{hybrid.healthy_remote_backends}` healthy / "
                    f"`{hybrid.degraded_remote_backends}` degraded / "
                    f"`{hybrid.unavailable_remote_backends}` unavailable"
                ),
                (
                    "- Remote concurrency: "
                    f"`{hybrid.remote_in_flight_requests}` in flight / "
                    f"`{hybrid.remote_concurrency_cap or 'unbounded'}` cap"
                ),
            ]
        )
        if hybrid.notes:
            lines.append(f"- Notes: {'; '.join(hybrid.notes)}")
    if artifact.environment.remote_workers is not None:
        remote_workers = artifact.environment.remote_workers
        lines.extend(
            [
                "",
                "## Remote Worker Lifecycle",
                f"- Secure registration required: `{remote_workers.secure_registration_required}`",
                f"- Auth mode: `{remote_workers.auth_mode.value}`",
                (
                    "- Registered workers: "
                    f"`{remote_workers.registered_instance_count}` registered / "
                    f"`{remote_workers.ready_instance_count}` ready / "
                    f"`{remote_workers.stale_instance_count}` stale"
                ),
                (
                    "- Lifecycle posture: "
                    f"`{remote_workers.draining_instance_count}` draining / "
                    f"`{remote_workers.unhealthy_instance_count}` unhealthy / "
                    f"`{remote_workers.lost_instance_count}` lost / "
                    f"`{remote_workers.retired_instance_count}` retired"
                ),
            ]
        )
        if remote_workers.notes:
            lines.append(f"- Notes: {'; '.join(remote_workers.notes)}")
    lines.extend(
        [
            "",
            "## Per-Scenario Table",
            "| Family | Requests | Successes | Failures | P50 Latency (ms) | Fallbacks |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if artifact.summary.family_summaries:
        for family, summary in artifact.summary.family_summaries.items():
            lines.append(
                f"| `{family.value}` | `{summary.request_count}` | `{summary.success_count}` | "
                f"`{summary.failure_count}` | `{summary.p50_latency_ms:.3f}` | "
                f"`{summary.fallback_count}` |"
            )
    else:
        lines.append(
            f"| `{artifact.scenario.family.value}` | `{artifact.request_count}` | "
            f"`{artifact.summary.success_count}` | `{artifact.summary.failure_count}` | "
            f"`{artifact.summary.p50_latency_ms:.3f}` | `{artifact.summary.fallback_count}` |"
        )
    lines.extend(
        [
            "",
            "## Route and Backend Distributions",
            f"- Routes: {_format_distribution(route_distribution)}",
            f"- Backends: {_format_distribution(artifact.summary.chosen_backend_counts)}",
            "",
            "## Fallback and Error Summary",
            f"- Fallback count: `{artifact.summary.fallback_count}`",
            (
                "- Error categories: "
                f"{_format_empty_distribution(error_categories, empty_label='none')}"
            ),
            "",
            "## Phase 4 Control Plane",
            (
                "- Admission outcomes: "
                f"{_format_empty_distribution(admission_outcomes, empty_label='none observed')}"
            ),
            (f"- Queue waits: {_format_queue_wait_summary(queue_wait_summary)}"),
            (
                "- Breaker phases: "
                f"{_format_empty_distribution(breaker_phases, empty_label='none observed')}"
            ),
            (
                "- Breaker reasons: "
                f"{_format_empty_distribution(breaker_reasons, empty_label='none observed')}"
            ),
            (
                "- Session affinity: "
                f"{_format_empty_distribution(affinity_dispositions, empty_label='none observed')}"
            ),
            (
                "- Canary rollout: "
                f"{_format_empty_distribution(rollout_dispositions, empty_label='none observed')}"
            ),
            (
                "- Canary policies: "
                f"{_format_empty_distribution(canary_policies, empty_label='none observed')}"
            ),
            (
                "- Shadow traffic: "
                f"{_format_empty_distribution(shadow_dispositions, empty_label='none observed')}"
            ),
            (
                "- Shadow targets: "
                f"{_format_empty_distribution(shadow_targets, empty_label='none observed')}"
            ),
            (
                "- Control-plane notes: "
                f"{_format_empty_distribution(control_plane_notes, empty_label='none observed')}"
            ),
        ]
    )
    if workload.shared_prefix is not None:
        lines.append(f"- Shared prefix: `{workload.shared_prefix}`")
    if workload.pattern is WorkloadPattern.BURSTY:
        lines.append(f"- Burst size: `{workload.burst_size}`")
    return "\n".join(lines)


def render_comparison_report_markdown(artifact: BenchmarkComparisonArtifact) -> str:
    """Render a compact Markdown report for a comparison artifact."""

    lines = [
        f"# Switchyard Comparison Report: {artifact.run_id}",
        "",
        "## Run Metadata",
        f"- Schema version: `{artifact.schema_version.value}`",
        f"- Timestamp: `{artifact.timestamp.isoformat()}`",
        "",
        "## Scenario",
        f"- Name: `{artifact.scenario_name}`",
        f"- Model: `{artifact.model}`",
        f"- Workload shape: `{artifact.workload_shape.value}`",
        f"- Request count: `{artifact.request_count}`",
        f"- Best latency result: `{artifact.best_result_by_latency}`",
    ]
    if artifact.best_result_by_throughput is not None:
        lines.append(f"- Best throughput result: `{artifact.best_result_by_throughput}`")
    lines.extend(
        [
            "",
            "## Top Takeaways",
            f"- Lowest latency label: `{artifact.best_result_by_latency}`",
            f"- Best throughput label: `{artifact.best_result_by_throughput or 'unavailable'}`",
            "",
            "## Results",
            "| Label | Avg Latency (ms) | Successes | Failures | Backends |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result in artifact.results:
        lines.append(
            f"| `{result.comparison_label}` | `{result.summary.avg_latency_ms:.3f}` | "
            f"`{result.summary.success_count}` | `{result.summary.failure_count}` | "
            f"`{', '.join(result.backends_involved)}` |"
        )
    return "\n".join(lines)


def render_target_comparison_report_markdown(
    artifact: BenchmarkTargetComparisonArtifact,
) -> str:
    """Render a compact Markdown report for a two-target comparison."""

    best_latency_run = (
        artifact.comparison_summary.best_result_by_latency
        if artifact.comparison_summary
        else artifact.left.run_id
    )
    best_throughput_run = (
        artifact.comparison_summary.best_result_by_throughput
        if artifact.comparison_summary and artifact.comparison_summary.best_result_by_throughput
        else "unavailable"
    )
    lines = [
        f"# Switchyard Target Comparison: {artifact.comparison_id}",
        "",
        "## Run Metadata",
        f"- Schema version: `{artifact.schema_version.value}`",
        f"- Timestamp: `{artifact.timestamp.isoformat()}`",
        "",
        "## Source",
        f"- Source kind: `{artifact.source_kind.value}`",
        f"- Source name: `{artifact.source_name}`",
        f"- Request count: `{artifact.request_count}`",
        "",
        "## Comparability",
        f"- Directly comparable: `{artifact.comparability_assessment.directly_comparable}`",
        f"- Left run kind: `{artifact.comparability_assessment.left_run_kind.value}`",
        f"- Right run kind: `{artifact.comparability_assessment.right_run_kind.value}`",
        "- Shared evidence classes: "
        + _format_empty_distribution(
            _enum_list_distribution(artifact.comparability_assessment.shared_evidence_classes),
            empty_label="none",
        ),
        "",
        "## Targets",
        (
            f"- Left: `{artifact.left.execution_target.target_type.value}`"
            f" -> `{artifact.left.run_id}`"
        ),
        (
            f"- Right: `{artifact.right.execution_target.target_type.value}`"
            f" -> `{artifact.right.run_id}`"
        ),
        "",
        "## Aggregate Metrics",
        "| Metric | Left | Right | Delta |",
        "| --- | --- | --- | --- |",
        (
            f"| Success rate | `{artifact.left.success_rate:.6f}` | "
            f"`{artifact.right.success_rate:.6f}` | "
            f"`{artifact.delta.success_rate_delta:.6f}` |"
        ),
        (
            f"| Fallback rate | `{artifact.left.fallback_rate:.6f}` | "
            f"`{artifact.right.fallback_rate:.6f}` | "
            f"`{artifact.delta.fallback_rate_delta:.6f}` |"
        ),
        (
            f"| P50 latency (ms) | `{artifact.left.p50_latency_ms:.3f}` | "
            f"`{artifact.right.p50_latency_ms:.3f}` | "
            f"`{artifact.delta.p50_latency_delta_ms:.3f}` |"
        ),
        (
            f"| P95 latency (ms) | `{artifact.left.p95_latency_ms:.3f}` | "
            f"`{artifact.right.p95_latency_ms:.3f}` | "
            f"`{artifact.delta.p95_latency_delta_ms:.3f}` |"
        ),
    ]
    if artifact.comparability_assessment.limitations:
        lines.append("- Limitations: " + "; ".join(artifact.comparability_assessment.limitations))
    if artifact.left.evidence_summary is not None or artifact.right.evidence_summary is not None:
        left_evidence = (
            {}
            if artifact.left.evidence_summary is None
            else _enum_key_distribution(artifact.left.evidence_summary.evidence_class_counts)
        )
        right_evidence = (
            {}
            if artifact.right.evidence_summary is None
            else _enum_key_distribution(artifact.right.evidence_summary.evidence_class_counts)
        )
        lines.extend(
            [
                "",
                "## Evidence Posture",
                "- Left evidence classes: "
                + _format_empty_distribution(left_evidence, empty_label="none"),
                "- Right evidence classes: "
                + _format_empty_distribution(right_evidence, empty_label="none"),
            ]
        )
    if artifact.delta.avg_tokens_per_second_delta is not None:
        lines.append(
            f"| Avg tokens/sec | `{artifact.left.avg_tokens_per_second}` | "
            f"`{artifact.right.avg_tokens_per_second}` | "
            f"`{artifact.delta.avg_tokens_per_second_delta:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Hybrid Evaluation",
        ]
    )
    if artifact.delta.hybrid_summary is not None:
        hybrid = artifact.delta.hybrid_summary
        lines.extend(
            [
                (
                    "- Outcome counts: "
                    f"`{hybrid.beneficial_count}` beneficial / "
                    f"`{hybrid.harmful_count}` harmful / "
                    f"`{hybrid.inconclusive_count}` inconclusive / "
                    f"`{hybrid.unsupported_count}` unsupported"
                ),
                (
                    "- Evidence quality: "
                    f"`{hybrid.direct_observation_count}` direct / "
                    f"`{hybrid.predictor_estimate_count}` predictor / "
                    f"`{hybrid.low_confidence_count}` low-confidence"
                ),
                (
                    "- Delta posture: "
                    f"observed penalty delta=`{hybrid.observed_network_penalty_delta_ms}` "
                    f"modeled cost delta=`{hybrid.modeled_cost_delta}` "
                    f"budget exhausted delta=`{hybrid.budget_exhausted_delta}`"
                ),
            ]
        )
        if hybrid.notes:
            lines.append(f"- Notes: {'; '.join(hybrid.notes)}")
    else:
        lines.append("- No hybrid-specific evidence was captured for this comparison.")
    lines.extend(
        [
            "",
            "## Route and Backend Distributions",
            f"- Left routes: {_format_distribution(artifact.left.route_distribution)}",
            f"- Right routes: {_format_distribution(artifact.right.route_distribution)}",
            f"- Left backends: {_format_distribution(artifact.left.backend_distribution)}",
            f"- Right backends: {_format_distribution(artifact.right.backend_distribution)}",
            "",
            "## Top Takeaways",
            f"- Best latency run: `{best_latency_run}`",
            f"- Best throughput run: `{best_throughput_run}`",
            "",
            "## Notable Per-Scenario Deltas",
            (
                "| Key | Latency Delta (ms) | Hybrid Outcome | Evidence | "
                "Success Changed | Backend Changed | Route Changed |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for delta in artifact.delta.notable_scenario_deltas[:10]:
        lines.append(
            f"| `{delta.key}` | `{delta.latency_delta_ms:.3f}` | "
            f"`{delta.hybrid_outcome.value}` | `{delta.evidence_kind.value}` | "
            f"`{delta.success_changed}` | `{delta.backend_changed}` | "
            f"`{delta.route_changed}` |"
        )
    if len(artifact.delta.notable_scenario_deltas) == 0:
        lines.append(
            "| none | `0.000` | `unsupported` | `unsupported` | `False` | `False` | `False` |"
        )
    return "\n".join(lines)


def render_simulation_report_markdown(artifact: CounterfactualSimulationArtifact) -> str:
    """Render a compact Markdown report for an offline simulation artifact."""

    recommendation = (
        "Keep baseline"
        if artifact.summary.changed_count == 0
        else (
            "Bounded rollout candidate"
            if artifact.summary.changed_count >= max(1, artifact.summary.request_count // 4)
            else "Recommendation mode candidate"
        )
    )
    lines = [
        f"# Switchyard Simulation Report: {artifact.simulation_id}",
        "",
        "## Run Metadata",
        f"- Schema version: `{artifact.schema_version.value}`",
        f"- Timestamp: `{artifact.timestamp.isoformat()}`",
        f"- Policy ID: `{artifact.policy.policy_id}`",
        f"- Policy version: `{artifact.policy.policy_version}`",
        f"- Mode: `{artifact.policy.mode.value}`",
        f"- Objective: `{artifact.policy.objective.value}`",
        f"- Minimum evidence count: `{artifact.policy.min_evidence_count}`",
        f"- Evaluation source runs: `{', '.join(artifact.source_run_ids) or 'none'}`",
        f"- Historical source runs: `{', '.join(artifact.historical_source_run_ids) or 'none'}`",
        "",
        "## Summary",
        f"- Requests evaluated: `{artifact.summary.request_count}`",
        f"- Route changes recommended: `{artifact.summary.changed_count}`",
        f"- Direct observations used: `{artifact.summary.direct_observation_count}`",
        f"- Predictor estimates used: `{artifact.summary.predictor_estimate_count}`",
        f"- Low-confidence estimates: `{artifact.summary.low_confidence_count}`",
        f"- Unsupported requests: `{artifact.summary.unsupported_count}`",
        f"- Guardrail blocks: `{artifact.summary.guardrail_block_count}`",
        f"- Insufficient-data requests: `{artifact.summary.insufficient_data_count}`",
        f"- Projected average latency: `{artifact.summary.projected_avg_latency_ms}`",
        f"- Projected error rate: `{artifact.summary.projected_error_rate}`",
        (f"- Projected average tokens/sec: `{artifact.summary.projected_avg_tokens_per_second}`"),
        "",
        "## Backend Shifts",
        f"- Observed backends: {_format_distribution(artifact.summary.observed_backend_counts)}",
        (
            "- Recommended backends: "
            f"{_format_distribution(artifact.summary.recommended_backend_counts)}"
        ),
        "",
        "## Recommendation",
        f"- Operator posture: `{recommendation}`",
    ]
    if artifact.policy.rationale:
        lines.extend(["", "## Policy Rationale"])
        lines.extend(f"- {item}" for item in artifact.policy.rationale)
    if artifact.summary.limitation_notes:
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in artifact.summary.limitation_notes)
    if artifact.summary.bucket_summaries:
        lines.extend(
            [
                "",
                "## Bucket Summary",
                (
                    "| Dimension | Bucket | Requests | Changed | Direct | Predictor | "
                    "Low Confidence | Unsupported |"
                ),
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for bucket in artifact.summary.bucket_summaries[:12]:
            lines.append(
                f"| `{bucket.dimension.value}` | `{bucket.bucket_key}` | "
                f"`{bucket.request_count}` | "
                f"`{bucket.changed_count}` | `{bucket.direct_observation_count}` | "
                f"`{bucket.predictor_estimate_count}` | `{bucket.low_confidence_count}` | "
                f"`{bucket.unsupported_count}` |"
            )
    return "\n".join(lines)


def render_simulation_comparison_report_markdown(
    artifact: CounterfactualSimulationComparisonArtifact,
) -> str:
    """Render a compact Markdown report for a multi-policy simulation comparison."""

    best_policy = max(
        artifact.evaluations,
        key=lambda evaluation: (
            evaluation.summary.request_count - evaluation.summary.unsupported_count,
            -(evaluation.summary.projected_avg_latency_ms or float("inf")),
            -(evaluation.summary.projected_error_rate or 1.0),
        ),
    ).policy.policy_id
    lines = [
        f"# Switchyard Simulation Comparison Report: {artifact.simulation_comparison_id}",
        "",
        "## Sources",
        f"- Benchmark runs: `{', '.join(artifact.source_run_ids) or 'none'}`",
        f"- Captured traces: `{', '.join(artifact.source_trace_ids) or 'none'}`",
        f"- Historical runs: `{', '.join(artifact.historical_source_run_ids) or 'none'}`",
        f"- Historical traces: `{', '.join(artifact.historical_source_trace_ids) or 'none'}`",
        "",
        "## Policy Table",
        (
            "| Policy | Objective | Requests | Changed | Direct | Predictor | "
            "Low Confidence | Unsupported | Avg Latency |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for evaluation in artifact.evaluations:
        summary = evaluation.summary
        lines.append(
            f"| `{evaluation.policy.policy_id}` | `{evaluation.policy.objective.value}` | "
            f"`{summary.request_count}` | `{summary.changed_count}` | "
            f"`{summary.direct_observation_count}` | `{summary.predictor_estimate_count}` | "
            f"`{summary.low_confidence_count}` | `{summary.unsupported_count}` | "
            f"`{summary.projected_avg_latency_ms}` |"
        )
    lines.extend(["", "## Recommendation", f"- Best supported policy: `{best_policy}`"])
    if artifact.limitation_notes:
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in artifact.limitation_notes)
    return "\n".join(lines)


def render_loaded_artifact_markdown(
    artifact: (
        BenchmarkRunArtifact
        | BenchmarkComparisonArtifact
        | BenchmarkTargetComparisonArtifact
        | CounterfactualSimulationArtifact
        | CounterfactualSimulationComparisonArtifact
        | PolicyRecommendationReportArtifact
        | OptimizationCandidateConfigurationArtifact
        | OptimizationTrialArtifact
        | OptimizationCampaignArtifact
    ),
) -> str:
    """Render markdown for any supported benchmark artifact type."""

    if isinstance(artifact, OptimizationCampaignArtifact):
        return render_optimization_campaign_report_markdown(artifact)
    if isinstance(artifact, OptimizationTrialArtifact):
        return render_optimization_trial_report_markdown(artifact)
    if isinstance(artifact, OptimizationCandidateConfigurationArtifact):
        return render_optimization_candidate_report_markdown(artifact)
    if isinstance(artifact, BenchmarkRunArtifact):
        return render_run_report_markdown(artifact)
    if isinstance(artifact, BenchmarkTargetComparisonArtifact):
        return render_target_comparison_report_markdown(artifact)
    if isinstance(artifact, CounterfactualSimulationComparisonArtifact):
        return render_simulation_comparison_report_markdown(artifact)
    if isinstance(artifact, CounterfactualSimulationArtifact):
        return render_simulation_report_markdown(artifact)
    if isinstance(artifact, PolicyRecommendationReportArtifact):
        return render_policy_recommendation_report_markdown(artifact)
    return render_comparison_report_markdown(artifact)


def render_artifact_bundle_markdown(
    artifacts: list[
        BenchmarkRunArtifact
        | BenchmarkComparisonArtifact
        | BenchmarkTargetComparisonArtifact
        | CounterfactualSimulationArtifact
        | CounterfactualSimulationComparisonArtifact
        | PolicyRecommendationReportArtifact
        | OptimizationCandidateConfigurationArtifact
        | OptimizationTrialArtifact
        | OptimizationCampaignArtifact
    ],
) -> str:
    """Render a compact markdown bundle for one or more artifacts."""

    sections = [render_loaded_artifact_markdown(artifact) for artifact in artifacts]
    return "\n\n---\n\n".join(sections)


def render_optimization_candidate_report_markdown(
    artifact: OptimizationCandidateConfigurationArtifact,
) -> str:
    """Render a compact Markdown report for one candidate configuration artifact."""

    lines = [
        f"# Switchyard Optimization Candidate: {artifact.candidate_configuration_id}",
        "",
        f"- Campaign: `{artifact.campaign_id}`",
        f"- Candidate kind: `{artifact.candidate.candidate_kind.value}`",
        f"- Config profile: `{artifact.config_profile_id}`",
        f"- Baseline profile: `{artifact.baseline_config_profile_id}`",
        f"- Knob changes: `{len(artifact.knob_changes)}`",
        f"- Objectives in scope: `{len(artifact.objectives_in_scope)}`",
        f"- Constraints in scope: `{len(artifact.constraints_in_scope)}`",
        (
            "- Generation strategy: "
            f"`{artifact.generation.strategy.value if artifact.generation is not None else 'none'}`"
        ),
        (
            "- Eligibility: "
            f"`{_candidate_eligibility_label(artifact)}`"
        ),
        (
            "- Expected evidence kinds: "
            f"`{', '.join(kind.value for kind in artifact.expected_evidence_kinds) or 'none'}`"
        ),
    ]
    if artifact.knob_changes:
        lines.extend(["", "## Knob Changes"])
        lines.extend(
            f"- `{change.knob_id}`: `{change.baseline_value}` -> `{change.candidate_value}`"
            for change in artifact.knob_changes
        )
    return "\n".join(lines)


def render_optimization_trial_report_markdown(
    artifact: OptimizationTrialArtifact,
) -> str:
    """Render a compact Markdown report for one optimization trial artifact."""

    evidence_kinds = ", ".join(
        sorted({record.evidence_kind.value for record in artifact.evidence_records})
    )
    remote_budget_assessments = [
        assessment
        for assessment in artifact.constraint_assessments
        if assessment.dimension
        in {
            OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
            OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE,
        }
    ]
    lines = [
        f"# Switchyard Optimization Trial: {artifact.trial_artifact_id}",
        "",
        f"- Campaign: `{artifact.campaign_id}`",
        f"- Candidate config: `{artifact.candidate_configuration.candidate_configuration_id}`",
        f"- Config profile: `{artifact.candidate_configuration.config_profile_id}`",
        f"- Trial status: `{artifact.result_status.value}`",
        f"- Evidence kinds: `{evidence_kinds or 'none'}`",
        f"- Objective assessments: `{len(artifact.objective_assessments)}`",
        f"- Constraint assessments: `{len(artifact.constraint_assessments)}`",
    ]
    if artifact.recommendation_summary is not None:
        recommendation = artifact.recommendation_summary
        lines.extend(
            [
                f"- Recommendation: `{recommendation.disposition.value}`",
                f"- Recommendation label: `{recommendation.recommendation_label.value}`",
                (
                    "- Recommendation evidence kinds: "
                    f"`{', '.join(kind.value for kind in recommendation.evidence_kinds) or 'none'}`"
                ),
            ]
        )
    if artifact.promotion_decision is not None:
        lines.append(f"- Promotion decision: `{artifact.promotion_decision.disposition.value}`")
    if artifact.candidate_configuration.knob_changes:
        lines.extend(["", "## Candidate Diff"])
        lines.extend(
            (f"- `{change.knob_id}`: `{change.baseline_value}` -> `{change.candidate_value}`")
            for change in artifact.candidate_configuration.knob_changes
        )
    if artifact.recommendation_summary is not None:
        recommendation = artifact.recommendation_summary
        lines.extend(
            [
                "",
                "## Workload Impact",
                (f"- Helps: `{', '.join(recommendation.benefited_workload_families) or 'none'}`"),
                (f"- Hurts: `{', '.join(recommendation.regressed_workload_families) or 'none'}`"),
            ]
        )
    if remote_budget_assessments:
        lines.extend(["", "## Remote Budget Posture"])
        lines.extend(
            (
                f"- `{assessment.constraint_id}`: "
                f"`{'satisfied' if assessment.satisfied else 'violated'}`"
            )
            for assessment in remote_budget_assessments
        )
    if artifact.stale_reason is not None:
        lines.append(f"- Stale reason: `{artifact.stale_reason}`")
    if artifact.invalidation_reason is not None:
        lines.append(f"- Invalidation reason: `{artifact.invalidation_reason}`")
    return "\n".join(lines)


def render_optimization_campaign_report_markdown(
    artifact: OptimizationCampaignArtifact,
) -> str:
    """Render a compact Markdown report for one optimization campaign artifact."""

    evidence_kinds = ", ".join(
        sorted({record.evidence_kind.value for record in artifact.evidence_records})
    )
    recommendation_counts: dict[str, int] = {}
    helped_workload_families = sorted(
        {
            family
            for recommendation in artifact.recommendation_summaries
            for family in recommendation.benefited_workload_families
        }
    )
    hurt_workload_families = sorted(
        {
            family
            for recommendation in artifact.recommendation_summaries
            for family in recommendation.regressed_workload_families
        }
    )
    remote_budget_constraints = sorted(
        {
            assessment.constraint_id
            for trial in artifact.trials
            for assessment in trial.constraint_assessments
            if assessment.dimension
            in {
                OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE,
            }
        }
    )
    for recommendation in artifact.recommendation_summaries:
        recommendation_counts[recommendation.disposition.value] = (
            recommendation_counts.get(recommendation.disposition.value, 0) + 1
        )
    lines = [
        f"# Switchyard Optimization Campaign: {artifact.campaign_artifact_id}",
        "",
        f"- Campaign ID: `{artifact.campaign.campaign_id}`",
        f"- Optimization profile: `{artifact.campaign.optimization_profile_id}`",
        f"- Objective: `{artifact.campaign.objective.value}`",
        f"- Status: `{artifact.result_status.value}`",
        f"- Candidate configurations: `{1 + len(artifact.candidate_configurations)}`",
        f"- Trials: `{len(artifact.trials)}`",
        f"- Recommendations: `{len(artifact.recommendation_summaries)}`",
        f"- Promotion decisions: `{len(artifact.promotion_decisions)}`",
        f"- Evidence kinds: `{evidence_kinds or 'none'}`",
        (
            "- Recommendation status counts: "
            f"`{_recommendation_counts_label(recommendation_counts)}`"
        ),
        (f"- Helps workload families: `{', '.join(helped_workload_families) or 'none'}`"),
        (f"- Hurts workload families: `{', '.join(hurt_workload_families) or 'none'}`"),
        (
            "- Remote budget constraints involved: "
            f"`{', '.join(remote_budget_constraints) or 'none'}`"
        ),
    ]
    if artifact.campaign.default_workload_set_ids:
        lines.append(f"- Workload sets: `{', '.join(artifact.campaign.default_workload_set_ids)}`")
    if artifact.trials:
        lines.extend(["", "## Trials"])
        lines.extend(
            (
                f"- `{trial.trial_artifact_id}`: "
                f"`{trial.candidate_configuration.config_profile_id}` "
                f"recommendation=`{_trial_recommendation_label(trial)}` "
                f"evidence=`{_trial_evidence_label(trial)}`"
            )
            for trial in artifact.trials
        )
    if artifact.stale_reason is not None:
        lines.append(f"- Stale reason: `{artifact.stale_reason}`")
    if artifact.invalidation_reason is not None:
        lines.append(f"- Invalidation reason: `{artifact.invalidation_reason}`")
    return "\n".join(lines)


def render_forge_campaign_inspection_markdown(
    inspection: ForgeCampaignInspectionResponse,
) -> str:
    """Render a detailed Markdown report for campaign inspection summaries."""

    lines = ["# Switchyard Forge Stage A Inspection", ""]
    if not inspection.campaigns:
        lines.append("- No campaign artifacts were provided.")
        return "\n".join(lines)
    for campaign in inspection.campaigns:
        lines.extend(
            [
                f"## Campaign `{campaign.campaign_artifact_id}`",
                f"- Campaign ID: `{campaign.campaign_id}`",
                f"- Optimization profile: `{campaign.optimization_profile_id}`",
                f"- Objective: `{campaign.objective}`",
                f"- Status: `{campaign.result_status.value}`",
                f"- Trustworthy: `{campaign.trustworthy}`",
                (
                    "- Evidence kinds: "
                    f"`{', '.join(kind.value for kind in campaign.evidence_kinds) or 'none'}`"
                ),
                (
                    "- Recommendation counts: "
                    f"`{_recommendation_counts_label(campaign.recommendation_status_counts)}`"
                ),
                (
                    "- Helps workload families: "
                    f"`{', '.join(campaign.helped_workload_families) or 'none'}`"
                ),
                (
                    "- Hurts workload families: "
                    f"`{', '.join(campaign.hurt_workload_families) or 'none'}`"
                ),
                f"- Remote budget involved: `{campaign.remote_budget_involved}`",
                (
                    "- Remote budget constraints: "
                    f"`{', '.join(campaign.remote_budget_constraint_ids) or 'none'}`"
                ),
            ]
        )

        if campaign.honesty_warnings:
            lines.extend(["", "### Honesty Warnings"])
            for warning in campaign.honesty_warnings:
                lines.append(
                    f"- [{warning.severity}] `{warning.kind.value}`: {warning.message}"
                )

        lines.extend(["", "### Trials"])
        for trial in campaign.trials:
            lines.extend(
                [
                    "",
                    f"#### Trial `{trial.trial_artifact_id}`",
                    f"- Candidate: `{trial.candidate_configuration_id}`",
                    f"- Config profile: `{trial.config_profile_id}`",
                    f"- Baseline profile: `{trial.baseline_config_profile_id}`",
                    f"- Candidate kind: `{trial.candidate_kind.value}`",
                    f"- Status: `{trial.trial_status.value}`",
                    (
                        "- Recommendation: "
                        f"`{_inspection_trial_recommendation_label(trial)}`"
                    ),
                ]
            )
            if trial.recommendation_label is not None:
                lines.append(
                    f"- Recommendation label: `{trial.recommendation_label.value}`"
                )
            lines.append(
                "- Evidence kinds: "
                f"`{', '.join(kind.value for kind in trial.evidence_kinds) or 'none'}`"
            )
            lines.append(
                "- Helps workload families: "
                f"`{', '.join(trial.helped_workload_families) or 'none'}`"
            )
            lines.append(
                "- Hurts workload families: "
                f"`{', '.join(trial.hurt_workload_families) or 'none'}`"
            )
            if trial.routing_policy is not None:
                lines.append(f"- Routing policy: `{trial.routing_policy}`")
            if trial.comparison_rank is not None:
                lines.append(
                    f"- Comparison rank: `{trial.comparison_rank}` "
                    f"pareto=`{trial.pareto_optimal}` "
                    f"dominated=`{trial.dominated}`"
                )
            if trial.remote_budget_involved:
                lines.append(
                    "- Remote budget outcomes: "
                    f"`{', '.join(trial.remote_budget_constraint_outcomes) or 'none'}`"
                )
            if trial.diff_entries:
                lines.append("")
                lines.append("##### Candidate Diff")
                for diff in trial.diff_entries:
                    lines.append(
                        f"- `{diff.knob_id}`: "
                        f"`{diff.baseline_value}` -> `{diff.candidate_value}`"
                    )
            lines.extend(
                [
                    "",
                    "##### Provenance",
                    f"- Campaign: `{trial.provenance.campaign_id}`",
                    f"- Trial artifact: `{trial.provenance.trial_artifact_id}`",
                    (
                        "- Candidate configuration: "
                        f"`{trial.provenance.candidate_configuration_id}`"
                    ),
                    f"- Baseline profile: `{trial.provenance.baseline_config_profile_id}`",
                ]
            )
            if trial.provenance.generation_strategy is not None:
                lines.append(
                    "- Generation strategy: "
                    f"`{trial.provenance.generation_strategy}`"
                )
            if trial.provenance.eligibility_status is not None:
                lines.append(
                    f"- Eligibility: `{trial.provenance.eligibility_status}`"
                )
        lines.append("")
    return "\n".join(lines).rstrip()


def render_forge_promotion_runtime_markdown(
    summary: ForgePromotionRuntimeSummary,
) -> str:
    """Render a detailed Markdown report for live Forge promotion state."""

    lines = [
        "# Switchyard Forge Stage A Promotion",
        "",
        f"- Rollout artifact: `{summary.rollout_artifact_id or 'none'}`",
        f"- Lifecycle state: `{_promotion_lifecycle_label(summary)}`",
        f"- Applied: `{summary.applied}`",
        f"- Rollback available: `{summary.rollback_available}`",
        f"- Requires operator review: `{summary.requires_operator_review}`",
        "",
        "## Config Profiles",
        f"- Baseline: `{summary.baseline_config_profile_id}`",
        f"- Active: `{summary.active_config_profile_id}`",
        f"- Candidate: `{summary.candidate_config_profile_id or 'none'}`",
        "",
        "## Rollout Controls",
        f"- Rollout mode: `{summary.rollout_mode.value}`",
        f"- Canary percentage: `{summary.canary_percentage}`",
        f"- Promotion disposition: `{summary.promotion_disposition.value}`",
        (
            "- Evidence kinds: "
            f"`{', '.join(kind.value for kind in summary.evidence_kinds) or 'none'}`"
        ),
    ]
    if summary.candidate_kind is not None:
        lines.append(f"- Candidate kind: `{summary.candidate_kind.value}`")
    if summary.routing_policy is not None:
        lines.append(f"- Routing policy: `{summary.routing_policy}`")

    if summary.campaign_id is not None:
        lines.extend(
            [
                "",
                "## Campaign Identity",
                f"- Campaign: `{summary.campaign_id}`",
                f"- Campaign artifact: `{summary.campaign_artifact_id or 'none'}`",
                f"- Trial artifact: `{summary.trial_artifact_id or 'none'}`",
                f"- Candidate: `{summary.candidate_configuration_id or 'none'}`",
            ]
        )

    if summary.applied_knob_changes:
        lines.extend(["", "## Applied Knob Changes"])
        for knob in summary.applied_knob_changes:
            applied_label = "applied" if knob.applied else "NOT applied"
            lines.append(
                f"- `{knob.knob_id}`: "
                f"`{knob.baseline_value}` -> `{knob.candidate_value}` "
                f"({applied_label})"
            )

    if summary.blocked_knob_changes:
        lines.extend(["", "## Blocked Knob Changes"])
        for knob in summary.blocked_knob_changes:
            lines.append(
                f"- `{knob.knob_id}`: {', '.join(knob.notes) or 'blocked'}"
            )

    if summary.comparison is not None:
        comp = summary.comparison
        lines.extend(
            [
                "",
                "## Canary Comparison",
                f"- Comparison artifact: `{comp.comparison_artifact_id}`",
                f"- Recommendation: `{comp.recommendation_disposition.value}`",
                f"- Label: `{comp.recommendation_label.value}`",
                (
                    f"- Rank: `{comp.rank}` "
                    f"pareto=`{comp.pareto_optimal}` "
                    f"dominated=`{comp.dominated}`"
                ),
                (
                    "- Evidence kinds: "
                    f"`{', '.join(kind.value for kind in comp.evidence_kinds) or 'none'}`"
                ),
                (
                    "- Improved objectives: "
                    f"`{', '.join(comp.improved_objective_ids) or 'none'}`"
                ),
                (
                    "- Regressed objectives: "
                    f"`{', '.join(comp.regressed_objective_ids) or 'none'}`"
                ),
                (
                    "- Satisfied constraints: "
                    f"`{', '.join(comp.satisfied_constraint_ids) or 'none'}`"
                ),
                (
                    "- Violated constraints: "
                    f"`{', '.join(comp.violated_constraint_ids) or 'none'}`"
                ),
                (
                    "- Helps workload families: "
                    f"`{', '.join(comp.benefited_workload_families) or 'none'}`"
                ),
                (
                    "- Hurts workload families: "
                    f"`{', '.join(comp.regressed_workload_families) or 'none'}`"
                ),
            ]
        )
        if comp.rationale:
            lines.extend(["", "### Rationale"])
            for reason in comp.rationale:
                lines.append(f"- {reason}")

    if summary.lifecycle_events:
        lines.extend(["", "## Lifecycle Events"])
        for event in summary.lifecycle_events:
            lines.append(
                f"- `{event.lifecycle_state.value}` at "
                f"`{event.recorded_at.isoformat()}` "
                f"disposition=`{event.promotion_disposition.value}`"
            )

    return "\n".join(lines)


def _candidate_eligibility_label(
    artifact: OptimizationCandidateConfigurationArtifact,
) -> str:
    if artifact.eligibility is None:
        return "unknown"
    return artifact.eligibility.status.value


def _recommendation_counts_label(counts: Mapping[str, int]) -> str:
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return rendered or "none"


def _trial_recommendation_label(artifact: OptimizationTrialArtifact) -> str:
    if artifact.recommendation_summary is None:
        return "none"
    return artifact.recommendation_summary.disposition.value


def _trial_evidence_label(artifact: OptimizationTrialArtifact) -> str:
    rendered = ", ".join(
        sorted({record.evidence_kind.value for record in artifact.evidence_records})
    )
    return rendered or "none"


def _inspection_trial_recommendation_label(trial: ForgeTrialInspectionSummary) -> str:
    if trial.recommendation_disposition is None:
        return "none"
    return trial.recommendation_disposition.value


def _promotion_lifecycle_label(summary: ForgePromotionRuntimeSummary) -> str:
    if summary.lifecycle_state is None:
        return "none"
    return summary.lifecycle_state.value


def write_json_model(model: BaseModel, output_path: Path) -> Path:
    """Write any serializable Pydantic benchmark model to stable JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            model.model_dump(mode="json", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _format_distribution(distribution: Mapping[str, int]) -> str:
    """Render a deterministic inline distribution summary."""

    if not distribution:
        return "`unavailable`"
    return ", ".join(f"`{key}`: `{value}`" for key, value in sorted(distribution.items()))


def _format_empty_distribution(
    distribution: Mapping[str, int],
    *,
    empty_label: str,
) -> str:
    """Render a deterministic inline distribution with a caller-provided empty label."""

    if not distribution:
        return f"`{empty_label}`"
    return _format_distribution(distribution)


def _enum_key_distribution(distribution: Mapping[Any, int]) -> dict[str, int]:
    return {
        key.value if hasattr(key, "value") else str(key): value
        for key, value in distribution.items()
    }


def _enum_list_distribution(values: Sequence[Any]) -> dict[str, int]:
    return _enum_key_distribution(Counter(values))


def default_output_path(base_dir: Path, artifact: BenchmarkRunArtifact) -> Path:
    """Return the default output path for an artifact."""

    return base_dir / f"{artifact.run_id}.json"


def default_comparison_output_path(
    base_dir: Path,
    artifact: BenchmarkTargetComparisonArtifact,
) -> Path:
    """Return the default output path for a two-target comparison artifact."""

    return base_dir / f"{artifact.comparison_id}.json"


def build_run_id(
    *,
    run_timestamp: datetime,
    policy: RoutingPolicy,
    suffix: str | None = None,
) -> str:
    """Build a reproducible run identifier."""

    run_id = f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}_{policy.value}"
    if suffix is None:
        return run_id
    return f"{run_id}_{suffix.replace(':', '_')}"


def _comparison_id(
    *,
    left_artifact: BenchmarkRunArtifact,
    right_artifact: BenchmarkRunArtifact,
) -> str:
    digest = hashlib.sha1(f"{left_artifact.run_id}|{right_artifact.run_id}".encode()).hexdigest()[
        :12
    ]
    return f"{left_artifact.timestamp.strftime('%Y%m%dT%H%M%SZ')}_compare_{digest}"


def _comparison_side_summary(
    artifact: BenchmarkRunArtifact,
) -> BenchmarkComparisonSideSummary:
    request_count = artifact.request_count
    return BenchmarkComparisonSideSummary(
        run_id=artifact.run_id,
        execution_target=artifact.execution_target
        or ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias=artifact.model_alias or artifact.scenario.model,
            routing_policy=artifact.policy,
        ),
        request_count=request_count,
        success_rate=_safe_rate(artifact.summary.success_count, request_count),
        error_rate=_safe_rate(artifact.summary.failure_count, request_count),
        fallback_rate=_safe_rate(artifact.summary.fallback_count, request_count),
        p50_latency_ms=artifact.summary.p50_latency_ms,
        p95_latency_ms=artifact.summary.p95_latency_ms,
        p50_ttft_ms=artifact.summary.p50_ttft_ms,
        p95_ttft_ms=artifact.summary.p95_ttft_ms,
        avg_tokens_per_second=artifact.summary.avg_tokens_per_second,
        p95_tokens_per_second=artifact.summary.p95_tokens_per_second,
        route_distribution=_route_distribution(artifact.records),
        backend_distribution=dict(sorted(artifact.summary.chosen_backend_counts.items())),
        hybrid_summary=artifact.summary.hybrid_summary,
        evidence_summary=_resolved_evidence_summary_for_artifact(artifact),
    )


def _comparability_assessment(
    *,
    left_artifact: BenchmarkRunArtifact,
    right_artifact: BenchmarkRunArtifact,
) -> BenchmarkComparabilityAssessment:
    left_summary = (
        _resolved_evidence_summary_for_artifact(left_artifact) or BenchmarkEvidenceSummary()
    )
    right_summary = (
        _resolved_evidence_summary_for_artifact(right_artifact) or BenchmarkEvidenceSummary()
    )
    shared_classes = sorted(
        set(left_summary.evidence_class_counts) & set(right_summary.evidence_class_counts),
        key=lambda item: item.value,
    )
    limitations: list[str] = []
    directly_comparable = True
    if left_summary.run_kind != right_summary.run_kind:
        limitations.append("comparison spans different run kinds and is not apples-to-apples")
        directly_comparable = False
    if (
        left_summary.observed_cloud_request_count > 0
        and right_summary.observed_cloud_request_count == 0
    ) or (
        right_summary.observed_cloud_request_count > 0
        and left_summary.observed_cloud_request_count == 0
    ):
        limitations.append("only one side contains direct observed cloud-worker execution")
        directly_comparable = False
    if (left_summary.mock_request_count > 0 and right_summary.mock_request_count == 0) or (
        right_summary.mock_request_count > 0 and left_summary.mock_request_count == 0
    ):
        limitations.append("comparison mixes mock-remote evidence with non-mock evidence")
        directly_comparable = False
    if (
        left_summary.cloud_provider_counts
        and right_summary.cloud_provider_counts
        and set(left_summary.cloud_provider_counts) != set(right_summary.cloud_provider_counts)
    ):
        limitations.append("observed/configured cloud providers differ across the two sides")
        directly_comparable = False
    if (
        left_summary.cloud_runtime_version_counts
        and right_summary.cloud_runtime_version_counts
        and set(left_summary.cloud_runtime_version_counts)
        != set(right_summary.cloud_runtime_version_counts)
    ):
        limitations.append("cloud runtime versions differ across the two sides")
        directly_comparable = False
    for limitation in [
        *left_artifact.comparability_limitations,
        *right_artifact.comparability_limitations,
    ]:
        if limitation not in limitations:
            limitations.append(limitation)
    return BenchmarkComparabilityAssessment(
        directly_comparable=directly_comparable,
        left_run_kind=left_summary.run_kind,
        right_run_kind=right_summary.run_kind,
        shared_evidence_classes=shared_classes,
        limitations=limitations,
    )


def _resolved_evidence_summary_for_artifact(
    artifact: BenchmarkRunArtifact,
) -> BenchmarkEvidenceSummary | None:
    if artifact.summary.evidence_summary is not None:
        return artifact.summary.evidence_summary
    if not artifact.records:
        return None
    for record in artifact.records:
        record.cloud_worker_evidence = _cloud_worker_evidence_for_record(record)
        record.evidence_class = _classify_record_evidence(record)
        record.confidence_notes = _confidence_notes_for_record(record)
        record.comparability_limitations = _record_comparability_limitations(record)
    return _summarize_record_evidence(artifact.records)


def _route_distribution(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        route_backend = (
            record.route_decision.backend_name
            if record.route_decision is not None
            else record.backend_name
        )
        counts[route_backend] = counts.get(route_backend, 0) + 1
    return dict(sorted(counts.items()))


def _error_category_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    """Summarize request error categories from recorded request outcomes."""

    counts: dict[str, int] = {}
    for record in records:
        if record.success:
            continue
        category = record.error_category or "unknown"
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _admission_outcome_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        admission_decision = _admission_decision_for_record(record)
        if admission_decision is None:
            continue
        counts[admission_decision.state.value] += 1
    return dict(sorted(counts.items()))


def _queue_wait_summary(records: list[BenchmarkRequestRecord]) -> dict[str, float]:
    queue_waits = [
        admission_decision.queue_wait_ms
        for record in records
        if (admission_decision := _admission_decision_for_record(record)) is not None
        and admission_decision.queue_wait_ms is not None
    ]
    if not queue_waits:
        return {}
    return {
        "count": float(len(queue_waits)),
        "avg_ms": round(_average(queue_waits), 3),
        "p95_ms": round(_percentile(queue_waits, 95), 3),
    }


def _breaker_phase_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        breaker_phase = _breaker_phase_for_record(record)
        if breaker_phase is None:
            continue
        counts[breaker_phase.value] += 1
    return dict(sorted(counts.items()))


def _breaker_reason_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        breaker_state = _breaker_state_for_record(record)
        if breaker_state is None or breaker_state.reason is None:
            continue
        counts[breaker_state.reason] += 1
    return dict(sorted(counts.items()))


def _affinity_disposition_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    return _annotation_counts(
        records,
        field_name="affinity_disposition",
        empty_value=AffinityDisposition.NOT_REQUESTED.value,
    )


def _rollout_disposition_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    return _annotation_counts(
        records,
        field_name="rollout_disposition",
        empty_value=RolloutDisposition.NONE.value,
    )


def _shadow_disposition_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    return _annotation_counts(
        records,
        field_name="shadow_disposition",
        empty_value=ShadowDisposition.DISABLED.value,
    )


def _canary_policy_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if (
            record.control_plane_metadata is not None
            and record.control_plane_metadata.canary_policy
        ):
            counts[record.control_plane_metadata.canary_policy.policy_name] += 1
            continue
        if record.route_decision is not None and record.route_decision.canary_policy is not None:
            counts[record.route_decision.canary_policy.policy_name] += 1
    return dict(sorted(counts.items()))


def _shadow_target_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        shadow_policy = None
        if record.control_plane_metadata is not None:
            shadow_policy = record.control_plane_metadata.shadow_policy
        if shadow_policy is None and record.route_decision is not None:
            shadow_policy = record.route_decision.shadow_policy
        if shadow_policy is None:
            continue
        target = shadow_policy.target_backend or shadow_policy.target_alias
        if target is None:
            continue
        counts[target] += 1
    return dict(sorted(counts.items()))


def _control_plane_note_counts(records: list[BenchmarkRequestRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        annotations = record.route_decision.annotations if record.route_decision else None
        if annotations is None:
            continue
        for note in annotations.notes:
            counts[note] += 1
    if not counts:
        return {}
    most_common = counts.most_common(5)
    return dict(sorted((note, count) for note, count in most_common))


def _annotation_counts(
    records: list[BenchmarkRequestRecord],
    *,
    field_name: str,
    empty_value: str,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        annotations = record.route_decision.annotations if record.route_decision else None
        if annotations is None:
            continue
        value = getattr(annotations, field_name)
        resolved = value.value if hasattr(value, "value") else str(value)
        if resolved == empty_value:
            continue
        counts[resolved] += 1
    return dict(sorted(counts.items()))


def _admission_decision_for_record(
    record: BenchmarkRequestRecord,
) -> AdmissionDecision | None:
    if (
        record.control_plane_metadata is not None
        and record.control_plane_metadata.admission_decision
    ):
        return record.control_plane_metadata.admission_decision
    if record.route_decision is not None:
        return record.route_decision.admission_decision
    return None


def _breaker_state_for_record(
    record: BenchmarkRequestRecord,
) -> CircuitBreakerState | None:
    if (
        record.control_plane_metadata is not None
        and record.control_plane_metadata.circuit_breaker_state
    ):
        return record.control_plane_metadata.circuit_breaker_state
    if record.route_decision is not None:
        return record.route_decision.circuit_breaker_state
    return None


def _breaker_phase_for_record(record: BenchmarkRequestRecord) -> CircuitBreakerPhase | None:
    if record.route_decision is not None and record.route_decision.annotations is not None:
        if record.route_decision.annotations.breaker_phase is not None:
            return record.route_decision.annotations.breaker_phase
    breaker_state = _breaker_state_for_record(record)
    return breaker_state.phase if breaker_state is not None else None


def _format_queue_wait_summary(summary: dict[str, float]) -> str:
    if not summary:
        return "`none observed`"
    return (
        f"`count={int(summary['count'])}, "
        f"avg_ms={summary['avg_ms']:.3f}, "
        f"p95_ms={summary['p95_ms']:.3f}`"
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _optional_delta(
    right: float | None,
    left: float | None,
) -> float | None:
    if right is None and left is None:
        return None
    resolved_right = 0.0 if right is None else right
    resolved_left = 0.0 if left is None else left
    return round(resolved_right - resolved_left, 3)


def _distribution_delta(
    left: dict[str, int],
    right: dict[str, int],
) -> dict[str, int]:
    keys = sorted(set(left) | set(right))
    return {
        key: right.get(key, 0) - left.get(key, 0)
        for key in keys
        if right.get(key, 0) - left.get(key, 0) != 0
    }


def _scenario_deltas(
    left_records: list[BenchmarkRequestRecord],
    right_records: list[BenchmarkRequestRecord],
) -> list[ScenarioDelta]:
    left_by_key = {
        _comparison_record_key(record=record, index=index): record
        for index, record in enumerate(left_records)
    }
    right_by_key = {
        _comparison_record_key(record=record, index=index): record
        for index, record in enumerate(right_records)
    }
    deltas: list[ScenarioDelta] = []
    for key in sorted(set(left_by_key) & set(right_by_key)):
        left = left_by_key[key]
        right = right_by_key[key]
        delta = ScenarioDelta(
            key=key,
            left_request_id=left.request_id,
            right_request_id=right.request_id,
            latency_delta_ms=round(right.latency_ms - left.latency_ms, 3),
            ttft_delta_ms=_optional_delta(right.ttft_ms, left.ttft_ms),
            tokens_per_second_delta=_optional_delta(
                right.tokens_per_second,
                left.tokens_per_second,
            ),
            modeled_cost_delta=_optional_delta(
                _modeled_cost_for_record(right),
                _modeled_cost_for_record(left),
            ),
            success_changed=left.success != right.success,
            backend_changed=left.backend_name != right.backend_name,
            route_changed=_route_backend_for_record(left) != _route_backend_for_record(right),
            evidence_kind=_comparison_evidence_kind(left, right),
            hybrid_outcome=_hybrid_comparison_outcome(left, right),
            condition_sources=_comparison_condition_sources(left, right),
            notes=_hybrid_comparison_notes(left, right),
        )
        if (
            delta.success_changed
            or delta.backend_changed
            or delta.route_changed
            or abs(delta.latency_delta_ms) >= 1.0
            or delta.hybrid_outcome is not HybridComparisonOutcome.INCONCLUSIVE
        ):
            deltas.append(delta)
    return deltas


def _comparison_record_key(
    *,
    record: BenchmarkRequestRecord,
    index: int,
) -> str:
    if record.workload_item_id is not None:
        return f"workload:{record.workload_item_id}"
    if record.source_trace_record_id is not None:
        return f"trace:{record.source_trace_record_id}"
    if record.source_request_id is not None:
        return f"request:{record.source_request_id}"
    return f"index:{index:04d}"


def _route_backend_for_record(record: BenchmarkRequestRecord) -> str:
    if record.route_decision is not None:
        return record.route_decision.backend_name
    return record.backend_name


def _modeled_cost_for_record(record: BenchmarkRequestRecord) -> float | None:
    if record.hybrid_context is None:
        return None
    if record.hybrid_context.injected_condition is not None:
        return record.hybrid_context.injected_condition.modeled_cost
    if record.hybrid_context.predictor_condition is not None:
        return record.hybrid_context.predictor_condition.modeled_cost
    return record.hybrid_context.observed_modeled_cost


def _summarize_hybrid_records(
    records: list[BenchmarkRequestRecord],
) -> HybridBenchmarkSummary | None:
    if not records:
        return None
    local_only_count = 0
    hybrid_spillover_count = 0
    remote_only_count = 0
    remote_blocked_count = 0
    remote_cold_count = 0
    remote_warm_count = 0
    observed_runtime_count = 0
    injected_condition_count = 0
    predictor_estimate_count = 0
    low_confidence_count = 0
    unsupported_count = 0
    budget_exhausted_count = 0
    budget_disabled_count = 0
    observed_placement_evidence_count = 0
    estimated_placement_evidence_count = 0
    observed_cost_evidence_count = 0
    estimated_cost_evidence_count = 0
    observed_penalties: list[float] = []
    injected_penalties: list[float] = []
    predicted_penalties: list[float] = []
    modeled_costs: list[float] = []
    notes: list[str] = []
    for record in records:
        context = record.hybrid_context
        if context is None:
            unsupported_count += 1
            continue
        observed_path = context.observed_execution_path
        if observed_path is HybridExecutionPath.LOCAL_ONLY:
            local_only_count += 1
        elif observed_path is HybridExecutionPath.HYBRID_SPILLOVER:
            hybrid_spillover_count += 1
        elif observed_path is HybridExecutionPath.REMOTE_ONLY:
            remote_only_count += 1
        elif observed_path is HybridExecutionPath.REMOTE_BLOCKED:
            remote_blocked_count += 1
        if observed_path is not HybridExecutionPath.UNKNOWN:
            observed_runtime_count += 1
        if context.observed_remote_temperature is RemoteTemperature.COLD:
            remote_cold_count += 1
        elif context.observed_remote_temperature is RemoteTemperature.WARM:
            remote_warm_count += 1
        if context.observed_network_penalty_ms is not None:
            observed_penalties.append(context.observed_network_penalty_ms)
        if context.observed_modeled_cost is not None:
            modeled_costs.append(context.observed_modeled_cost)
        if context.observed_placement_evidence is not None:
            if context.observed_placement_evidence.source is CloudEvidenceSource.OBSERVED_RUNTIME:
                observed_placement_evidence_count += 1
            else:
                estimated_placement_evidence_count += 1
        if context.observed_cost_evidence is not None:
            if context.observed_cost_evidence.source is CloudEvidenceSource.OBSERVED_RUNTIME:
                observed_cost_evidence_count += 1
            else:
                estimated_cost_evidence_count += 1
        if context.observed_budget_outcome is RemoteBudgetOutcome.EXHAUSTED:
            budget_exhausted_count += 1
        elif context.observed_budget_outcome is RemoteBudgetOutcome.DISABLED:
            budget_disabled_count += 1
        injected = context.injected_condition
        if injected is not None:
            injected_condition_count += 1
            if injected.network_penalty_ms is not None:
                injected_penalties.append(injected.network_penalty_ms)
            if injected.modeled_cost is not None:
                modeled_costs.append(injected.modeled_cost)
            if injected.placement_evidence is not None:
                estimated_placement_evidence_count += 1
            if injected.cost_evidence is not None:
                estimated_cost_evidence_count += 1
            if injected.budget_outcome is RemoteBudgetOutcome.EXHAUSTED:
                budget_exhausted_count += 1
            elif injected.budget_outcome is RemoteBudgetOutcome.DISABLED:
                budget_disabled_count += 1
            if injected.confidence is RecommendationConfidence.LOW:
                low_confidence_count += 1
        predictor = context.predictor_condition
        if predictor is not None:
            predictor_estimate_count += 1
            if predictor.network_penalty_ms is not None:
                predicted_penalties.append(predictor.network_penalty_ms)
            if predictor.modeled_cost is not None:
                modeled_costs.append(predictor.modeled_cost)
            if predictor.placement_evidence is not None:
                if predictor.placement_evidence.source is CloudEvidenceSource.OBSERVED_RUNTIME:
                    observed_placement_evidence_count += 1
                else:
                    estimated_placement_evidence_count += 1
            if predictor.cost_evidence is not None:
                if predictor.cost_evidence.source is CloudEvidenceSource.OBSERVED_RUNTIME:
                    observed_cost_evidence_count += 1
                else:
                    estimated_cost_evidence_count += 1
            if predictor.confidence in {
                RecommendationConfidence.LOW,
                RecommendationConfidence.INSUFFICIENT,
            }:
                low_confidence_count += 1
        if observed_path is HybridExecutionPath.UNKNOWN and injected is None and predictor is None:
            unsupported_count += 1
    if observed_runtime_count == 0 and injected_condition_count > 0:
        notes.append("hybrid evidence relied on injected/mock conditions")
    if predictor_estimate_count > 0:
        notes.append("predictor-based remote cost or network estimates were included")
    if unsupported_count > 0:
        notes.append(f"{unsupported_count} requests lacked usable hybrid evidence")
    return HybridBenchmarkSummary(
        local_only_count=local_only_count,
        hybrid_spillover_count=hybrid_spillover_count,
        remote_only_count=remote_only_count,
        remote_blocked_count=remote_blocked_count,
        remote_cold_count=remote_cold_count,
        remote_warm_count=remote_warm_count,
        observed_runtime_count=observed_runtime_count,
        injected_condition_count=injected_condition_count,
        predictor_estimate_count=predictor_estimate_count,
        low_confidence_count=low_confidence_count,
        unsupported_count=unsupported_count,
        budget_exhausted_count=budget_exhausted_count,
        budget_disabled_count=budget_disabled_count,
        observed_placement_evidence_count=observed_placement_evidence_count,
        estimated_placement_evidence_count=estimated_placement_evidence_count,
        observed_cost_evidence_count=observed_cost_evidence_count,
        estimated_cost_evidence_count=estimated_cost_evidence_count,
        avg_observed_network_penalty_ms=(
            None if not observed_penalties else _average(observed_penalties)
        ),
        avg_injected_network_penalty_ms=(
            None if not injected_penalties else _average(injected_penalties)
        ),
        avg_predicted_network_penalty_ms=(
            None if not predicted_penalties else _average(predicted_penalties)
        ),
        total_modeled_cost=None if not modeled_costs else round(sum(modeled_costs), 6),
        avg_modeled_cost=None if not modeled_costs else _average(modeled_costs),
        notes=notes,
    )


def _hybrid_comparison_summary(
    left_records: list[BenchmarkRequestRecord],
    right_records: list[BenchmarkRequestRecord],
) -> HybridComparisonSummary | None:
    deltas = _scenario_deltas(left_records, right_records)
    if not deltas:
        return None
    beneficial_count = sum(
        1 for delta in deltas if delta.hybrid_outcome is HybridComparisonOutcome.BENEFICIAL
    )
    harmful_count = sum(
        1 for delta in deltas if delta.hybrid_outcome is HybridComparisonOutcome.HARMFUL
    )
    inconclusive_count = sum(
        1 for delta in deltas if delta.hybrid_outcome is HybridComparisonOutcome.INCONCLUSIVE
    )
    unsupported_count = sum(
        1 for delta in deltas if delta.hybrid_outcome is HybridComparisonOutcome.UNSUPPORTED
    )
    direct_count = sum(
        1 for delta in deltas if delta.evidence_kind is SimulationEvidenceKind.DIRECT_OBSERVATION
    )
    predictor_count = sum(
        1 for delta in deltas if delta.evidence_kind is SimulationEvidenceKind.PREDICTOR_ESTIMATE
    )
    low_confidence_count = sum(
        1
        for delta in deltas
        if delta.evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE
    )
    observed_penalty_deltas = [
        penalty_delta
        for left, right in _paired_records(left_records, right_records)
        if (penalty_delta := _observed_network_penalty_delta(left, right)) is not None
    ]
    modeled_cost_deltas = [
        delta.modeled_cost_delta for delta in deltas if delta.modeled_cost_delta is not None
    ]
    budget_exhausted_delta = sum(1 for record in right_records if _budget_exhausted(record)) - sum(
        1 for record in left_records if _budget_exhausted(record)
    )
    notes: list[str] = []
    if beneficial_count > 0:
        notes.append(f"{beneficial_count} requests showed hybrid benefit on the right-hand side")
    if harmful_count > 0:
        notes.append(f"{harmful_count} requests regressed under the right-hand side")
    if unsupported_count > 0:
        notes.append(f"{unsupported_count} requests remained unsupported or inconclusive")
    return HybridComparisonSummary(
        beneficial_count=beneficial_count,
        harmful_count=harmful_count,
        inconclusive_count=inconclusive_count,
        unsupported_count=unsupported_count,
        direct_observation_count=direct_count,
        predictor_estimate_count=predictor_count,
        low_confidence_count=low_confidence_count,
        observed_network_penalty_delta_ms=(
            None if not observed_penalty_deltas else _average(observed_penalty_deltas)
        ),
        modeled_cost_delta=None if not modeled_cost_deltas else _average(modeled_cost_deltas),
        budget_exhausted_delta=budget_exhausted_delta,
        notes=notes,
    )


def _paired_records(
    left_records: list[BenchmarkRequestRecord],
    right_records: list[BenchmarkRequestRecord],
) -> list[tuple[BenchmarkRequestRecord, BenchmarkRequestRecord]]:
    left_by_key = {
        _comparison_record_key(record=record, index=index): record
        for index, record in enumerate(left_records)
    }
    right_by_key = {
        _comparison_record_key(record=record, index=index): record
        for index, record in enumerate(right_records)
    }
    return [
        (left_by_key[key], right_by_key[key])
        for key in sorted(set(left_by_key) & set(right_by_key))
    ]


def _observed_network_penalty_delta(
    left: BenchmarkRequestRecord,
    right: BenchmarkRequestRecord,
) -> float | None:
    left_penalty = (
        None if left.hybrid_context is None else left.hybrid_context.observed_network_penalty_ms
    )
    right_penalty = (
        None if right.hybrid_context is None else right.hybrid_context.observed_network_penalty_ms
    )
    return _optional_delta(right_penalty, left_penalty)


def _budget_exhausted(record: BenchmarkRequestRecord) -> bool:
    context = record.hybrid_context
    if context is None:
        return False
    if context.observed_budget_outcome is RemoteBudgetOutcome.EXHAUSTED:
        return True
    if (
        context.injected_condition is not None
        and context.injected_condition.budget_outcome is RemoteBudgetOutcome.EXHAUSTED
    ):
        return True
    return bool(
        context.predictor_condition is not None
        and context.predictor_condition.budget_outcome is RemoteBudgetOutcome.EXHAUSTED
    )


def _comparison_condition_sources(
    left: BenchmarkRequestRecord,
    right: BenchmarkRequestRecord,
) -> list[HybridConditionSource]:
    sources: list[HybridConditionSource] = []
    for record in (left, right):
        context = record.hybrid_context
        if context is None:
            continue
        if context.observed_execution_path is not HybridExecutionPath.UNKNOWN:
            sources.append(HybridConditionSource.OBSERVED_RUNTIME)
        if context.injected_condition is not None:
            sources.append(HybridConditionSource.INJECTED_MOCK)
        if context.predictor_condition is not None:
            sources.append(HybridConditionSource.PREDICTOR_ESTIMATE)
    return sorted(set(sources), key=lambda item: item.value)


def _comparison_evidence_kind(
    left: BenchmarkRequestRecord,
    right: BenchmarkRequestRecord,
) -> SimulationEvidenceKind:
    sources = _comparison_condition_sources(left, right)
    if HybridConditionSource.OBSERVED_RUNTIME in sources:
        return SimulationEvidenceKind.DIRECT_OBSERVATION
    if HybridConditionSource.PREDICTOR_ESTIMATE in sources:
        return SimulationEvidenceKind.PREDICTOR_ESTIMATE
    if HybridConditionSource.INJECTED_MOCK in sources:
        return SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE
    return SimulationEvidenceKind.UNSUPPORTED


def _hybrid_comparison_outcome(
    left: BenchmarkRequestRecord,
    right: BenchmarkRequestRecord,
) -> HybridComparisonOutcome:
    evidence_kind = _comparison_evidence_kind(left, right)
    if evidence_kind is SimulationEvidenceKind.UNSUPPORTED:
        return HybridComparisonOutcome.UNSUPPORTED
    if right.success and not left.success:
        return HybridComparisonOutcome.BENEFICIAL
    if left.success and not right.success:
        return HybridComparisonOutcome.HARMFUL
    right_cost = _modeled_cost_for_record(right)
    left_cost = _modeled_cost_for_record(left)
    if right.latency_ms + 5.0 < left.latency_ms:
        return HybridComparisonOutcome.BENEFICIAL
    if right.latency_ms > left.latency_ms + 5.0:
        return HybridComparisonOutcome.HARMFUL
    if right_cost is not None and left_cost is not None and right_cost + 0.001 < left_cost:
        return HybridComparisonOutcome.BENEFICIAL
    if right_cost is not None and left_cost is not None and right_cost > left_cost + 0.001:
        return HybridComparisonOutcome.HARMFUL
    return HybridComparisonOutcome.INCONCLUSIVE


def _hybrid_comparison_notes(
    left: BenchmarkRequestRecord,
    right: BenchmarkRequestRecord,
) -> list[str]:
    notes: list[str] = []
    evidence_kind = _comparison_evidence_kind(left, right)
    if evidence_kind is SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE:
        notes.append("comparison relied on injected/mock remote conditions")
    elif evidence_kind is SimulationEvidenceKind.PREDICTOR_ESTIMATE:
        notes.append("comparison included predictor-based remote estimates")
    elif evidence_kind is SimulationEvidenceKind.UNSUPPORTED:
        notes.append("comparison lacked enough hybrid evidence")
    if _budget_exhausted(right) and not _budget_exhausted(left):
        notes.append("right-hand side hit remote budget exhaustion")
    return notes


def get_git_sha() -> str | None:
    """Return the current git SHA when available."""

    try:
        return (
            check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                stderr=DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except (CalledProcessError, FileNotFoundError):
        return None


def parse_prometheus_text(text: str) -> list[PrometheusSample]:
    """Parse a small Prometheus text response into samples."""

    samples: list[PrometheusSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_part, _, value_part = line.rpartition(" ")
        if not value_part:
            continue
        name, labels = _parse_metric_labels(metric_part)
        samples.append(PrometheusSample(name=name, labels=labels, value=float(value_part)))
    return samples


def _parse_metric_labels(metric_part: str) -> tuple[str, dict[str, str]]:
    if "{" not in metric_part:
        return metric_part, {}

    name, _, raw_labels = metric_part.partition("{")
    raw_labels = raw_labels.removesuffix("}")
    labels: dict[str, str] = {}
    for entry in _split_prometheus_labels(raw_labels):
        key, _, raw_value = entry.partition("=")
        labels[key] = _unescape_prometheus_label_value(raw_value.strip('"'))
    return name, labels


def _split_prometheus_labels(raw_labels: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    escaped = False
    for char in raw_labels:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == '"':
            current.append(char)
            in_quotes = not in_quotes
            continue
        if char == "," and not in_quotes:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _unescape_prometheus_label_value(value: str) -> str:
    return value.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _hybrid_context_for_record(
    *,
    metadata: Mapping[str, str] | None,
    route_decision: RouteDecision | None,
    backend_name: str,
    routing_policy: RoutingPolicy,
) -> HybridExecutionContext | None:
    injected = _condition_profile_from_metadata(
        metadata=metadata,
        prefix="injected",
        source=HybridConditionSource.INJECTED_MOCK,
    )
    predictor = _predictor_condition_from_route_decision(route_decision)
    observed_path, observed_budget, reason_codes = _observed_hybrid_outcome(
        route_decision=route_decision,
        backend_name=backend_name,
        routing_policy=routing_policy,
    )
    observed_temperature = _observed_remote_temperature(
        route_decision=route_decision,
        injected=injected,
    )
    observed_network_penalty_ms = _observed_network_penalty(route_decision)
    observed_cost = _observed_modeled_cost(route_decision)
    observed_placement_evidence = _observed_placement_evidence(route_decision)
    observed_cost_evidence = _observed_cost_evidence(route_decision)
    if (
        observed_path is HybridExecutionPath.UNKNOWN
        and observed_temperature is RemoteTemperature.UNKNOWN
        and observed_budget is RemoteBudgetOutcome.UNKNOWN
        and observed_network_penalty_ms is None
        and observed_cost is None
        and observed_placement_evidence is None
        and observed_cost_evidence is None
        and injected is None
        and predictor is None
    ):
        return None
    return HybridExecutionContext(
        observed_execution_path=observed_path,
        observed_remote_temperature=observed_temperature,
        observed_budget_outcome=observed_budget,
        observed_network_penalty_ms=observed_network_penalty_ms,
        observed_modeled_cost=observed_cost,
        observed_placement_evidence=observed_placement_evidence,
        observed_cost_evidence=observed_cost_evidence,
        reason_codes=reason_codes,
        injected_condition=injected,
        predictor_condition=predictor,
    )


def _condition_profile_from_metadata(
    *,
    metadata: Mapping[str, str] | None,
    prefix: str,
    source: HybridConditionSource,
) -> HybridConditionProfile | None:
    if metadata is None:
        return None
    key_prefix = f"{prefix}_"
    path = _safe_hybrid_execution_path(metadata.get(f"{key_prefix}execution_path"))
    temperature = _safe_remote_temperature(metadata.get(f"{key_prefix}remote_temperature"))
    budget = _safe_remote_budget_outcome(metadata.get(f"{key_prefix}budget_outcome"))
    network_penalty = _maybe_parse_float(metadata.get(f"{key_prefix}network_penalty_ms"))
    cold_start_penalty = _maybe_parse_float(metadata.get(f"{key_prefix}cold_start_penalty_ms"))
    modeled_cost = _maybe_parse_float(metadata.get(f"{key_prefix}modeled_cost"))
    confidence = _safe_recommendation_confidence(metadata.get(f"{key_prefix}confidence"))
    if (
        path is HybridExecutionPath.UNKNOWN
        and temperature is RemoteTemperature.UNKNOWN
        and budget is RemoteBudgetOutcome.UNKNOWN
        and network_penalty is None
        and cold_start_penalty is None
        and modeled_cost is None
        and confidence is None
    ):
        return None
    notes = []
    expected_signal = metadata.get("expected_signal")
    if expected_signal is not None:
        notes.append(f"scenario signal={expected_signal}")
    placement_evidence = _placement_evidence_from_metadata(metadata=metadata, prefix=prefix)
    cost_evidence = _cost_evidence_from_metadata(metadata=metadata, prefix=prefix)
    return HybridConditionProfile(
        source=source,
        execution_path=path,
        remote_temperature=temperature,
        budget_outcome=budget,
        network_penalty_ms=network_penalty,
        cold_start_penalty_ms=cold_start_penalty,
        modeled_cost=modeled_cost,
        placement_evidence=placement_evidence,
        cost_evidence=cost_evidence,
        confidence=confidence,
        notes=notes,
    )


def _predictor_condition_from_route_decision(
    route_decision: RouteDecision | None,
) -> HybridConditionProfile | None:
    if route_decision is None or route_decision.selected_deployment is None:
        return None
    deployment = route_decision.selected_deployment
    network_penalty = deployment.readiness_hints.estimated_cold_start_ms
    expected_rtt = (
        deployment.instances[0].network_characteristics.expected_rtt_ms
        if deployment.instances
        else None
    )
    if expected_rtt is not None:
        network_penalty = (
            float(expected_rtt)
            if network_penalty is None
            else float(network_penalty) + float(expected_rtt)
        )
    modeled_cost = deployment.cost_profile.relative_cost_index
    confidence = (
        RecommendationConfidence.LOW
        if deployment.readiness_hints.cold_start_likely or expected_rtt is None
        else RecommendationConfidence.MEDIUM
    )
    execution_path = (
        HybridExecutionPath.REMOTE_ONLY
        if deployment.execution_mode
        in {
            ExecutionModeLabel.REMOTE_WORKER,
            ExecutionModeLabel.EXTERNAL_SERVICE,
        }
        else HybridExecutionPath.LOCAL_ONLY
    )
    temperature = (
        RemoteTemperature.COLD
        if deployment.readiness_hints.cold_start_likely
        else RemoteTemperature.UNKNOWN
    )
    budget_outcome = (
        RemoteBudgetOutcome.UNKNOWN
        if deployment.cost_profile.budget_bucket is None
        else RemoteBudgetOutcome.WITHIN_BUDGET
    )
    if (
        network_penalty is None
        and modeled_cost is None
        and temperature is RemoteTemperature.UNKNOWN
    ):
        return None
    return HybridConditionProfile(
        source=HybridConditionSource.PREDICTOR_ESTIMATE,
        execution_path=execution_path,
        remote_temperature=temperature,
        budget_outcome=budget_outcome,
        network_penalty_ms=network_penalty,
        cold_start_penalty_ms=deployment.readiness_hints.estimated_cold_start_ms,
        modeled_cost=None if modeled_cost is None else float(modeled_cost),
        placement_evidence=_deployment_placement_evidence(
            route_decision,
            source=CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE,
        ),
        cost_evidence=_deployment_cost_evidence(
            route_decision,
            source=CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE,
        ),
        confidence=confidence,
        notes=["derived from selected deployment hints"],
    )


def _observed_hybrid_outcome(
    *,
    route_decision: RouteDecision | None,
    backend_name: str,
    routing_policy: RoutingPolicy,
) -> tuple[HybridExecutionPath, RemoteBudgetOutcome, list[str]]:
    reason_codes: list[str] = []
    if route_decision is not None and route_decision.explanation is not None:
        reason_codes.extend(
            code.value for code in route_decision.explanation.selection_reason_codes
        )
    if route_decision is not None and route_decision.admission_decision is not None:
        reason = route_decision.admission_decision.reason_code
        if reason is not None:
            reason_codes.append(reason.value)
    if (
        route_decision is not None
        and route_decision.admission_decision is not None
        and route_decision.admission_decision.reason_code is not None
    ):
        if route_decision.admission_decision.reason_code.value == "remote_budget_exhausted":
            return HybridExecutionPath.REMOTE_BLOCKED, RemoteBudgetOutcome.EXHAUSTED, reason_codes
        if route_decision.admission_decision.reason_code.value == "remote_spillover_not_permitted":
            return HybridExecutionPath.REMOTE_BLOCKED, RemoteBudgetOutcome.DISABLED, reason_codes
    deployment = None if route_decision is None else route_decision.selected_deployment
    remote_selected = False
    if deployment is not None and deployment.execution_mode in {
        ExecutionModeLabel.REMOTE_WORKER,
        ExecutionModeLabel.EXTERNAL_SERVICE,
    }:
        remote_selected = True
    elif backend_name.startswith("remote-worker:"):
        remote_selected = True
    if remote_selected:
        if routing_policy in {
            RoutingPolicy.BURST_TO_REMOTE,
            RoutingPolicy.LOCAL_PREFERRED,
            RoutingPolicy.LATENCY_SLO,
            RoutingPolicy.QUALITY_ON_DEMAND,
            RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY,
        }:
            return (
                HybridExecutionPath.HYBRID_SPILLOVER,
                RemoteBudgetOutcome.WITHIN_BUDGET,
                reason_codes,
            )
        return HybridExecutionPath.REMOTE_ONLY, RemoteBudgetOutcome.WITHIN_BUDGET, reason_codes
    if routing_policy in {RoutingPolicy.REMOTE_DISABLED, RoutingPolicy.LOCAL_ONLY}:
        return HybridExecutionPath.LOCAL_ONLY, RemoteBudgetOutcome.DISABLED, reason_codes
    return HybridExecutionPath.LOCAL_ONLY, RemoteBudgetOutcome.UNKNOWN, reason_codes


def _observed_remote_temperature(
    *,
    route_decision: RouteDecision | None,
    injected: HybridConditionProfile | None,
) -> RemoteTemperature:
    if (
        route_decision is not None
        and route_decision.selected_deployment is not None
        and route_decision.selected_deployment.readiness_hints.cold_start_likely
    ):
        return RemoteTemperature.COLD
    if injected is not None:
        return injected.remote_temperature
    return RemoteTemperature.UNKNOWN


def _observed_network_penalty(route_decision: RouteDecision | None) -> float | None:
    if route_decision is None or route_decision.selected_deployment is None:
        return None
    deployment = route_decision.selected_deployment
    values = [
        value
        for value in (
            deployment.readiness_hints.estimated_cold_start_ms,
            deployment.instances[0].network_characteristics.expected_rtt_ms
            if deployment.instances
            else None,
        )
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values), 3)


def _observed_modeled_cost(route_decision: RouteDecision | None) -> float | None:
    if (
        route_decision is None
        or route_decision.selected_deployment is None
        or route_decision.selected_deployment.cost_profile.relative_cost_index is None
    ):
        return None
    return float(route_decision.selected_deployment.cost_profile.relative_cost_index)


def _observed_placement_evidence(
    route_decision: RouteDecision | None,
) -> CloudPlacementEvidence | None:
    source = _resolved_placement_source(route_decision)
    if source is None:
        return None
    return _deployment_placement_evidence(route_decision, source=source)


def _observed_cost_evidence(route_decision: RouteDecision | None) -> CloudCostEvidence | None:
    source = _resolved_cost_source(route_decision)
    if source is None:
        return None
    return _deployment_cost_evidence(route_decision, source=source)


def _deployment_placement_evidence(
    route_decision: RouteDecision | None,
    *,
    source: CloudEvidenceSource,
) -> CloudPlacementEvidence | None:
    if route_decision is None or route_decision.selected_deployment is None:
        return None
    if source is CloudEvidenceSource.OBSERVED_RUNTIME:
        instance = _observed_backend_instance(route_decision)
        if instance is not None and (
            instance.placement.provider is not None
            or instance.placement.region is not None
            or instance.placement.zone is not None
        ):
            return CloudPlacementEvidence(
                source=source,
                provider=instance.placement.provider,
                region=instance.placement.region,
                zone=instance.placement.zone,
                notes=["derived from observed backend instance metadata"],
            )
    deployment = route_decision.selected_deployment
    if (
        deployment.placement.provider is None
        and deployment.placement.region is None
        and deployment.placement.zone is None
    ):
        return None
    return CloudPlacementEvidence(
        source=source,
        provider=deployment.placement.provider,
        region=deployment.placement.region,
        zone=deployment.placement.zone,
        notes=["derived from selected deployment metadata"],
    )


def _deployment_cost_evidence(
    route_decision: RouteDecision | None,
    *,
    source: CloudEvidenceSource,
) -> CloudCostEvidence | None:
    if route_decision is None or route_decision.selected_deployment is None:
        return None
    if source is CloudEvidenceSource.OBSERVED_RUNTIME:
        instance = _observed_backend_instance(route_decision)
        if instance is not None and (
            instance.cost_profile.relative_cost_index is not None
            or instance.cost_profile.budget_bucket is not None
            or instance.cost_profile.currency is not None
        ):
            return CloudCostEvidence(
                source=source,
                relative_cost_index=instance.cost_profile.relative_cost_index,
                budget_bucket=instance.cost_profile.budget_bucket,
                currency=instance.cost_profile.currency,
                notes=["derived from observed backend instance metadata"],
            )
    cost_profile = route_decision.selected_deployment.cost_profile
    if (
        cost_profile.relative_cost_index is None
        and cost_profile.budget_bucket is None
        and cost_profile.currency is None
    ):
        return None
    return CloudCostEvidence(
        source=source,
        relative_cost_index=cost_profile.relative_cost_index,
        budget_bucket=cost_profile.budget_bucket,
        currency=cost_profile.currency,
        notes=["derived from selected deployment metadata"],
    )


def _placement_evidence_from_metadata(
    *,
    metadata: Mapping[str, str],
    prefix: str,
) -> CloudPlacementEvidence | None:
    key_prefix = f"{prefix}_"
    provider = metadata.get(f"{key_prefix}provider")
    region = metadata.get(f"{key_prefix}region")
    zone = metadata.get(f"{key_prefix}zone")
    if provider is None and region is None and zone is None:
        return None
    return CloudPlacementEvidence(
        source=CloudEvidenceSource.INJECTED_MOCK,
        provider=provider,
        region=region,
        zone=zone,
        notes=["derived from benchmark metadata"],
    )


def _cost_evidence_from_metadata(
    *,
    metadata: Mapping[str, str],
    prefix: str,
) -> CloudCostEvidence | None:
    key_prefix = f"{prefix}_"
    relative_cost_index = _maybe_parse_float(metadata.get(f"{key_prefix}modeled_cost"))
    budget_bucket = metadata.get(f"{key_prefix}budget_bucket")
    currency = metadata.get(f"{key_prefix}currency")
    if relative_cost_index is None and budget_bucket is None and currency is None:
        return None
    return CloudCostEvidence(
        source=CloudEvidenceSource.INJECTED_MOCK,
        relative_cost_index=relative_cost_index,
        budget_bucket=budget_bucket,
        currency=currency,
        notes=["derived from benchmark metadata"],
    )


def _resolved_placement_source(route_decision: RouteDecision | None) -> CloudEvidenceSource | None:
    if route_decision is None or route_decision.selected_deployment is None:
        return None
    deployment = route_decision.selected_deployment
    if (
        route_decision.execution_observation is not None
        and route_decision.execution_observation.backend_instance_id
    ):
        instance_id = route_decision.execution_observation.backend_instance_id
        for instance in deployment.instances:
            if instance.instance_id == instance_id and (
                instance.placement.provider is not None
                or instance.placement.region is not None
                or instance.placement.zone is not None
            ):
                return CloudEvidenceSource.OBSERVED_RUNTIME
    if (
        deployment.placement.provider is not None
        or deployment.placement.region is not None
        or deployment.placement.zone is not None
    ):
        return CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE
    return None


def _resolved_cost_source(route_decision: RouteDecision | None) -> CloudEvidenceSource | None:
    if route_decision is None or route_decision.selected_deployment is None:
        return None
    observed_instance = _observed_backend_instance(route_decision)
    if observed_instance is not None and (
        observed_instance.cost_profile.relative_cost_index is not None
        or observed_instance.cost_profile.budget_bucket is not None
        or observed_instance.cost_profile.currency is not None
    ):
        return CloudEvidenceSource.OBSERVED_RUNTIME
    cost_profile = route_decision.selected_deployment.cost_profile
    if (
        cost_profile.relative_cost_index is None
        and cost_profile.budget_bucket is None
        and cost_profile.currency is None
    ):
        return None
    return CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE


def _observed_backend_instance(
    route_decision: RouteDecision | None,
) -> BackendInstance | None:
    if (
        route_decision is None
        or route_decision.selected_deployment is None
        or route_decision.execution_observation is None
        or route_decision.execution_observation.backend_instance_id is None
    ):
        return None
    instance_id = route_decision.execution_observation.backend_instance_id
    for instance in route_decision.selected_deployment.instances:
        if instance.instance_id == instance_id:
            return instance
    return None


def _maybe_parse_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    try:
        return round(float(raw_value), 6)
    except ValueError:
        return None


def _safe_hybrid_execution_path(raw_value: str | None) -> HybridExecutionPath:
    if raw_value is None:
        return HybridExecutionPath.UNKNOWN
    try:
        return HybridExecutionPath(raw_value)
    except ValueError:
        return HybridExecutionPath.UNKNOWN


def _safe_remote_temperature(raw_value: str | None) -> RemoteTemperature:
    if raw_value is None:
        return RemoteTemperature.UNKNOWN
    try:
        return RemoteTemperature(raw_value)
    except ValueError:
        return RemoteTemperature.UNKNOWN


def _safe_remote_budget_outcome(raw_value: str | None) -> RemoteBudgetOutcome:
    if raw_value is None:
        return RemoteBudgetOutcome.UNKNOWN
    try:
        return RemoteBudgetOutcome(raw_value)
    except ValueError:
        return RemoteBudgetOutcome.UNKNOWN


def _safe_recommendation_confidence(raw_value: str | None) -> RecommendationConfidence | None:
    if raw_value is None:
        return None
    try:
        return RecommendationConfidence(raw_value)
    except ValueError:
        return None


async def _run_non_streaming_gateway_request(
    *,
    benchmark_client: httpx.AsyncClient,
    request_id: str,
    payload: dict[str, object],
    headers: dict[str, str],
    model_alias: str,
    workload_item_id: str | None,
    scenario_family: WorkloadScenarioFamily | None,
    request_metadata: Mapping[str, str] | None,
    started_at: datetime,
    started_perf: float,
) -> BenchmarkRequestRecord:
    response = await benchmark_client.post("/v1/chat/completions", json=payload, headers=headers)
    completed_at = datetime.now(UTC)
    latency_ms = (perf_counter() - started_perf) * 1000
    response_payload = response.json()
    route_decision = _parse_route_decision_header(
        response.headers.get("x-switchyard-route-decision")
    )
    if response.status_code >= 400:
        return BenchmarkRequestRecord(
            request_id=request_id,
            workload_item_id=workload_item_id,
            scenario_family=scenario_family,
            tenant_id=headers.get("x-switchyard-tenant-id", "default"),
            request_class=_request_class_from_headers(headers),
            session_id=headers.get("x-switchyard-session-id"),
            backend_name="error",
            model_alias=model_alias,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=round(latency_ms, 3),
            routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
            route_decision=route_decision,
            control_plane_metadata=_control_plane_metadata_from_response(
                response_headers=response.headers,
                route_decision=route_decision,
                tenant_id=headers.get("x-switchyard-tenant-id", "default"),
                session_id=headers.get("x-switchyard-session-id"),
            ),
            hybrid_context=_hybrid_context_for_record(
                metadata=request_metadata,
                route_decision=route_decision,
                backend_name="error",
                routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
            ),
            success=False,
            status_code=response.status_code,
            cache_observation=CacheObservation(),
            error=_extract_error_message(response_payload),
            error_category=_categorize_status_code(response.status_code),
        )

    usage_payload = response_payload["usage"]
    usage = UsageStats.model_validate(usage_payload)
    output_tokens = usage.completion_tokens
    return BenchmarkRequestRecord(
        request_id=request_id,
        workload_item_id=workload_item_id,
        scenario_family=scenario_family,
        tenant_id=headers.get("x-switchyard-tenant-id", "default"),
        request_class=_request_class_from_headers(headers),
        session_id=headers.get("x-switchyard-session-id"),
        backend_name=response_payload["backend_name"],
        backend_type=_infer_backend_type(response_payload["backend_name"]),
        model_alias=model_alias,
        model_identifier=response_payload.get("model", model_alias),
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=round(latency_ms, 3),
        output_tokens=output_tokens,
        tokens_per_second=compute_tokens_per_second(
            output_tokens=output_tokens,
            total_latency_ms=latency_ms,
        ),
        routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
        route_decision=route_decision,
        control_plane_metadata=_control_plane_metadata_from_response(
            response_headers=response.headers,
            route_decision=route_decision,
            tenant_id=headers.get("x-switchyard-tenant-id", "default"),
            session_id=headers.get("x-switchyard-session-id"),
        ),
        hybrid_context=_hybrid_context_for_record(
            metadata=request_metadata,
            route_decision=route_decision,
            backend_name=response_payload["backend_name"],
            routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
        ),
        cache_observation=CacheObservation(),
        success=True,
        status_code=response.status_code,
        usage=usage,
    )


async def _run_streaming_gateway_request(
    *,
    benchmark_client: httpx.AsyncClient,
    request_id: str,
    payload: dict[str, object],
    headers: dict[str, str],
    model_alias: str,
    workload_item_id: str | None,
    scenario_family: WorkloadScenarioFamily | None,
    request_metadata: Mapping[str, str] | None,
    started_at: datetime,
    started_perf: float,
) -> BenchmarkRequestRecord:
    backend_name = "unknown"
    ttft_ms: float | None = None
    output_fragments: list[str] = []
    status_code = 500
    error: str | None = None
    async with benchmark_client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as response:
        route_decision = _parse_route_decision_header(
            response.headers.get("x-switchyard-route-decision")
        )
        status_code = response.status_code
        if response.status_code >= 400:
            error = (await response.aread()).decode()
        else:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw_event = line.removeprefix("data: ")
                if raw_event == "[DONE]":
                    break
                chunk_payload = json.loads(raw_event)
                backend_name = chunk_payload.get("backend_name", backend_name)
                for choice in chunk_payload["choices"]:
                    delta = choice["delta"]
                    content = delta.get("content")
                    if content is not None:
                        if ttft_ms is None and content.strip() != "":
                            ttft_ms = (perf_counter() - started_perf) * 1000
                        output_fragments.append(content)

    completed_at = datetime.now(UTC)
    latency_ms = (perf_counter() - started_perf) * 1000
    if status_code >= 400:
        return BenchmarkRequestRecord(
            request_id=request_id,
            workload_item_id=workload_item_id,
            scenario_family=scenario_family,
            tenant_id=headers.get("x-switchyard-tenant-id", "default"),
            request_class=_request_class_from_headers(headers),
            session_id=headers.get("x-switchyard-session-id"),
            backend_name=backend_name,
            model_alias=model_alias,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=round(latency_ms, 3),
            routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
            route_decision=route_decision,
            control_plane_metadata=_control_plane_metadata_from_response(
                response_headers=response.headers,
                route_decision=route_decision,
                tenant_id=headers.get("x-switchyard-tenant-id", "default"),
                session_id=headers.get("x-switchyard-session-id"),
            ),
            hybrid_context=_hybrid_context_for_record(
                metadata=request_metadata,
                route_decision=route_decision,
                backend_name=backend_name,
                routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
            ),
            success=False,
            status_code=status_code,
            cache_observation=CacheObservation(),
            error=error or "streaming request failed",
            error_category=_categorize_status_code(status_code),
        )

    output_tokens = estimate_token_count("".join(output_fragments).strip())
    return BenchmarkRequestRecord(
        request_id=request_id,
        workload_item_id=workload_item_id,
        scenario_family=scenario_family,
        tenant_id=headers.get("x-switchyard-tenant-id", "default"),
        request_class=_request_class_from_headers(headers),
        session_id=headers.get("x-switchyard-session-id"),
        backend_name=backend_name,
        backend_type=_infer_backend_type(backend_name),
        model_alias=model_alias,
        model_identifier=model_alias,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=round(latency_ms, 3),
        ttft_ms=None if ttft_ms is None else round(ttft_ms, 3),
        output_tokens=output_tokens,
        tokens_per_second=compute_tokens_per_second(
            output_tokens=output_tokens,
            total_latency_ms=latency_ms,
        ),
        routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
        route_decision=route_decision,
        control_plane_metadata=_control_plane_metadata_from_response(
            response_headers=response.headers,
            route_decision=route_decision,
            tenant_id=headers.get("x-switchyard-tenant-id", "default"),
            session_id=headers.get("x-switchyard-session-id"),
        ),
        hybrid_context=_hybrid_context_for_record(
            metadata=request_metadata,
            route_decision=route_decision,
            backend_name=backend_name,
            routing_policy=RoutingPolicy(headers["x-switchyard-routing-policy"]),
        ),
        cache_observation=CacheObservation(),
        success=True,
        status_code=status_code,
    )


async def _get_metrics_snapshot(
    *,
    benchmark_client: httpx.AsyncClient,
    metrics_path: str | None,
) -> list[PrometheusSample] | None:
    if metrics_path is None:
        return None

    response = await benchmark_client.get(metrics_path)
    if response.status_code != 200:
        return None
    return parse_prometheus_text(response.text)


def _merge_server_metrics_into_records(
    *,
    records: list[BenchmarkRequestRecord],
    before: list[PrometheusSample] | None,
    after: list[PrometheusSample] | None,
) -> None:
    if after is None:
        return

    previous_indexes = {
        sample.labels["index"]
        for sample in before or []
        if sample.name == "switchyard_backend_request_latency_ms" and "index" in sample.labels
    }
    execution_samples: dict[str, dict[str, object]] = {}
    for sample in after:
        if sample.name not in {
            "switchyard_backend_request_latency_ms",
            "switchyard_backend_ttft_ms",
            "switchyard_backend_output_tokens",
            "switchyard_backend_tokens_per_second",
            "switchyard_route_decision_latency_ms",
        }:
            continue
        index = sample.labels.get("index")
        if index is None or index in previous_indexes:
            continue
        entry = execution_samples.setdefault(index, {})
        if sample.name == "switchyard_backend_request_latency_ms":
            entry.update(sample.labels)
            entry["latency_ms"] = sample.value
        elif sample.name == "switchyard_backend_ttft_ms":
            entry["ttft_ms"] = sample.value
        elif sample.name == "switchyard_backend_output_tokens":
            entry["output_tokens"] = sample.value
        elif sample.name == "switchyard_backend_tokens_per_second":
            entry["tokens_per_second"] = sample.value
        elif sample.name == "switchyard_route_decision_latency_ms":
            entry["routing_policy"] = sample.labels.get("policy")
            entry["route_candidate_count"] = sample.labels.get("candidate_backend_count")
            entry["route_selected_backend"] = sample.labels.get("chosen_backend")
            entry["route_reason"] = sample.labels.get("route_reason")

    for record, index in zip(records, sorted(execution_samples, key=int), strict=False):
        sample_values = execution_samples[index]
        record.backend_name = str(sample_values.get("backend_name", record.backend_name))
        record.backend_type = str(
            sample_values.get("backend_type", record.backend_type or "unknown")
        )
        record.model_alias = str(sample_values.get("model", record.model_alias or "unknown"))
        model_identifier = sample_values.get("model_identifier")
        if isinstance(model_identifier, str):
            record.model_identifier = model_identifier
        ttft_value = sample_values.get("ttft_ms")
        if isinstance(ttft_value, float):
            record.ttft_ms = round(ttft_value, 3)
        output_value = sample_values.get("output_tokens")
        if isinstance(output_value, float):
            record.output_tokens = int(output_value)
        tps_value = sample_values.get("tokens_per_second")
        if isinstance(tps_value, float):
            record.tokens_per_second = round(tps_value, 6)
        policy_value = sample_values.get("routing_policy")
        if isinstance(policy_value, str):
            record.routing_policy = RoutingPolicy(policy_value)
        candidate_count_value = sample_values.get("route_candidate_count")
        if isinstance(candidate_count_value, str) and candidate_count_value.isdigit():
            record.route_candidate_count = int(candidate_count_value)
        selected_backend = sample_values.get("route_selected_backend")
        if isinstance(selected_backend, str) and selected_backend:
            record.fallback_used = selected_backend != record.backend_name
        route_reason = sample_values.get("route_reason")
        if isinstance(route_reason, str) and route_reason:
            record.route_reason = route_reason


def _build_request_for_scenario(
    *,
    scenario: BenchmarkScenario,
    index: int,
) -> ChatCompletionRequest:
    prompt_template = scenario.prompt_template or "Synthetic benchmark request {index}"
    workload = scenario.workload_generation
    prompt = prompt_template.format(index=index)
    prompt = _apply_workload_pattern(
        prompt=prompt,
        index=index,
        request_count=scenario.request_count,
        workload=workload,
    )
    return ChatCompletionRequest(
        model=scenario.model,
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content=prompt,
            )
        ],
        max_output_tokens=scenario.max_output_tokens,
        temperature=0.7 if scenario.temperature is None else scenario.temperature,
        top_p=1.0 if scenario.top_p is None else scenario.top_p,
        stream=scenario.stream,
    )


def _build_artifact(
    *,
    scenario: BenchmarkScenario,
    records: list[BenchmarkRequestRecord],
    run_timestamp: datetime,
    environment: BenchmarkEnvironmentMetadata,
    run_id_suffix: str | None = None,
    execution_target: ExecutionTarget | None = None,
    run_config: BenchmarkRunConfig | None = None,
    settings: Settings | None = None,
) -> BenchmarkRunArtifact:
    resolved_run_config = run_config or BenchmarkRunConfig()
    if settings is not None:
        resolved_run_config = attach_benchmark_config_snapshot(
            settings=settings,
            run_config=resolved_run_config,
        )
    return BenchmarkRunArtifact(
        run_id=build_run_id(
            run_timestamp=run_timestamp,
            policy=scenario.policy,
            suffix=run_id_suffix,
        ),
        timestamp=run_timestamp,
        git_sha=get_git_sha(),
        scenario=scenario,
        policy=scenario.policy,
        backends_involved=sorted({record.backend_name for record in records}),
        backend_types_involved=sorted(
            {record.backend_type for record in records if record.backend_type is not None}
        ),
        model_aliases_involved=sorted(
            {record.model_alias for record in records if record.model_alias is not None}
        ),
        request_count=len(records),
        summary=summarize_records(records),
        environment=environment,
        execution_target=execution_target,
        run_config=resolved_run_config,
        records=records,
    )


async def _build_http_benchmark_environment(
    *,
    benchmark_mode: str,
    benchmark_client: httpx.AsyncClient,
    gateway_base_url: str,
    metrics_path: str | None,
    stream: bool,
    timeout_seconds: float,
    deployment_target: BenchmarkDeploymentTarget | None,
    deployment_profile: DeploymentProfile | None,
    config_profile_name: str | None,
    control_plane_image: BackendImageMetadata | None,
    runtime_inspection_path: str | None,
    metadata: dict[str, str],
) -> BenchmarkEnvironmentMetadata:
    topology = [
        DeployedTopologyEndpoint(
            endpoint_id="control-plane-gateway",
            role="control_plane_gateway",
            address=gateway_base_url,
        )
    ]
    worker_inventory: list[BackendInstance] = []
    remote_worker_snapshot = None
    hybrid_execution = None
    remote_workers = None
    topology_capture_source: str | None = None
    capture_metadata = dict(metadata)
    if runtime_inspection_path is not None:
        try:
            runtime_response = await benchmark_client.get(runtime_inspection_path)
            runtime_response.raise_for_status()
            runtime_snapshot = RuntimeInspectionResponse.model_validate(runtime_response.json())
            topology.extend(_runtime_topology_endpoints(runtime_snapshot))
            worker_inventory = _runtime_worker_inventory(runtime_snapshot)
            hybrid_execution = runtime_snapshot.hybrid_execution
            remote_workers = runtime_snapshot.remote_workers
            remote_worker_snapshot = runtime_snapshot.remote_worker_registry
            topology_capture_source = "gateway_admin_runtime"
            capture_metadata["topology_captured_at"] = runtime_snapshot.captured_at.isoformat()
        except (httpx.HTTPError, ValueError) as exc:
            capture_metadata["topology_capture_error"] = str(exc)
    return BenchmarkEnvironmentMetadata(
        benchmark_mode=benchmark_mode,
        gateway_base_url=gateway_base_url,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        metrics_url=None if metrics_path is None else f"{gateway_base_url}{metrics_path}",
        stream=stream,
        timeout_seconds=timeout_seconds,
        deployed_topology=topology,
        worker_instance_inventory=worker_inventory,
        hybrid_execution=hybrid_execution,
        remote_workers=remote_workers,
        remote_worker_snapshot=remote_worker_snapshot,
        control_plane_image=control_plane_image,
        topology_capture_source=topology_capture_source,
        metadata=capture_metadata,
    )


def _runtime_topology_endpoints(
    runtime_snapshot: RuntimeInspectionResponse,
) -> list[DeployedTopologyEndpoint]:
    endpoints: list[DeployedTopologyEndpoint] = []
    for backend in runtime_snapshot.backends:
        for instance in backend.instances:
            endpoints.append(
                DeployedTopologyEndpoint(
                    endpoint_id=instance.instance_id,
                    role="worker_instance",
                    address=instance.endpoint,
                    transport=_safe_worker_transport(instance.transport),
                    execution_mode=_safe_execution_mode(instance.execution_mode),
                    locality_class=_safe_locality_class(instance.locality_class),
                    provider=instance.provider,
                    region=instance.region,
                    zone=instance.zone,
                    network_profile=_safe_network_profile(instance.network_profile),
                    metadata={
                        "backend_name": backend.backend_name,
                        "backend_type": backend.backend_type,
                        "transport": instance.transport,
                        "source_of_truth": instance.source_of_truth,
                        "health_state": instance.health_state,
                        "load_state": instance.load_state,
                        "runtime_label": (
                            instance.runtime.runtime_label
                            if instance.runtime is not None
                            else backend.runtime.runtime_label
                            if backend.runtime is not None
                            else "unknown"
                        ),
                    },
                )
            )
    return endpoints


def _runtime_worker_inventory(
    runtime_snapshot: RuntimeInspectionResponse,
) -> list[BackendInstance]:
    inventory: list[BackendInstance] = []
    for backend in runtime_snapshot.backends:
        backend_type = _safe_backend_type(backend.backend_type)
        for instance in backend.instances:
            source_of_truth = _safe_instance_source(instance.source_of_truth)
            inventory.append(
                BackendInstance(
                    instance_id=instance.instance_id,
                    endpoint=BackendNetworkEndpoint(
                        base_url=instance.endpoint,
                        transport=_safe_worker_transport(instance.transport),
                    ),
                    source_of_truth=source_of_truth,
                    backend_type=backend_type,
                    device_class=_safe_device_class(
                        instance.device_class,
                        backend_type=backend_type,
                    ),
                    runtime=(
                        instance.runtime.model_copy(deep=True)
                        if instance.runtime is not None
                        else backend.runtime.model_copy(deep=True)
                        if backend.runtime is not None
                        else None
                    ),
                    gpu=instance.gpu.model_copy(deep=True) if instance.gpu is not None else None,
                    locality=instance.locality,
                    locality_class=_safe_locality_class(instance.locality_class),
                    execution_mode=_safe_execution_mode(instance.execution_mode),
                    placement=CloudPlacementMetadata(
                        provider=instance.provider,
                        region=instance.region,
                        zone=instance.zone,
                    ),
                    cost_profile=CostBudgetProfile(),
                    readiness_hints=ReadinessHints(),
                    trust=TrustMetadata(
                        auth_state=_safe_auth_state(instance.auth_state),
                        trust_state=_safe_trust_state(instance.trust_state),
                    ),
                    network_characteristics=NetworkCharacteristics(
                        profile=_safe_network_profile(instance.network_profile)
                    ),
                    tags=list(instance.tags),
                    registration=BackendRegistrationMetadata(
                        state=_safe_registration_state(
                            raw_value=instance.registration_state,
                            fallback_source=source_of_truth,
                        ),
                        last_heartbeat_at=instance.last_seen_at,
                        source="admin_runtime",
                    ),
                    health=BackendHealth(
                        state=_safe_backend_health_state(instance.health_state),
                        load_state=_safe_backend_load_state(instance.load_state),
                    ),
                    observed_capacity=(
                        instance.observed_capacity.model_copy(deep=True)
                        if instance.observed_capacity is not None
                        else None
                    ),
                    last_seen_at=instance.last_seen_at,
                    metadata={
                        "backend_name": backend.backend_name,
                        "runtime_health_state": instance.health_state,
                        "runtime_load_state": instance.load_state,
                    },
                )
            )
    return inventory


def _safe_backend_type(raw_value: str) -> BackendType | None:
    try:
        return BackendType(raw_value)
    except ValueError:
        return None


def _safe_instance_source(raw_value: str) -> BackendInstanceSource:
    try:
        return BackendInstanceSource(raw_value)
    except ValueError:
        return BackendInstanceSource.STATIC_CONFIG


def _safe_worker_transport(raw_value: str) -> WorkerTransportType:
    try:
        return WorkerTransportType(raw_value)
    except ValueError:
        return WorkerTransportType.HTTP


def _safe_locality_class(raw_value: str | None) -> WorkerLocalityClass:
    if raw_value is None:
        return WorkerLocalityClass.UNKNOWN
    try:
        return WorkerLocalityClass(raw_value)
    except ValueError:
        return WorkerLocalityClass.UNKNOWN


def _safe_execution_mode(raw_value: str | None) -> ExecutionModeLabel:
    if raw_value is None:
        return ExecutionModeLabel.HOST_NATIVE
    try:
        return ExecutionModeLabel(raw_value)
    except ValueError:
        return ExecutionModeLabel.HOST_NATIVE


def _safe_network_profile(raw_value: str | None) -> NetworkProfile:
    if raw_value is None:
        return NetworkProfile.UNKNOWN
    try:
        return NetworkProfile(raw_value)
    except ValueError:
        return NetworkProfile.UNKNOWN


def _safe_auth_state(raw_value: str | None) -> WorkerAuthState:
    if raw_value is None:
        return WorkerAuthState.UNKNOWN
    try:
        return WorkerAuthState(raw_value)
    except ValueError:
        return WorkerAuthState.UNKNOWN


def _safe_trust_state(raw_value: str | None) -> WorkerTrustState:
    if raw_value is None:
        return WorkerTrustState.UNKNOWN
    try:
        return WorkerTrustState(raw_value)
    except ValueError:
        return WorkerTrustState.UNKNOWN


def _safe_backend_health_state(raw_value: str) -> BackendHealthState:
    try:
        return BackendHealthState(raw_value)
    except ValueError:
        return BackendHealthState.UNAVAILABLE


def _safe_backend_load_state(raw_value: str) -> BackendLoadState:
    try:
        return BackendLoadState(raw_value)
    except ValueError:
        return BackendLoadState.COLD


def _registration_state_for_instance_source(
    source_of_truth: BackendInstanceSource,
) -> WorkerRegistrationState:
    if source_of_truth is BackendInstanceSource.REGISTERED:
        return WorkerRegistrationState.REGISTERED
    if source_of_truth is BackendInstanceSource.DISCOVERED:
        return WorkerRegistrationState.DISCOVERED
    return WorkerRegistrationState.STATIC


def _safe_registration_state(
    raw_value: str | None,
    *,
    fallback_source: BackendInstanceSource,
) -> WorkerRegistrationState:
    if raw_value is not None:
        try:
            return WorkerRegistrationState(raw_value)
        except ValueError:
            pass
    return _registration_state_for_instance_source(fallback_source)


def _device_class_for_backend_type(backend_type: BackendType | None) -> DeviceClass:
    if backend_type is BackendType.MLX_LM or backend_type is BackendType.VLLM_METAL:
        return DeviceClass.APPLE_GPU
    if backend_type is BackendType.MOCK:
        return DeviceClass.CPU
    if backend_type is BackendType.VLLM_CUDA:
        return DeviceClass.NVIDIA_GPU
    return DeviceClass.REMOTE


def _safe_device_class(
    raw_value: str | None,
    *,
    backend_type: BackendType | None,
) -> DeviceClass:
    if raw_value is not None:
        try:
            return DeviceClass(raw_value)
        except ValueError:
            pass
    return _device_class_for_backend_type(backend_type)


def _worker_inventory_summary(
    inventory: list[BackendInstance],
) -> dict[str, int]:
    counts = {"local": 0, "remote": 0, "external": 0}
    for instance in inventory:
        if instance.execution_mode is ExecutionModeLabel.EXTERNAL_SERVICE:
            counts["external"] += 1
        elif instance.execution_mode is ExecutionModeLabel.REMOTE_WORKER:
            counts["remote"] += 1
        else:
            counts["local"] += 1
    return counts


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * (percentile / 100)) - 1)
    return round(sorted_values[index], 3)


def _output_tokens_for_record(record: BenchmarkRequestRecord) -> int:
    if record.output_tokens is not None:
        return record.output_tokens
    if record.usage is not None:
        return record.usage.completion_tokens
    return 0


def _infer_backend_type(backend_name: str) -> str:
    if backend_name.startswith("mlx-lm:"):
        return BackendType.MLX_LM.value
    if backend_name.startswith("vllm-metal:"):
        return BackendType.VLLM_METAL.value
    if backend_name.startswith("mock"):
        return BackendType.MOCK.value
    return "unknown"


def _compact_route_reason(explanation: RouteExplanation | None) -> str | None:
    if explanation is None:
        return None
    return explanation.compact_reason()


def _comparison_label(
    *,
    policy: RoutingPolicy,
    internal_backend_pin: str | None,
) -> str:
    if internal_backend_pin is None:
        return policy.value
    return f"{policy.value}:pinned:{internal_backend_pin}"


def _apply_workload_pattern(
    *,
    prompt: str,
    index: int,
    request_count: int,
    workload: WorkloadGenerationConfig,
) -> str:
    if workload.pattern is WorkloadPattern.UNIFORM:
        return prompt
    if workload.pattern is WorkloadPattern.REPEATED_PREFIX:
        shared_prefix = workload.shared_prefix or "shared-prefix"
        return f"{shared_prefix}\n{prompt}"

    burst_number = (index // workload.burst_size) + 1
    burst_total = math.ceil(request_count / workload.burst_size)
    burst_position = (index % workload.burst_size) + 1
    rng = random.Random(workload.seed + index)
    burst_tag = rng.choice(("burst", "spike", "wave"))
    return (
        f"[{burst_tag} {burst_number}/{burst_total} item {burst_position}/{workload.burst_size}] "
        f"{prompt}"
    )


def _extract_error_message(payload: object) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return "request failed"


def _parse_route_decision_header(raw_value: str | None) -> RouteDecision | None:
    if raw_value is None or raw_value == "":
        return None
    try:
        return RouteDecision.model_validate_json(raw_value)
    except Exception:
        return None


def _parse_admission_decision_header(raw_value: str | None) -> AdmissionDecision | None:
    if raw_value is None or raw_value == "":
        return None
    try:
        return AdmissionDecision.model_validate_json(raw_value)
    except Exception:
        return None


def _categorize_status_code(status_code: int) -> str:
    if status_code >= 500:
        return "server_error"
    if status_code >= 400:
        return "client_error"
    return "ok"


def _request_class_from_headers(headers: Mapping[str, str]) -> RequestClass:
    raw_value = headers.get("x-switchyard-request-class", RequestClass.STANDARD.value)
    return RequestClass(raw_value)


def _control_plane_metadata_from_response(
    *,
    response_headers: Mapping[str, str],
    route_decision: RouteDecision | None,
    tenant_id: str,
    session_id: str | None,
) -> ControlPlaneReportMetadata | None:
    admission_decision = _parse_admission_decision_header(
        response_headers.get("x-switchyard-admission-decision")
    )
    if route_decision is None and admission_decision is None:
        return None
    if route_decision is not None and route_decision.telemetry_metadata is not None:
        return ControlPlaneReportMetadata(
            tenant_id=tenant_id,
            session_id=session_id,
            admission_decision=route_decision.admission_decision or admission_decision,
            circuit_breaker_state=route_decision.circuit_breaker_state,
            sticky_route=route_decision.sticky_route,
            canary_policy=route_decision.canary_policy,
            shadow_policy=route_decision.shadow_policy,
            shadow_decision=route_decision.shadow_decision,
            policy_reference=route_decision.policy_reference,
            topology_reference=route_decision.topology_reference,
            execution_observation=route_decision.execution_observation,
            telemetry_metadata=route_decision.telemetry_metadata,
        )
    return ControlPlaneReportMetadata(
        tenant_id=tenant_id,
        session_id=session_id,
        admission_decision=admission_decision,
    )


async def _run_workload_warmup(
    *,
    benchmark_client: httpx.AsyncClient,
    scenario: BenchmarkScenario,
    warmup: BenchmarkWarmupConfig,
    execution_target: ExecutionTarget,
) -> None:
    if not warmup.enabled or warmup.request_count == 0:
        return

    items = _scenario_items(scenario)
    for index in range(min(warmup.request_count, len(items))):
        item = items[index]
        headers = _workload_headers(
            request_id=f"{scenario.name}_warmup_{index:04d}",
            scenario=scenario,
            execution_target=execution_target,
            tenant_id=item.metadata.get("tenant_id", "default"),
            request_class=item.metadata.get("request_class", RequestClass.STANDARD.value),
            session_id=item.metadata.get("session_id"),
        )
        payload = _build_request_from_workload_item(
            scenario=scenario,
            item=item,
            model_alias=execution_target.model_alias,
        ).model_dump(mode="json", exclude_none=True)
        if scenario.stream:
            await _consume_streaming_warmup_request(
                benchmark_client=benchmark_client,
                payload=payload,
                headers=headers,
            )
            continue
        await benchmark_client.post("/v1/chat/completions", json=payload, headers=headers)


async def _execute_workload_items(
    *,
    benchmark_client: httpx.AsyncClient,
    scenario: BenchmarkScenario,
    execution_target: ExecutionTarget,
) -> list[BenchmarkRequestRecord]:
    items = _scenario_items(scenario)
    if _is_burst_workload(items):
        burst_window_records: list[BenchmarkRequestRecord] = []
        for burst_group in _burst_groups(items):
            burst_records = await asyncio.gather(
                *[
                    _execute_workload_item(
                        benchmark_client=benchmark_client,
                        scenario=scenario,
                        execution_target=execution_target,
                        item=item,
                        index=index,
                    )
                    for index, item in burst_group
                ]
            )
            burst_window_records.extend(burst_records)
        return burst_window_records

    records: list[BenchmarkRequestRecord] = []
    for index, item in enumerate(items):
        records.append(
            await _execute_workload_item(
                benchmark_client=benchmark_client,
                scenario=scenario,
                execution_target=execution_target,
                item=item,
                index=index,
            )
        )
    return records


def _resolved_workload_warmup(
    *,
    scenario: BenchmarkScenario,
    warmup: BenchmarkWarmupConfig | None,
) -> BenchmarkWarmupConfig:
    resolved = warmup or BenchmarkWarmupConfig()
    if not any(
        item.family is WorkloadScenarioFamily.REPEATED_PREFIX for item in _scenario_items(scenario)
    ):
        return resolved
    if resolved.request_count > 0:
        return resolved
    return BenchmarkWarmupConfig(enabled=True, request_count=1, concurrency=1)


async def _execute_workload_item(
    *,
    benchmark_client: httpx.AsyncClient,
    scenario: BenchmarkScenario,
    execution_target: ExecutionTarget,
    item: WorkloadItem,
    index: int,
) -> BenchmarkRequestRecord:
    request_id = f"{scenario.name}_{index:04d}"
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    payload = _build_request_from_workload_item(
        scenario=scenario,
        item=item,
        model_alias=execution_target.model_alias,
    ).model_dump(mode="json", exclude_none=True)
    headers = _workload_headers(
        request_id=request_id,
        scenario=scenario,
        execution_target=execution_target,
        tenant_id=item.metadata.get("tenant_id", "default"),
        request_class=item.metadata.get("request_class", RequestClass.STANDARD.value),
        session_id=item.metadata.get("session_id"),
    )
    if scenario.stream:
        return await _run_streaming_gateway_request(
            benchmark_client=benchmark_client,
            request_id=request_id,
            payload=payload,
            headers=headers,
            model_alias=execution_target.model_alias,
            workload_item_id=item.item_id,
            scenario_family=item.family,
            request_metadata=item.metadata,
            started_at=started_at,
            started_perf=started_perf,
        )
    return await _run_non_streaming_gateway_request(
        benchmark_client=benchmark_client,
        request_id=request_id,
        payload=payload,
        headers=headers,
        model_alias=execution_target.model_alias,
        workload_item_id=item.item_id,
        scenario_family=item.family,
        request_metadata=item.metadata,
        started_at=started_at,
        started_perf=started_perf,
    )


async def _run_trace_replay_warmup(
    *,
    benchmark_client: httpx.AsyncClient,
    traces: list[CapturedTraceRecord],
    replay_plan: ReplayPlan,
    scenario: BenchmarkScenario,
) -> None:
    if not replay_plan.warmup.enabled or replay_plan.warmup.request_count == 0:
        return

    traces_by_id = {trace.record_id: trace for trace in traces}
    for request in replay_plan.requests[: replay_plan.warmup.request_count]:
        trace = traces_by_id[request.source_trace_record_id]
        payload = _build_request_from_trace(
            trace=trace,
            model_alias=replay_plan.execution_target.model_alias,
        ).model_dump(mode="json", exclude_none=True)
        headers = _replay_headers(
            request_id=f"{request.replay_request_id}_warmup",
            scenario=scenario,
            execution_target=replay_plan.execution_target,
            tenant_id=request.tenant_id,
            request_class=request.request_class.value,
            session_id=request.session_id,
        )
        if request.stream:
            await _consume_streaming_warmup_request(
                benchmark_client=benchmark_client,
                payload=payload,
                headers=headers,
            )
            continue
        await benchmark_client.post("/v1/chat/completions", json=payload, headers=headers)


async def _execute_trace_replay(
    *,
    benchmark_client: httpx.AsyncClient,
    traces: list[CapturedTraceRecord],
    replay_plan: ReplayPlan,
    scenario: BenchmarkScenario,
) -> list[BenchmarkRequestRecord]:
    traces_by_id = {trace.record_id: trace for trace in traces}
    if replay_plan.replay_mode is ReplayMode.FIXED_CONCURRENCY:
        semaphore = asyncio.Semaphore(replay_plan.concurrency)

        async def _run_with_semaphore(
            request: ReplayRequest,
        ) -> tuple[int, BenchmarkRequestRecord]:
            async with semaphore:
                record = await _execute_one_replay_request(
                    benchmark_client=benchmark_client,
                    trace=traces_by_id[request.source_trace_record_id],
                    replay_request=request,
                    scenario=scenario,
                    execution_target=replay_plan.execution_target,
                )
                return request.order_index, record

        results = await asyncio.gather(
            *[_run_with_semaphore(request) for request in replay_plan.requests]
        )
        return [record for _, record in sorted(results, key=lambda item: item[0])]

    records: list[BenchmarkRequestRecord] = []
    for request in replay_plan.requests:
        records.append(
            await _execute_one_replay_request(
                benchmark_client=benchmark_client,
                trace=traces_by_id[request.source_trace_record_id],
                replay_request=request,
                scenario=scenario,
                execution_target=replay_plan.execution_target,
            )
        )
    return records


async def _consume_streaming_warmup_request(
    *,
    benchmark_client: httpx.AsyncClient,
    payload: dict[str, object],
    headers: dict[str, str],
) -> None:
    async with benchmark_client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as response:
        await response.aread()


def _build_request_from_workload_item(
    *,
    scenario: BenchmarkScenario,
    item: WorkloadItem,
    model_alias: str,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model_alias,
        messages=[ChatMessage(role=ChatRole.USER, content=item.prompt)],
        max_output_tokens=scenario.max_output_tokens,
        temperature=0.7 if scenario.temperature is None else scenario.temperature,
        top_p=1.0 if scenario.top_p is None else scenario.top_p,
        stream=scenario.stream,
    )


def _build_request_from_trace(
    *,
    trace: CapturedTraceRecord,
    model_alias: str,
) -> ChatCompletionRequest:
    payload = trace.normalized_request_payload
    if payload is None:
        msg = f"trace {trace.record_id} does not contain a replayable request payload"
        raise ValueError(msg)
    request_payload = dict(payload)
    request_payload["model"] = model_alias
    try:
        return ChatCompletionRequest.model_validate(request_payload)
    except Exception as exc:
        msg = f"trace {trace.record_id} capture mode {trace.capture_mode.value} is not replayable"
        raise ValueError(msg) from exc


def _workload_headers(
    *,
    request_id: str,
    scenario: BenchmarkScenario,
    execution_target: ExecutionTarget,
    tenant_id: str = "default",
    request_class: str = RequestClass.STANDARD.value,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "x-request-id": request_id,
        "x-switchyard-workload-shape": scenario.workload_shape.value,
        "x-switchyard-tenant-id": tenant_id,
        "x-switchyard-request-class": request_class,
    }
    if session_id is not None:
        headers["x-switchyard-session-id"] = session_id
    if execution_target.routing_policy is not None:
        headers["x-switchyard-routing-policy"] = execution_target.routing_policy.value
    else:
        headers["x-switchyard-routing-policy"] = scenario.policy.value
    if execution_target.pinned_backend is not None:
        headers["x-switchyard-internal-backend-pin"] = execution_target.pinned_backend
    return headers


def _replay_headers(
    *,
    request_id: str,
    scenario: BenchmarkScenario,
    execution_target: ExecutionTarget,
    tenant_id: str = "default",
    request_class: str = RequestClass.STANDARD.value,
    session_id: str | None = None,
) -> dict[str, str]:
    return _workload_headers(
        request_id=request_id,
        scenario=scenario,
        execution_target=execution_target,
        tenant_id=tenant_id,
        request_class=request_class,
        session_id=session_id,
    )


def _scenario_items(scenario: BenchmarkScenario) -> list[WorkloadItem]:
    if scenario.items:
        return scenario.items
    return [
        WorkloadItem(
            item_id=f"{scenario.name}_{index:04d}",
            family=scenario.family,
            prompt=_build_request_for_scenario(scenario=scenario, index=index).messages[0].content,
            metadata={"generated": "true"},
        )
        for index in range(scenario.request_count)
    ]


def _is_burst_workload(items: list[WorkloadItem]) -> bool:
    return any(item.family is WorkloadScenarioFamily.CONCURRENCY_BURST for item in items)


def _burst_groups(
    items: list[WorkloadItem],
) -> list[list[tuple[int, WorkloadItem]]]:
    groups: list[list[tuple[int, WorkloadItem]]] = []
    current_key: str | None = None
    current_group: list[tuple[int, WorkloadItem]] = []
    for index, item in enumerate(items):
        burst_group = item.metadata.get("burst_group")
        key = burst_group if item.family is WorkloadScenarioFamily.CONCURRENCY_BURST else None
        if key != current_key and current_group:
            groups.append(current_group)
            current_group = []
        current_group.append((index, item))
        current_key = key
    if current_group:
        groups.append(current_group)
    return groups


def _build_replay_scenario(
    *,
    traces: list[CapturedTraceRecord],
    execution_target: ExecutionTarget,
    source_run_id: str,
) -> BenchmarkScenario:
    first_trace = traces[0]
    inferred_policy = (
        execution_target.routing_policy
        or first_trace.execution_target.routing_policy
        or RoutingPolicy.BALANCED
    )
    stream_values = {trace.stream for trace in traces}
    return BenchmarkScenario(
        name=f"trace_replay_{source_run_id}",
        model=execution_target.model_alias,
        model_alias=execution_target.model_alias,
        family=WorkloadScenarioFamily.MIXED,
        policy=inferred_policy,
        workload_shape=WorkloadShape.INTERACTIVE,
        request_count=len(traces),
        input_messages_per_request=1,
        stream=len(stream_values) == 1 and True in stream_values,
        prompt_template="trace replay",
    )


def _planned_replay_requests(
    *,
    traces: list[CapturedTraceRecord],
    replay_mode: ReplayMode,
) -> list[ReplayRequest]:
    indexed_traces = list(enumerate(traces))
    if replay_mode is ReplayMode.PRESERVE_ORDER_WITHOUT_ORIGINAL_TIMING:
        indexed_traces.sort(
            key=lambda item: (
                item[1].request_timestamp,
                item[0],
            )
        )

    planned_requests: list[ReplayRequest] = []
    previous_timestamp: datetime | None = None
    for order_index, (_, trace) in enumerate(indexed_traces):
        interarrival_ms: float | None = None
        if previous_timestamp is not None:
            interarrival_ms = round(
                max(
                    (trace.request_timestamp - previous_timestamp).total_seconds() * 1000,
                    0.0,
                ),
                3,
            )
        planned_requests.append(
            ReplayRequest(
                replay_request_id=f"replay_{trace.request_id}",
                source_request_id=trace.request_id,
                source_trace_record_id=trace.record_id,
                order_index=order_index,
                original_request_timestamp=trace.request_timestamp,
                original_interarrival_ms=interarrival_ms,
                scheduled_offset_ms=None,
                stream=trace.stream,
                tenant_id=trace.tenant_id,
                request_class=trace.request_class,
                session_id=trace.session_id,
                request_features=trace.request_features,
                policy_reference=trace.policy_reference,
                topology_reference=trace.topology_reference,
                hybrid_context=trace.hybrid_context,
                metadata={
                    "original_capture_mode": trace.capture_mode.value,
                    **trace.metadata,
                },
            )
        )
        previous_timestamp = trace.request_timestamp
    return planned_requests


async def _execute_one_replay_request(
    *,
    benchmark_client: httpx.AsyncClient,
    trace: CapturedTraceRecord,
    replay_request: ReplayRequest,
    scenario: BenchmarkScenario,
    execution_target: ExecutionTarget,
) -> BenchmarkRequestRecord:
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    payload = _build_request_from_trace(
        trace=trace,
        model_alias=execution_target.model_alias,
    ).model_dump(mode="json", exclude_none=True)
    headers = _replay_headers(
        request_id=replay_request.replay_request_id,
        scenario=scenario,
        execution_target=execution_target,
        tenant_id=replay_request.tenant_id,
        request_class=replay_request.request_class.value,
        session_id=replay_request.session_id,
    )
    if replay_request.stream:
        record = await _run_streaming_gateway_request(
            benchmark_client=benchmark_client,
            request_id=replay_request.replay_request_id,
            payload=payload,
            headers=headers,
            model_alias=execution_target.model_alias,
            workload_item_id=None,
            scenario_family=None,
            request_metadata=replay_request.metadata,
            started_at=started_at,
            started_perf=started_perf,
        )
    else:
        record = await _run_non_streaming_gateway_request(
            benchmark_client=benchmark_client,
            request_id=replay_request.replay_request_id,
            payload=payload,
            headers=headers,
            model_alias=execution_target.model_alias,
            workload_item_id=None,
            scenario_family=None,
            request_metadata=replay_request.metadata,
            started_at=started_at,
            started_perf=started_perf,
        )
    record.source_request_id = replay_request.source_request_id
    record.source_trace_record_id = replay_request.source_trace_record_id
    record.replay_correlation_id = replay_request.replay_request_id
    return record


def _workload_run_suffix(execution_target: ExecutionTarget) -> str | None:
    if execution_target.target_type is ExecutionTargetType.PINNED_BACKEND:
        return f"workload_{execution_target.pinned_backend}"
    if execution_target.target_type is ExecutionTargetType.ROUTING_POLICY:
        return f"workload_{execution_target.model_alias}"
    return f"workload_{execution_target.model_alias}"


def _trace_replay_run_suffix(
    execution_target: ExecutionTarget,
    replay_mode: ReplayMode,
) -> str:
    target_suffix = execution_target.model_alias
    if execution_target.target_type is ExecutionTargetType.PINNED_BACKEND:
        target_suffix = execution_target.pinned_backend or execution_target.model_alias
    return f"replay_{replay_mode.value}_{target_suffix}"
