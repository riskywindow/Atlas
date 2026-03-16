"""Benchmark artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from subprocess import DEVNULL, CalledProcessError, check_output
from time import perf_counter

from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.adapters.registry import AdapterRegistry
from switchyard.router.service import RouterService
from switchyard.schemas.backend import BackendCapabilities, BackendType, DeviceClass
from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import RequestContext, RoutingPolicy, WorkloadShape


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Result of a benchmark run including the artifact and written path."""

    artifact: BenchmarkRunArtifact
    output_path: Path


def build_synthetic_scenario(
    *,
    request_count: int,
    model: str = "mock-chat",
    policy: RoutingPolicy = RoutingPolicy.BALANCED,
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE,
) -> BenchmarkScenario:
    """Construct a small synthetic Phase 0 benchmark scenario."""

    return BenchmarkScenario(
        name="synthetic_phase0",
        model=model,
        policy=policy,
        workload_shape=workload_shape,
        request_count=request_count,
        input_messages_per_request=1,
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


async def run_synthetic_benchmark(
    *,
    scenario: BenchmarkScenario,
    registry: AdapterRegistry | None = None,
    timestamp: datetime | None = None,
) -> BenchmarkRunArtifact:
    """Run a small benchmark directly against the router and backend layer."""

    resolved_registry = registry or build_default_registry()
    router = RouterService(resolved_registry)
    run_timestamp = timestamp or datetime.now(UTC)
    records: list[BenchmarkRequestRecord] = []

    for index in range(scenario.request_count):
        request_id = f"{scenario.name}_{index:04d}"
        request = ChatCompletionRequest(
            model=scenario.model,
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content=f"Synthetic benchmark request {index}",
                )
            ],
        )
        context = RequestContext(
            request_id=request_id,
            policy=scenario.policy,
            workload_shape=scenario.workload_shape,
        )
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        try:
            decision = await router.route(request, context)
            adapter = resolved_registry.get(decision.backend_name)
            response = await adapter.generate(request, context)
        except Exception as exc:
            completed_at = datetime.now(UTC)
            latency_ms = (perf_counter() - started_perf) * 1000
            records.append(
                BenchmarkRequestRecord(
                    request_id=request_id,
                    backend_name="unrouted",
                    started_at=started_at,
                    completed_at=completed_at,
                    latency_ms=round(latency_ms, 3),
                    success=False,
                    status_code=503,
                    error=str(exc),
                )
            )
            continue

        completed_at = datetime.now(UTC)
        latency_ms = (perf_counter() - started_perf) * 1000
        records.append(
            BenchmarkRequestRecord(
                request_id=request_id,
                backend_name=decision.backend_name,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=round(latency_ms, 3),
                success=True,
                status_code=200,
                usage=response.usage,
            )
        )

    summary = summarize_records(records)
    return BenchmarkRunArtifact(
        run_id=build_run_id(run_timestamp=run_timestamp, policy=scenario.policy),
        timestamp=run_timestamp,
        git_sha=get_git_sha(),
        scenario=scenario,
        policy=scenario.policy,
        backends_involved=sorted({record.backend_name for record in records}),
        request_count=len(records),
        summary=summary,
        records=records,
    )


def summarize_records(records: list[BenchmarkRequestRecord]) -> BenchmarkSummary:
    """Summarize per-request benchmark records."""

    latencies = [record.latency_ms for record in records]
    success_count = sum(1 for record in records if record.success)
    failure_count = len(records) - success_count
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    sorted_latencies = sorted(latencies)

    return BenchmarkSummary(
        request_count=len(records),
        success_count=success_count,
        failure_count=failure_count,
        avg_latency_ms=round(mean(latencies), 3) if latencies else 0.0,
        p95_latency_ms=sorted_latencies[p95_index] if sorted_latencies else 0.0,
    )


def write_artifact(artifact: BenchmarkRunArtifact, output_path: Path) -> Path:
    """Write a benchmark artifact to disk in a stable JSON format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            artifact.model_dump(mode="json", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def default_output_path(base_dir: Path, artifact: BenchmarkRunArtifact) -> Path:
    """Return the default output path for an artifact."""

    return base_dir / f"{artifact.run_id}.json"


def build_run_id(*, run_timestamp: datetime, policy: RoutingPolicy) -> str:
    """Build a reproducible run identifier."""

    return f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}_{policy.value}"


def get_git_sha() -> str | None:
    """Return the current git SHA when available."""

    try:
        return (
            check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                stderr=DEVNULL,
                text=True,
            )
            .strip()
            or None
        )
    except (CalledProcessError, FileNotFoundError):
        return None
