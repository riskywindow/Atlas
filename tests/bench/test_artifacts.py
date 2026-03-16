from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from switchyard.bench.artifacts import (
    build_run_id,
    build_synthetic_scenario,
    run_synthetic_benchmark,
    write_artifact,
)
from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkSummary,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import RoutingPolicy


@pytest.mark.asyncio
async def test_run_synthetic_benchmark_builds_artifact() -> None:
    scenario = build_synthetic_scenario(request_count=2, policy=RoutingPolicy.BALANCED)
    artifact = await run_synthetic_benchmark(
        scenario=scenario,
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert artifact.run_id == "20260315T120000Z_balanced"
    assert artifact.request_count == 2
    assert artifact.summary.success_count == 2
    assert artifact.backends_involved == ["mock-local-fast"]


def test_write_artifact_outputs_stable_json(tmp_path: Path) -> None:
    scenario = build_synthetic_scenario(request_count=1)

    started_at = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 3, 15, 12, 0, 0, 1000, tzinfo=UTC)
    benchmark_artifact = BenchmarkRunArtifact(
        run_id=build_run_id(
            run_timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
            policy=RoutingPolicy.BALANCED,
        ),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        git_sha="abcdef123456",
        scenario=scenario,
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-local-fast"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=1.0,
            p95_latency_ms=1.0,
        ),
        records=[
            BenchmarkRequestRecord(
                request_id="synthetic_phase0_0000",
                backend_name="mock-local-fast",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=1.0,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=4, completion_tokens=4, total_tokens=8),
            )
        ],
    )
    output_path = tmp_path / "artifact.json"

    write_artifact(benchmark_artifact, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "20260315T120000Z_balanced"
    assert payload["summary"]["avg_latency_ms"] == 1.0
    assert output_path.read_text(encoding="utf-8").endswith("\n")
