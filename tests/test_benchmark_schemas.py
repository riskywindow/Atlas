from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import RoutingPolicy, WorkloadShape


def test_benchmark_artifact_serializes() -> None:
    started_at = datetime(2026, 3, 15, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=25)
    scenario = BenchmarkScenario(
        name="smoke",
        model="mock-chat",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        request_count=1,
    )
    record = BenchmarkRequestRecord(
        request_id="req_1",
        backend_name="mock-a",
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=25.0,
        success=True,
        status_code=200,
        usage=UsageStats(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    artifact = BenchmarkRunArtifact(
        run_id="run_1",
        scenario=scenario,
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-a"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=25.0,
            p95_latency_ms=25.0,
        ),
        records=[record],
    )

    payload = artifact.model_dump(mode="json")

    assert payload["scenario"]["name"] == "smoke"
    assert payload["records"][0]["latency_ms"] == 25.0


def test_benchmark_record_rejects_invalid_failure_shape() -> None:
    started_at = datetime(2026, 3, 15, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=10)

    try:
        BenchmarkRequestRecord(
            request_id="req_2",
            backend_name="mock-a",
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=10.0,
            success=False,
            status_code=500,
        )
    except ValidationError as exc:
        assert "error message" in str(exc)
    else:
        raise AssertionError("BenchmarkRequestRecord should require an error for failures")
