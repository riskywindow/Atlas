from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from typer.testing import CliRunner

from switchyard.adapters.registry import AdapterRegistry
from switchyard.bench import cli
from switchyard.bench.cli import app
from switchyard.bench.simulation import compare_candidate_policies_offline
from switchyard.bench.workloads import build_workload_manifest
from switchyard.config import GenerationDefaults, LocalModelConfig, Settings, WarmupSettings
from switchyard.schemas.backend import BackendImageMetadata, BackendType, DeploymentProfile
from switchyard.schemas.benchmark import (
    BenchmarkComparisonArtifact,
    BenchmarkComparisonDeltaSummary,
    BenchmarkComparisonSideSummary,
    BenchmarkDeploymentTarget,
    BenchmarkEnvironmentMetadata,
    BenchmarkPolicyComparison,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
    BenchmarkTargetComparisonArtifact,
    BenchmarkWarmupConfig,
    CapturedTraceRecord,
    ComparisonSourceKind,
    CounterfactualObjective,
    ExecutionTarget,
    ExecutionTargetType,
    ExplainablePolicySpec,
    ReplayMode,
    TraceCaptureMode,
    WorkloadGenerationConfig,
    WorkloadScenarioFamily,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import RoutingPolicy, WorkloadShape


def test_run_synthetic_cli_writes_artifact(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-synthetic",
            "--request-count",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["request_count"] == 2
    assert payload["policy"] == "balanced"
    assert payload["summary"]["success_count"] == 2


def test_run_synthetic_cli_writes_markdown_report_when_requested(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-synthetic",
            "--request-count",
            "2",
            "--workload-pattern",
            "repeated_prefix",
            "--shared-prefix",
            "phase3-prefix",
            "--markdown-report",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    report_path = artifact_path.with_suffix(".md")
    assert report_path.exists()
    assert "phase3-prefix" in report_path.read_text(encoding="utf-8")


def test_run_gateway_cli_requires_a_model_when_none_is_configured(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-gateway",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "No model alias provided" in (result.stdout + result.stderr)


def test_generate_workload_cli_writes_manifest(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "generate-workload",
            "--family",
            "mixed",
            "--model-alias",
            "chat-shared",
            "--request-count",
            "3",
            "--seed",
            "9",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["family"] == WorkloadScenarioFamily.MIXED.value
    assert payload["model_alias"] == "chat-shared"
    assert len(payload["items"]) == 3


def test_generate_report_cli_writes_markdown_from_run_artifact(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact = BenchmarkRunArtifact(
        run_id="report-run",
        scenario=BenchmarkScenario(
            name="report-scenario",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=1,
        ),
        policy=RoutingPolicy.BALANCED,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.LOGICAL_ALIAS,
            model_alias="chat-shared",
        ),
        backends_involved=["mock-a"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=5.0,
            p50_latency_ms=5.0,
            p95_latency_ms=5.0,
            avg_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=4,
            avg_output_tokens=4.0,
            avg_tokens_per_second=8.0,
            p95_tokens_per_second=8.0,
            fallback_count=0,
            chosen_backend_counts={"mock-a": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="workload_manifest",
            platform="macOS-14",
            machine="arm64",
            python_version="3.12.0",
        ),
        records=[
            BenchmarkRequestRecord(
                request_id="req-1",
                workload_item_id="item-1",
                backend_name="mock-a",
                backend_type="mock",
                model_alias="chat-shared",
                started_at=datetime(2026, 3, 16, tzinfo=UTC),
                completed_at=datetime(2026, 3, 16, tzinfo=UTC),
                latency_ms=5.0,
                output_tokens=4,
                tokens_per_second=8.0,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=5, completion_tokens=4, total_tokens=9),
            )
        ],
    )
    artifact_path = tmp_path / "report-run.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    report_path = tmp_path / "report-run.md"

    result = runner.invoke(
        app,
        [
            "generate-report",
            str(artifact_path),
            "--output-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert Path(result.stdout.strip()) == report_path
    markdown = report_path.read_text(encoding="utf-8")
    assert "# Switchyard Benchmark Report: report-run" in markdown
    assert "## Aggregate Metrics" in markdown
    assert "## Route and Backend Distributions" in markdown


def test_simulate_policy_cli_writes_artifact_and_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    artifact = BenchmarkRunArtifact(
        run_id="simulation-source",
        scenario=BenchmarkScenario(
            name="simulation-scenario",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=1,
        ),
        policy=RoutingPolicy.BALANCED,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.LOGICAL_ALIAS,
            model_alias="chat-shared",
        ),
        backends_involved=["mock-a"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=15.0,
            p50_latency_ms=15.0,
            p95_latency_ms=15.0,
            avg_ttft_ms=5.0,
            p50_ttft_ms=5.0,
            p95_ttft_ms=5.0,
            total_output_tokens=4,
            avg_output_tokens=4.0,
            avg_tokens_per_second=8.0,
            p95_tokens_per_second=8.0,
            fallback_count=0,
            chosen_backend_counts={"mock-a": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="synthetic"),
        records=[
            BenchmarkRequestRecord(
                request_id="req-1",
                backend_name="mock-a",
                backend_type="mock",
                model_alias="chat-shared",
                started_at=datetime(2026, 3, 16, tzinfo=UTC),
                completed_at=datetime(2026, 3, 16, tzinfo=UTC),
                latency_ms=15.0,
                output_tokens=4,
                tokens_per_second=8.0,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=5, completion_tokens=4, total_tokens=9),
            )
        ],
    )
    artifact_path = tmp_path / "source.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "simulate-policy",
            str(artifact_path),
            "--policy-id",
            "adaptive-balanced-v1",
            "--markdown-report",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    simulation_path = Path(result.stdout.strip())
    payload = json.loads(simulation_path.read_text(encoding="utf-8"))
    assert payload["policy"]["policy_id"] == "adaptive-balanced-v1"
    assert payload["summary"]["request_count"] == 1
    assert "policy_recommendation" in payload["metadata"]
    report_path = simulation_path.with_suffix(".md")
    assert report_path.exists()
    assert "# Switchyard Simulation Report:" in report_path.read_text(encoding="utf-8")


def test_compare_offline_policies_cli_writes_comparison_artifact(tmp_path: Path) -> None:
    runner = CliRunner()
    trace = CapturedTraceRecord(
        record_id="trace-compare",
        request_id="trace-compare-req",
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.LOGICAL_ALIAS,
            model_alias="chat-shared",
        ),
        logical_alias="chat-shared",
        chosen_backend="mock-a",
        latency_ms=12.0,
        status_code=200,
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "compare-offline-policies",
            "--trace-path",
            str(trace_path),
            "--routing-policy",
            "balanced",
            "--candidate-policy",
            "adaptive-fast:latency",
            "--markdown-report",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    comparison_path = Path(result.stdout.strip())
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["simulation_comparison_id"]
    assert len(payload["evaluations"]) == 2
    report_path = comparison_path.with_suffix(".md")
    assert report_path.exists()
    assert "# Switchyard Simulation Comparison Report:" in report_path.read_text(
        encoding="utf-8"
    )


def test_recommend_policies_cli_writes_report_and_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    comparison = compare_candidate_policies_offline(
        policies=[
            ExplainablePolicySpec(
                policy_id="balanced-offline",
                objective=CounterfactualObjective.BALANCED,
            )
        ],
        evaluation_artifacts=[],
        evaluation_trace_records=[
            CapturedTraceRecord(
                record_id="trace-guidance",
                request_id="trace-guidance-req",
                execution_target=ExecutionTarget(
                    target_type=ExecutionTargetType.LOGICAL_ALIAS,
                    model_alias="chat-shared",
                ),
                logical_alias="chat-shared",
                chosen_backend="mock-a",
                latency_ms=12.0,
                status_code=200,
            )
        ],
        history_artifacts=[],
        history_trace_records=[],
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )
    artifact_path = tmp_path / "comparison.json"
    artifact_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "recommend-policies",
            str(artifact_path),
            "--markdown-report",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    report_path = Path(result.stdout.strip())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["recommendation_report_id"]
    assert "recommendations" in payload
    markdown_path = report_path.with_suffix(".md")
    assert markdown_path.exists()
    assert "# Switchyard Policy Recommendation Report:" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_run_workload_cli_writes_artifact(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    runner = CliRunner()
    manifest = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=2,
        seed=3,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    async def fake_run_workload_manifest_benchmark(
        *,
        scenario: BenchmarkScenario,
        gateway_base_url: str,
        execution_target: ExecutionTarget,
        warmup: BenchmarkWarmupConfig,
        metrics_path: str | None,
        timeout_seconds: float,
        deployment_target: BenchmarkDeploymentTarget | None,
        deployment_profile: DeploymentProfile | None,
        config_profile_name: str | None,
        control_plane_image: BackendImageMetadata | None,
        runtime_inspection_path: str | None,
        settings: Settings,
    ) -> BenchmarkRunArtifact:
        assert scenario.model_alias == "chat-shared"
        assert execution_target.target_type is ExecutionTargetType.PINNED_BACKEND
        assert execution_target.pinned_backend == "mock-local-fast"
        assert deployment_target is BenchmarkDeploymentTarget.COMPOSE
        assert deployment_profile is DeploymentProfile.COMPOSE
        assert config_profile_name == "compose-smoke"
        assert control_plane_image is not None
        assert control_plane_image.image_tag == "switchyard/control-plane:dev"
        assert runtime_inspection_path == "/admin/runtime"
        assert settings is not None
        return BenchmarkRunArtifact(
            run_id="workload-run",
            scenario=BenchmarkScenario.model_validate(manifest.model_dump(mode="python")),
            policy=RoutingPolicy.BALANCED,
            execution_target=execution_target,
            backends_involved=["mock-local-fast"],
            backend_types_involved=["mock"],
            model_aliases_involved=["chat-shared"],
            request_count=1,
            summary=BenchmarkSummary(
                request_count=1,
                success_count=1,
                failure_count=0,
                avg_latency_ms=5.0,
                p50_latency_ms=5.0,
                p95_latency_ms=5.0,
                avg_ttft_ms=None,
                p95_ttft_ms=None,
                total_output_tokens=4,
                avg_output_tokens=4.0,
                avg_tokens_per_second=8.0,
                p95_tokens_per_second=8.0,
                fallback_count=0,
                chosen_backend_counts={"mock-local-fast": 1},
            ),
            environment=BenchmarkEnvironmentMetadata(
                benchmark_mode="workload_manifest",
                gateway_base_url=gateway_base_url,
                timeout_seconds=timeout_seconds,
            ),
            records=[
                BenchmarkRequestRecord(
                    request_id="req-1",
                    workload_item_id=manifest.items[0].item_id,
                    backend_name="mock-local-fast",
                    backend_type="mock",
                    model_alias="chat-shared",
                    started_at=datetime(2026, 3, 16, tzinfo=UTC),
                    completed_at=datetime(2026, 3, 16, tzinfo=UTC),
                    latency_ms=5.0,
                    output_tokens=4,
                    tokens_per_second=8.0,
                    success=True,
                    status_code=200,
                    usage=UsageStats(prompt_tokens=5, completion_tokens=4, total_tokens=9),
                )
            ],
        )

    monkeypatch.setattr(
        cli,
        "run_workload_manifest_benchmark",
        fake_run_workload_manifest_benchmark,
    )

    result = runner.invoke(
        app,
        [
            "run-workload",
            "--manifest-path",
            str(manifest_path),
            "--gateway-base-url",
            "http://testserver",
            "--pinned-backend",
            "mock-local-fast",
            "--deployment-target",
            "compose",
            "--deployment-profile",
            "compose",
            "--config-profile-name",
            "compose-smoke",
            "--control-plane-image-tag",
            "switchyard/control-plane:dev",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["environment"]["benchmark_mode"] == "workload_manifest"
    assert payload["execution_target"]["pinned_backend"] == "mock-local-fast"
    assert payload["records"][0]["workload_item_id"] == manifest.items[0].item_id


def test_replay_traces_cli_writes_artifact(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    runner = CliRunner()
    target = ExecutionTarget(
        target_type=ExecutionTargetType.LOGICAL_ALIAS,
        model_alias="chat-shared",
    )
    trace = CapturedTraceRecord(
        record_id="trace-1",
        request_id="req-1",
        execution_target=target,
        fallback_used=False,
        status_code=200,
        latency_ms=10.0,
        normalized_request_payload={
            "model": "chat-shared",
            "messages": [{"role": "user", "content": "trace prompt"}],
            "stream": False,
        },
    )
    trace_path = tmp_path / "gateway-traces.jsonl"
    trace_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")

    async def fake_run_trace_replay_benchmark(
        *,
        traces: list[CapturedTraceRecord],
        gateway_base_url: str,
        execution_target: ExecutionTarget,
        replay_mode: ReplayMode,
        concurrency: int,
        warmup: BenchmarkWarmupConfig,
        source_run_id: str,
        metrics_path: str | None,
        timeout_seconds: float,
        deployment_target: BenchmarkDeploymentTarget | None,
        deployment_profile: DeploymentProfile | None,
        config_profile_name: str | None,
        control_plane_image: BackendImageMetadata | None,
        runtime_inspection_path: str | None,
        settings: Settings,
    ) -> BenchmarkRunArtifact:
        assert traces[0].record_id == "trace-1"
        assert execution_target.target_type is ExecutionTargetType.ROUTING_POLICY
        assert execution_target.routing_policy is RoutingPolicy.BALANCED
        assert replay_mode is ReplayMode.FIXED_CONCURRENCY
        assert concurrency == 2
        assert source_run_id == "gateway-traces"
        assert deployment_target is BenchmarkDeploymentTarget.KIND
        assert deployment_profile is DeploymentProfile.KIND
        assert config_profile_name == "kind-smoke"
        assert control_plane_image is not None
        assert control_plane_image.image_tag == "switchyard/control-plane:kind-dev"
        assert runtime_inspection_path is None
        assert settings is not None
        return BenchmarkRunArtifact(
            run_id="trace-replay-run",
            scenario=BenchmarkScenario(
                name="trace-replay",
                model="chat-shared",
                policy=RoutingPolicy.BALANCED,
                workload_shape=WorkloadShape.INTERACTIVE,
                request_count=1,
            ),
            policy=RoutingPolicy.BALANCED,
            execution_target=execution_target,
            backends_involved=["mock-replay"],
            backend_types_involved=["mock"],
            model_aliases_involved=["chat-shared"],
            request_count=1,
            summary=BenchmarkSummary(
                request_count=1,
                success_count=1,
                failure_count=0,
                avg_latency_ms=5.0,
                p50_latency_ms=5.0,
                p95_latency_ms=5.0,
                avg_ttft_ms=None,
                p95_ttft_ms=None,
                total_output_tokens=4,
                avg_output_tokens=4.0,
                avg_tokens_per_second=8.0,
                p95_tokens_per_second=8.0,
                fallback_count=0,
                chosen_backend_counts={"mock-replay": 1},
            ),
            environment=BenchmarkEnvironmentMetadata(
                benchmark_mode="trace_replay",
                gateway_base_url=gateway_base_url,
                timeout_seconds=timeout_seconds,
            ),
            records=[
                BenchmarkRequestRecord(
                    request_id="replay_req-1",
                    source_request_id="req-1",
                    source_trace_record_id="trace-1",
                    replay_correlation_id="replay_req-1",
                    backend_name="mock-replay",
                    backend_type="mock",
                    model_alias="chat-shared",
                    started_at=datetime(2026, 3, 16, tzinfo=UTC),
                    completed_at=datetime(2026, 3, 16, tzinfo=UTC),
                    latency_ms=5.0,
                    output_tokens=4,
                    tokens_per_second=8.0,
                    success=True,
                    status_code=200,
                    usage=UsageStats(prompt_tokens=5, completion_tokens=4, total_tokens=9),
                )
            ],
        )

    monkeypatch.setattr(
        cli,
        "run_trace_replay_benchmark",
        fake_run_trace_replay_benchmark,
    )

    result = runner.invoke(
        app,
        [
            "replay-traces",
            "--trace-path",
            str(trace_path),
            "--gateway-base-url",
            "http://testserver",
            "--policy",
            "balanced",
            "--replay-mode",
            "fixed_concurrency",
            "--concurrency",
            "2",
            "--deployment-target",
            "kind",
            "--deployment-profile",
            "kind",
            "--config-profile-name",
            "kind-smoke",
            "--control-plane-image-tag",
            "switchyard/control-plane:kind-dev",
            "--runtime-inspection-path",
            "none",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["environment"]["benchmark_mode"] == "trace_replay"
    assert payload["execution_target"]["routing_policy"] == "balanced"
    assert payload["records"][0]["source_trace_record_id"] == "trace-1"


def test_replay_traces_cli_rejects_metadata_only_trace(tmp_path: Path) -> None:
    runner = CliRunner()
    target = ExecutionTarget(
        target_type=ExecutionTargetType.LOGICAL_ALIAS,
        model_alias="chat-shared",
    )
    trace = CapturedTraceRecord(
        record_id="trace-1",
        request_id="req-1",
        execution_target=target,
        capture_mode=TraceCaptureMode.METADATA_ONLY,
        fallback_used=False,
        status_code=200,
        latency_ms=10.0,
        normalized_request_payload={
            "model": "chat-shared",
            "message_count": 1,
            "roles": ["user"],
            "stream": False,
        },
    )
    trace_path = tmp_path / "gateway-traces.jsonl"
    trace_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "replay-traces",
            "--trace-path",
            str(trace_path),
            "--gateway-base-url",
            "http://testserver",
        ],
    )

    assert result.exit_code != 0
    assert "metadata_only is not replayable" in (result.stdout + result.stderr)


def test_compare_workload_cli_writes_comparison_artifact(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    manifest = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=2,
        seed=5,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    async def fake_compare_workload_execution_targets(
        *,
        scenario: BenchmarkScenario,
        gateway_base_url: str,
        left_target: ExecutionTarget,
        right_target: ExecutionTarget,
        warmup: BenchmarkWarmupConfig | None,
        metrics_path: str | None,
        timeout_seconds: float,
    ) -> BenchmarkTargetComparisonArtifact:
        assert scenario.model_alias == "chat-shared"
        assert left_target.target_type is ExecutionTargetType.ROUTING_POLICY
        assert left_target.routing_policy is RoutingPolicy.BALANCED
        assert right_target.target_type is ExecutionTargetType.PINNED_BACKEND
        assert right_target.pinned_backend == "mock-b"
        return BenchmarkTargetComparisonArtifact(
            comparison_id="workload-compare",
            source_kind=ComparisonSourceKind.WORKLOAD_MANIFEST,
            source_name="manifest",
            request_count=2,
            left=BenchmarkComparisonSideSummary(
                run_id="left-run",
                execution_target=left_target,
                request_count=2,
                success_rate=1.0,
                error_rate=0.0,
                fallback_rate=0.0,
                p50_latency_ms=5.0,
                p95_latency_ms=6.0,
                p50_ttft_ms=2.0,
                p95_ttft_ms=3.0,
                avg_tokens_per_second=10.0,
                p95_tokens_per_second=9.0,
                route_distribution={"mock-a": 2},
                backend_distribution={"mock-a": 2},
            ),
            right=BenchmarkComparisonSideSummary(
                run_id="right-run",
                execution_target=right_target,
                request_count=2,
                success_rate=1.0,
                error_rate=0.0,
                fallback_rate=0.0,
                p50_latency_ms=7.0,
                p95_latency_ms=8.0,
                p50_ttft_ms=2.0,
                p95_ttft_ms=3.0,
                avg_tokens_per_second=9.0,
                p95_tokens_per_second=8.0,
                route_distribution={"mock-b": 2},
                backend_distribution={"mock-b": 2},
            ),
            delta=BenchmarkComparisonDeltaSummary(
                request_count_delta=0,
                success_rate_delta=0.0,
                error_rate_delta=0.0,
                fallback_rate_delta=0.0,
                p50_latency_delta_ms=2.0,
                p95_latency_delta_ms=2.0,
                p50_ttft_delta_ms=0.0,
                p95_ttft_delta_ms=0.0,
                avg_tokens_per_second_delta=-1.0,
                p95_tokens_per_second_delta=-1.0,
                route_distribution_delta={"mock-a": -2, "mock-b": 2},
                backend_distribution_delta={"mock-a": -2, "mock-b": 2},
            ),
        )

    monkeypatch.setattr(
        cli,
        "compare_workload_execution_targets",
        fake_compare_workload_execution_targets,
    )

    result = runner.invoke(
        app,
        [
            "compare-workload",
            "--manifest-path",
            str(manifest_path),
            "--gateway-base-url",
            "http://testserver",
            "--left-policy",
            "balanced",
            "--right-pinned-backend",
            "mock-b",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(Path(result.stdout.strip()).read_text(encoding="utf-8"))
    assert payload["source_kind"] == "workload_manifest"
    assert payload["right"]["execution_target"]["pinned_backend"] == "mock-b"


def test_compare_synthetic_cli_writes_comparison_artifact(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "compare-synthetic",
            "--request-count",
            "2",
            "--output-dir",
            str(tmp_path),
            "--policy",
            "balanced",
            "--pinned-backend",
            "mock-local-fast",
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["request_count"] == 2
    assert payload["scenario_name"] == "synthetic_policy_comparison"
    assert payload["best_policy_by_latency"] == "balanced"
    assert {result["comparison_label"] for result in payload["results"]} == {
        "balanced",
        "balanced:pinned:mock-local-fast",
    }


def test_compare_synthetic_cli_writes_markdown_report_when_requested(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "compare-synthetic",
            "--request-count",
            "2",
            "--output-dir",
            str(tmp_path),
            "--policy",
            "balanced",
            "--workload-pattern",
            "bursty",
            "--burst-size",
            "2",
            "--markdown-report",
        ],
    )

    assert result.exit_code == 0
    artifact_path = Path(result.stdout.strip())
    report_path = artifact_path.with_suffix(".md")
    assert report_path.exists()
    assert "## Results" in report_path.read_text(encoding="utf-8")


def test_compare_synthetic_command_uses_configured_registry_when_local_models_exist(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                serving_target="chat-shared",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        )
    )
    registry = AdapterRegistry()
    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "Settings", lambda: settings)

    def fake_build_registry_from_settings(_settings: Settings) -> AdapterRegistry:
        return registry

    monkeypatch.setattr(cli, "build_registry_from_settings", fake_build_registry_from_settings)

    async def fake_compare_synthetic_policies(
        *,
        request_count: int,
        model: str,
        workload_shape: WorkloadShape,
        policies: list[RoutingPolicy] | None,
        pinned_backends: list[str] | None,
        workload_generation: WorkloadGenerationConfig,
        registry: AdapterRegistry | None,
    ) -> BenchmarkComparisonArtifact:
        seen["request_count"] = request_count
        seen["model"] = model
        seen["workload_shape"] = workload_shape
        seen["policies"] = policies
        seen["pinned_backends"] = pinned_backends
        seen["workload_generation"] = workload_generation
        seen["registry"] = registry
        return BenchmarkComparisonArtifact(
            run_id="comparison",
            timestamp=datetime(2026, 3, 16, tzinfo=UTC),
            scenario_name="synthetic_policy_comparison",
            model=model,
            workload_shape=workload_shape,
            request_count=request_count,
            results=[
                BenchmarkPolicyComparison(
                    comparison_label="balanced",
                    policy=RoutingPolicy.BALANCED,
                    run_id="run-balanced",
                    backends_involved=["mlx-lm:mlx-chat"],
                    summary=BenchmarkSummary(
                        request_count=request_count,
                        success_count=request_count,
                        failure_count=0,
                        avg_latency_ms=5.0,
                        p50_latency_ms=5.0,
                        p95_latency_ms=5.0,
                        avg_ttft_ms=None,
                        p95_ttft_ms=None,
                        total_output_tokens=4 * request_count,
                        avg_output_tokens=4.0,
                        avg_tokens_per_second=10.0,
                        p95_tokens_per_second=10.0,
                        fallback_count=0,
                        chosen_backend_counts={"mlx-lm:mlx-chat": request_count},
                    ),
                )
            ],
            best_policy_by_latency=RoutingPolicy.BALANCED,
            best_result_by_latency="balanced",
        )

    monkeypatch.setattr(cli, "compare_synthetic_policies", fake_compare_synthetic_policies)

    output_path = asyncio.run(
        cli._compare_synthetic_command(
            request_count=2,
            output_dir=tmp_path,
            model="chat-shared",
            workload_shape=WorkloadShape.INTERACTIVE,
            policies=[RoutingPolicy.BALANCED],
            pinned_backends=["mlx-lm:mlx-chat"],
            workload_generation=WorkloadGenerationConfig(),
            markdown_report=False,
        )
    )

    assert output_path == tmp_path / "comparison.json"
    assert seen["registry"] is registry
    assert seen["model"] == "chat-shared"
    assert seen["policies"] == [RoutingPolicy.BALANCED]
    assert seen["pinned_backends"] == ["mlx-lm:mlx-chat"]
    assert seen["workload_generation"] == WorkloadGenerationConfig()
