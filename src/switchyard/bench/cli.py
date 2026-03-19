"""Benchmark CLI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import typer

from switchyard.adapters.factory import build_registry_from_settings
from switchyard.adapters.registry import AdapterRegistry
from switchyard.bench.artifacts import (
    BenchmarkRunResult,
    build_gateway_scenario,
    build_replay_plan,
    build_synthetic_scenario,
    compare_synthetic_policies,
    compare_trace_execution_targets,
    compare_workload_execution_targets,
    default_comparison_output_path,
    default_generated_report_path,
    default_markdown_report_path,
    default_output_path,
    load_benchmark_artifact_model,
    load_captured_traces,
    render_artifact_bundle_markdown,
    render_comparison_report_markdown,
    render_loaded_artifact_markdown,
    render_run_report_markdown,
    render_target_comparison_report_markdown,
    run_gateway_benchmark,
    run_synthetic_benchmark,
    run_trace_replay_benchmark,
    run_workload_manifest_benchmark,
    validate_replayable_traces,
    write_artifact,
    write_json_model,
    write_markdown_report,
)
from switchyard.bench.recommendations import build_policy_recommendation_report
from switchyard.bench.simulation import (
    compare_candidate_policies_offline,
    compatibility_policy_spec,
    recommend_policy_from_simulation,
    simulate_policy_counterfactual,
)
from switchyard.bench.workloads import build_workload_manifest, default_workload_manifest_path
from switchyard.config import Settings
from switchyard.schemas.backend import BackendImageMetadata, DeploymentProfile
from switchyard.schemas.benchmark import (
    AdaptivePolicyGuardrails,
    AdaptivePolicyMode,
    BenchmarkDeploymentTarget,
    BenchmarkRunArtifact,
    BenchmarkWarmupConfig,
    CapturedTraceRecord,
    CounterfactualObjective,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    ExecutionTarget,
    ExecutionTargetType,
    ExplainablePolicySpec,
    ReplayMode,
    WorkloadGenerationConfig,
    WorkloadPattern,
    WorkloadScenario,
    WorkloadScenarioFamily,
)
from switchyard.schemas.routing import RoutingPolicy, WorkloadShape

app = typer.Typer(help="Switchyard benchmark utilities.")


@app.command("generate-workload")
def generate_workload(
    family: WorkloadScenarioFamily = typer.Option(
        WorkloadScenarioFamily.SHORT_CHAT,
        case_sensitive=False,
        help="Built-in workload scenario family.",
    ),
    model_alias: str = typer.Option(
        "mock-chat",
        help="Logical model alias to attach to generated workload items.",
    ),
    request_count: int = typer.Option(4, min=1, help="Number of workload items to generate."),
    seed: int = typer.Option(0, min=0, help="Deterministic generation seed."),
    workload_shape: WorkloadShape = typer.Option(
        WorkloadShape.INTERACTIVE,
        case_sensitive=False,
        help="Workload shape metadata for the manifest.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output manifest. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
) -> None:
    """Generate a deterministic workload manifest and write it to disk."""

    output_path = _generate_workload_command(
        family=family,
        model_alias=model_alias,
        request_count=request_count,
        seed=seed,
        workload_shape=workload_shape,
        output_dir=output_dir,
    )
    typer.echo(output_path)


@app.command("run-synthetic")
def run_synthetic(
    request_count: int = typer.Option(3, min=1, help="Number of synthetic requests to issue."),
    policy: RoutingPolicy = typer.Option(
        RoutingPolicy.BALANCED,
        case_sensitive=False,
        help="Routing policy to exercise.",
    ),
    workload_pattern: WorkloadPattern = typer.Option(
        WorkloadPattern.UNIFORM,
        case_sensitive=False,
        help="Synthetic workload prompt pattern.",
    ),
    workload_seed: int = typer.Option(
        0,
        min=0,
        help="Deterministic seed for synthetic workload generation.",
    ),
    shared_prefix: str | None = typer.Option(
        None,
        help="Shared prompt prefix for repeated-prefix workloads.",
    ),
    burst_size: int = typer.Option(
        1,
        min=1,
        help="Burst size for bursty synthetic workloads.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
) -> None:
    """Run a small synthetic benchmark and write a JSON artifact."""

    result = asyncio.run(
        _run_synthetic_command(
            request_count=request_count,
            policy=policy,
            workload_generation=WorkloadGenerationConfig(
                pattern=workload_pattern,
                seed=workload_seed,
                shared_prefix=shared_prefix,
                burst_size=burst_size,
            ),
            output_dir=output_dir,
            markdown_report=markdown_report,
        )
    )
    typer.echo(result.output_path)


@app.command("run-gateway")
def run_gateway(
    model: str | None = typer.Option(
        None,
        help=(
            "Model alias to send to the gateway. Defaults to "
            "SWITCHYARD_DEFAULT_MODEL_ALIAS or the first local model."
        ),
    ),
    gateway_base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        help="Base URL for the deployed Switchyard gateway.",
    ),
    request_count: int = typer.Option(
        3,
        min=1,
        max=8,
        help="Number of requests to issue. Keep this small for laptop-friendly runs.",
    ),
    policy: RoutingPolicy = typer.Option(
        RoutingPolicy.BALANCED,
        case_sensitive=False,
        help="Routing policy to request from the gateway.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    metrics_path: str | None = typer.Option(
        "/metrics",
        help="Prometheus-style metrics path. Use 'none' to skip metrics scraping.",
    ),
    timeout_seconds: float = typer.Option(
        60.0,
        min=1.0,
        max=3600.0,
        help="Per-request timeout for the benchmark client.",
    ),
    deployment_target: BenchmarkDeploymentTarget | None = typer.Option(
        None,
        case_sensitive=False,
        help="Deployment shape exercised by this run, such as local_dev, compose, or kind.",
    ),
    deployment_profile: DeploymentProfile | None = typer.Option(
        None,
        case_sensitive=False,
        help="Typed deployment profile for the exercised control-plane topology.",
    ),
    config_profile_name: str | None = typer.Option(
        None,
        help="Relevant config/profile name recorded into the artifact.",
    ),
    control_plane_image_tag: str | None = typer.Option(
        None,
        help="Optional control-plane image tag or release label.",
    ),
    control_plane_git_sha: str | None = typer.Option(
        None,
        help="Optional control-plane git revision recorded into the artifact.",
    ),
    runtime_inspection_path: str | None = typer.Option(
        "/admin/runtime",
        help="Path used to snapshot deployed runtime topology. Use 'none' to skip it.",
    ),
) -> None:
    """Run a lightweight benchmark against the local gateway and write a JSON artifact."""

    resolved_metrics_path = None if metrics_path == "none" else metrics_path
    resolved_runtime_inspection_path = (
        None if runtime_inspection_path == "none" else runtime_inspection_path
    )
    result = asyncio.run(
        _run_gateway_command(
            model=model,
            gateway_base_url=gateway_base_url,
            request_count=request_count,
            policy=policy,
            output_dir=output_dir,
            metrics_path=resolved_metrics_path,
            timeout_seconds=timeout_seconds,
            deployment_target=deployment_target,
            deployment_profile=deployment_profile,
            config_profile_name=config_profile_name,
            control_plane_image_tag=control_plane_image_tag,
            control_plane_git_sha=control_plane_git_sha,
            runtime_inspection_path=resolved_runtime_inspection_path,
        )
    )
    typer.echo(result.output_path)


@app.command("run-workload")
def run_workload(
    manifest_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    gateway_base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        help="Base URL for the deployed Switchyard gateway.",
    ),
    model_alias: str | None = typer.Option(
        None,
        help="Optional logical model alias override. Defaults to the manifest alias.",
    ),
    policy: RoutingPolicy | None = typer.Option(
        None,
        case_sensitive=False,
        help="Optional routing policy override. Defaults to the manifest policy.",
    ),
    pinned_backend: str | None = typer.Option(
        None,
        help="Optional explicit backend pin for the workload run.",
    ),
    warmup_request_count: int = typer.Option(
        0,
        min=0,
        max=8,
        help="Number of warmup requests to send before measured workload requests.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    metrics_path: str | None = typer.Option(
        "/metrics",
        help="Prometheus-style metrics path. Use 'none' to skip metrics scraping.",
    ),
    timeout_seconds: float = typer.Option(
        60.0,
        min=1.0,
        max=3600.0,
        help="Per-request timeout for the benchmark client.",
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
    deployment_target: BenchmarkDeploymentTarget | None = typer.Option(
        None,
        case_sensitive=False,
        help="Deployment shape exercised by this run, such as local_dev, compose, or kind.",
    ),
    deployment_profile: DeploymentProfile | None = typer.Option(
        None,
        case_sensitive=False,
        help="Typed deployment profile for the exercised control-plane topology.",
    ),
    config_profile_name: str | None = typer.Option(
        None,
        help="Relevant config/profile name recorded into the artifact.",
    ),
    control_plane_image_tag: str | None = typer.Option(
        None,
        help="Optional control-plane image tag or release label.",
    ),
    control_plane_git_sha: str | None = typer.Option(
        None,
        help="Optional control-plane git revision recorded into the artifact.",
    ),
    runtime_inspection_path: str | None = typer.Option(
        "/admin/runtime",
        help="Path used to snapshot deployed runtime topology. Use 'none' to skip it.",
    ),
) -> None:
    """Run a generated workload manifest against the existing gateway path."""

    resolved_metrics_path = None if metrics_path == "none" else metrics_path
    resolved_runtime_inspection_path = (
        None if runtime_inspection_path == "none" else runtime_inspection_path
    )
    result = asyncio.run(
        _run_workload_command(
            manifest_path=manifest_path,
            gateway_base_url=gateway_base_url,
            model_alias=model_alias,
            policy=policy,
            pinned_backend=pinned_backend,
            warmup_request_count=warmup_request_count,
            output_dir=output_dir,
            metrics_path=resolved_metrics_path,
            timeout_seconds=timeout_seconds,
            markdown_report=markdown_report,
            deployment_target=deployment_target,
            deployment_profile=deployment_profile,
            config_profile_name=config_profile_name,
            control_plane_image_tag=control_plane_image_tag,
            control_plane_git_sha=control_plane_git_sha,
            runtime_inspection_path=resolved_runtime_inspection_path,
        )
    )
    typer.echo(result.output_path)


@app.command("replay-traces")
def replay_traces(
    trace_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    gateway_base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        help="Base URL for the deployed Switchyard gateway.",
    ),
    model_alias: str | None = typer.Option(
        None,
        help="Optional logical model alias override. Defaults to the trace alias.",
    ),
    policy: RoutingPolicy | None = typer.Option(
        None,
        case_sensitive=False,
        help="Optional routing policy override for replay.",
    ),
    pinned_backend: str | None = typer.Option(
        None,
        help="Optional explicit backend pin for replay.",
    ),
    replay_mode: ReplayMode = typer.Option(
        ReplayMode.SEQUENTIAL,
        case_sensitive=False,
        help="Replay dispatch mode.",
    ),
    concurrency: int = typer.Option(
        1,
        min=1,
        max=32,
        help="Concurrent in-flight replay requests for fixed_concurrency mode.",
    ),
    warmup_request_count: int = typer.Option(
        0,
        min=0,
        max=8,
        help="Number of warmup requests to send before measured replay requests.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    metrics_path: str | None = typer.Option(
        "/metrics",
        help="Prometheus-style metrics path. Use 'none' to skip metrics scraping.",
    ),
    timeout_seconds: float = typer.Option(
        60.0,
        min=1.0,
        max=3600.0,
        help="Per-request timeout for the replay client.",
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
    deployment_target: BenchmarkDeploymentTarget | None = typer.Option(
        None,
        case_sensitive=False,
        help="Deployment shape exercised by this run, such as local_dev, compose, or kind.",
    ),
    deployment_profile: DeploymentProfile | None = typer.Option(
        None,
        case_sensitive=False,
        help="Typed deployment profile for the exercised control-plane topology.",
    ),
    config_profile_name: str | None = typer.Option(
        None,
        help="Relevant config/profile name recorded into the artifact.",
    ),
    control_plane_image_tag: str | None = typer.Option(
        None,
        help="Optional control-plane image tag or release label.",
    ),
    control_plane_git_sha: str | None = typer.Option(
        None,
        help="Optional control-plane git revision recorded into the artifact.",
    ),
    runtime_inspection_path: str | None = typer.Option(
        "/admin/runtime",
        help="Path used to snapshot deployed runtime topology. Use 'none' to skip it.",
    ),
) -> None:
    """Replay captured traces through the existing gateway path."""

    resolved_metrics_path = None if metrics_path == "none" else metrics_path
    resolved_runtime_inspection_path = (
        None if runtime_inspection_path == "none" else runtime_inspection_path
    )
    result = asyncio.run(
        _replay_traces_command(
            trace_path=trace_path,
            gateway_base_url=gateway_base_url,
            model_alias=model_alias,
            policy=policy,
            pinned_backend=pinned_backend,
            replay_mode=replay_mode,
            concurrency=concurrency,
            warmup_request_count=warmup_request_count,
            output_dir=output_dir,
            metrics_path=resolved_metrics_path,
            timeout_seconds=timeout_seconds,
            markdown_report=markdown_report,
            deployment_target=deployment_target,
            deployment_profile=deployment_profile,
            config_profile_name=config_profile_name,
            control_plane_image_tag=control_plane_image_tag,
            control_plane_git_sha=control_plane_git_sha,
            runtime_inspection_path=resolved_runtime_inspection_path,
        )
    )
    typer.echo(result.output_path)


@app.command("compare-synthetic")
def compare_synthetic(
    request_count: int = typer.Option(3, min=1, help="Number of requests per policy."),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    model: str = typer.Option("mock-chat", help="Model alias to benchmark."),
    workload_shape: WorkloadShape = typer.Option(
        WorkloadShape.INTERACTIVE,
        case_sensitive=False,
        help="Workload shape to attach to each synthetic request.",
    ),
    policy: list[RoutingPolicy] | None = typer.Option(
        None,
        "--policy",
        case_sensitive=False,
        help="Optional policy filter. Repeat to compare a subset instead of all policies.",
    ),
    pinned_backend: list[str] | None = typer.Option(
        None,
        "--pinned-backend",
        help=(
            "Optional internal backend pin baseline. Repeat to compare the alias against one or "
            "more explicit backend deployments."
        ),
    ),
    workload_pattern: WorkloadPattern = typer.Option(
        WorkloadPattern.UNIFORM,
        case_sensitive=False,
        help="Synthetic workload prompt pattern.",
    ),
    workload_seed: int = typer.Option(
        0,
        min=0,
        help="Deterministic seed for synthetic workload generation.",
    ),
    shared_prefix: str | None = typer.Option(
        None,
        help="Shared prompt prefix for repeated-prefix workloads.",
    ),
    burst_size: int = typer.Option(
        1,
        min=1,
        help="Burst size for bursty synthetic workloads.",
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
) -> None:
    """Compare all routing policies with the synthetic benchmark runner."""

    result = asyncio.run(
        _compare_synthetic_command(
            request_count=request_count,
            output_dir=output_dir,
            model=model,
            workload_shape=workload_shape,
            policies=policy,
            pinned_backends=pinned_backend,
            workload_generation=WorkloadGenerationConfig(
                pattern=workload_pattern,
                seed=workload_seed,
                shared_prefix=shared_prefix,
                burst_size=burst_size,
            ),
            markdown_report=markdown_report,
        )
    )
    typer.echo(result)


@app.command("compare-workload")
def compare_workload(
    manifest_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    gateway_base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        help="Base URL for the local Switchyard gateway.",
    ),
    left_model_alias: str | None = typer.Option(
        None,
        help="Optional left-side model alias override.",
    ),
    left_policy: RoutingPolicy | None = typer.Option(None, case_sensitive=False),
    left_pinned_backend: str | None = typer.Option(None),
    right_model_alias: str | None = typer.Option(
        None,
        help="Optional right-side model alias override.",
    ),
    right_policy: RoutingPolicy | None = typer.Option(None, case_sensitive=False),
    right_pinned_backend: str | None = typer.Option(None),
    warmup_request_count: int = typer.Option(0, min=0, max=8),
    output_dir: Path | None = typer.Option(None),
    metrics_path: str | None = typer.Option("/metrics"),
    timeout_seconds: float = typer.Option(60.0, min=1.0, max=3600.0),
    markdown_report: bool = typer.Option(False),
) -> None:
    """Compare the same workload manifest across two execution targets."""

    resolved_metrics_path = None if metrics_path == "none" else metrics_path
    output_path = asyncio.run(
        _compare_workload_command(
            manifest_path=manifest_path,
            gateway_base_url=gateway_base_url,
            left_model_alias=left_model_alias,
            left_policy=left_policy,
            left_pinned_backend=left_pinned_backend,
            right_model_alias=right_model_alias,
            right_policy=right_policy,
            right_pinned_backend=right_pinned_backend,
            warmup_request_count=warmup_request_count,
            output_dir=output_dir,
            metrics_path=resolved_metrics_path,
            timeout_seconds=timeout_seconds,
            markdown_report=markdown_report,
        )
    )
    typer.echo(output_path)


@app.command("compare-traces")
def compare_traces(
    trace_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    gateway_base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        help="Base URL for the local Switchyard gateway.",
    ),
    left_model_alias: str | None = typer.Option(None),
    left_policy: RoutingPolicy | None = typer.Option(None, case_sensitive=False),
    left_pinned_backend: str | None = typer.Option(None),
    right_model_alias: str | None = typer.Option(None),
    right_policy: RoutingPolicy | None = typer.Option(None, case_sensitive=False),
    right_pinned_backend: str | None = typer.Option(None),
    replay_mode: ReplayMode = typer.Option(ReplayMode.SEQUENTIAL, case_sensitive=False),
    concurrency: int = typer.Option(1, min=1, max=32),
    warmup_request_count: int = typer.Option(0, min=0, max=8),
    output_dir: Path | None = typer.Option(None),
    metrics_path: str | None = typer.Option("/metrics"),
    timeout_seconds: float = typer.Option(60.0, min=1.0, max=3600.0),
    markdown_report: bool = typer.Option(False),
) -> None:
    """Compare the same trace set across two execution targets."""

    resolved_metrics_path = None if metrics_path == "none" else metrics_path
    output_path = asyncio.run(
        _compare_traces_command(
            trace_path=trace_path,
            gateway_base_url=gateway_base_url,
            left_model_alias=left_model_alias,
            left_policy=left_policy,
            left_pinned_backend=left_pinned_backend,
            right_model_alias=right_model_alias,
            right_policy=right_policy,
            right_pinned_backend=right_pinned_backend,
            replay_mode=replay_mode,
            concurrency=concurrency,
            warmup_request_count=warmup_request_count,
            output_dir=output_dir,
            metrics_path=resolved_metrics_path,
            timeout_seconds=timeout_seconds,
            markdown_report=markdown_report,
        )
    )
    typer.echo(output_path)


@app.command("simulate-policy")
def simulate_policy(
    artifact_path: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="One or more benchmark run artifacts to evaluate.",
    ),
    history_artifact_path: list[Path] | None = typer.Option(
        None,
        "--history-artifact-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional historical benchmark artifacts used as the estimator corpus.",
    ),
    policy_id: str = typer.Option(
        "adaptive-balanced-v1",
        help="Stable identifier for the simulated explainable policy.",
    ),
    objective: CounterfactualObjective = typer.Option(
        CounterfactualObjective.BALANCED,
        case_sensitive=False,
        help="Counterfactual scoring objective.",
    ),
    mode: AdaptivePolicyMode = typer.Option(
        AdaptivePolicyMode.RECOMMEND,
        case_sensitive=False,
        help="Safe rollout posture represented by the simulation output.",
    ),
    min_evidence_count: int = typer.Option(
        3,
        min=1,
        help="Minimum historical evidence count required by the policy.",
    ),
    require_sufficient_data: bool = typer.Option(
        True,
        help="Reject candidates without enough historical evidence.",
    ),
    max_predicted_error_rate: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Optional guardrail for predicted error rate.",
    ),
    max_predicted_latency_regression_ms: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional guardrail for latency regression versus the observed backend.",
    ),
    require_observed_backend_evidence: bool = typer.Option(
        False,
        help="Require sufficient evidence for the observed backend before recommending change.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
) -> None:
    """Run offline counterfactual policy simulation against benchmark artifacts."""

    output_path = _simulate_policy_command(
        artifact_paths=artifact_path,
        history_artifact_paths=history_artifact_path,
        policy_id=policy_id,
        objective=objective,
        mode=mode,
        min_evidence_count=min_evidence_count,
        require_sufficient_data=require_sufficient_data,
        max_predicted_error_rate=max_predicted_error_rate,
        max_predicted_latency_regression_ms=max_predicted_latency_regression_ms,
        require_observed_backend_evidence=require_observed_backend_evidence,
        output_dir=output_dir,
        markdown_report=markdown_report,
    )
    typer.echo(output_path)


@app.command("compare-offline-policies")
def compare_offline_policies(
    artifact_path: list[Path] = typer.Argument(
        [],
        exists=True,
        dir_okay=False,
        readable=True,
        help="Benchmark/replay run artifacts used as evaluation inputs.",
    ),
    trace_path: list[Path] | None = typer.Option(
        None,
        "--trace-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Captured trace jsonl files used as evaluation inputs.",
    ),
    history_artifact_path: list[Path] | None = typer.Option(
        None,
        "--history-artifact-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional benchmark/replay artifacts used as historical evidence only.",
    ),
    history_trace_path: list[Path] | None = typer.Option(
        None,
        "--history-trace-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional captured trace files used as historical evidence only.",
    ),
    routing_policy: list[RoutingPolicy] | None = typer.Option(
        None,
        "--routing-policy",
        case_sensitive=False,
        help="Fixed compatibility policy to compare offline. Repeat for several policies.",
    ),
    candidate_policy: list[str] | None = typer.Option(
        None,
        "--candidate-policy",
        help="Custom policy in the form policy_id:objective. Repeat to compare several.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
) -> None:
    """Compare several candidate policies offline against authoritative artifacts."""

    output_path = _compare_offline_policies_command(
        artifact_paths=artifact_path,
        trace_paths=trace_path or [],
        history_artifact_paths=history_artifact_path or [],
        history_trace_paths=history_trace_path or [],
        routing_policies=routing_policy or [],
        candidate_policies=candidate_policy or [],
        output_dir=output_dir,
        markdown_report=markdown_report,
    )
    typer.echo(output_path)


@app.command("generate-report")
def generate_report(
    artifact_path: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="One or more benchmark artifacts to render as markdown.",
    ),
    output_path: Path | None = typer.Option(
        None,
        dir_okay=False,
        help="Optional markdown output path. Defaults next to the input artifact(s).",
    ),
) -> None:
    """Render one or more benchmark artifacts into a compact markdown report."""

    rendered_path = _generate_report_command(
        artifact_paths=artifact_path,
        output_path=output_path,
    )
    typer.echo(rendered_path)


@app.command("recommend-policies")
def recommend_policies(
    artifact_path: list[Path] = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Benchmark and simulation artifacts used as authoritative recommendation evidence.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help=(
            "Directory for the recommendation artifact. "
            "Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR."
        ),
    ),
    markdown_report: bool = typer.Option(
        False,
        help="Also write a compact Markdown report next to the JSON artifact.",
    ),
) -> None:
    """Generate human-readable routing guidance from authoritative artifacts."""

    output_path = _recommend_policies_command(
        artifact_paths=artifact_path,
        output_dir=output_dir,
        markdown_report=markdown_report,
    )
    typer.echo(output_path)


async def _run_synthetic_command(
    *,
    request_count: int,
    policy: RoutingPolicy,
    workload_generation: WorkloadGenerationConfig,
    output_dir: Path | None = None,
    markdown_report: bool = False,
) -> BenchmarkRunResult:
    """Run the synthetic benchmark command and return the artifact path."""

    settings = Settings()
    scenario = build_synthetic_scenario(
        request_count=request_count,
        policy=policy,
        workload_generation=workload_generation,
    )
    artifact = await run_synthetic_benchmark(
        scenario=scenario,
        registry=_resolve_benchmark_registry(settings),
        settings=settings,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    artifact_path = write_artifact(artifact, default_output_path(resolved_output_dir, artifact))
    if markdown_report:
        write_markdown_report(
            render_run_report_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.run_id),
        )
    return BenchmarkRunResult(artifact=artifact, output_path=artifact_path)


async def _run_gateway_command(
    *,
    model: str | None,
    gateway_base_url: str,
    request_count: int,
    policy: RoutingPolicy,
    output_dir: Path | None,
    metrics_path: str | None,
    timeout_seconds: float,
    deployment_target: BenchmarkDeploymentTarget | None,
    deployment_profile: DeploymentProfile | None,
    config_profile_name: str | None,
    control_plane_image_tag: str | None,
    control_plane_git_sha: str | None,
    runtime_inspection_path: str | None,
) -> BenchmarkRunResult:
    """Run the gateway benchmark command and return the artifact path."""

    settings = Settings()
    resolved_model = model or _resolve_default_model(settings)
    scenario = build_gateway_scenario(
        model=resolved_model,
        request_count=request_count,
        policy=policy,
    )
    artifact = await run_gateway_benchmark(
        scenario=scenario,
        gateway_base_url=gateway_base_url,
        settings=settings,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        control_plane_image=_control_plane_image_metadata(
            image_tag=control_plane_image_tag,
            git_sha=control_plane_git_sha,
        ),
        runtime_inspection_path=runtime_inspection_path,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    artifact_path = write_artifact(artifact, default_output_path(resolved_output_dir, artifact))
    return BenchmarkRunResult(artifact=artifact, output_path=artifact_path)


async def _compare_synthetic_command(
    *,
    request_count: int,
    output_dir: Path | None,
    model: str,
    workload_shape: WorkloadShape,
    policies: list[RoutingPolicy] | None,
    pinned_backends: list[str] | None,
    workload_generation: WorkloadGenerationConfig,
    markdown_report: bool,
) -> Path:
    """Run the synthetic policy comparison command and return the artifact path."""

    settings = Settings()
    artifact = await compare_synthetic_policies(
        request_count=request_count,
        model=model,
        workload_shape=workload_shape,
        policies=policies,
        pinned_backends=pinned_backends,
        workload_generation=workload_generation,
        registry=_resolve_benchmark_registry(settings),
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = resolved_output_dir / f"{artifact.run_id}.json"
    written_path = write_json_model(artifact, output_path)
    if markdown_report:
        write_markdown_report(
            render_comparison_report_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.run_id),
        )
    return written_path


async def _run_workload_command(
    *,
    manifest_path: Path,
    gateway_base_url: str,
    model_alias: str | None,
    policy: RoutingPolicy | None,
    pinned_backend: str | None,
    warmup_request_count: int,
    output_dir: Path | None,
    metrics_path: str | None,
    timeout_seconds: float,
    markdown_report: bool,
    deployment_target: BenchmarkDeploymentTarget | None,
    deployment_profile: DeploymentProfile | None,
    config_profile_name: str | None,
    control_plane_image_tag: str | None,
    control_plane_git_sha: str | None,
    runtime_inspection_path: str | None,
) -> BenchmarkRunResult:
    """Run a workload manifest benchmark command and return the artifact path."""

    settings = Settings()
    scenario = WorkloadScenario.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    resolved_model_alias = model_alias or scenario.model_alias or scenario.model
    if pinned_backend is not None:
        execution_target = ExecutionTarget(
            target_type=ExecutionTargetType.PINNED_BACKEND,
            model_alias=resolved_model_alias,
            pinned_backend=pinned_backend,
        )
    elif policy is None:
        execution_target = ExecutionTarget(
            target_type=ExecutionTargetType.LOGICAL_ALIAS,
            model_alias=resolved_model_alias,
        )
    else:
        execution_target = ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias=resolved_model_alias,
            routing_policy=policy,
        )
    warmup = BenchmarkWarmupConfig(
        enabled=warmup_request_count > 0,
        request_count=warmup_request_count,
    )
    artifact = await run_workload_manifest_benchmark(
        scenario=scenario,
        gateway_base_url=gateway_base_url,
        execution_target=execution_target,
        warmup=warmup,
        settings=settings,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        control_plane_image=_control_plane_image_metadata(
            image_tag=control_plane_image_tag,
            git_sha=control_plane_git_sha,
        ),
        runtime_inspection_path=runtime_inspection_path,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    artifact_path = write_artifact(artifact, default_output_path(resolved_output_dir, artifact))
    if markdown_report:
        write_markdown_report(
            render_run_report_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.run_id),
    )
    return BenchmarkRunResult(artifact=artifact, output_path=artifact_path)


async def _replay_traces_command(
    *,
    trace_path: Path,
    gateway_base_url: str,
    model_alias: str | None,
    policy: RoutingPolicy | None,
    pinned_backend: str | None,
    replay_mode: ReplayMode,
    concurrency: int,
    warmup_request_count: int,
    output_dir: Path | None,
    metrics_path: str | None,
    timeout_seconds: float,
    markdown_report: bool,
    deployment_target: BenchmarkDeploymentTarget | None,
    deployment_profile: DeploymentProfile | None,
    config_profile_name: str | None,
    control_plane_image_tag: str | None,
    control_plane_git_sha: str | None,
    runtime_inspection_path: str | None,
) -> BenchmarkRunResult:
    """Replay captured traces and return the written artifact path."""

    settings = Settings()
    traces = load_captured_traces(trace_path)
    if not traces:
        msg = "No trace records found in the provided trace file."
        raise typer.BadParameter(msg)
    resolved_model_alias = (
        model_alias
        or traces[0].logical_alias
        or traces[0].execution_target.model_alias
    )
    try:
        validate_replayable_traces(traces, model_alias=resolved_model_alias)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if pinned_backend is not None:
        execution_target = ExecutionTarget(
            target_type=ExecutionTargetType.PINNED_BACKEND,
            model_alias=resolved_model_alias,
            pinned_backend=pinned_backend,
        )
    elif policy is not None:
        execution_target = ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias=resolved_model_alias,
            routing_policy=policy,
        )
    else:
        execution_target = ExecutionTarget(
            target_type=ExecutionTargetType.LOGICAL_ALIAS,
            model_alias=resolved_model_alias,
        )
    warmup = BenchmarkWarmupConfig(
        enabled=warmup_request_count > 0,
        request_count=warmup_request_count,
    )
    source_run_id = trace_path.stem
    build_replay_plan(
        traces=traces,
        execution_target=execution_target,
        replay_mode=replay_mode,
        concurrency=concurrency,
        warmup=warmup,
        source_run_id=source_run_id,
        settings=settings,
    )
    artifact = await run_trace_replay_benchmark(
        traces=traces,
        gateway_base_url=gateway_base_url,
        execution_target=execution_target,
        replay_mode=replay_mode,
        concurrency=concurrency,
        warmup=warmup,
        source_run_id=source_run_id,
        settings=settings,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
        deployment_target=deployment_target,
        deployment_profile=deployment_profile,
        config_profile_name=config_profile_name,
        control_plane_image=_control_plane_image_metadata(
            image_tag=control_plane_image_tag,
            git_sha=control_plane_git_sha,
        ),
        runtime_inspection_path=runtime_inspection_path,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    artifact_path = write_artifact(artifact, default_output_path(resolved_output_dir, artifact))
    if markdown_report:
        write_markdown_report(
            render_run_report_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.run_id),
        )
    return BenchmarkRunResult(artifact=artifact, output_path=artifact_path)


async def _compare_workload_command(
    *,
    manifest_path: Path,
    gateway_base_url: str,
    left_model_alias: str | None,
    left_policy: RoutingPolicy | None,
    left_pinned_backend: str | None,
    right_model_alias: str | None,
    right_policy: RoutingPolicy | None,
    right_pinned_backend: str | None,
    warmup_request_count: int,
    output_dir: Path | None,
    metrics_path: str | None,
    timeout_seconds: float,
    markdown_report: bool,
) -> Path:
    settings = Settings()
    scenario = WorkloadScenario.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    base_alias = scenario.model_alias or scenario.model
    left_target = _resolve_execution_target(
        model_alias=left_model_alias or base_alias,
        policy=left_policy,
        pinned_backend=left_pinned_backend,
    )
    right_target = _resolve_execution_target(
        model_alias=right_model_alias or base_alias,
        policy=right_policy,
        pinned_backend=right_pinned_backend,
    )
    warmup = BenchmarkWarmupConfig(
        enabled=warmup_request_count > 0,
        request_count=warmup_request_count,
    )
    artifact = await compare_workload_execution_targets(
        scenario=scenario,
        gateway_base_url=gateway_base_url,
        left_target=left_target,
        right_target=right_target,
        warmup=warmup,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = write_json_model(
        artifact,
        default_comparison_output_path(resolved_output_dir, artifact),
    )
    if markdown_report:
        write_markdown_report(
            render_target_comparison_report_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.comparison_id),
    )
    return output_path


def _simulate_policy_command(
    *,
    artifact_paths: list[Path],
    history_artifact_paths: list[Path] | None,
    policy_id: str,
    objective: CounterfactualObjective,
    mode: AdaptivePolicyMode,
    min_evidence_count: int,
    require_sufficient_data: bool,
    max_predicted_error_rate: float | None,
    max_predicted_latency_regression_ms: float | None,
    require_observed_backend_evidence: bool,
    output_dir: Path | None,
    markdown_report: bool,
) -> Path:
    settings = Settings()
    evaluation_artifacts = _load_run_artifacts(artifact_paths)
    historical_artifacts = (
        evaluation_artifacts
        if not history_artifact_paths
        else _load_run_artifacts(history_artifact_paths)
    )
    policy = ExplainablePolicySpec(
        policy_id=policy_id,
        objective=objective,
        mode=mode,
        min_evidence_count=min_evidence_count,
        guardrails=AdaptivePolicyGuardrails(
            require_sufficient_data=require_sufficient_data,
            max_predicted_error_rate=max_predicted_error_rate,
            max_predicted_latency_regression_ms=max_predicted_latency_regression_ms,
            require_observed_backend_evidence=require_observed_backend_evidence,
        ),
        rationale=[
            "offline simulation uses typed historical route estimates from benchmark artifacts",
            "guardrails keep adaptive recommendations explainable and reversible",
        ],
    )
    artifact = simulate_policy_counterfactual(
        evaluation_artifacts=evaluation_artifacts,
        history_artifacts=historical_artifacts,
        policy=policy,
    )
    artifact.metadata["policy_recommendation"] = recommend_policy_from_simulation(artifact)
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = write_json_model(
        artifact,
        resolved_output_dir / f"{artifact.simulation_id}.json",
    )
    if markdown_report:
        write_markdown_report(
            render_loaded_artifact_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.simulation_id),
        )
    return output_path


def _compare_offline_policies_command(
    *,
    artifact_paths: list[Path],
    trace_paths: list[Path],
    history_artifact_paths: list[Path],
    history_trace_paths: list[Path],
    routing_policies: list[RoutingPolicy],
    candidate_policies: list[str],
    output_dir: Path | None,
    markdown_report: bool,
) -> Path:
    settings = Settings()
    if not artifact_paths and not trace_paths:
        msg = "Provide at least one evaluation artifact or trace path."
        raise typer.BadParameter(msg)
    policies = [compatibility_policy_spec(policy) for policy in routing_policies]
    policies.extend(_parse_candidate_policy_specs(candidate_policies))
    if not policies:
        msg = "Provide at least one --routing-policy or --candidate-policy."
        raise typer.BadParameter(msg)
    evaluation_artifacts = _load_run_artifacts(artifact_paths)
    evaluation_traces = _load_trace_files(trace_paths)
    historical_artifacts = (
        evaluation_artifacts
        if not history_artifact_paths
        else _load_run_artifacts(history_artifact_paths)
    )
    historical_traces = (
        evaluation_traces if not history_trace_paths else _load_trace_files(history_trace_paths)
    )
    artifact = compare_candidate_policies_offline(
        policies=policies,
        evaluation_artifacts=evaluation_artifacts,
        evaluation_trace_records=evaluation_traces,
        history_artifacts=historical_artifacts,
        history_trace_records=historical_traces,
    )
    artifact.metadata["best_supported_policy"] = max(
        artifact.evaluations,
        key=lambda evaluation: (
            evaluation.summary.request_count - evaluation.summary.unsupported_count,
            -(evaluation.summary.projected_avg_latency_ms or float("inf")),
        ),
    ).policy.policy_id
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = write_json_model(
        artifact,
        resolved_output_dir / f"{artifact.simulation_comparison_id}.json",
    )
    if markdown_report:
        write_markdown_report(
            render_loaded_artifact_markdown(artifact),
            default_markdown_report_path(
                resolved_output_dir,
                artifact.simulation_comparison_id,
            ),
        )
    return output_path


async def _compare_traces_command(
    *,
    trace_path: Path,
    gateway_base_url: str,
    left_model_alias: str | None,
    left_policy: RoutingPolicy | None,
    left_pinned_backend: str | None,
    right_model_alias: str | None,
    right_policy: RoutingPolicy | None,
    right_pinned_backend: str | None,
    replay_mode: ReplayMode,
    concurrency: int,
    warmup_request_count: int,
    output_dir: Path | None,
    metrics_path: str | None,
    timeout_seconds: float,
    markdown_report: bool,
) -> Path:
    settings = Settings()
    traces = load_captured_traces(trace_path)
    if not traces:
        msg = "No trace records found in the provided trace file."
        raise typer.BadParameter(msg)
    base_alias = traces[0].logical_alias or traces[0].execution_target.model_alias
    try:
        validate_replayable_traces(traces, model_alias=base_alias)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    left_target = _resolve_execution_target(
        model_alias=left_model_alias or base_alias,
        policy=left_policy,
        pinned_backend=left_pinned_backend,
    )
    right_target = _resolve_execution_target(
        model_alias=right_model_alias or base_alias,
        policy=right_policy,
        pinned_backend=right_pinned_backend,
    )
    warmup = BenchmarkWarmupConfig(
        enabled=warmup_request_count > 0,
        request_count=warmup_request_count,
    )
    artifact = await compare_trace_execution_targets(
        traces=traces,
        gateway_base_url=gateway_base_url,
        left_target=left_target,
        right_target=right_target,
        replay_mode=replay_mode,
        concurrency=concurrency,
        warmup=warmup,
        source_run_id=trace_path.stem,
        metrics_path=metrics_path,
        timeout_seconds=timeout_seconds,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = write_json_model(
        artifact,
        default_comparison_output_path(resolved_output_dir, artifact),
    )
    if markdown_report:
        write_markdown_report(
            render_target_comparison_report_markdown(artifact),
            default_markdown_report_path(resolved_output_dir, artifact.comparison_id),
        )
    return output_path


def _generate_workload_command(
    *,
    family: WorkloadScenarioFamily,
    model_alias: str,
    request_count: int,
    seed: int,
    workload_shape: WorkloadShape,
    output_dir: Path | None,
) -> Path:
    """Generate a workload manifest and return the written path."""

    settings = Settings()
    scenario = build_workload_manifest(
        family=family,
        model_alias=model_alias,
        request_count=request_count,
        seed=seed,
        workload_shape=workload_shape,
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = default_workload_manifest_path(resolved_output_dir, scenario)
    return write_json_model(scenario, output_path)


def _generate_report_command(
    *,
    artifact_paths: list[Path],
    output_path: Path | None,
) -> Path:
    """Render benchmark artifact markdown from authoritative JSON artifacts."""

    artifacts = [load_benchmark_artifact_model(path) for path in artifact_paths]
    markdown = render_artifact_bundle_markdown(artifacts)
    resolved_output_path = output_path or default_generated_report_path(artifact_paths)
    return write_markdown_report(markdown, resolved_output_path)


def _recommend_policies_command(
    *,
    artifact_paths: list[Path],
    output_dir: Path | None,
    markdown_report: bool,
) -> Path:
    """Build a typed recommendation report from authoritative artifact inputs."""

    settings = Settings()
    loaded_artifacts = []
    for path in artifact_paths:
        artifact = load_benchmark_artifact_model(path)
        if not isinstance(
            artifact,
            (
                BenchmarkRunArtifact,
                CounterfactualSimulationArtifact,
                CounterfactualSimulationComparisonArtifact,
            ),
        ):
            msg = f"{path} is not a benchmark run or simulation artifact"
            raise typer.BadParameter(msg)
        loaded_artifacts.append(artifact)
    report = build_policy_recommendation_report(
        loaded_artifacts,
        report_id=f"policy-recommendations-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        timestamp=datetime.now(UTC),
    )
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    output_path = write_json_model(
        report,
        resolved_output_dir / f"{report.recommendation_report_id}.json",
    )
    if markdown_report:
        write_markdown_report(
            render_loaded_artifact_markdown(report),
            default_markdown_report_path(
                resolved_output_dir,
                report.recommendation_report_id,
            ),
        )
    return output_path


def _load_trace_files(trace_paths: list[Path]) -> list[CapturedTraceRecord]:
    traces: list[CapturedTraceRecord] = []
    for path in trace_paths:
        traces.extend(load_captured_traces(path))
    return traces


def _load_run_artifacts(artifact_paths: list[Path]) -> list[BenchmarkRunArtifact]:
    artifacts: list[BenchmarkRunArtifact] = []
    for path in artifact_paths:
        artifact = load_benchmark_artifact_model(path)
        if not isinstance(artifact, BenchmarkRunArtifact):
            msg = f"{path} is not a benchmark run artifact"
            raise typer.BadParameter(msg)
        artifacts.append(artifact)
    return artifacts


def _parse_candidate_policy_specs(policy_specs: list[str]) -> list[ExplainablePolicySpec]:
    parsed: list[ExplainablePolicySpec] = []
    for spec in policy_specs:
        policy_id, separator, objective = spec.partition(":")
        if not separator:
            msg = f"Invalid candidate policy '{spec}'. Use policy_id:objective."
            raise typer.BadParameter(msg)
        parsed.append(
            ExplainablePolicySpec(
                policy_id=policy_id,
                objective=CounterfactualObjective(objective),
                rationale=[
                    "custom offline candidate policy parsed from CLI input",
                    "results remain counterfactual and depend on historical evidence quality",
                ],
            )
        )
    return parsed


def _control_plane_image_metadata(
    *,
    image_tag: str | None,
    git_sha: str | None,
) -> BackendImageMetadata | None:
    if image_tag is None and git_sha is None:
        return None
    return BackendImageMetadata(image_tag=image_tag, git_sha=git_sha)


def _resolve_execution_target(
    *,
    model_alias: str,
    policy: RoutingPolicy | None,
    pinned_backend: str | None,
) -> ExecutionTarget:
    if pinned_backend is not None:
        return ExecutionTarget(
            target_type=ExecutionTargetType.PINNED_BACKEND,
            model_alias=model_alias,
            pinned_backend=pinned_backend,
        )
    if policy is not None:
        return ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias=model_alias,
            routing_policy=policy,
        )
    return ExecutionTarget(
        target_type=ExecutionTargetType.LOGICAL_ALIAS,
        model_alias=model_alias,
    )


def _resolve_default_model(settings: Settings) -> str:
    if settings.default_model_alias is not None:
        return settings.default_model_alias
    if settings.local_models:
        return settings.local_models[0].alias

    msg = "No model alias provided and no local model is configured."
    raise typer.BadParameter(msg)


def _resolve_benchmark_registry(settings: Settings) -> AdapterRegistry | None:
    if not settings.local_models:
        return None
    return build_registry_from_settings(settings)


def main() -> None:
    """Entrypoint for `python -m switchyard.bench.cli`."""

    app()


if __name__ == "__main__":
    main()
