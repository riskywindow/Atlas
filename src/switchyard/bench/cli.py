"""Benchmark CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from switchyard.bench.artifacts import (
    BenchmarkRunResult,
    build_synthetic_scenario,
    default_output_path,
    run_synthetic_benchmark,
    write_artifact,
)
from switchyard.config import Settings
from switchyard.schemas.routing import RoutingPolicy

app = typer.Typer(help="Switchyard benchmark utilities.")


@app.command("run-synthetic")
def run_synthetic(
    request_count: int = typer.Option(3, min=1, help="Number of synthetic requests to issue."),
    policy: RoutingPolicy = typer.Option(
        RoutingPolicy.BALANCED,
        case_sensitive=False,
        help="Routing policy to exercise.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for the output artifact. Defaults to SWITCHYARD_BENCHMARK_OUTPUT_DIR.",
    ),
) -> None:
    """Run a small synthetic benchmark and write a JSON artifact."""

    result = asyncio.run(
        _run_synthetic_command(
            request_count=request_count,
            policy=policy,
            output_dir=output_dir,
        )
    )
    typer.echo(result.output_path)


async def _run_synthetic_command(
    *,
    request_count: int,
    policy: RoutingPolicy,
    output_dir: Path | None = None,
) -> BenchmarkRunResult:
    """Run the synthetic benchmark command and return the artifact path."""

    settings = Settings()
    scenario = build_synthetic_scenario(request_count=request_count, policy=policy)
    artifact = await run_synthetic_benchmark(scenario=scenario)
    resolved_output_dir = output_dir or settings.benchmark_output_dir
    artifact_path = write_artifact(artifact, default_output_path(resolved_output_dir, artifact))
    return BenchmarkRunResult(artifact=artifact, output_path=artifact_path)


def main() -> None:
    """Entrypoint for `python -m switchyard.bench.cli`."""

    app()


if __name__ == "__main__":
    main()
