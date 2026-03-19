"""CLI for serving the concrete dependency-gated vLLM-CUDA remote worker."""

from __future__ import annotations

from collections.abc import Callable

import typer
import uvicorn

from switchyard.worker.config import RemoteWorkerRuntimeSettings
from switchyard.worker.vllm_cuda import create_vllm_cuda_worker_app

app = typer.Typer(help="Serve the concrete Linux/NVIDIA vLLM-CUDA worker process.")

ServerRunner = Callable[..., None]


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind address override."),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Bind port override.",
    ),
) -> None:
    """Serve the concrete vLLM-CUDA worker using environment-driven settings."""

    settings = RemoteWorkerRuntimeSettings()
    serve_vllm_cuda_worker_process(
        settings=settings,
        host=host,
        port=port,
    )


def serve_vllm_cuda_worker_process(
    *,
    settings: RemoteWorkerRuntimeSettings,
    host: str | None = None,
    port: int | None = None,
    server_runner: ServerRunner = uvicorn.run,
) -> None:
    """Run the concrete vLLM-CUDA remote worker process."""

    app_instance = create_vllm_cuda_worker_app(settings)
    server_runner(
        app_instance,
        host=host or settings.host,
        port=port or settings.port,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    """Entrypoint for `python -m switchyard.worker.vllm_cuda_cli`."""

    app()


if __name__ == "__main__":
    main()
