"""CLI for a CI-safe remote worker stub image."""

from __future__ import annotations

from collections.abc import Callable

import typer
import uvicorn

from switchyard.worker.config import RemoteWorkerRuntimeSettings
from switchyard.worker.fake import create_fake_remote_worker_app

app = typer.Typer(help="Serve a deterministic remote worker stub over the shared protocol.")

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
    """Serve a fake remote worker using environment-driven runtime metadata."""

    settings = RemoteWorkerRuntimeSettings()
    serve_fake_remote_worker_process(
        settings=settings,
        host=host,
        port=port,
    )


def serve_fake_remote_worker_process(
    *,
    settings: RemoteWorkerRuntimeSettings,
    host: str | None = None,
    port: int | None = None,
    server_runner: ServerRunner = uvicorn.run,
) -> None:
    """Run the fake remote worker process for CI/local integration tests."""

    app_instance = create_fake_remote_worker_app(settings.to_fake_worker_config())
    server_runner(
        app_instance,
        host=host or settings.host,
        port=port or settings.port,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    """Entrypoint for `python -m switchyard.worker.fake_cli`."""

    app()


if __name__ == "__main__":
    main()
