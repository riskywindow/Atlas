"""CLI entrypoints for the portable control plane."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import httpx
import typer
import uvicorn

from switchyard.config import Settings
from switchyard.diagnostics import collect_deployment_diagnostics
from switchyard.gateway.app import create_app
from switchyard.schemas.admin import DeploymentDiagnosticsResponse

app = typer.Typer(help="Switchyard control-plane entrypoints.")


@app.command("gateway")
def gateway(
    host: str = typer.Option("0.0.0.0", "--host", help="Gateway bind address."),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="Gateway bind port."),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level."),
) -> None:
    """Run the FastAPI gateway."""

    uvicorn.run(
        "switchyard.gateway:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )


@app.command("check-config")
def check_config(
    create_gateway: bool = typer.Option(
        True,
        "--create-gateway/--no-create-gateway",
        help="Also instantiate the gateway app after loading settings.",
    ),
) -> None:
    """Load settings and optionally build the gateway for smoke validation."""

    settings = Settings()
    if create_gateway:
        create_app(settings=settings)
    typer.echo("ok")


@app.command("doctor")
def doctor(
    gateway_base_url: str | None = typer.Option(
        None,
        "--gateway-base-url",
        help="Optional deployed gateway URL. When set, query /admin/deployment there.",
    ),
    admin_path: str = typer.Option(
        "/admin/deployment",
        "--admin-path",
        help="Admin diagnostics path on a deployed gateway.",
    ),
    fail_on_issues: bool = typer.Option(
        False,
        "--fail-on-issues/--no-fail-on-issues",
        help="Exit non-zero when diagnostics include unreachable or error statuses.",
    ),
) -> None:
    """Run local preflight diagnostics or query deployed diagnostics from a gateway."""

    if gateway_base_url is None:
        payload = asyncio.run(_doctor_local())
    else:
        payload = asyncio.run(
            _doctor_remote(
                gateway_base_url=gateway_base_url,
                admin_path=admin_path,
            )
        )
    typer.echo(json.dumps(payload, indent=2))
    if fail_on_issues and _diagnostics_has_issues(payload):
        raise typer.Exit(code=1)


def main() -> None:
    """Entrypoint for `python -m switchyard.control_plane.cli`."""

    app()


if __name__ == "__main__":
    main()


async def _doctor_local() -> dict[str, object]:
    settings = Settings()
    return (await collect_deployment_diagnostics(settings)).model_dump(mode="json")


async def _doctor_remote(
    *,
    gateway_base_url: str,
    admin_path: str,
) -> dict[str, object]:
    async with httpx.AsyncClient(
        base_url=gateway_base_url,
        timeout=httpx.Timeout(10.0),
    ) as client:
        response = await client.get(admin_path)
        response.raise_for_status()
        payload = DeploymentDiagnosticsResponse.model_validate(response.json())
        return cast(dict[str, object], payload.model_dump(mode="json"))


def _diagnostics_has_issues(payload: object) -> bool:
    if isinstance(payload, dict):
        status = payload.get("status")
        if status in {"unreachable", "error"}:
            return True
        return any(_diagnostics_has_issues(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_diagnostics_has_issues(value) for value in payload)
    return False
