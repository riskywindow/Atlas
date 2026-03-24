"""CLI entrypoints for the portable control plane."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import httpx
import typer
import uvicorn

from switchyard.bench.artifacts import (
    render_forge_campaign_inspection_markdown,
    render_forge_promotion_runtime_markdown,
)
from switchyard.bench.campaigns import inspect_forge_stage_a_campaigns
from switchyard.config import Settings
from switchyard.diagnostics import collect_deployment_diagnostics
from switchyard.gateway.app import create_app
from switchyard.optimization import (
    build_forge_stage_a_campaign,
    build_optimization_profile,
)
from switchyard.schemas.admin import (
    DeploymentDiagnosticsResponse,
)
from switchyard.schemas.forge import (
    ForgePromotionApplyRequest,
    ForgePromotionCompareRequest,
    ForgePromotionDecisionRequest,
    ForgePromotionProposeRequest,
    ForgePromotionRuntimeSummary,
)
from switchyard.schemas.optimization import (
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationTrialArtifact,
)

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


@app.command("export-optimization-profile")
def export_optimization_profile() -> None:
    """Export the optimization-ready control-plane knob surface as JSON."""

    profile = build_optimization_profile(Settings())
    typer.echo(profile.model_dump_json(indent=2))


@app.command("export-forge-stage-a-campaign")
def export_forge_stage_a_campaign() -> None:
    """Export the typed Forge Stage A campaign inspection snapshot as JSON."""

    campaign = build_forge_stage_a_campaign(Settings())
    typer.echo(campaign.model_dump_json(indent=2))


@app.command("export-forge-stage-a-promotion")
def export_forge_stage_a_promotion(
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to query.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion",
        "--admin-path",
        help="Forge promotion admin path on the gateway.",
    ),
    markdown: bool = typer.Option(
        False,
        "--markdown",
        help="Render a compact markdown summary instead of raw JSON.",
    ),
) -> None:
    """Export the current live Forge Stage A promotion state from a gateway."""

    payload = asyncio.run(
        _fetch_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
        )
    )
    if markdown:
        summary = ForgePromotionRuntimeSummary.model_validate(payload)
        typer.echo(render_forge_promotion_runtime_markdown(summary))
        return
    typer.echo(json.dumps(payload, indent=2))


@app.command("inspect-forge-stage-a-campaign")
def inspect_forge_stage_a_campaign(
    artifact_path: Path = typer.Option(
        ...,
        "--artifact-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to an optimization campaign artifact JSON file.",
    ),
    comparison_artifact_path: Path | None = typer.Option(
        None,
        "--comparison-artifact-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional path to an optimization campaign comparison artifact JSON file.",
    ),
    markdown: bool = typer.Option(
        False,
        "--markdown",
        help="Render a compact markdown inspection report instead of JSON.",
    ),
) -> None:
    """Inspect one Forge Stage A campaign artifact without reading raw internals."""

    campaign_artifact = _load_campaign_artifact(artifact_path)
    comparison_artifacts = (
        []
        if comparison_artifact_path is None
        else [_load_campaign_comparison_artifact(comparison_artifact_path)]
    )
    inspection = inspect_forge_stage_a_campaigns(
        campaign_artifacts=[campaign_artifact],
        comparison_artifacts=comparison_artifacts,
    )
    if markdown:
        typer.echo(render_forge_campaign_inspection_markdown(inspection))
        return
    typer.echo(inspection.model_dump_json(indent=2))


@app.command("propose-forge-stage-a-promotion")
def propose_forge_stage_a_promotion(
    artifact_path: Path = typer.Option(
        ...,
        "--artifact-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a trial or campaign artifact JSON file.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    candidate_configuration_id: str | None = typer.Option(
        None,
        "--candidate-configuration-id",
        help="Required when the artifact contains multiple candidate trials.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/propose",
        "--admin-path",
        help="Forge promotion propose path on the gateway.",
    ),
) -> None:
    """Propose one reviewed Forge Stage A trial to a running gateway."""

    trial_artifact = _load_trial_artifact(
        artifact_path=artifact_path,
        candidate_configuration_id=candidate_configuration_id,
    )
    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionProposeRequest(
                trial_artifact=trial_artifact,
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("approve-forge-stage-a-promotion")
def approve_forge_stage_a_promotion(
    rollout_artifact_id: str = typer.Option(
        ...,
        "--rollout-artifact-id",
        help="Proposed Forge rollout artifact ID.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/approve",
        "--admin-path",
        help="Forge promotion approve path on the gateway.",
    ),
) -> None:
    """Approve one proposed Forge Stage A rollout."""

    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionDecisionRequest(
                rollout_artifact_id=rollout_artifact_id,
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("apply-forge-stage-a-promotion")
def apply_forge_stage_a_promotion(
    rollout_artifact_id: str = typer.Option(
        ...,
        "--rollout-artifact-id",
        help="Approved Forge rollout artifact ID.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    canary_percentage: float | None = typer.Option(
        None,
        "--canary-percentage",
        min=0.0,
        max=100.0,
        help="Optional bounded canary percentage override.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/apply",
        "--admin-path",
        help="Forge promotion apply path on the gateway.",
    ),
) -> None:
    """Activate one approved Forge Stage A rollout as a canary."""

    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionApplyRequest(
                rollout_artifact_id=rollout_artifact_id,
                canary_percentage=canary_percentage,
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("compare-forge-stage-a-promotion")
def compare_forge_stage_a_promotion(
    rollout_artifact_id: str = typer.Option(
        ...,
        "--rollout-artifact-id",
        help="Active Forge rollout artifact ID.",
    ),
    artifact_path: Path = typer.Option(
        ...,
        "--artifact-path",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to an optimization campaign comparison artifact JSON file.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/compare",
        "--admin-path",
        help="Forge promotion compare path on the gateway.",
    ),
) -> None:
    """Attach artifact-backed comparison evidence to an active Forge rollout."""

    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionCompareRequest(
                rollout_artifact_id=rollout_artifact_id,
                comparison_artifact=_load_campaign_comparison_artifact(artifact_path),
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("promote-default-forge-stage-a-promotion")
def promote_default_forge_stage_a_promotion(
    rollout_artifact_id: str = typer.Option(
        ...,
        "--rollout-artifact-id",
        help="Compared Forge rollout artifact ID.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Optional operator reason recorded alongside the promotion.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/promote-default",
        "--admin-path",
        help="Forge promotion default-promotion path on the gateway.",
    ),
) -> None:
    """Promote a compared Forge rollout to the runtime default."""

    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionDecisionRequest(
                rollout_artifact_id=rollout_artifact_id,
                reason=reason,
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("reject-forge-stage-a-promotion")
def reject_forge_stage_a_promotion(
    rollout_artifact_id: str = typer.Option(
        ...,
        "--rollout-artifact-id",
        help="Active Forge rollout artifact ID.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        help="Optional operator reason recorded alongside the rejection.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/reject",
        "--admin-path",
        help="Forge promotion reject path on the gateway.",
    ),
) -> None:
    """Reject one proposed or active Forge rollout."""

    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionDecisionRequest(
                rollout_artifact_id=rollout_artifact_id,
                reason=reason,
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


@app.command("reset-forge-stage-a-promotion")
def reset_forge_stage_a_promotion(
    rollout_artifact_id: str = typer.Option(
        ...,
        "--rollout-artifact-id",
        help="Active Forge rollout artifact ID.",
    ),
    gateway_base_url: str = typer.Option(
        ...,
        "--gateway-base-url",
        help="Gateway base URL to mutate.",
    ),
    admin_path: str = typer.Option(
        "/admin/forge/stage-a/promotion/reset",
        "--admin-path",
        help="Forge promotion reset path on the gateway.",
    ),
) -> None:
    """Reset the live Forge Stage A promotion controller on a gateway."""

    payload = asyncio.run(
        _post_gateway_json(
            gateway_base_url=gateway_base_url,
            path=admin_path,
            payload=ForgePromotionDecisionRequest(
                rollout_artifact_id=rollout_artifact_id,
            ).model_dump(mode="json"),
        )
    )
    typer.echo(json.dumps(payload, indent=2))


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


async def _fetch_gateway_json(
    *,
    gateway_base_url: str,
    path: str,
) -> dict[str, object]:
    async with httpx.AsyncClient(
        base_url=gateway_base_url,
        timeout=httpx.Timeout(10.0),
    ) as client:
        response = await client.get(path)
        response.raise_for_status()
        return cast(dict[str, object], response.json())


async def _post_gateway_json(
    *,
    gateway_base_url: str,
    path: str,
    payload: dict[str, object],
) -> dict[str, object]:
    async with httpx.AsyncClient(
        base_url=gateway_base_url,
        timeout=httpx.Timeout(10.0),
    ) as client:
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return cast(dict[str, object], response.json())


def _load_trial_artifact(
    *,
    artifact_path: Path,
    candidate_configuration_id: str | None,
) -> OptimizationTrialArtifact:
    payload = json.loads(artifact_path.read_text())
    if "trial_artifact_id" in payload:
        return OptimizationTrialArtifact.model_validate(payload)
    if "campaign_artifact_id" not in payload:
        msg = "artifact must be an optimization trial or campaign artifact"
        raise typer.BadParameter(msg)
    campaign = OptimizationCampaignArtifact.model_validate(payload)
    if candidate_configuration_id is None:
        promotable_trials = [
            trial
            for trial in campaign.trials
            if trial.promotion_decision is not None
            and trial.promotion_decision.disposition.value
            in {
                "recommend_canary",
                "approved_canary",
            }
        ]
        if len(promotable_trials) != 1:
            msg = (
                "campaign artifact contains multiple candidate trials; "
                "set --candidate-configuration-id"
            )
            raise typer.BadParameter(msg)
        return promotable_trials[0]
    for trial in campaign.trials:
        if trial.candidate_configuration.candidate_configuration_id == candidate_configuration_id:
            return trial
    msg = "candidate_configuration_id was not found in the provided campaign artifact"
    raise typer.BadParameter(msg)


def _load_campaign_artifact(artifact_path: Path) -> OptimizationCampaignArtifact:
    try:
        return OptimizationCampaignArtifact.model_validate_json(artifact_path.read_text())
    except ValueError as exc:
        msg = "artifact did not contain a valid optimization campaign artifact"
        raise typer.BadParameter(msg) from exc


def _load_campaign_comparison_artifact(
    artifact_path: Path,
) -> OptimizationCampaignComparisonArtifact:
    try:
        return OptimizationCampaignComparisonArtifact.model_validate_json(artifact_path.read_text())
    except ValueError as exc:
        msg = "artifact did not contain a valid optimization campaign comparison artifact"
        raise typer.BadParameter(msg) from exc


def _diagnostics_has_issues(payload: object) -> bool:
    if isinstance(payload, dict):
        status = payload.get("status")
        if status in {"unreachable", "error"}:
            return True
        return any(_diagnostics_has_issues(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_diagnostics_has_issues(value) for value in payload)
    return False
