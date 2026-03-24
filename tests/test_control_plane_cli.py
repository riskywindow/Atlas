from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import switchyard.control_plane.cli as control_plane_cli
from switchyard.config import Settings
from switchyard.optimization import build_baseline_optimization_config_profile
from switchyard.schemas.benchmark import CounterfactualObjective, RecommendationConfidence
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactSourceType,
    OptimizationCampaignArtifact,
    OptimizationCampaignMetadata,
    OptimizationCandidateConfigurationArtifact,
    OptimizationComparisonOperator,
    OptimizationConstraintAssessment,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationEvidenceRecord,
    OptimizationKnobChange,
    OptimizationPromotionDecision,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import PolicyRolloutMode, RoutingPolicy


def test_gateway_command_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        called["app"] = app
        called["kwargs"] = kwargs

    monkeypatch.setattr("switchyard.control_plane.cli.uvicorn.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        ["gateway", "--host", "127.0.0.1", "--port", "8010", "--log-level", "DEBUG"],
    )

    assert result.exit_code == 0
    assert called["app"] == "switchyard.gateway:create_app"
    assert called["kwargs"] == {
        "factory": True,
        "host": "127.0.0.1",
        "port": 8010,
        "log_level": "debug",
    }


def test_check_config_command_loads_settings_and_builds_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    called: dict[str, object] = {}

    monkeypatch.setattr(control_plane_cli, "Settings", lambda: settings)

    def fake_create_app(*, settings: Settings) -> object:
        called["settings"] = settings
        return object()

    monkeypatch.setattr(control_plane_cli, "create_app", fake_create_app)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["check-config"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert called["settings"] is settings


def test_doctor_command_runs_local_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_doctor_local() -> dict[str, object]:
        return {
            "diagnostics_source": "config_preflight",
            "worker_deployments": [],
            "supporting_services": [],
        }

    monkeypatch.setattr(control_plane_cli, "_doctor_local", fake_doctor_local)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["doctor"])

    assert result.exit_code == 0
    assert '"diagnostics_source": "config_preflight"' in result.stdout


def test_doctor_command_can_fail_on_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_doctor_local() -> dict[str, object]:
        return {
            "diagnostics_source": "config_preflight",
            "worker_deployments": [
                {"configured_instances": [{"probe": {"status": "unreachable"}}]}
            ],
        }

    monkeypatch.setattr(control_plane_cli, "_doctor_local", fake_doctor_local)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["doctor", "--fail-on-issues"])

    assert result.exit_code == 1


def test_doctor_command_queries_remote_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_doctor_remote(*, gateway_base_url: str, admin_path: str) -> dict[str, object]:
        assert gateway_base_url == "http://testserver"
        assert admin_path == "/admin/deployment"
        return {
            "diagnostics_source": "runtime",
            "worker_deployments": [],
            "supporting_services": [],
        }

    monkeypatch.setattr(control_plane_cli, "_doctor_remote", fake_doctor_remote)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        ["doctor", "--gateway-base-url", "http://testserver"],
    )

    assert result.exit_code == 0
    assert '"diagnostics_source": "runtime"' in result.stdout


def test_export_optimization_profile_command(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    monkeypatch.setattr(control_plane_cli, "Settings", lambda: settings)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["export-optimization-profile"])

    assert result.exit_code == 0
    assert '"profile_id": "phase9-stage-a-baseline"' in result.stdout
    assert '"knob_id": "default_routing_policy"' in result.stdout


def test_export_forge_stage_a_campaign_command(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    monkeypatch.setattr(control_plane_cli, "Settings", lambda: settings)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["export-forge-stage-a-campaign"])

    assert result.exit_code == 0
    assert '"campaign_id": "phase9-stage-a-baseline-forge-stage-a"' in result.stdout
    assert '"trial_role": "baseline"' in result.stdout
    assert '"candidate_kind": "routing_policy"' in result.stdout


def test_export_forge_stage_a_promotion_command(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(*, gateway_base_url: str, path: str) -> dict[str, object]:
        assert gateway_base_url == "http://testserver"
        assert path == "/admin/forge/stage-a/promotion"
        return {"active_config_profile_id": "phase9-baseline", "applied": False}

    monkeypatch.setattr(control_plane_cli, "_fetch_gateway_json", fake_fetch)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        ["export-forge-stage-a-promotion", "--gateway-base-url", "http://testserver"],
    )

    assert result.exit_code == 0
    assert '"active_config_profile_id": "phase9-baseline"' in result.stdout


def test_export_forge_stage_a_promotion_command_supports_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(*, gateway_base_url: str, path: str) -> dict[str, object]:
        assert gateway_base_url == "http://testserver"
        assert path == "/admin/forge/stage-a/promotion"
        return {
            "baseline_config_profile_id": _baseline_config_profile_id(),
            "active_config_profile_id": _baseline_config_profile_id(),
            "rollout_mode": "disabled",
            "canary_percentage": 0.0,
            "promotion_disposition": "no_action",
            "evidence_kinds": ["observed"],
            "applied": False,
        }

    monkeypatch.setattr(control_plane_cli, "_fetch_gateway_json", fake_fetch)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        [
            "export-forge-stage-a-promotion",
            "--gateway-base-url",
            "http://testserver",
            "--markdown",
        ],
    )

    assert result.exit_code == 0
    assert "# Switchyard Forge Stage A Promotion" in result.stdout
    assert "- Rollout mode: `disabled`" in result.stdout


def test_propose_forge_stage_a_promotion_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    posted: dict[str, object] = {}

    async def fake_post(
        *,
        gateway_base_url: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        posted["gateway_base_url"] = gateway_base_url
        posted["path"] = path
        posted["payload"] = payload
        return {
            "rollout_artifact_id": "rollout-001",
            "candidate_config_profile_id": "phase9-local-preferred",
            "lifecycle_state": "proposed",
        }

    monkeypatch.setattr(control_plane_cli, "_post_gateway_json", fake_post)
    artifact_path = tmp_path / "trial.json"
    artifact_path.write_text(json.dumps(build_promotable_trial_artifact().model_dump(mode="json")))

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        [
            "propose-forge-stage-a-promotion",
            "--artifact-path",
            str(artifact_path),
            "--gateway-base-url",
            "http://testserver",
        ],
    )

    assert result.exit_code == 0
    assert posted["gateway_base_url"] == "http://testserver"
    assert posted["path"] == "/admin/forge/stage-a/promotion/propose"
    request_payload = posted["payload"]
    assert isinstance(request_payload, dict)
    assert "trial_artifact" in request_payload
    assert '"lifecycle_state": "proposed"' in result.stdout


def test_inspect_forge_stage_a_campaign_command_renders_markdown(tmp_path: Path) -> None:
    artifact_path = tmp_path / "campaign.json"
    artifact_path.write_text(
        json.dumps(build_campaign_artifact().model_dump(mode="json")),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        [
            "inspect-forge-stage-a-campaign",
            "--artifact-path",
            str(artifact_path),
            "--markdown",
        ],
    )

    assert result.exit_code == 0
    assert "# Switchyard Forge Stage A Inspection" in result.stdout
    assert "Recommendation: `promote_candidate`" in result.stdout
    assert "Helps workload families: `repeated_prefix`" in result.stdout


def test_apply_forge_stage_a_promotion_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted: dict[str, object] = {}

    async def fake_post(
        *,
        gateway_base_url: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        posted["gateway_base_url"] = gateway_base_url
        posted["path"] = path
        posted["payload"] = payload
        return {"active_config_profile_id": "phase9-local-preferred", "applied": True}

    monkeypatch.setattr(control_plane_cli, "_post_gateway_json", fake_post)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        [
            "apply-forge-stage-a-promotion",
            "--rollout-artifact-id",
            "rollout-001",
            "--gateway-base-url",
            "http://testserver",
            "--canary-percentage",
            "12.5",
        ],
    )

    assert result.exit_code == 0
    assert posted["gateway_base_url"] == "http://testserver"
    assert posted["path"] == "/admin/forge/stage-a/promotion/apply"
    request_payload = posted["payload"]
    assert isinstance(request_payload, dict)
    assert request_payload["rollout_artifact_id"] == "rollout-001"
    assert request_payload["canary_percentage"] == 12.5
    assert '"applied": true' in result.stdout.lower()


def test_reset_forge_stage_a_promotion_command(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        *,
        gateway_base_url: str,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert gateway_base_url == "http://testserver"
        assert path == "/admin/forge/stage-a/promotion/reset"
        assert payload == {"rollout_artifact_id": "rollout-001", "reason": None, "notes": []}
        return {"active_config_profile_id": "phase9-baseline", "applied": False}

    monkeypatch.setattr(control_plane_cli, "_post_gateway_json", fake_post)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        [
            "reset-forge-stage-a-promotion",
            "--rollout-artifact-id",
            "rollout-001",
            "--gateway-base-url",
            "http://testserver",
        ],
    )

    assert result.exit_code == 0
    assert '"applied": false' in result.stdout.lower()


def build_promotable_trial_artifact(
    *,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    canary_percentage: float = 10.0,
) -> OptimizationTrialArtifact:
    trial_identity = OptimizationTrialIdentity(
        trial_id=f"trial-{config_profile_id}",
        candidate_id=f"routing-policy:{routing_policy.value}",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )
    candidate_configuration = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=f"candidate-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        candidate=trial_identity,
        baseline_config_profile_id=_baseline_config_profile_id(),
        config_profile_id=config_profile_id,
        knob_changes=[
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value=RoutingPolicy.BALANCED.value,
                candidate_value=routing_policy.value,
            )
        ],
    )
    return OptimizationTrialArtifact(
        trial_artifact_id=f"trial-artifact-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-phase9-baseline",
        candidate_configuration=candidate_configuration,
        trial_identity=trial_identity,
        evidence_records=[
            OptimizationEvidenceRecord(
                evidence_id=f"evidence-observed-{config_profile_id}",
                evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
                source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
                source_artifact_id="benchmark-phase9-001",
            ),
            OptimizationEvidenceRecord(
                evidence_id=f"evidence-replayed-{config_profile_id}",
                evidence_kind=OptimizationArtifactEvidenceKind.REPLAYED,
                source_type=OptimizationArtifactSourceType.REPLAY_PLAN,
                source_artifact_id="replay-phase9-001",
            ),
        ],
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="remote-share-cap",
                dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=25.0,
                evaluated_value=18.0,
                satisfied=True,
                evidence_kinds=[OptimizationArtifactEvidenceKind.OBSERVED],
            )
        ],
        recommendation_summary=OptimizationRecommendationSummary(
            recommendation_summary_id=f"recommendation-{config_profile_id}",
            disposition=OptimizationRecommendationDisposition.PROMOTE_CANDIDATE,
            recommendation_label=OptimizationRecommendationLabel.PROMOTION_ELIGIBLE,
            confidence=RecommendationConfidence.MEDIUM,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=config_profile_id,
            evidence_kinds=[
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.REPLAYED,
            ],
            benefited_workload_families=["repeated_prefix"],
            regressed_workload_families=["burst_parallel"],
            reason_codes=[OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED],
            rationale=["candidate improved the primary objective"],
        ),
        promotion_decision=OptimizationPromotionDecision(
            promotion_decision_id=f"promotion-{config_profile_id}",
            disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=config_profile_id,
            rollout_mode=PolicyRolloutMode.CANARY,
            canary_percentage=canary_percentage,
        ),
    )


def build_campaign_artifact() -> OptimizationCampaignArtifact:
    baseline_trial_identity = OptimizationTrialIdentity(
        trial_id="trial-baseline",
        candidate_id="routing-policy:balanced",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=_baseline_config_profile_id(),
        routing_policy=RoutingPolicy.BALANCED,
    )
    baseline_candidate = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id="candidate-phase9-baseline",
        campaign_id="campaign-phase9-001",
        candidate=baseline_trial_identity,
        baseline_config_profile_id=_baseline_config_profile_id(),
        config_profile_id=_baseline_config_profile_id(),
    )
    trial = build_promotable_trial_artifact()
    assert trial.recommendation_summary is not None
    assert trial.promotion_decision is not None
    return OptimizationCampaignArtifact(
        campaign_artifact_id="campaign-artifact-phase9-001",
        campaign=OptimizationCampaignMetadata(
            campaign_id="campaign-phase9-001",
            optimization_profile_id="phase9-stage-a-baseline",
            objective=CounterfactualObjective.LATENCY,
            evidence_sources=[
                ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
                ForgeEvidenceSourceKind.REPLAYED_BENCHMARK,
            ],
            required_evidence_sources=[ForgeEvidenceSourceKind.REPLAYED_BENCHMARK],
            default_workload_set_ids=["phase9-cache-locality"],
        ),
        baseline_candidate_configuration=baseline_candidate,
        candidate_configurations=[trial.candidate_configuration],
        trials=[trial],
        evidence_records=list(trial.evidence_records),
        recommendation_summaries=[trial.recommendation_summary],
        promotion_decisions=[trial.promotion_decision],
    )


def _baseline_config_profile_id() -> str:
    return build_baseline_optimization_config_profile(Settings()).config_profile_id
