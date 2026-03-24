"""Tests for the offline campaign executor."""

from __future__ import annotations

import pytest

from switchyard.bench.campaigns import ForgeStageAExecutionResult
from switchyard.bench.executor import (
    CampaignExecutionMode,
    CampaignExecutionPlan,
    ConfigOverlay,
    OverlayValidationStatus,
    apply_overlay_to_settings,
    execute_campaign,
    validate_overlay,
)
from switchyard.config import Settings
from switchyard.optimization import build_optimization_profile
from switchyard.schemas.benchmark import (
    CounterfactualObjective,
    WorkloadScenarioFamily,
)
from switchyard.schemas.optimization import (
    OptimizationArtifactStatus,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationStrategy,
)
from switchyard.schemas.routing import RoutingPolicy


def _default_settings() -> Settings:
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.optimization.worker_launch_presets = ()
    settings.phase4.policy_rollout.candidate_policy_id = None
    settings.phase4.policy_rollout.canary_percentage = 10.0
    return settings


# -- overlay validation --


def test_validate_overlay_accepts_valid_routing_policy_change() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-valid",
        changes={"default_routing_policy": "latency_first"},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is True
    assert result.issues == []
    assert "default_routing_policy" in result.validated_knob_ids


def test_validate_overlay_rejects_unknown_knob() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-unknown",
        changes={"nonexistent_knob": "some_value"},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is False
    assert len(result.issues) == 1
    assert result.issues[0].status is OverlayValidationStatus.UNKNOWN_KNOB
    assert result.issues[0].knob_id == "nonexistent_knob"


def test_validate_overlay_rejects_domain_violation() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-domain",
        changes={"default_routing_policy": "nonexistent_policy"},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is False
    assert len(result.issues) == 1
    assert result.issues[0].status is OverlayValidationStatus.DOMAIN_VIOLATION


def test_validate_overlay_accepts_numeric_knob_in_range() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-numeric",
        changes={"policy_rollout_canary_percentage": 5.0},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is True
    assert "policy_rollout_canary_percentage" in result.validated_knob_ids


def test_validate_overlay_rejects_numeric_out_of_range() -> None:
    settings = _default_settings()
    settings.optimization.max_rollout_canary_percentage = 25.0
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-out-of-range",
        changes={"policy_rollout_canary_percentage": 99.0},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is False
    assert result.issues[0].status is OverlayValidationStatus.DOMAIN_VIOLATION


def test_validate_overlay_accepts_boolean_knob() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-bool",
        changes={"hybrid_prefer_local": False},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is True
    assert "hybrid_prefer_local" in result.validated_knob_ids


def test_validate_overlay_rejects_wrong_type_for_boolean() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-bad-bool",
        changes={"hybrid_prefer_local": "yes"},
    )

    result = validate_overlay(overlay, profile)

    assert result.valid is False
    assert result.issues[0].status is OverlayValidationStatus.DOMAIN_VIOLATION


# -- apply_overlay_to_settings --


def test_apply_overlay_does_not_mutate_original_settings() -> None:
    settings = _default_settings()
    original_policy = settings.default_routing_policy
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-apply",
        changes={"default_routing_policy": "latency_first"},
    )

    cloned = apply_overlay_to_settings(settings, overlay, profile)

    assert settings.default_routing_policy == original_policy
    assert cloned.default_routing_policy == RoutingPolicy.LATENCY_FIRST


def test_apply_overlay_raises_on_invalid_overlay() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-invalid",
        changes={"nonexistent_knob": "value"},
    )

    with pytest.raises(ValueError, match="invalid"):
        apply_overlay_to_settings(settings, overlay, profile)


def test_apply_overlay_applies_canary_percentage() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-canary",
        changes={"policy_rollout_canary_percentage": 5.0},
    )

    cloned = apply_overlay_to_settings(settings, overlay, profile)

    assert cloned.phase4.policy_rollout.canary_percentage == 5.0
    assert settings.phase4.policy_rollout.canary_percentage == 10.0


def test_apply_overlay_applies_hybrid_knob() -> None:
    settings = _default_settings()
    profile = build_optimization_profile(settings)
    overlay = ConfigOverlay(
        overlay_id="test-hybrid",
        changes={"hybrid_spillover_enabled": False},
    )

    cloned = apply_overlay_to_settings(settings, overlay, profile)

    assert cloned.phase7.hybrid_execution.spillover_enabled is False


# -- execute_campaign --


@pytest.mark.asyncio
async def test_execute_campaign_runs_baseline_and_produces_artifacts() -> None:
    settings = _default_settings()
    plan = CampaignExecutionPlan(
        plan_id="test-baseline",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        model_alias="mock-chat",
        request_count_per_scenario=5,
        seed=42,
    )

    result = await execute_campaign(settings=settings, plan=plan)

    assert result.plan_id == "test-baseline"
    assert len(result.baseline_artifacts) == 1
    assert result.baseline_artifacts[0].request_count == 5
    assert isinstance(result.campaign_result, ForgeStageAExecutionResult)
    campaign = result.campaign_result.campaign_artifact
    assert campaign.result_status in {
        OptimizationArtifactStatus.COMPLETE,
        OptimizationArtifactStatus.PARTIAL,
    }


@pytest.mark.asyncio
async def test_execute_campaign_with_routing_policy_overlay() -> None:
    settings = _default_settings()
    overlay = ConfigOverlay(
        overlay_id="try-latency-first",
        changes={"default_routing_policy": "latency_first"},
    )
    plan = CampaignExecutionPlan(
        plan_id="test-overlay",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        model_alias="mock-chat",
        request_count_per_scenario=5,
        seed=42,
        candidate_overlays=[overlay],
    )

    result = await execute_campaign(settings=settings, plan=plan)

    assert len(result.overlay_validation_results) == 1
    assert result.overlay_validation_results[0].valid is True
    campaign = result.campaign_result.campaign_artifact
    assert len(campaign.trials) >= 1


@pytest.mark.asyncio
async def test_execute_campaign_rejects_invalid_overlay() -> None:
    settings = _default_settings()
    overlay = ConfigOverlay(
        overlay_id="bad-overlay",
        changes={"nonexistent_knob": "bad_value"},
    )
    plan = CampaignExecutionPlan(
        plan_id="test-reject",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        model_alias="mock-chat",
        request_count_per_scenario=5,
        candidate_overlays=[overlay],
    )

    with pytest.raises(ValueError, match="invalid config overlays"):
        await execute_campaign(settings=settings, plan=plan)


@pytest.mark.asyncio
async def test_execute_campaign_multiple_workload_families() -> None:
    settings = _default_settings()
    plan = CampaignExecutionPlan(
        plan_id="test-multi-workload",
        workload_families=[
            WorkloadScenarioFamily.SHORT_CHAT,
            WorkloadScenarioFamily.LONG_PROMPT,
        ],
        model_alias="mock-chat",
        request_count_per_scenario=3,
        seed=99,
    )

    result = await execute_campaign(settings=settings, plan=plan)

    assert len(result.baseline_artifacts) == 2


@pytest.mark.asyncio
async def test_execute_campaign_with_candidate_generation() -> None:
    settings = _default_settings()
    plan = CampaignExecutionPlan(
        plan_id="test-candidate-gen",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        model_alias="mock-chat",
        request_count_per_scenario=5,
        seed=42,
        candidate_generation_config=OptimizationCandidateGenerationConfig(
            strategies=[
                OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
            ],
            allowed_knob_ids=["default_routing_policy"],
            seed=42,
        ),
    )

    result = await execute_campaign(settings=settings, plan=plan)

    assert result.campaign_result.candidate_generation is not None


# -- reproducibility metadata --


@pytest.mark.asyncio
async def test_execute_campaign_records_reproducibility_metadata() -> None:
    settings = _default_settings()
    plan = CampaignExecutionPlan(
        plan_id="test-repro",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        model_alias="mock-chat",
        request_count_per_scenario=3,
        seed=123,
    )

    result = await execute_campaign(settings=settings, plan=plan)

    repro = result.reproducibility
    assert repro.plan_id == "test-repro"
    assert repro.seed == 123
    assert repro.execution_mode is CampaignExecutionMode.OFFLINE_SYNTHETIC
    assert repro.settings_fingerprint
    assert len(repro.settings_fingerprint) == 16
    assert repro.optimization_profile_id
    assert repro.workload_families == ["short_chat"]
    assert repro.executed_at is not None


@pytest.mark.asyncio
async def test_execute_campaign_reproducibility_fingerprint_changes_with_policy() -> None:
    settings1 = _default_settings()
    settings2 = _default_settings()
    settings2.default_routing_policy = RoutingPolicy.LATENCY_FIRST

    plan = CampaignExecutionPlan(
        plan_id="test-fingerprint",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        request_count_per_scenario=3,
        seed=42,
    )

    result1 = await execute_campaign(settings=settings1, plan=plan)
    result2 = await execute_campaign(settings=settings2, plan=plan)

    assert (
        result1.reproducibility.settings_fingerprint
        != result2.reproducibility.settings_fingerprint
    )


# -- does not mutate caller settings --


@pytest.mark.asyncio
async def test_execute_campaign_does_not_mutate_caller_settings() -> None:
    settings = _default_settings()
    original_policies = settings.optimization.allowlisted_routing_policies
    overlay = ConfigOverlay(
        overlay_id="add-quality",
        changes={"default_routing_policy": "latency_first"},
    )
    plan = CampaignExecutionPlan(
        plan_id="test-no-mutate",
        workload_families=[WorkloadScenarioFamily.SHORT_CHAT],
        request_count_per_scenario=3,
        candidate_overlays=[overlay],
    )

    await execute_campaign(settings=settings, plan=plan)

    assert settings.optimization.allowlisted_routing_policies == original_policies
