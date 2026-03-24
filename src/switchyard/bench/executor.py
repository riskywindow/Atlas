"""Offline campaign executor for Forge Stage A.

Orchestrates end-to-end campaign runs: applies config overlays to an isolated
Settings copy, runs synthetic benchmarks against mock/local backends, feeds
the resulting artifacts through the existing campaign comparison pipeline, and
records reproducibility metadata.

The executor never mutates the caller's Settings instance. Config overlays are
validated against the typed optimization surface before execution.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from switchyard.bench.artifacts import (
    build_default_registry,
    run_synthetic_benchmark,
)
from switchyard.bench.campaigns import (
    ForgeStageAExecutionResult,
    execute_forge_stage_a_campaign,
)
from switchyard.bench.workloads import build_workload_manifest
from switchyard.config import Settings
from switchyard.optimization import build_optimization_profile
from switchyard.schemas.benchmark import (
    BenchmarkRunArtifact,
    BenchmarkScenario,
    WorkloadScenarioFamily,
)
from switchyard.schemas.optimization import (
    OptimizationCandidateGenerationConfig,
    OptimizationKnobSurface,
    OptimizationProfile,
)
from switchyard.schemas.routing import RoutingPolicy


class CampaignExecutionMode(StrEnum):
    """How the campaign executor should produce trial evidence."""

    OFFLINE_SYNTHETIC = "offline_synthetic"
    REPLAY_BACKED = "replay_backed"


class OverlayValidationStatus(StrEnum):
    """Result of validating one config overlay against the optimization surface."""

    VALID = "valid"
    UNKNOWN_KNOB = "unknown_knob"
    DOMAIN_VIOLATION = "domain_violation"
    SCOPE_VIOLATION = "scope_violation"


@dataclass(frozen=True, slots=True)
class OverlayValidationIssue:
    """One validation issue found in a config overlay."""

    knob_id: str
    status: OverlayValidationStatus
    detail: str


@dataclass(frozen=True, slots=True)
class ConfigOverlay:
    """An explicit, bounded set of config changes for a candidate trial.

    Each entry maps a knob_id (from the optimization surface) to a candidate
    value. The overlay is validated against the surface before execution.
    """

    overlay_id: str
    changes: dict[str, bool | int | float | str]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ConfigOverlayValidationResult:
    """Result of validating a ConfigOverlay against the optimization surface."""

    overlay_id: str
    valid: bool
    issues: list[OverlayValidationIssue]
    validated_knob_ids: list[str]


@dataclass(frozen=True, slots=True)
class CampaignExecutionPlan:
    """Describes what the campaign executor should run.

    Binds a workload set to a baseline config plus zero or more candidate
    config overlays. The executor runs benchmarks for the baseline, then
    feeds all artifacts through the existing offline comparison pipeline.
    """

    plan_id: str
    workload_families: list[WorkloadScenarioFamily]
    model_alias: str = "mock-chat"
    request_count_per_scenario: int = 10
    seed: int = 42
    execution_mode: CampaignExecutionMode = CampaignExecutionMode.OFFLINE_SYNTHETIC
    candidate_overlays: list[ConfigOverlay] = field(default_factory=list)
    candidate_generation_config: OptimizationCandidateGenerationConfig | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ReproducibilityRecord:
    """Reproducibility metadata recorded alongside campaign results."""

    plan_id: str
    execution_mode: CampaignExecutionMode
    settings_fingerprint: str
    optimization_profile_id: str
    overlay_ids: list[str]
    workload_families: list[str]
    seed: int
    git_sha: str | None
    executed_at: datetime
    notes: list[str]


@dataclass(frozen=True, slots=True)
class CampaignExecutionResult:
    """Full result bundle from one campaign executor run."""

    plan_id: str
    campaign_result: ForgeStageAExecutionResult
    baseline_artifacts: list[BenchmarkRunArtifact]
    overlay_validation_results: list[ConfigOverlayValidationResult]
    reproducibility: ReproducibilityRecord
    notes: list[str]


def validate_overlay(
    overlay: ConfigOverlay,
    profile: OptimizationProfile,
) -> ConfigOverlayValidationResult:
    """Validate a config overlay against the typed optimization surface.

    Returns a validation result indicating which knob changes are safe and
    which are rejected.
    """

    knob_by_id: dict[str, OptimizationKnobSurface] = {
        knob.knob_id: knob for knob in profile.knobs
    }
    issues: list[OverlayValidationIssue] = []
    validated: list[str] = []

    for knob_id, value in overlay.changes.items():
        knob = knob_by_id.get(knob_id)
        if knob is None:
            issues.append(
                OverlayValidationIssue(
                    knob_id=knob_id,
                    status=OverlayValidationStatus.UNKNOWN_KNOB,
                    detail=f"knob '{knob_id}' is not declared in the optimization surface",
                )
            )
            continue
        if not _value_in_domain(knob, value):
            issues.append(
                OverlayValidationIssue(
                    knob_id=knob_id,
                    status=OverlayValidationStatus.DOMAIN_VIOLATION,
                    detail=(
                        f"value {value!r} is outside the domain for knob '{knob_id}'"
                    ),
                )
            )
            continue
        validated.append(knob_id)

    return ConfigOverlayValidationResult(
        overlay_id=overlay.overlay_id,
        valid=len(issues) == 0,
        issues=issues,
        validated_knob_ids=validated,
    )


def apply_overlay_to_settings(
    settings: Settings,
    overlay: ConfigOverlay,
    profile: OptimizationProfile,
) -> Settings:
    """Apply a validated config overlay to a deep copy of Settings.

    Returns a new Settings instance with the overlay applied. The original
    Settings is never mutated. Only knobs declared in the optimization
    surface are applied.

    Raises ValueError if the overlay contains invalid knob changes.
    """

    validation = validate_overlay(overlay, profile)
    if not validation.valid:
        detail = "; ".join(issue.detail for issue in validation.issues)
        msg = f"config overlay '{overlay.overlay_id}' is invalid: {detail}"
        raise ValueError(msg)

    cloned = settings.model_copy(deep=True)
    knob_by_id = {knob.knob_id: knob for knob in profile.knobs}

    for knob_id, value in overlay.changes.items():
        knob = knob_by_id[knob_id]
        _apply_knob_to_settings(cloned, knob, value)

    return cloned


async def execute_campaign(
    *,
    settings: Settings,
    plan: CampaignExecutionPlan,
) -> CampaignExecutionResult:
    """Execute an end-to-end offline optimization campaign.

    1. Validates all candidate overlays against the optimization surface.
    2. Rejects plans with any invalid overlays.
    3. Runs synthetic benchmarks for the baseline workload scenarios.
    4. Applies each valid overlay to an isolated Settings copy and records
       the routing-policy candidate in the optimization profile.
    5. Feeds all artifacts into the existing Forge Stage A comparison pipeline.
    6. Returns the full result bundle with reproducibility metadata.
    """

    profile = build_optimization_profile(settings)
    timestamp = datetime.now(UTC)

    # Validate overlays
    overlay_validations = [
        validate_overlay(overlay, profile) for overlay in plan.candidate_overlays
    ]
    invalid_overlays = [v for v in overlay_validations if not v.valid]
    if invalid_overlays:
        detail = "; ".join(
            f"{v.overlay_id}: {', '.join(i.detail for i in v.issues)}"
            for v in invalid_overlays
        )
        msg = f"campaign plan contains invalid config overlays: {detail}"
        raise ValueError(msg)

    # Build effective settings with candidate routing policies from overlays
    effective_settings = _build_effective_settings(
        settings, plan.candidate_overlays, profile
    )

    # Run baseline synthetic benchmarks across all workload families
    baseline_artifacts = await _run_workload_benchmarks(
        settings=effective_settings,
        plan=plan,
        timestamp=timestamp,
    )

    # Feed artifacts into existing campaign pipeline
    campaign_result = execute_forge_stage_a_campaign(
        settings=effective_settings,
        evaluation_artifacts=baseline_artifacts,
        history_artifacts=baseline_artifacts,
        candidate_generation_config=plan.candidate_generation_config,
        timestamp=timestamp,
    )

    reproducibility = _build_reproducibility_record(
        plan=plan,
        settings=effective_settings,
        profile=profile,
        timestamp=timestamp,
    )

    notes = [
        "campaign execution used offline synthetic benchmarks with mock backends",
        "config overlays were validated against the typed optimization surface",
        "original settings were not mutated; overlays were applied to isolated copies",
    ]
    if plan.candidate_overlays:
        notes.append(
            f"{len(plan.candidate_overlays)} candidate overlay(s) were applied "
            "to the effective optimization profile"
        )

    return CampaignExecutionResult(
        plan_id=plan.plan_id,
        campaign_result=campaign_result,
        baseline_artifacts=baseline_artifacts,
        overlay_validation_results=overlay_validations,
        reproducibility=reproducibility,
        notes=notes,
    )


def _build_effective_settings(
    settings: Settings,
    overlays: Sequence[ConfigOverlay],
    profile: OptimizationProfile,
) -> Settings:
    """Build effective settings by folding overlay routing policies into the allowlist.

    Overlays that change `default_routing_policy` add their candidate policy to
    the allowlisted set so the campaign comparison pipeline can evaluate them.
    Other knob changes are recorded but do not mutate the baseline settings
    for the benchmark run itself; the existing counterfactual simulation
    handles policy comparison offline.
    """

    cloned = settings.model_copy(deep=True)
    extra_policies: list[RoutingPolicy] = []

    for overlay in overlays:
        policy_value = overlay.changes.get("default_routing_policy")
        if isinstance(policy_value, str):
            try:
                candidate_policy = RoutingPolicy(policy_value)
            except ValueError:
                continue
            if candidate_policy not in cloned.optimization.allowlisted_routing_policies:
                extra_policies.append(candidate_policy)

    if extra_policies:
        cloned.optimization.allowlisted_routing_policies = (
            *cloned.optimization.allowlisted_routing_policies,
            *extra_policies,
        )

    return cloned


async def _run_workload_benchmarks(
    *,
    settings: Settings,
    plan: CampaignExecutionPlan,
    timestamp: datetime,
) -> list[BenchmarkRunArtifact]:
    """Run synthetic benchmarks for each workload family in the plan."""

    registry = build_default_registry()
    artifacts: list[BenchmarkRunArtifact] = []

    for family in plan.workload_families:
        scenario = _build_scenario_for_family(
            family=family,
            model_alias=plan.model_alias,
            request_count=plan.request_count_per_scenario,
            seed=plan.seed,
            policy=settings.default_routing_policy,
        )
        artifact = await run_synthetic_benchmark(
            scenario=scenario,
            registry=registry,
            settings=settings,
            timestamp=timestamp,
        )
        artifacts.append(artifact)

    return artifacts


def _build_scenario_for_family(
    *,
    family: WorkloadScenarioFamily,
    model_alias: str,
    request_count: int,
    seed: int,
    policy: RoutingPolicy,
) -> BenchmarkScenario:
    """Build a benchmark scenario from a workload family specification."""

    workload = build_workload_manifest(
        family=family,
        model_alias=model_alias,
        request_count=request_count,
        seed=seed,
        policy=policy,
    )
    return BenchmarkScenario(
        name=workload.name,
        model=workload.model,
        model_alias=workload.model_alias,
        family=workload.family,
        policy=workload.policy,
        workload_shape=workload.workload_shape,
        request_count=workload.request_count,
        prompt_template=workload.prompt_template,
        workload_generation=workload.workload_generation,
        items=workload.items,
        scenario_seed=workload.scenario_seed,
    )


def _build_reproducibility_record(
    *,
    plan: CampaignExecutionPlan,
    settings: Settings,
    profile: OptimizationProfile,
    timestamp: datetime,
) -> ReproducibilityRecord:
    """Build reproducibility metadata for the campaign execution."""

    settings_payload = {
        "default_routing_policy": settings.default_routing_policy.value,
        "profile_id": profile.profile_id,
        "allowlisted_policies": [
            p.value for p in settings.optimization.allowlisted_routing_policies
        ],
        "objective": settings.optimization.objective.value,
    }
    settings_fingerprint = hashlib.sha256(
        json.dumps(settings_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return ReproducibilityRecord(
        plan_id=plan.plan_id,
        execution_mode=plan.execution_mode,
        settings_fingerprint=settings_fingerprint,
        optimization_profile_id=profile.profile_id,
        overlay_ids=[overlay.overlay_id for overlay in plan.candidate_overlays],
        workload_families=[f.value for f in plan.workload_families],
        seed=plan.seed,
        git_sha=_current_git_sha(),
        executed_at=timestamp,
        notes=[
            "reproducibility record captures the execution environment at campaign time",
            "settings fingerprint is derived from routing policy, profile, and objective",
        ],
    )


def _value_in_domain(
    knob: OptimizationKnobSurface,
    value: bool | int | float | str,
) -> bool:
    """Check whether a value falls within a knob's declared domain."""

    from switchyard.schemas.optimization import OptimizationKnobType

    if knob.knob_type is OptimizationKnobType.BOOLEAN:
        return isinstance(value, bool)
    if knob.knob_type is OptimizationKnobType.ENUM:
        return isinstance(value, str) and value in knob.allowed_values
    if knob.knob_type is OptimizationKnobType.INTEGER:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if knob.min_value is not None and value < int(knob.min_value):
            return False
        if knob.max_value is not None and value > int(knob.max_value):
            return False
        return True
    if knob.knob_type is OptimizationKnobType.FLOAT:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        numeric = float(value)
        if knob.min_value is not None and numeric < float(knob.min_value):
            return False
        if knob.max_value is not None and numeric > float(knob.max_value):
            return False
        return True
    return False


def _apply_knob_to_settings(
    settings: Settings,
    knob: OptimizationKnobSurface,
    value: bool | int | float | str,
) -> None:
    """Apply a single knob change to a Settings instance in-place.

    Only touches the knob's declared config_path. Unsupported paths are
    silently skipped to keep the overlay application bounded.
    """

    path = knob.config_path
    _KNOB_APPLIERS = {
        "default_routing_policy": lambda s, v: setattr(
            s, "default_routing_policy", RoutingPolicy(v)
        ),
        "phase4.policy_rollout.mode": lambda s, v: setattr(
            s.phase4.policy_rollout, "mode", __import__(
                "switchyard.schemas.routing", fromlist=["PolicyRolloutMode"]
            ).PolicyRolloutMode(v)
        ),
        "phase4.policy_rollout.canary_percentage": lambda s, v: setattr(
            s.phase4.policy_rollout, "canary_percentage", float(v)
        ),
        "phase4.shadow_routing.default_sampling_rate": lambda s, v: setattr(
            s.phase4.shadow_routing, "default_sampling_rate", float(v)
        ),
        "phase4.canary_routing.default_percentage": lambda s, v: setattr(
            s.phase4.canary_routing, "default_percentage", float(v)
        ),
        "phase4.admission_control.global_concurrency_cap": lambda s, v: setattr(
            s.phase4.admission_control, "global_concurrency_cap", int(v)
        ),
        "phase4.admission_control.global_queue_size": lambda s, v: setattr(
            s.phase4.admission_control, "global_queue_size", int(v)
        ),
        "phase4.admission_control.queue_timeout_seconds": lambda s, v: setattr(
            s.phase4.admission_control, "queue_timeout_seconds", float(v)
        ),
        "phase4.circuit_breakers.failure_threshold": lambda s, v: setattr(
            s.phase4.circuit_breakers, "failure_threshold", int(v)
        ),
        "phase4.circuit_breakers.open_cooldown_seconds": lambda s, v: setattr(
            s.phase4.circuit_breakers, "open_cooldown_seconds", float(v)
        ),
        "phase4.session_affinity.ttl_seconds": lambda s, v: setattr(
            s.phase4.session_affinity, "ttl_seconds", float(v)
        ),
        "phase7.hybrid_execution.prefer_local": lambda s, v: setattr(
            s.phase7.hybrid_execution, "prefer_local", bool(v)
        ),
        "phase7.hybrid_execution.spillover_enabled": lambda s, v: setattr(
            s.phase7.hybrid_execution, "spillover_enabled", bool(v)
        ),
        "phase7.hybrid_execution.max_remote_share_percent": lambda s, v: setattr(
            s.phase7.hybrid_execution, "max_remote_share_percent", float(v)
        ),
        "phase7.hybrid_execution.remote_request_budget_per_minute": lambda s, v: setattr(
            s.phase7.hybrid_execution, "remote_request_budget_per_minute", int(v)
        ),
        "phase7.hybrid_execution.remote_concurrency_cap": lambda s, v: setattr(
            s.phase7.hybrid_execution, "remote_concurrency_cap", int(v)
        ),
        "phase7.hybrid_execution.remote_cooldown_seconds": lambda s, v: setattr(
            s.phase7.hybrid_execution, "remote_cooldown_seconds", float(v)
        ),
    }

    applier = _KNOB_APPLIERS.get(path)
    if applier is not None:
        applier(settings, value)


def _current_git_sha() -> str | None:
    """Return the current git SHA if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None
