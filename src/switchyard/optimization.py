"""Helpers for exporting optimization-ready control-plane surfaces."""

from __future__ import annotations

import hashlib
import json

from switchyard.config import Settings
from switchyard.schemas.benchmark import (
    BenchmarkConfigFingerprint,
    BenchmarkConfigKnob,
    BenchmarkConfigKnobCategory,
    BenchmarkConfigSnapshot,
    BenchmarkRunConfig,
    CounterfactualObjective,
    WorkloadScenarioFamily,
)
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    ForgePromotionPlan,
    ForgeStageACampaign,
    ForgeTrialLineage,
    ForgeTrialRole,
    OptimizationCampaignMetadata,
    OptimizationCandidateConfigurationArtifact,
    OptimizationComparisonOperator,
    OptimizationConfigProfile,
    OptimizationConfigProfileChange,
    OptimizationConfigProfileDiff,
    OptimizationConfigProfileProvenance,
    OptimizationConfigProfileRole,
    OptimizationConfigProfileSourceKind,
    OptimizationConfigProfileValidation,
    OptimizationConfigProfileValidationIssue,
    OptimizationConfigProfileValidationIssueKind,
    OptimizationConstraint,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationDomainKind,
    OptimizationEvidenceProfile,
    OptimizationExcludedDimension,
    OptimizationGoal,
    OptimizationKnobDomain,
    OptimizationKnobGroup,
    OptimizationKnobSurface,
    OptimizationKnobType,
    OptimizationObjectiveMetric,
    OptimizationObjectiveTarget,
    OptimizationProfile,
    OptimizationRecommendationDisposition,
    OptimizationScope,
    OptimizationScopeKind,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
    OptimizationWorkloadSet,
    OptimizationWorkloadSourceKind,
)
from switchyard.schemas.routing import PolicyRolloutMode, RequestClass, WorkloadTag


def build_optimization_profile(settings: Settings) -> OptimizationProfile:
    """Build a typed snapshot of tunable routing knobs for Phase 9 Stage A workflows."""

    optimization = settings.optimization
    rollout = settings.phase4.policy_rollout
    hybrid = settings.phase7.hybrid_execution

    notes = [
        "profile is informational and does not mutate live routing behavior",
        "Phase 9 Stage A should pair this profile with benchmark, replay, and simulation artifacts",
    ]
    if optimization.promotion_requires_operator_review:
        notes.append("optimized policies still require explicit operator review before promotion")
    if hybrid.enabled and hybrid.prefer_local:
        notes.append(
            "local-first posture remains the default even when remote execution is enabled"
        )
    workload_sets = _build_workload_sets(settings)
    return OptimizationProfile(
        profile_id=optimization.profile_id,
        active_routing_policy=settings.default_routing_policy,
        active_rollout_mode=rollout.mode,
        allowlisted_routing_policies=list(optimization.allowlisted_routing_policies),
        allowlisted_rollout_modes=list(optimization.allowlisted_rollout_modes),
        candidate_policy_id=rollout.candidate_policy_id,
        shadow_policy_id=rollout.shadow_policy_id,
        hybrid_remote_enabled=hybrid.enabled,
        worker_launch_presets=list(optimization.worker_launch_presets),
        evidence=OptimizationEvidenceProfile(
            objective=optimization.objective,
            min_evidence_count=optimization.min_evidence_count,
            max_predicted_error_rate=optimization.max_predicted_error_rate,
            max_predicted_latency_regression_ms=optimization.max_predicted_latency_regression_ms,
            require_observed_backend_evidence=optimization.require_observed_backend_evidence,
            promotion_requires_operator_review=optimization.promotion_requires_operator_review,
            notes=[
                (
                    "evidence thresholds are intended for offline policy comparison and "
                    "recommendation flows"
                )
            ],
        ),
        knobs=_build_knob_surfaces(settings),
        objectives=_build_objectives(settings, workload_sets=workload_sets),
        constraints=_build_constraints(settings),
        workload_sets=workload_sets,
        campaign=_build_campaign_metadata(
            settings=settings,
            profile_id=optimization.profile_id,
            workload_sets=workload_sets,
        ),
        baseline_trial=_build_baseline_trial(settings=settings),
        candidate_trials=_build_candidate_trials(settings=settings),
        excluded_dimensions=_build_excluded_dimensions(),
        notes=notes,
    )


def build_baseline_optimization_config_profile(
    settings: Settings,
    *,
    profile_version: int = 1,
) -> OptimizationConfigProfile:
    """Build the explicit baseline config profile from the current optimization surface."""

    profile = build_optimization_profile(settings)
    baseline_trial = profile.baseline_trial
    if baseline_trial is None:
        msg = "optimization profile must expose baseline_trial before building a config profile"
        raise ValueError(msg)
    return OptimizationConfigProfile(
        config_profile_id=baseline_trial.config_profile_id,
        profile_version=profile_version,
        profile_role=OptimizationConfigProfileRole.BASELINE,
        optimization_profile_id=profile.profile_id,
        baseline_config_profile_id=baseline_trial.config_profile_id,
        applies_to=list(baseline_trial.applies_to),
        candidate_kind=baseline_trial.candidate_kind,
        routing_policy=profile.active_routing_policy,
        rollout_policy_id=profile.candidate_policy_id,
        changes=[],
        validation=OptimizationConfigProfileValidation(
            compatible=True,
            notes=["baseline profile does not override any exported tunable knobs"],
        ),
        diff=OptimizationConfigProfileDiff(
            baseline_config_profile_id=baseline_trial.config_profile_id,
            config_profile_id=baseline_trial.config_profile_id,
            profile_scope=list(baseline_trial.applies_to),
            notes=["baseline-versus-baseline diff is empty by design"],
        ),
        provenance=OptimizationConfigProfileProvenance(
            source_kind=OptimizationConfigProfileSourceKind.SETTINGS_BASELINE,
            optimization_profile_id=profile.profile_id,
            baseline_config_profile_id=baseline_trial.config_profile_id,
            candidate_id=baseline_trial.candidate_id,
            candidate_kind=baseline_trial.candidate_kind,
            notes=[
                "profile is derived from the current resolved settings and optimization surface"
            ],
        ),
        notes=["baseline profile anchors later candidate or promoted profile diffs"],
    )


def build_candidate_optimization_config_profile(
    *,
    settings: Settings,
    candidate_configuration: OptimizationCandidateConfigurationArtifact,
    profile_version: int = 1,
) -> OptimizationConfigProfile:
    """Build an explicit candidate config profile from one candidate artifact."""

    return _build_optimization_config_profile(
        settings=settings,
        candidate_configuration=candidate_configuration,
        trial_artifact=None,
        campaign_artifact_id=None,
        profile_version=profile_version,
    )


def build_trial_optimization_config_profile(
    *,
    settings: Settings,
    trial_artifact: OptimizationTrialArtifact,
    campaign_artifact_id: str | None = None,
    profile_version: int = 1,
) -> OptimizationConfigProfile:
    """Build an explicit promotion-ready config profile from one reviewed trial."""

    return _build_optimization_config_profile(
        settings=settings,
        candidate_configuration=trial_artifact.candidate_configuration,
        trial_artifact=trial_artifact,
        campaign_artifact_id=campaign_artifact_id,
        profile_version=profile_version,
    )


def _build_optimization_config_profile(
    *,
    settings: Settings,
    candidate_configuration: OptimizationCandidateConfigurationArtifact,
    trial_artifact: OptimizationTrialArtifact | None,
    campaign_artifact_id: str | None,
    profile_version: int,
) -> OptimizationConfigProfile:
    profile = build_optimization_profile(settings)
    baseline_profile = build_baseline_optimization_config_profile(
        settings,
        profile_version=1,
    )
    validation_issues: list[OptimizationConfigProfileValidationIssue] = []
    if (
        trial_artifact is not None
        and trial_artifact.candidate_configuration.candidate_configuration_id
        != candidate_configuration.candidate_configuration_id
    ):
        validation_issues.append(
            OptimizationConfigProfileValidationIssue(
                issue_kind=OptimizationConfigProfileValidationIssueKind.PROVENANCE_MISMATCH,
                detail="trial_artifact.candidate_configuration must match candidate_configuration",
            )
        )
    if (
        candidate_configuration.baseline_config_profile_id
        != baseline_profile.config_profile_id
    ):
        validation_issues.append(
            OptimizationConfigProfileValidationIssue(
                issue_kind=OptimizationConfigProfileValidationIssueKind.PROVENANCE_MISMATCH,
                detail=(
                    "candidate baseline_config_profile_id does not match the "
                    "current resolved baseline profile"
                ),
            )
        )
    if (
        candidate_configuration.eligibility is not None
        and not candidate_configuration.eligibility.eligible
    ):
        validation_issues.append(
            OptimizationConfigProfileValidationIssue(
                issue_kind=OptimizationConfigProfileValidationIssueKind.ELIGIBILITY_BLOCKED,
                detail=(
                    "candidate eligibility blocked safe profile materialization: "
                    + ", ".join(
                        candidate_configuration.eligibility.rejection_reasons
                        or ["unknown"]
                    )
                ),
            )
        )

    changes: list[OptimizationConfigProfileChange] = []
    validated_knob_ids: list[str] = []
    surface_by_id = {knob.knob_id: knob for knob in profile.knobs}
    profile_scope = list(candidate_configuration.candidate.applies_to)
    for change in candidate_configuration.knob_changes:
        knob = surface_by_id.get(change.knob_id)
        if knob is None:
            validation_issues.append(
                OptimizationConfigProfileValidationIssue(
                    issue_kind=OptimizationConfigProfileValidationIssueKind.UNKNOWN_KNOB,
                    knob_id=change.knob_id,
                    detail=(
                        "candidate touches a knob that is not declared in the "
                        "optimization surface"
                    ),
                )
            )
            continue
        if change.config_path != knob.config_path:
            validation_issues.append(
                OptimizationConfigProfileValidationIssue(
                    issue_kind=OptimizationConfigProfileValidationIssueKind.CONFIG_PATH_MISMATCH,
                    knob_id=change.knob_id,
                    detail="candidate config_path does not match the declared knob config_path",
                )
            )
        if not _scopes_within(change.applies_to, knob.applies_to):
            validation_issues.append(
                OptimizationConfigProfileValidationIssue(
                    issue_kind=OptimizationConfigProfileValidationIssueKind.SCOPE_NOT_ALLOWED,
                    knob_id=change.knob_id,
                    detail="candidate scope exceeds the declared knob scope boundary",
                )
            )
        if not _scopes_within(change.applies_to, profile_scope):
            validation_issues.append(
                OptimizationConfigProfileValidationIssue(
                    issue_kind=OptimizationConfigProfileValidationIssueKind.SCOPE_NOT_ALLOWED,
                    knob_id=change.knob_id,
                    detail="knob change scope exceeds the enclosing config profile scope",
                )
            )
        if not _value_within_knob_domain(knob=knob, value=change.candidate_value):
            validation_issues.append(
                OptimizationConfigProfileValidationIssue(
                    issue_kind=OptimizationConfigProfileValidationIssueKind.DOMAIN_VIOLATION,
                    knob_id=change.knob_id,
                    detail="candidate value is outside the exported knob domain",
                )
            )
        if not any(
            issue.knob_id == change.knob_id
            for issue in validation_issues
            if issue.issue_kind
            in {
                OptimizationConfigProfileValidationIssueKind.CONFIG_PATH_MISMATCH,
                OptimizationConfigProfileValidationIssueKind.SCOPE_NOT_ALLOWED,
                OptimizationConfigProfileValidationIssueKind.DOMAIN_VIOLATION,
            }
        ):
            validated_knob_ids.append(change.knob_id)
        changes.append(
            OptimizationConfigProfileChange(
                knob_id=change.knob_id,
                config_path=change.config_path,
                group=knob.group,
                knob_type=knob.knob_type,
                applies_to=list(change.applies_to),
                supported_scopes=list(knob.applies_to),
                baseline_value=change.baseline_value,
                candidate_value=change.candidate_value,
                mutable_at_runtime=knob.mutable_at_runtime,
                notes=[
                    *list(change.notes),
                    *list(knob.notes),
                ],
            )
        )

    return OptimizationConfigProfile(
        config_profile_id=candidate_configuration.config_profile_id,
        profile_version=profile_version,
        profile_role=_config_profile_role(trial_artifact),
        optimization_profile_id=profile.profile_id,
        baseline_config_profile_id=candidate_configuration.baseline_config_profile_id,
        applies_to=profile_scope,
        candidate_kind=candidate_configuration.candidate.candidate_kind,
        routing_policy=candidate_configuration.candidate.routing_policy,
        rollout_policy_id=candidate_configuration.candidate.rollout_policy_id,
        worker_launch_preset=candidate_configuration.candidate.worker_launch_preset,
        changes=changes,
        validation=OptimizationConfigProfileValidation(
            validated_knob_ids=validated_knob_ids,
            issues=validation_issues,
            notes=[
                (
                    "config profiles only materialize declared optimization knobs; "
                    "unknown or out-of-scope changes remain incompatible"
                )
            ],
        ),
        diff=_config_profile_diff(
            baseline_config_profile_id=candidate_configuration.baseline_config_profile_id,
            config_profile_id=candidate_configuration.config_profile_id,
            changes=changes,
            profile_scope=profile_scope,
        ),
        provenance=_config_profile_provenance(
            profile=profile,
            candidate_configuration=candidate_configuration,
            trial_artifact=trial_artifact,
            campaign_artifact_id=campaign_artifact_id,
        ),
        notes=_config_profile_notes(
            trial_artifact=trial_artifact,
            validation_issues=validation_issues,
        ),
    )


def build_forge_stage_a_campaign(settings: Settings) -> ForgeStageACampaign:
    """Build the first typed Forge Stage A campaign inspection snapshot."""

    profile = build_optimization_profile(settings)
    evaluation_sources = _forge_evaluation_sources()
    required_evaluation_sources = _forge_required_evaluation_sources(
        require_observed_backend_evidence=profile.evidence.require_observed_backend_evidence
    )
    recommended_rollout_mode = _recommended_forge_rollout_mode(profile=profile)
    baseline_trial_id = _forge_trial_id(
        profile_id=profile.profile_id,
        role=ForgeTrialRole.BASELINE,
        label=profile.active_routing_policy.value,
    )
    trial_lineage = [
        ForgeTrialLineage(
            trial_id=baseline_trial_id,
            parent_trial_id=None,
            trial_role=ForgeTrialRole.BASELINE,
            candidate_kind=ForgeCandidateKind.CONFIG_PROFILE,
            config_profile_id=_forge_config_profile_id(
                profile_id=profile.profile_id,
                label="baseline",
            ),
            routing_policy=profile.active_routing_policy,
            rollout_mode=profile.active_rollout_mode,
            evaluation_sources=evaluation_sources,
            required_evaluation_sources=required_evaluation_sources,
            explainable_recommendations_only=True,
            notes=[
                "baseline trial captures the active config profile before any tuning campaign",
                "observed runtime, replay, and simulation evidence stay separated by source",
            ],
        )
    ]
    if profile.candidate_policy_id is not None:
        trial_lineage.append(
            ForgeTrialLineage(
                trial_id=_forge_trial_id(
                    profile_id=profile.profile_id,
                    role=ForgeTrialRole.CANDIDATE,
                    label=profile.candidate_policy_id,
                ),
                parent_trial_id=baseline_trial_id,
                trial_role=ForgeTrialRole.CANDIDATE,
                candidate_kind=ForgeCandidateKind.ROLLOUT_POLICY,
                config_profile_id=_forge_config_profile_id(
                    profile_id=profile.profile_id,
                    label=profile.candidate_policy_id,
                ),
                rollout_policy_id=profile.candidate_policy_id,
                rollout_mode=recommended_rollout_mode,
                evaluation_sources=evaluation_sources,
                required_evaluation_sources=required_evaluation_sources,
                explainable_recommendations_only=True,
                notes=[
                    "candidate rollout policy remains inspectable before any live promotion",
                    "promotion should stay behind operator-reviewed rollout controls",
                ],
            )
        )
    for routing_policy in profile.allowlisted_routing_policies:
        if routing_policy == profile.active_routing_policy:
            continue
        trial_lineage.append(
            ForgeTrialLineage(
                trial_id=_forge_trial_id(
                    profile_id=profile.profile_id,
                    role=ForgeTrialRole.CANDIDATE,
                    label=routing_policy.value,
                ),
                parent_trial_id=baseline_trial_id,
                trial_role=ForgeTrialRole.CANDIDATE,
                candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
                config_profile_id=_forge_config_profile_id(
                    profile_id=profile.profile_id,
                    label=routing_policy.value,
                ),
                routing_policy=routing_policy,
                rollout_mode=recommended_rollout_mode,
                evaluation_sources=evaluation_sources,
                required_evaluation_sources=required_evaluation_sources,
                explainable_recommendations_only=True,
                notes=[
                    "routing-policy candidate should be compared offline before canary exposure",
                    "candidate generation remains backend-agnostic and Mac-first compatible",
                ],
            )
        )
    promotion = ForgePromotionPlan(
        config_profile_id=_forge_config_profile_id(
            profile_id=profile.profile_id,
            label="promotion-canary",
        ),
        rollout_mode=recommended_rollout_mode,
        max_canary_percentage=settings.optimization.max_rollout_canary_percentage,
        requires_operator_review=profile.evidence.promotion_requires_operator_review,
        reversible_controls=[
            "phase4.policy_rollout.mode",
            "phase4.policy_rollout.kill_switch_enabled",
            "/admin/policy-rollout",
            "/admin/policy-rollout/reset",
        ],
        notes=[
            "promotion is a config-profile and canary step, not an automatic cutover",
            "Forge Stage B kernel generation remains out of scope for this campaign",
        ],
    )
    return ForgeStageACampaign(
        campaign_id=_forge_campaign_id(profile_id=profile.profile_id),
        optimization_profile_id=profile.profile_id,
        objective=profile.evidence.objective,
        active_routing_policy=profile.active_routing_policy,
        active_rollout_mode=profile.active_rollout_mode,
        baseline_trial_id=baseline_trial_id,
        evaluation_sources=evaluation_sources,
        required_evaluation_sources=required_evaluation_sources,
        candidate_count=max(0, len(trial_lineage) - 1),
        explainable_recommendations_only=True,
        automatic_promotion_enabled=False,
        promotion=promotion,
        trial_lineage=trial_lineage,
        notes=[
            "campaign inspection is read-only and does not create or mutate trials",
            "observed runtime evidence is kept distinct from replayed or simulated evidence",
            "Stage A covers recommendation and safe promotion planning only",
        ],
    )


def attach_benchmark_config_snapshot(
    *,
    settings: Settings,
    run_config: BenchmarkRunConfig,
) -> BenchmarkRunConfig:
    """Attach a canonical immutable config snapshot and fingerprint to one run config."""

    snapshot = build_benchmark_config_snapshot(settings=settings, run_config=run_config)
    return run_config.model_copy(
        update={
            "immutable_config": snapshot,
            "config_fingerprint": snapshot.fingerprint,
        },
        deep=True,
    )


def build_benchmark_config_snapshot(
    *,
    settings: Settings,
    run_config: BenchmarkRunConfig,
) -> BenchmarkConfigSnapshot:
    """Build the bounded benchmark-facing configuration truth for one run."""

    profile = build_optimization_profile(settings)
    knobs = _benchmark_runner_knobs(run_config) + _settings_knobs(settings)
    notes = [
        "captures bounded benchmark-relevant control-plane and worker-launch knobs",
        "batching knobs are absent because the current architecture does not expose them",
    ]
    return BenchmarkConfigSnapshot(
        profile_id=profile.profile_id,
        fingerprint=_fingerprint_benchmark_knobs(
            profile_id=profile.profile_id,
            knobs=knobs,
            notes=notes,
        ),
        knobs=knobs,
        notes=notes,
    )


def _config_profile_role(
    trial_artifact: OptimizationTrialArtifact | None,
) -> OptimizationConfigProfileRole:
    if (
        trial_artifact is not None
        and trial_artifact.recommendation_summary is not None
        and trial_artifact.recommendation_summary.disposition
        is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    ):
        return OptimizationConfigProfileRole.PROMOTED
    return OptimizationConfigProfileRole.CANDIDATE


def _config_profile_provenance(
    *,
    profile: OptimizationProfile,
    candidate_configuration: OptimizationCandidateConfigurationArtifact,
    trial_artifact: OptimizationTrialArtifact | None,
    campaign_artifact_id: str | None,
) -> OptimizationConfigProfileProvenance:
    evidence_kinds = (
        []
        if trial_artifact is None
        else sorted(
            {
                record.evidence_kind for record in trial_artifact.evidence_records
            },
            key=lambda item: item.value,
        )
    )
    return OptimizationConfigProfileProvenance(
        source_kind=(
            OptimizationConfigProfileSourceKind.REVIEWED_TRIAL
            if trial_artifact is not None
            else OptimizationConfigProfileSourceKind.CANDIDATE_CONFIGURATION
        ),
        optimization_profile_id=profile.profile_id,
        baseline_config_profile_id=candidate_configuration.baseline_config_profile_id,
        parent_config_profile_id=candidate_configuration.baseline_config_profile_id,
        campaign_id=(
            candidate_configuration.campaign_id
            if trial_artifact is None
            else trial_artifact.campaign_id
        ),
        campaign_artifact_id=campaign_artifact_id,
        trial_artifact_id=(
            None if trial_artifact is None else trial_artifact.trial_artifact_id
        ),
        candidate_configuration_id=candidate_configuration.candidate_configuration_id,
        candidate_id=candidate_configuration.candidate.candidate_id,
        candidate_kind=candidate_configuration.candidate.candidate_kind,
        recommendation_summary_id=(
            None
            if trial_artifact is None or trial_artifact.recommendation_summary is None
            else trial_artifact.recommendation_summary.recommendation_summary_id
        ),
        recommendation_disposition=(
            None
            if trial_artifact is None or trial_artifact.recommendation_summary is None
            else trial_artifact.recommendation_summary.disposition
        ),
        promotion_decision_id=(
            None
            if trial_artifact is None or trial_artifact.promotion_decision is None
            else trial_artifact.promotion_decision.promotion_decision_id
        ),
        evidence_kinds=evidence_kinds,
        notes=[
            (
                "profile provenance stays attached to the originating campaign and "
                "candidate configuration instead of relying on ad hoc operator notes"
            )
        ],
    )


def _config_profile_diff(
    *,
    baseline_config_profile_id: str,
    config_profile_id: str,
    changes: list[OptimizationConfigProfileChange],
    profile_scope: list[OptimizationScope],
) -> OptimizationConfigProfileDiff:
    return OptimizationConfigProfileDiff(
        baseline_config_profile_id=baseline_config_profile_id,
        config_profile_id=config_profile_id,
        changed_knob_ids=sorted(change.knob_id for change in changes),
        changed_groups=sorted(
            {change.group for change in changes},
            key=lambda item: item.value,
        ),
        mutable_runtime_knob_ids=sorted(
            change.knob_id for change in changes if change.mutable_at_runtime
        ),
        immutable_knob_ids=sorted(
            change.knob_id for change in changes if not change.mutable_at_runtime
        ),
        profile_scope=list(profile_scope),
        notes=[
            "diff is derived only from declared knob changes, not hidden process state",
        ],
    )


def _config_profile_notes(
    *,
    trial_artifact: OptimizationTrialArtifact | None,
    validation_issues: list[OptimizationConfigProfileValidationIssue],
) -> list[str]:
    notes = [
        "profile only captures declared tunable knobs and explicit scope",
    ]
    if validation_issues:
        notes.append("profile is incompatible until validation issues are resolved")
    if (
        trial_artifact is not None
        and trial_artifact.recommendation_summary is not None
        and trial_artifact.recommendation_summary.disposition
        is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    ):
        notes.append("trial recommendation marked this candidate promotion-eligible")
    return notes


def _scopes_within(
    candidate_scopes: list[OptimizationScope],
    allowed_scopes: list[OptimizationScope],
) -> bool:
    if not candidate_scopes:
        return True
    candidate_keys = _scope_keys(candidate_scopes)
    allowed_keys = _scope_keys(allowed_scopes)
    if (OptimizationScopeKind.GLOBAL.value, None) in allowed_keys:
        return candidate_keys == {(OptimizationScopeKind.GLOBAL.value, None)}
    return candidate_keys.issubset(allowed_keys)


def _scope_keys(scopes: list[OptimizationScope]) -> set[tuple[str, str | None]]:
    return {(scope.kind.value, scope.target) for scope in scopes}


def _value_within_knob_domain(
    *,
    knob: OptimizationKnobSurface,
    value: bool | int | float | str | list[str] | None,
) -> bool:
    if value is None:
        return knob.domain.nullable
    if knob.knob_type is OptimizationKnobType.BOOLEAN:
        return isinstance(value, bool)
    if knob.knob_type is OptimizationKnobType.ENUM:
        return isinstance(value, str) and value in knob.allowed_values
    if knob.knob_type is OptimizationKnobType.STRING_LIST:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if knob.knob_type is OptimizationKnobType.INTEGER:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if knob.min_value is not None and value < int(knob.min_value):
            return False
        if knob.max_value is not None and value > int(knob.max_value):
            return False
        return True
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    if knob.min_value is not None and numeric < float(knob.min_value):
        return False
    if knob.max_value is not None and numeric > float(knob.max_value):
        return False
    return True


def _build_knob_surfaces(settings: Settings) -> list[OptimizationKnobSurface]:
    optimization = settings.optimization
    rollout = settings.phase4.policy_rollout
    canary = settings.phase4.canary_routing
    shadow = settings.phase4.shadow_routing
    admission = settings.phase4.admission_control
    circuit = settings.phase4.circuit_breakers
    session_affinity = settings.phase4.session_affinity
    hybrid = settings.phase7.hybrid_execution
    return [
        OptimizationKnobSurface(
            knob_id="default_routing_policy",
            config_path="default_routing_policy",
            group=OptimizationKnobGroup.ROUTING_POLICY,
            knob_type=OptimizationKnobType.ENUM,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.ENUM,
                allowed_values=[
                    policy.value for policy in optimization.allowlisted_routing_policies
                ],
            ),
            current_value=settings.default_routing_policy.value,
            mutable_at_runtime=False,
            notes=["baseline compatibility policy for live routing"],
        ),
        OptimizationKnobSurface(
            knob_id="policy_rollout_mode",
            config_path="phase4.policy_rollout.mode",
            group=OptimizationKnobGroup.POLICY_ROLLOUT,
            knob_type=OptimizationKnobType.ENUM,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.ENUM,
                allowed_values=[mode.value for mode in optimization.allowlisted_rollout_modes],
            ),
            current_value=rollout.mode.value,
            mutable_at_runtime=True,
            notes=["runtime rollout controller may override this safely through the admin surface"],
        ),
        OptimizationKnobSurface(
            knob_id="policy_rollout_canary_percentage",
            config_path="phase4.policy_rollout.canary_percentage",
            group=OptimizationKnobGroup.POLICY_ROLLOUT,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=optimization.max_rollout_canary_percentage,
            ),
            current_value=rollout.canary_percentage,
            mutable_at_runtime=True,
            notes=["caps later policy promotion experiments to bounded slices"],
        ),
        OptimizationKnobSurface(
            knob_id="shadow_sampling_rate",
            config_path="phase4.shadow_routing.default_sampling_rate",
            group=OptimizationKnobGroup.POLICY_ROLLOUT,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=optimization.max_shadow_sampling_rate,
            ),
            current_value=shadow.default_sampling_rate,
            mutable_at_runtime=False,
            notes=["non-binding observational traffic only"],
        ),
        OptimizationKnobSurface(
            knob_id="canary_default_percentage",
            config_path="phase4.canary_routing.default_percentage",
            group=OptimizationKnobGroup.POLICY_ROLLOUT,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=optimization.max_rollout_canary_percentage,
            ),
            current_value=canary.default_percentage,
            mutable_at_runtime=False,
            notes=["applies to backend canaries rather than scorer rollout"],
        ),
        OptimizationKnobSurface(
            knob_id="admission_global_concurrency_cap",
            config_path="phase4.admission_control.global_concurrency_cap",
            group=OptimizationKnobGroup.ADMISSION_CONTROL,
            knob_type=OptimizationKnobType.INTEGER,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.INTEGER_RANGE,
                min_value=1,
                max_value=optimization.max_global_concurrency_cap,
            ),
            current_value=admission.global_concurrency_cap,
            mutable_at_runtime=False,
            notes=["kept explicit so offline tuning does not ignore overload posture"],
        ),
        OptimizationKnobSurface(
            knob_id="admission_global_queue_size",
            config_path="phase4.admission_control.global_queue_size",
            group=OptimizationKnobGroup.ADMISSION_CONTROL,
            knob_type=OptimizationKnobType.INTEGER,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.INTEGER_RANGE,
                min_value=0,
                max_value=optimization.max_global_concurrency_cap,
            ),
            current_value=admission.global_queue_size,
            mutable_at_runtime=False,
            notes=["queue depth affects overload shedding behavior under sustained load"],
        ),
        OptimizationKnobSurface(
            knob_id="admission_queue_timeout_seconds",
            config_path="phase4.admission_control.queue_timeout_seconds",
            group=OptimizationKnobGroup.ADMISSION_CONTROL,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.1,
                max_value=3600.0,
            ),
            current_value=admission.queue_timeout_seconds,
            mutable_at_runtime=False,
            notes=["bounded queue wait time trades latency tail against shed rate"],
        ),
        OptimizationKnobSurface(
            knob_id="circuit_failure_threshold",
            config_path="phase4.circuit_breakers.failure_threshold",
            group=OptimizationKnobGroup.BACKEND_PROTECTION,
            knob_type=OptimizationKnobType.INTEGER,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.INTEGER_RANGE,
                min_value=1,
                max_value=1000,
            ),
            current_value=circuit.failure_threshold,
            mutable_at_runtime=False,
            notes=["lower thresholds trip faster but risk false positives under burst errors"],
        ),
        OptimizationKnobSurface(
            knob_id="circuit_open_cooldown_seconds",
            config_path="phase4.circuit_breakers.open_cooldown_seconds",
            group=OptimizationKnobGroup.BACKEND_PROTECTION,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=optimization.max_circuit_open_cooldown_seconds,
            ),
            current_value=circuit.open_cooldown_seconds,
            mutable_at_runtime=False,
            notes=["backend protection remains outside the HTTP layer"],
        ),
        OptimizationKnobSurface(
            knob_id="session_affinity_ttl_seconds",
            config_path="phase4.session_affinity.ttl_seconds",
            group=OptimizationKnobGroup.SESSION_AFFINITY,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=86_400.0,
            ),
            current_value=session_affinity.ttl_seconds,
            mutable_at_runtime=False,
            notes=["session stickiness is tunable, but bounded to explicit TTL limits"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_prefer_local",
            config_path="phase7.hybrid_execution.prefer_local",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.BOOLEAN,
            domain=OptimizationKnobDomain(domain_kind=OptimizationDomainKind.BOOLEAN),
            current_value=hybrid.prefer_local,
            mutable_at_runtime=False,
            notes=["local-first default should remain inspectable to optimizers"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_spillover_enabled",
            config_path="phase7.hybrid_execution.spillover_enabled",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.BOOLEAN,
            domain=OptimizationKnobDomain(domain_kind=OptimizationDomainKind.BOOLEAN),
            current_value=hybrid.spillover_enabled,
            mutable_at_runtime=True,
            notes=["runtime operator controls may disable remote spillover immediately"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_max_remote_share_percent",
            config_path="phase7.hybrid_execution.max_remote_share_percent",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=optimization.max_remote_share_percent,
            ),
            current_value=hybrid.max_remote_share_percent,
            mutable_at_runtime=True,
            notes=["bounded remote share guardrail for hybrid execution"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_remote_request_budget_per_minute",
            config_path="phase7.hybrid_execution.remote_request_budget_per_minute",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.INTEGER,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.INTEGER_RANGE,
                min_value=1 if hybrid.remote_request_budget_per_minute is not None else 1,
                max_value=optimization.max_remote_request_budget_per_minute or 1_000_000,
                nullable=True,
            ),
            current_value=hybrid.remote_request_budget_per_minute,
            mutable_at_runtime=True,
            notes=["null means no explicit per-minute budget is configured"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_remote_concurrency_cap",
            config_path="phase7.hybrid_execution.remote_concurrency_cap",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.INTEGER,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.INTEGER_RANGE,
                min_value=1 if hybrid.remote_concurrency_cap is not None else 1,
                max_value=optimization.max_remote_concurrency_cap or 100_000,
                nullable=True,
            ),
            current_value=hybrid.remote_concurrency_cap,
            mutable_at_runtime=True,
            notes=["protects remote capacity and cost posture"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_remote_cooldown_seconds",
            config_path="phase7.hybrid_execution.remote_cooldown_seconds",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.FLOAT,
            domain=OptimizationKnobDomain(
                domain_kind=OptimizationDomainKind.FLOAT_RANGE,
                min_value=0.0,
                max_value=optimization.max_remote_cooldown_seconds,
            ),
            current_value=hybrid.remote_cooldown_seconds,
            mutable_at_runtime=True,
            notes=["cooldown should remain explicit after transport instability"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_allowed_remote_environments",
            config_path="phase7.hybrid_execution.allowed_remote_environments",
            group=OptimizationKnobGroup.HYBRID_EXECUTION,
            knob_type=OptimizationKnobType.STRING_LIST,
            domain=OptimizationKnobDomain(domain_kind=OptimizationDomainKind.STRING_LIST),
            current_value=list(hybrid.allowed_remote_environments),
            mutable_at_runtime=False,
            notes=["keeps remote enablement scoped to named deployment environments"],
        ),
    ]


def _build_objectives(
    settings: Settings,
    *,
    workload_sets: list[OptimizationWorkloadSet],
) -> list[OptimizationObjectiveTarget]:
    optimization = settings.optimization
    default_workload_id = workload_sets[0].workload_set_id
    objectives: list[OptimizationObjectiveTarget]
    if optimization.objective is CounterfactualObjective.LATENCY:
        objectives = [
            OptimizationObjectiveTarget(
                objective_id="latency_primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                workload_set_ids=[default_workload_id],
                evidence_sources=_forge_evaluation_sources(),
                notes=["primary latency objective across the default evaluation bundle"],
            )
        ]
    elif optimization.objective is CounterfactualObjective.THROUGHPUT:
        objectives = [
            OptimizationObjectiveTarget(
                objective_id="throughput_primary",
                metric=OptimizationObjectiveMetric.TOKENS_PER_SECOND,
                goal=OptimizationGoal.MAXIMIZE,
                workload_set_ids=[default_workload_id],
                evidence_sources=_forge_evaluation_sources(),
                notes=["primary throughput objective across the default evaluation bundle"],
            )
        ]
    elif optimization.objective is CounterfactualObjective.RELIABILITY:
        objectives = [
            OptimizationObjectiveTarget(
                objective_id="reliability_primary",
                metric=OptimizationObjectiveMetric.ERROR_RATE,
                goal=OptimizationGoal.MINIMIZE,
                workload_set_ids=[default_workload_id],
                evidence_sources=_forge_evaluation_sources(),
                notes=["primary reliability objective across the default evaluation bundle"],
            )
        ]
    else:
        objectives = [
            OptimizationObjectiveTarget(
                objective_id="balanced_latency",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                weight=0.45,
                workload_set_ids=[default_workload_id],
                evidence_sources=_forge_evaluation_sources(),
                notes=["balanced objective weighs latency alongside reliability and throughput"],
            ),
            OptimizationObjectiveTarget(
                objective_id="balanced_reliability",
                metric=OptimizationObjectiveMetric.ERROR_RATE,
                goal=OptimizationGoal.MINIMIZE,
                weight=0.35,
                workload_set_ids=[default_workload_id],
                evidence_sources=_forge_evaluation_sources(),
            ),
            OptimizationObjectiveTarget(
                objective_id="balanced_throughput",
                metric=OptimizationObjectiveMetric.TOKENS_PER_SECOND,
                goal=OptimizationGoal.MAXIMIZE,
                weight=0.20,
                workload_set_ids=[default_workload_id],
                evidence_sources=_forge_evaluation_sources(),
            ),
        ]

    objectives.append(
        OptimizationObjectiveTarget(
            objective_id="cache_locality_latency",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=0.15,
            applies_to=[
                OptimizationScope(
                    kind=OptimizationScopeKind.SCENARIO_FAMILY,
                    target=WorkloadScenarioFamily.REPEATED_PREFIX.value,
                )
            ],
            workload_set_ids=["phase9-cache-locality"],
            evidence_sources=_forge_evaluation_sources(),
            notes=[
                "cache and prefix-aware behavior is evaluated via repeated-prefix scenarios",
                (
                    "no direct prefix-locality tuning knob is exposed because the repo "
                    "does not configure one safely"
                ),
            ],
        )
    )
    if settings.phase7.hybrid_execution.enabled:
        objectives.append(
            OptimizationObjectiveTarget(
                objective_id="remote_share_guardrail",
                metric=OptimizationObjectiveMetric.REMOTE_SHARE_PERCENT,
                goal=OptimizationGoal.AT_MOST,
                target_value=settings.phase7.hybrid_execution.max_remote_share_percent,
                weight=0.25,
                applies_to=[
                    OptimizationScope(
                        kind=OptimizationScopeKind.SCENARIO_FAMILY,
                        target=WorkloadScenarioFamily.HYBRID_SPILLOVER.value,
                    )
                ],
                workload_set_ids=["phase9-hybrid-remote"],
                evidence_sources=_forge_evaluation_sources(),
                notes=["hybrid tuning should stay bounded by explicit remote-share guardrails"],
            )
        )
    return objectives


def _build_constraints(settings: Settings) -> list[OptimizationConstraint]:
    optimization = settings.optimization
    hybrid = settings.phase7.hybrid_execution
    constraints: list[OptimizationConstraint] = [
        OptimizationConstraint(
            constraint_id="max_rollout_canary_percentage",
            dimension=OptimizationConstraintDimension.CANARY_PERCENTAGE,
            strength=OptimizationConstraintStrength.HARD,
            operator=OptimizationComparisonOperator.LTE,
            threshold_value=optimization.max_rollout_canary_percentage,
            notes=["live promotion must stay bounded by explicit canary limits"],
        ),
        OptimizationConstraint(
            constraint_id="max_shadow_sampling_rate",
            dimension=OptimizationConstraintDimension.SHADOW_SAMPLING_RATE,
            strength=OptimizationConstraintStrength.HARD,
            operator=OptimizationComparisonOperator.LTE,
            threshold_value=optimization.max_shadow_sampling_rate,
            notes=["shadow traffic is observational only and should stay bounded"],
        ),
        OptimizationConstraint(
            constraint_id="promotion_requires_operator_review",
            dimension=OptimizationConstraintDimension.OPERATOR_REVIEW_REQUIRED,
            strength=OptimizationConstraintStrength.HARD,
            operator=OptimizationComparisonOperator.EQ,
            threshold_value=optimization.promotion_requires_operator_review,
        ),
    ]
    if optimization.max_predicted_error_rate is not None:
        constraints.append(
            OptimizationConstraint(
                constraint_id="max_predicted_error_rate",
                dimension=OptimizationConstraintDimension.PREDICTED_ERROR_RATE,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=optimization.max_predicted_error_rate,
            )
        )
    if optimization.max_predicted_latency_regression_ms is not None:
        constraints.append(
            OptimizationConstraint(
                constraint_id="max_predicted_latency_regression_ms",
                dimension=OptimizationConstraintDimension.PREDICTED_LATENCY_REGRESSION_MS,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=optimization.max_predicted_latency_regression_ms,
            )
        )
    constraints.append(
        OptimizationConstraint(
            constraint_id="max_remote_share_percent",
            dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
            strength=OptimizationConstraintStrength.HARD,
            operator=OptimizationComparisonOperator.LTE,
            threshold_value=optimization.max_remote_share_percent,
            notes=["hybrid remote share remains bounded even during spillover experiments"],
        )
    )
    if optimization.max_remote_request_budget_per_minute is not None:
        constraints.append(
            OptimizationConstraint(
                constraint_id="max_remote_request_budget_per_minute",
                dimension=OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=optimization.max_remote_request_budget_per_minute,
            )
        )
    if optimization.max_remote_concurrency_cap is not None:
        constraints.append(
            OptimizationConstraint(
                constraint_id="max_remote_concurrency_cap",
                dimension=OptimizationConstraintDimension.REMOTE_CONCURRENCY_CAP,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=optimization.max_remote_concurrency_cap,
            )
        )
    if hybrid.prefer_local:
        constraints.append(
            OptimizationConstraint(
                constraint_id="prefer_local_default",
                dimension=OptimizationConstraintDimension.LOCAL_PREFERENCE_ENABLED,
                strength=OptimizationConstraintStrength.SOFT,
                operator=OptimizationComparisonOperator.EQ,
                threshold_value=True,
                notes=[
                    (
                        "local-first posture is advisory for ranking, not an absolute "
                        "ban on remote help"
                    )
                ],
            )
        )
    return constraints


def _build_workload_sets(settings: Settings) -> list[OptimizationWorkloadSet]:
    serving_targets = [] if settings.default_model_alias is None else [settings.default_model_alias]
    return [
        OptimizationWorkloadSet(
            workload_set_id="phase9-default",
            source_kind=OptimizationWorkloadSourceKind.BUILT_IN_SCENARIO_FAMILY,
            serving_targets=serving_targets,
            scenario_families=[
                WorkloadScenarioFamily.SHORT_CHAT,
                WorkloadScenarioFamily.LONG_PROMPT,
                WorkloadScenarioFamily.REPEATED_PREFIX,
                WorkloadScenarioFamily.QUEUE_SATURATION,
                WorkloadScenarioFamily.TENANT_CONTENTION,
            ],
            request_classes=[RequestClass.STANDARD, RequestClass.LATENCY_SENSITIVE],
            notes=["default local-plus-routing evaluation bundle for bounded Phase 9 tuning"],
        ),
        OptimizationWorkloadSet(
            workload_set_id="phase9-cache-locality",
            source_kind=OptimizationWorkloadSourceKind.BUILT_IN_SCENARIO_FAMILY,
            serving_targets=serving_targets,
            scenario_families=[
                WorkloadScenarioFamily.REPEATED_PREFIX,
                WorkloadScenarioFamily.SESSION_STICKINESS,
            ],
            workload_tags=[WorkloadTag.REPEATED_PREFIX, WorkloadTag.SESSION_CONTINUATION],
            notes=[
                (
                    "cache-aware evaluation is modeled through repeated-prefix and "
                    "sticky-session scenarios"
                ),
                (
                    "the current repo does not expose prefix-locality service internals "
                    "as tuning knobs"
                ),
            ],
        ),
        OptimizationWorkloadSet(
            workload_set_id="phase9-hybrid-remote",
            source_kind=OptimizationWorkloadSourceKind.BUILT_IN_SCENARIO_FAMILY,
            serving_targets=serving_targets,
            scenario_families=[
                WorkloadScenarioFamily.HYBRID_SPILLOVER,
                WorkloadScenarioFamily.REMOTE_COLD_WARM,
                WorkloadScenarioFamily.REMOTE_BUDGET_GUARDRAIL,
                WorkloadScenarioFamily.REAL_CLOUD_VALIDATION,
            ],
            request_classes=[RequestClass.STANDARD, RequestClass.LATENCY_SENSITIVE],
            notes=["hybrid and cloud guardrail bundle for bounded remote execution tuning"],
        ),
    ]


def _build_campaign_metadata(
    *,
    settings: Settings,
    profile_id: str,
    workload_sets: list[OptimizationWorkloadSet],
) -> OptimizationCampaignMetadata:
    return OptimizationCampaignMetadata(
        campaign_id=_forge_campaign_id(profile_id=profile_id),
        optimization_profile_id=profile_id,
        objective=settings.optimization.objective,
        evidence_sources=_forge_evaluation_sources(),
        required_evidence_sources=_forge_required_evaluation_sources(
            require_observed_backend_evidence=settings.optimization.require_observed_backend_evidence
        ),
        default_workload_set_ids=[workload_set.workload_set_id for workload_set in workload_sets],
        promotion_requires_operator_review=settings.optimization.promotion_requires_operator_review,
        notes=[
            "campaign metadata is descriptive only; no optimization loop is executed here",
            (
                "safe config profiles and workload bundles are explicit so unsupported "
                "dimensions stay excluded"
            ),
        ],
    )


def _build_baseline_trial(*, settings: Settings) -> OptimizationTrialIdentity:
    profile_id = settings.optimization.profile_id
    return OptimizationTrialIdentity(
        trial_id=_forge_trial_id(
            profile_id=profile_id,
            role=ForgeTrialRole.BASELINE,
            label=settings.default_routing_policy.value,
        ),
        candidate_id=f"baseline:{settings.default_routing_policy.value}",
        candidate_kind=ForgeCandidateKind.CONFIG_PROFILE,
        config_profile_id=_forge_config_profile_id(
            profile_id=profile_id,
            label="baseline",
        ),
        notes=["baseline identity for the currently active Phase 9 optimization profile"],
    )


def _build_candidate_trials(*, settings: Settings) -> list[OptimizationTrialIdentity]:
    profile_id = settings.optimization.profile_id
    candidates: list[OptimizationTrialIdentity] = []
    rollout = settings.phase4.policy_rollout
    if rollout.candidate_policy_id is not None:
        candidates.append(
            OptimizationTrialIdentity(
                trial_id=_forge_trial_id(
                    profile_id=profile_id,
                    role=ForgeTrialRole.CANDIDATE,
                    label=rollout.candidate_policy_id,
                ),
                candidate_id=f"rollout-policy:{rollout.candidate_policy_id}",
                candidate_kind=ForgeCandidateKind.ROLLOUT_POLICY,
                config_profile_id=_forge_config_profile_id(
                    profile_id=profile_id,
                    label=rollout.candidate_policy_id,
                ),
                rollout_policy_id=rollout.candidate_policy_id,
                notes=["current rollout-policy candidate if one is configured"],
            )
        )
    for routing_policy in settings.optimization.allowlisted_routing_policies:
        if routing_policy == settings.default_routing_policy:
            continue
        candidates.append(
            OptimizationTrialIdentity(
                trial_id=_forge_trial_id(
                    profile_id=profile_id,
                    role=ForgeTrialRole.CANDIDATE,
                    label=routing_policy.value,
                ),
                candidate_id=f"routing-policy:{routing_policy.value}",
                candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
                config_profile_id=_forge_config_profile_id(
                    profile_id=profile_id,
                    label=routing_policy.value,
                ),
                routing_policy=routing_policy,
                notes=["safe routing-policy candidate exported from the allowlist"],
            )
        )
    for preset in settings.optimization.worker_launch_presets:
        candidates.append(
            OptimizationTrialIdentity(
                trial_id=_forge_trial_id(
                    profile_id=profile_id,
                    role=ForgeTrialRole.CANDIDATE,
                    label=preset.preset_name,
                ),
                candidate_id=f"runtime-profile:{preset.preset_name}",
                candidate_kind=ForgeCandidateKind.CONFIG_PROFILE,
                config_profile_id=_forge_config_profile_id(
                    profile_id=profile_id,
                    label=preset.preset_name,
                ),
                worker_launch_preset=preset.preset_name,
                applies_to=[
                    OptimizationScope(
                        kind=OptimizationScopeKind.WORKER_CLASS,
                        target=preset.scope.value,
                    )
                ],
                notes=["runtime launch profile candidate exported from existing safe presets"],
            )
        )
    return candidates


def _build_excluded_dimensions() -> list[OptimizationExcludedDimension]:
    return [
        OptimizationExcludedDimension(
            dimension_id="routing_policy_score_weights",
            reason=(
                "router scoring coefficients are internal implementation details, not "
                "config-backed tuning knobs"
            ),
        ),
        OptimizationExcludedDimension(
            dimension_id="prefix_locality_tracker_limits",
            reason=(
                "prefix-locality ttl and capacity are constructor-level service defaults "
                "and are not safely exposed through settings"
            ),
        ),
        OptimizationExcludedDimension(
            dimension_id="engine_batching_parameters",
            reason=(
                "runtime batching and engine-internal scheduler knobs are not exposed "
                "safely through the current worker contract"
            ),
        ),
        OptimizationExcludedDimension(
            dimension_id="per_backend_admission_overrides",
            reason=(
                "per-tenant and per-backend concurrency and queue limits are operator-scoped "
                "overrides and are not safely tunable as global optimization knobs"
            ),
        ),
    ]


def _forge_campaign_id(*, profile_id: str) -> str:
    candidate = f"{profile_id}-forge-stage-a"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:16]
    return f"forge-stage-a-{digest}"


def _forge_trial_id(
    *,
    profile_id: str,
    role: ForgeTrialRole,
    label: str,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "profile_id": profile_id,
                "role": role.value,
                "label": label,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"trial-{digest}"


def _forge_config_profile_id(*, profile_id: str, label: str) -> str:
    normalized_label = label.replace("_", "-")
    candidate = f"{profile_id}-{normalized_label}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    return f"forge-profile-{digest}"


def _forge_evaluation_sources() -> list[ForgeEvidenceSourceKind]:
    return [
        ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
        ForgeEvidenceSourceKind.REPLAYED_BENCHMARK,
        ForgeEvidenceSourceKind.REPLAYED_TRACE,
        ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION,
    ]


def _forge_required_evaluation_sources(
    *,
    require_observed_backend_evidence: bool,
) -> list[ForgeEvidenceSourceKind]:
    required = [
        ForgeEvidenceSourceKind.REPLAYED_BENCHMARK,
        ForgeEvidenceSourceKind.REPLAYED_TRACE,
        ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION,
    ]
    if require_observed_backend_evidence:
        required.insert(0, ForgeEvidenceSourceKind.OBSERVED_RUNTIME)
    return required


def _recommended_forge_rollout_mode(*, profile: OptimizationProfile) -> PolicyRolloutMode:
    preferred_order = (
        PolicyRolloutMode.CANARY,
        PolicyRolloutMode.SHADOW_ONLY,
        PolicyRolloutMode.REPORT_ONLY,
        PolicyRolloutMode.ACTIVE_GUARDED,
    )
    for mode in preferred_order:
        if mode in profile.allowlisted_rollout_modes:
            return mode
    return profile.active_rollout_mode


def _benchmark_runner_knobs(run_config: BenchmarkRunConfig) -> list[BenchmarkConfigKnob]:
    knobs: list[BenchmarkConfigKnob] = [
        BenchmarkConfigKnob(
            knob_id="benchmark.concurrency",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="run_config.concurrency",
            value=run_config.concurrency,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.timeout_seconds",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="run_config.timeout_seconds",
            value=run_config.timeout_seconds,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.canary_percentage",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="run_config.canary_percentage",
            value=run_config.canary_percentage,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.shadow_sampling_rate",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="run_config.shadow_sampling_rate",
            value=run_config.shadow_sampling_rate,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.trace_capture_mode",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="run_config.trace_capture_mode",
            value=run_config.trace_capture_mode.value,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.warmup.enabled",
            category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
            config_path="run_config.warmup.enabled",
            value=run_config.warmup.enabled,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.warmup.request_count",
            category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
            config_path="run_config.warmup.request_count",
            value=run_config.warmup.request_count,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.warmup.concurrency",
            category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
            config_path="run_config.warmup.concurrency",
            value=run_config.warmup.concurrency,
            source_scope="benchmark_runner",
        ),
    ]
    if run_config.replay_mode is not None:
        knobs.append(
            BenchmarkConfigKnob(
                knob_id="benchmark.replay_mode",
                category=BenchmarkConfigKnobCategory.SCHEDULING,
                config_path="run_config.replay_mode",
                value=run_config.replay_mode.value,
                source_scope="benchmark_runner",
            )
        )
    if run_config.session_affinity_ttl_seconds is not None:
        knobs.append(
            BenchmarkConfigKnob(
                knob_id="benchmark.session_affinity_ttl_seconds",
                category=BenchmarkConfigKnobCategory.ROUTING,
                config_path="run_config.session_affinity_ttl_seconds",
                value=run_config.session_affinity_ttl_seconds,
                source_scope="benchmark_runner",
            )
        )
    return knobs


def _settings_knobs(settings: Settings) -> list[BenchmarkConfigKnob]:
    phase4 = settings.phase4
    hybrid = settings.phase7.hybrid_execution
    optimization = settings.optimization
    knobs: list[BenchmarkConfigKnob] = [
        BenchmarkConfigKnob(
            knob_id="routing.default_policy",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="default_routing_policy",
            value=settings.default_routing_policy.value,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.global_concurrency_cap",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.global_concurrency_cap",
            value=phase4.admission_control.global_concurrency_cap,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.global_queue_size",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.global_queue_size",
            value=phase4.admission_control.global_queue_size,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.default_concurrency_cap",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.default_concurrency_cap",
            value=phase4.admission_control.default_concurrency_cap,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.default_queue_size",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.default_queue_size",
            value=phase4.admission_control.default_queue_size,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.queue_timeout_seconds",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.queue_timeout_seconds",
            value=phase4.admission_control.queue_timeout_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="circuit.failure_threshold",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.circuit_breakers.failure_threshold",
            value=phase4.circuit_breakers.failure_threshold,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="circuit.open_cooldown_seconds",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.circuit_breakers.open_cooldown_seconds",
            value=phase4.circuit_breakers.open_cooldown_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="rollout.mode",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.policy_rollout.mode",
            value=phase4.policy_rollout.mode.value,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="rollout.canary_percentage",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.policy_rollout.canary_percentage",
            value=phase4.policy_rollout.canary_percentage,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="canary.default_percentage",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.canary_routing.default_percentage",
            value=phase4.canary_routing.default_percentage,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="shadow.default_sampling_rate",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.shadow_routing.default_sampling_rate",
            value=phase4.shadow_routing.default_sampling_rate,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="affinity.ttl_seconds",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.session_affinity.ttl_seconds",
            value=phase4.session_affinity.ttl_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.prefer_local",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.prefer_local",
            value=hybrid.prefer_local,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.spillover_enabled",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.spillover_enabled",
            value=hybrid.spillover_enabled,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.max_remote_share_percent",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.max_remote_share_percent",
            value=hybrid.max_remote_share_percent,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.remote_request_budget_per_minute",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.remote_request_budget_per_minute",
            value=hybrid.remote_request_budget_per_minute,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.remote_concurrency_cap",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.remote_concurrency_cap",
            value=hybrid.remote_concurrency_cap,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.remote_cooldown_seconds",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.remote_cooldown_seconds",
            value=hybrid.remote_cooldown_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.allowed_remote_environments",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.allowed_remote_environments",
            value=list(hybrid.allowed_remote_environments),
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="search.allowlisted_routing_policies",
            category=BenchmarkConfigKnobCategory.SEARCH_SPACE,
            config_path="optimization.allowlisted_routing_policies",
            value=[policy.value for policy in optimization.allowlisted_routing_policies],
            source_scope="optimization_surface",
        ),
        BenchmarkConfigKnob(
            knob_id="search.allowlisted_rollout_modes",
            category=BenchmarkConfigKnobCategory.SEARCH_SPACE,
            config_path="optimization.allowlisted_rollout_modes",
            value=[mode.value for mode in optimization.allowlisted_rollout_modes],
            source_scope="optimization_surface",
        ),
    ]
    for model in sorted(settings.local_models, key=lambda item: item.alias):
        knobs.extend(
            [
                BenchmarkConfigKnob(
                    knob_id=f"serving.{model.alias}.configured_priority",
                    category=BenchmarkConfigKnobCategory.SERVING,
                    config_path=f"local_models[{model.alias}].configured_priority",
                    value=model.configured_priority,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"serving.{model.alias}.configured_weight",
                    category=BenchmarkConfigKnobCategory.SERVING,
                    config_path=f"local_models[{model.alias}].configured_weight",
                    value=model.configured_weight,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"worker_launch.{model.alias}.worker_transport",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"local_models[{model.alias}].worker_transport",
                    value=model.worker_transport.value,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"worker_launch.{model.alias}.warmup_enabled",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"local_models[{model.alias}].warmup.enabled",
                    value=model.warmup.enabled,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"worker_launch.{model.alias}.warmup_eager",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"local_models[{model.alias}].warmup.eager",
                    value=model.warmup.eager,
                    source_scope=model.alias,
                ),
            ]
        )
    for preset in sorted(optimization.worker_launch_presets, key=lambda item: item.preset_name):
        prefix = f"worker_preset.{preset.preset_name}"
        knobs.extend(
            [
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.scope",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"optimization.worker_launch_presets[{preset.preset_name}].scope",
                    value=preset.scope.value,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.warmup_mode",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}].warmup_mode"
                    ),
                    value=preset.warmup_mode,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.concurrency_limit",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}]"
                        ".concurrency_limit"
                    ),
                    value=preset.concurrency_limit,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.supports_streaming",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}]"
                        ".supports_streaming"
                    ),
                    value=preset.supports_streaming,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.stream_chunk_size",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}]"
                        ".stream_chunk_size"
                    ),
                    value=preset.stream_chunk_size,
                    source_scope=preset.preset_name,
                ),
            ]
        )
    return knobs


def _fingerprint_benchmark_knobs(
    *,
    profile_id: str,
    knobs: list[BenchmarkConfigKnob],
    notes: list[str],
) -> BenchmarkConfigFingerprint:
    payload = {
        "profile_id": profile_id,
        "knobs": [knob.model_dump(mode="json", exclude_none=True) for knob in knobs],
        "notes": notes,
    }
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return BenchmarkConfigFingerprint(
        algorithm="sha256_canonical_json",
        value=hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest(),
    )
