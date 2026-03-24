"""Explicit optimization-profile operations for Forge Stage A promotion workflows.

This module provides standalone, reusable functions for converting recommended
candidates into explicit optimization profiles, inspecting what a profile
touches and whether it stays within declared tunable boundaries, computing
standalone diffs between profiles, and checking scope compatibility.

These functions are independent of the ``ForgePromotionService`` lifecycle FSM
and can be used by offline inspection, CLI, and reporting workflows.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from switchyard.config import Settings
from switchyard.optimization import (
    build_optimization_profile,
    build_trial_optimization_config_profile,
)
from switchyard.schemas.optimization import (
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationConfigProfile,
    OptimizationConfigProfileChange,
    OptimizationConfigProfileDiff,
    OptimizationConfigProfileValidationIssueKind,
    OptimizationKnobGroup,
    OptimizationKnobSurface,
    OptimizationProfile,
    OptimizationScope,
    OptimizationScopeKind,
)

# ---------------------------------------------------------------------------
# Application-boundary schema
# ---------------------------------------------------------------------------


class ProfileApplicationBoundaryKnob(BaseModel):
    """One knob's status within the application boundary of a config profile."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(default="", max_length=256)
    group: OptimizationKnobGroup | None = None
    declared_tunable: bool = False
    runtime_mutable: bool = False
    scope_compatible: bool = True
    domain_valid: bool = True
    allowed: bool = True
    notes: list[str] = Field(default_factory=list)


class ProfileApplicationBoundary(BaseModel):
    """Explicit inspection of what a config profile touches and where it may apply.

    This is an operator-facing summary that tells you whether a promotion-ready
    profile stays within the declared optimization surface and is safe to apply
    at runtime.
    """

    model_config = ConfigDict(extra="forbid")

    config_profile_id: str = Field(min_length=1, max_length=128)
    optimization_profile_id: str = Field(min_length=1, max_length=128)
    profile_version: int = Field(ge=1)
    total_changes: int = Field(default=0, ge=0)
    tunable_knob_ids: list[str] = Field(default_factory=list)
    runtime_mutable_knob_ids: list[str] = Field(default_factory=list)
    immutable_knob_ids: list[str] = Field(default_factory=list)
    undeclared_knob_ids: list[str] = Field(default_factory=list)
    scope_incompatible_knob_ids: list[str] = Field(default_factory=list)
    domain_violation_knob_ids: list[str] = Field(default_factory=list)
    within_boundary: bool = True
    all_runtime_mutable: bool = True
    scope_compatible: bool = True
    knobs: list[ProfileApplicationBoundaryKnob] = Field(default_factory=list)
    profile_scope: list[OptimizationScope] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Profile-from-recommendation
# ---------------------------------------------------------------------------


def promote_recommendation_to_profile(
    *,
    settings: Settings,
    campaign_artifact: OptimizationCampaignArtifact,
    comparison_artifact: OptimizationCampaignComparisonArtifact,
    candidate_configuration_id: str,
    profile_version: int = 1,
) -> OptimizationConfigProfile:
    """Convert a recommended candidate into an explicit optimization profile.

    This function connects the offline comparison pipeline to the promotion
    pipeline without requiring the caller to navigate the campaign/trial
    artifact hierarchy manually.

    Raises ``ValueError`` if the candidate is not found in the campaign, does
    not appear in the comparison, or its comparison recommendation does not
    support promotion.
    """
    if comparison_artifact.campaign_id != campaign_artifact.campaign.campaign_id:
        msg = (
            "comparison_artifact.campaign_id does not match "
            "campaign_artifact.campaign.campaign_id"
        )
        raise ValueError(msg)

    # Locate the candidate comparison entry.
    candidate_comparison = next(
        (
            entry
            for entry in comparison_artifact.candidate_comparisons
            if entry.candidate_configuration_id == candidate_configuration_id
        ),
        None,
    )
    if candidate_comparison is None:
        msg = (
            f"candidate_configuration_id '{candidate_configuration_id}' is not present "
            "in the comparison artifact"
        )
        raise ValueError(msg)

    # Locate the matching trial artifact in the campaign.
    trial_artifact = next(
        (
            trial
            for trial in campaign_artifact.trials
            if trial.candidate_configuration.candidate_configuration_id
            == candidate_configuration_id
        ),
        None,
    )
    if trial_artifact is None:
        msg = (
            f"candidate_configuration_id '{candidate_configuration_id}' is not present "
            "in the campaign artifact trials"
        )
        raise ValueError(msg)

    return build_trial_optimization_config_profile(
        settings=settings,
        trial_artifact=trial_artifact,
        campaign_artifact_id=campaign_artifact.campaign_artifact_id,
        profile_version=profile_version,
    )


# ---------------------------------------------------------------------------
# Application-boundary validation
# ---------------------------------------------------------------------------


def validate_profile_application_boundary(
    *,
    settings: Settings,
    config_profile: OptimizationConfigProfile,
) -> ProfileApplicationBoundary:
    """Inspect what a config profile will touch and whether it is safe to apply.

    Returns a ``ProfileApplicationBoundary`` that explicitly lists which knobs
    are declared tunable, which are runtime-mutable, which are out of scope or
    out of domain, and which are entirely undeclared. The ``within_boundary``
    flag is ``True`` only when every change targets a declared, domain-valid,
    scope-compatible tunable knob.
    """
    surface = build_optimization_profile(settings)
    surface_by_id = {knob.knob_id: knob for knob in surface.knobs}

    tunable_knob_ids: list[str] = []
    runtime_mutable_knob_ids: list[str] = []
    immutable_knob_ids: list[str] = []
    undeclared_knob_ids: list[str] = []
    scope_incompatible_knob_ids: list[str] = []
    domain_violation_knob_ids: list[str] = []
    knob_entries: list[ProfileApplicationBoundaryKnob] = []

    for change in config_profile.changes:
        knob = surface_by_id.get(change.knob_id)
        if knob is None:
            undeclared_knob_ids.append(change.knob_id)
            knob_entries.append(
                ProfileApplicationBoundaryKnob(
                    knob_id=change.knob_id,
                    config_path=change.config_path,
                    declared_tunable=False,
                    allowed=False,
                    notes=["knob is not declared in the optimization surface"],
                )
            )
            continue

        declared_tunable = True
        tunable_knob_ids.append(change.knob_id)
        if knob.mutable_at_runtime:
            runtime_mutable_knob_ids.append(change.knob_id)
        else:
            immutable_knob_ids.append(change.knob_id)

        scope_ok = _scopes_within(change.applies_to, knob.applies_to)
        if not scope_ok:
            scope_incompatible_knob_ids.append(change.knob_id)

        domain_ok = _change_within_domain(change, knob)
        if not domain_ok:
            domain_violation_knob_ids.append(change.knob_id)

        allowed = scope_ok and domain_ok
        entry_notes: list[str] = []
        if not scope_ok:
            entry_notes.append("change scope exceeds the declared knob scope")
        if not domain_ok:
            entry_notes.append("candidate value is outside the declared domain")
        if not knob.mutable_at_runtime:
            entry_notes.append(
                "knob is declared tunable but not runtime-mutable; "
                "requires a restart or config reload"
            )

        knob_entries.append(
            ProfileApplicationBoundaryKnob(
                knob_id=change.knob_id,
                config_path=change.config_path,
                group=knob.group,
                declared_tunable=declared_tunable,
                runtime_mutable=knob.mutable_at_runtime,
                scope_compatible=scope_ok,
                domain_valid=domain_ok,
                allowed=allowed,
                notes=entry_notes,
            )
        )

    # Also check the profile-level validation issues.  Unknown knobs are
    # skipped during profile construction (not added to ``changes``) so they
    # only appear in validation issues.
    seen_knob_ids = {entry.knob_id for entry in knob_entries}
    for issue in config_profile.validation.issues:
        if (
            issue.issue_kind
            is OptimizationConfigProfileValidationIssueKind.UNKNOWN_KNOB
            and issue.knob_id is not None
            and issue.knob_id not in seen_knob_ids
        ):
            undeclared_knob_ids.append(issue.knob_id)
            seen_knob_ids.add(issue.knob_id)
            knob_entries.append(
                ProfileApplicationBoundaryKnob(
                    knob_id=issue.knob_id,
                    config_path="",
                    declared_tunable=False,
                    allowed=False,
                    notes=[
                        "knob is not declared in the optimization surface "
                        "(detected from validation issues)"
                    ],
                )
            )
        if (
            issue.issue_kind
            is OptimizationConfigProfileValidationIssueKind.SCOPE_NOT_ALLOWED
            and issue.knob_id is not None
            and issue.knob_id not in scope_incompatible_knob_ids
        ):
            scope_incompatible_knob_ids.append(issue.knob_id)
        if (
            issue.issue_kind
            is OptimizationConfigProfileValidationIssueKind.DOMAIN_VIOLATION
            and issue.knob_id is not None
            and issue.knob_id not in domain_violation_knob_ids
        ):
            domain_violation_knob_ids.append(issue.knob_id)

    within_boundary = (
        not undeclared_knob_ids
        and not scope_incompatible_knob_ids
        and not domain_violation_knob_ids
    )
    all_runtime_mutable = not immutable_knob_ids and not undeclared_knob_ids
    scope_compatible = not scope_incompatible_knob_ids

    notes: list[str] = []
    if within_boundary:
        notes.append(
            "all profile changes are within the declared optimization surface boundary"
        )
    else:
        notes.append(
            "profile includes changes outside the declared optimization surface boundary"
        )
    if not all_runtime_mutable:
        notes.append(
            "profile includes immutable knobs that require a restart to apply"
        )

    return ProfileApplicationBoundary(
        config_profile_id=config_profile.config_profile_id,
        optimization_profile_id=config_profile.optimization_profile_id,
        profile_version=config_profile.profile_version,
        total_changes=len(config_profile.changes),
        tunable_knob_ids=sorted(set(tunable_knob_ids)),
        runtime_mutable_knob_ids=sorted(set(runtime_mutable_knob_ids)),
        immutable_knob_ids=sorted(set(immutable_knob_ids)),
        undeclared_knob_ids=sorted(set(undeclared_knob_ids)),
        scope_incompatible_knob_ids=sorted(set(scope_incompatible_knob_ids)),
        domain_violation_knob_ids=sorted(set(domain_violation_knob_ids)),
        within_boundary=within_boundary,
        all_runtime_mutable=all_runtime_mutable,
        scope_compatible=scope_compatible,
        knobs=knob_entries,
        profile_scope=list(config_profile.applies_to),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Profile diff
# ---------------------------------------------------------------------------


def compute_profile_diff(
    *,
    baseline: OptimizationConfigProfile,
    candidate: OptimizationConfigProfile,
) -> OptimizationConfigProfileDiff:
    """Compute an explicit diff between two materialized config profiles.

    Unlike the diff embedded in ``OptimizationConfigProfile`` (which compares a
    candidate to the baseline at construction time), this function compares any
    two profiles after the fact. This is useful for comparing two promoted
    profiles, a baseline to a rollback snapshot, or two candidates.
    """
    baseline_changes_by_id = {c.knob_id: c for c in baseline.changes}
    candidate_changes_by_id = {c.knob_id: c for c in candidate.changes}

    all_knob_ids = sorted(
        set(baseline_changes_by_id.keys()) | set(candidate_changes_by_id.keys())
    )

    changed_knob_ids: list[str] = []
    changed_groups: set[OptimizationKnobGroup] = set()
    mutable_runtime_knob_ids: list[str] = []
    immutable_knob_ids: list[str] = []

    for knob_id in all_knob_ids:
        baseline_change = baseline_changes_by_id.get(knob_id)
        candidate_change = candidate_changes_by_id.get(knob_id)

        baseline_value = (
            baseline_change.candidate_value if baseline_change is not None else None
        )
        candidate_value = (
            candidate_change.candidate_value if candidate_change is not None else None
        )

        if baseline_value != candidate_value:
            changed_knob_ids.append(knob_id)
            change = candidate_change or baseline_change
            assert change is not None
            changed_groups.add(change.group)
            if change.mutable_at_runtime:
                mutable_runtime_knob_ids.append(knob_id)
            else:
                immutable_knob_ids.append(knob_id)

    # Merge scopes from both profiles.
    profile_scope = _merge_scopes(baseline.applies_to, candidate.applies_to)

    notes: list[str] = [
        "diff computed between two materialized config profiles",
    ]
    if not changed_knob_ids:
        notes.append("profiles are equivalent in their declared knob changes")

    return OptimizationConfigProfileDiff(
        baseline_config_profile_id=baseline.config_profile_id,
        config_profile_id=candidate.config_profile_id,
        changed_knob_ids=sorted(changed_knob_ids),
        changed_groups=sorted(changed_groups, key=lambda g: g.value),
        mutable_runtime_knob_ids=sorted(mutable_runtime_knob_ids),
        immutable_knob_ids=sorted(immutable_knob_ids),
        profile_scope=profile_scope,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Scope compatibility
# ---------------------------------------------------------------------------


def check_profile_scope_compatibility(
    *,
    config_profile: OptimizationConfigProfile,
    target_scope: OptimizationScope,
) -> bool:
    """Check whether a profile's declared scope is compatible with a target context.

    A profile is compatible with a target scope when:
    - The profile applies globally and the target is also global.
    - The profile applies globally and therefore covers any specific target.
    - The profile's scopes include the target scope exactly.

    Returns ``True`` if the profile can be applied in the given scope context.
    """
    if not config_profile.applies_to:
        # No declared scope means global by default.
        return True

    profile_keys = _scope_keys(config_profile.applies_to)

    # A global profile covers everything.
    if (OptimizationScopeKind.GLOBAL.value, None) in profile_keys:
        return True

    # A global target requires a global profile.
    if target_scope.kind is OptimizationScopeKind.GLOBAL:
        return (OptimizationScopeKind.GLOBAL.value, None) in profile_keys

    # Otherwise the target must appear in the profile's scopes.
    target_key = (target_scope.kind.value, target_scope.target)
    return target_key in profile_keys


def check_profile_scope_compatibility_for_knob(
    *,
    config_profile: OptimizationConfigProfile,
    knob_id: str,
    target_scope: OptimizationScope,
    optimization_surface: OptimizationProfile | None = None,
) -> bool:
    """Check whether a specific knob change within a profile is compatible with a scope.

    Checks both the change's own ``applies_to`` and, when provided, the knob's
    declared scope on the optimization surface.
    """
    change = next(
        (c for c in config_profile.changes if c.knob_id == knob_id),
        None,
    )
    if change is None:
        return False

    # Check the change's own scope.
    if change.applies_to:
        change_keys = _scope_keys(
            [OptimizationScope(**s.model_dump()) for s in change.applies_to]
        )
        if (OptimizationScopeKind.GLOBAL.value, None) not in change_keys:
            target_key = (target_scope.kind.value, target_scope.target)
            if (
                target_scope.kind is not OptimizationScopeKind.GLOBAL
                and target_key not in change_keys
            ):
                return False

    # Check the declared surface scope when available.
    if optimization_surface is not None:
        surface_knob = next(
            (k for k in optimization_surface.knobs if k.knob_id == knob_id),
            None,
        )
        if surface_knob is not None and surface_knob.applies_to:
            knob_keys = _scope_keys(surface_knob.applies_to)
            if (OptimizationScopeKind.GLOBAL.value, None) not in knob_keys:
                target_key = (target_scope.kind.value, target_scope.target)
                if (
                    target_scope.kind is not OptimizationScopeKind.GLOBAL
                    and target_key not in knob_keys
                ):
                    return False

    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scopes_within(
    candidate_scopes: list[OptimizationScope],
    allowed_scopes: list[OptimizationScope],
) -> bool:
    """Check that every candidate scope is covered by the allowed scopes."""
    if not candidate_scopes:
        return True
    candidate_keys = _scope_keys(candidate_scopes)
    allowed_keys = _scope_keys(allowed_scopes)
    if (OptimizationScopeKind.GLOBAL.value, None) in allowed_keys:
        return candidate_keys == {(OptimizationScopeKind.GLOBAL.value, None)}
    return candidate_keys.issubset(allowed_keys)


def _scope_keys(
    scopes: list[OptimizationScope],
) -> set[tuple[str, str | None]]:
    return {(scope.kind.value, scope.target) for scope in scopes}


def _merge_scopes(
    a: list[OptimizationScope],
    b: list[OptimizationScope],
) -> list[OptimizationScope]:
    """Merge two scope lists, deduplicating by (kind, target)."""
    seen: set[tuple[str, str | None]] = set()
    result: list[OptimizationScope] = []
    for scope in [*a, *b]:
        key = (scope.kind.value, scope.target)
        if key not in seen:
            seen.add(key)
            result.append(scope)
    return result


def _change_within_domain(
    change: OptimizationConfigProfileChange,
    knob: OptimizationKnobSurface,
) -> bool:
    """Check if a change's candidate value is within the knob's declared domain."""
    value = change.candidate_value
    if value is None:
        return knob.domain.nullable

    from switchyard.schemas.optimization import OptimizationKnobType

    if knob.knob_type is OptimizationKnobType.BOOLEAN:
        return isinstance(value, bool)
    if knob.knob_type is OptimizationKnobType.ENUM:
        return isinstance(value, str) and value in knob.allowed_values
    if knob.knob_type is OptimizationKnobType.STRING_LIST:
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if knob.knob_type is OptimizationKnobType.INTEGER:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if knob.min_value is not None and value < int(knob.min_value):
            return False
        if knob.max_value is not None and value > int(knob.max_value):
            return False
        return True
    # FLOAT
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    if knob.min_value is not None and numeric < float(knob.min_value):
        return False
    if knob.max_value is not None and numeric > float(knob.max_value):
        return False
    return True
