"""Campaign honesty assessment for hybrid local+remote Forge Stage A environments.

Validates that campaign results, recommendations, and promotion decisions remain
trustworthy given the current topology, budget posture, evidence mix, and workload
coverage.  Produces typed warnings that flow through to operator inspection surfaces
so that stale, budget-constrained, overfit, or inconsistent recommendations are
never silently trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.backend import BackendInstance
from switchyard.schemas.optimization import (
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactStatus,
    OptimizationCampaignArtifact,
    OptimizationConstraintDimension,
    OptimizationRecommendationDisposition,
    OptimizationTrialArtifact,
)


class CampaignHonestyWarningKind(StrEnum):
    """Category of a campaign-honesty warning."""

    BUDGET_BOUND_EXCEEDED = "budget_bound_exceeded"
    TOPOLOGY_DRIFT = "topology_drift"
    STALE_EVIDENCE = "stale_evidence"
    NARROW_WORKLOAD_COVERAGE = "narrow_workload_coverage"
    EVIDENCE_INCONSISTENCY = "evidence_inconsistency"
    OBSERVED_EVIDENCE_MISSING = "observed_evidence_missing"
    COST_SIGNAL_MISMATCH = "cost_signal_mismatch"


class CampaignHonestyWarning(BaseModel):
    """One typed honesty warning for a campaign or trial."""

    model_config = ConfigDict(extra="forbid")

    kind: CampaignHonestyWarningKind
    severity: str = Field(default="warning", pattern=r"^(info|warning|error)$")
    message: str = Field(min_length=1, max_length=512)
    affected_trial_ids: list[str] = Field(default_factory=list)
    affected_constraint_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CampaignHonestyAssessment(BaseModel):
    """Result of a campaign honesty check against current environment state."""

    model_config = ConfigDict(extra="forbid")

    campaign_artifact_id: str = Field(min_length=1, max_length=128)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trustworthy: bool = True
    recommended_status: OptimizationArtifactStatus = OptimizationArtifactStatus.COMPLETE
    recommended_status_reason: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    warnings: list[CampaignHonestyWarning] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Staleness thresholds
# ---------------------------------------------------------------------------

_DEFAULT_STALENESS_HOURS = 48
_DEFAULT_MIN_WORKLOAD_FAMILIES = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_campaign_honesty(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    current_worker_inventory: Sequence[BackendInstance] | None = None,
    current_remote_budget_per_minute: int | None = None,
    current_max_remote_share_percent: float | None = None,
    current_remote_concurrency_cap: int | None = None,
    now: datetime | None = None,
    staleness_hours: int = _DEFAULT_STALENESS_HOURS,
    min_workload_families: int = _DEFAULT_MIN_WORKLOAD_FAMILIES,
) -> CampaignHonestyAssessment:
    """Validate that a campaign artifact's results still deserve trust.

    Checks:
    - Remote budget constraints vs current budget posture (including concurrency cap)
    - Topology drift between campaign evidence and current workers
    - Staleness of evidence windows
    - Narrow workload coverage (overfit risk)
    - Missing observed evidence behind promotion recommendations
    - Inconsistency between replay-only and observed evidence
    - Mismatch between observed and configured cost signals
    """

    resolved_now = now or datetime.now(UTC)
    warnings: list[CampaignHonestyWarning] = []

    warnings.extend(
        _check_budget_bounds(
            campaign_artifact=campaign_artifact,
            current_remote_budget_per_minute=current_remote_budget_per_minute,
            current_max_remote_share_percent=current_max_remote_share_percent,
            current_remote_concurrency_cap=current_remote_concurrency_cap,
        )
    )
    warnings.extend(
        _check_topology_drift(
            campaign_artifact=campaign_artifact,
            current_worker_inventory=current_worker_inventory,
        )
    )
    warnings.extend(
        _check_staleness(
            campaign_artifact=campaign_artifact,
            now=resolved_now,
            staleness_hours=staleness_hours,
        )
    )
    warnings.extend(
        _check_workload_coverage(
            campaign_artifact=campaign_artifact,
            min_workload_families=min_workload_families,
        )
    )
    warnings.extend(
        _check_evidence_consistency(
            campaign_artifact=campaign_artifact,
        )
    )
    warnings.extend(
        _check_cost_signal_honesty(
            campaign_artifact=campaign_artifact,
        )
    )

    has_error = any(warning.severity == "error" for warning in warnings)
    has_warning = any(warning.severity == "warning" for warning in warnings)

    if has_error:
        recommended_status = OptimizationArtifactStatus.INVALIDATED
        recommended_reason = "campaign honesty assessment found invalidating conditions"
    elif has_warning:
        recommended_status = OptimizationArtifactStatus.STALE
        recommended_reason = "campaign honesty assessment found conditions that reduce trust"
    else:
        recommended_status = campaign_artifact.result_status
        recommended_reason = None

    trustworthy = not has_error and not has_warning

    notes = [
        "campaign honesty assessment is advisory and does not mutate the source artifact",
    ]
    if warnings:
        notes.append(
            f"found {len(warnings)} honesty warning(s) across budget, topology, "
            f"staleness, workload, and evidence checks"
        )
    else:
        notes.append("no honesty warnings found; campaign results appear trustworthy")

    return CampaignHonestyAssessment(
        campaign_artifact_id=campaign_artifact.campaign_artifact_id,
        assessed_at=resolved_now,
        trustworthy=trustworthy,
        recommended_status=recommended_status,
        recommended_status_reason=recommended_reason,
        warnings=warnings,
        notes=notes,
    )


def mark_campaign_stale(
    campaign_artifact: OptimizationCampaignArtifact,
    *,
    reason: str,
) -> OptimizationCampaignArtifact:
    """Return a copy of the campaign artifact with STALE status and reason."""

    return campaign_artifact.model_copy(
        update={
            "result_status": OptimizationArtifactStatus.STALE,
            "stale_reason": reason,
            "notes": [
                *campaign_artifact.notes,
                f"marked stale: {reason}",
            ],
        },
        deep=True,
    )


def mark_campaign_invalidated(
    campaign_artifact: OptimizationCampaignArtifact,
    *,
    reason: str,
) -> OptimizationCampaignArtifact:
    """Return a copy of the campaign artifact with INVALIDATED status and reason."""

    return campaign_artifact.model_copy(
        update={
            "result_status": OptimizationArtifactStatus.INVALIDATED,
            "invalidation_reason": reason,
            "notes": [
                *campaign_artifact.notes,
                f"invalidated: {reason}",
            ],
        },
        deep=True,
    )


def mark_trial_stale(
    trial_artifact: OptimizationTrialArtifact,
    *,
    reason: str,
) -> OptimizationTrialArtifact:
    """Return a copy of the trial artifact with STALE status and reason."""

    return trial_artifact.model_copy(
        update={
            "result_status": OptimizationArtifactStatus.STALE,
            "stale_reason": reason,
            "notes": [
                *trial_artifact.notes,
                f"marked stale: {reason}",
            ],
        },
        deep=True,
    )


def mark_trial_invalidated(
    trial_artifact: OptimizationTrialArtifact,
    *,
    reason: str,
) -> OptimizationTrialArtifact:
    """Return a copy of the trial artifact with INVALIDATED status and reason."""

    return trial_artifact.model_copy(
        update={
            "result_status": OptimizationArtifactStatus.INVALIDATED,
            "invalidation_reason": reason,
            "notes": [
                *trial_artifact.notes,
                f"invalidated: {reason}",
            ],
        },
        deep=True,
    )


# ---------------------------------------------------------------------------
# Budget-bound checks
# ---------------------------------------------------------------------------


def _check_budget_bounds(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    current_remote_budget_per_minute: int | None,
    current_max_remote_share_percent: float | None,
    current_remote_concurrency_cap: int | None = None,
) -> list[CampaignHonestyWarning]:
    warnings: list[CampaignHonestyWarning] = []

    for trial in campaign_artifact.trials:
        for assessment in trial.constraint_assessments:
            if (
                assessment.dimension
                is OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE
                and current_remote_budget_per_minute is not None
                and assessment.evaluated_value is not None
            ):
                evaluated = (
                    float(assessment.evaluated_value)
                    if not isinstance(assessment.evaluated_value, bool)
                    else None
                )
                if evaluated is not None and evaluated > current_remote_budget_per_minute:
                    warnings.append(
                        CampaignHonestyWarning(
                            kind=CampaignHonestyWarningKind.BUDGET_BOUND_EXCEEDED,
                            severity="warning",
                            message=(
                                f"trial {trial.trial_artifact_id} assumed remote budget "
                                f"{evaluated}/min but current budget is "
                                f"{current_remote_budget_per_minute}/min"
                            ),
                            affected_trial_ids=[trial.trial_artifact_id],
                            affected_constraint_ids=[assessment.constraint_id],
                        )
                    )

            if (
                assessment.dimension
                is OptimizationConstraintDimension.REMOTE_SHARE_PERCENT
                and current_max_remote_share_percent is not None
                and assessment.evaluated_value is not None
            ):
                evaluated = (
                    float(assessment.evaluated_value)
                    if not isinstance(assessment.evaluated_value, bool)
                    else None
                )
                if (
                    evaluated is not None
                    and evaluated > current_max_remote_share_percent
                ):
                    warnings.append(
                        CampaignHonestyWarning(
                            kind=CampaignHonestyWarningKind.BUDGET_BOUND_EXCEEDED,
                            severity="warning",
                            message=(
                                f"trial {trial.trial_artifact_id} assumed remote share "
                                f"{evaluated:.1f}% but current cap is "
                                f"{current_max_remote_share_percent:.1f}%"
                            ),
                            affected_trial_ids=[trial.trial_artifact_id],
                            affected_constraint_ids=[assessment.constraint_id],
                        )
                    )

            if (
                assessment.dimension
                is OptimizationConstraintDimension.REMOTE_CONCURRENCY_CAP
                and current_remote_concurrency_cap is not None
                and assessment.evaluated_value is not None
            ):
                evaluated = (
                    float(assessment.evaluated_value)
                    if not isinstance(assessment.evaluated_value, bool)
                    else None
                )
                if evaluated is not None and evaluated > current_remote_concurrency_cap:
                    warnings.append(
                        CampaignHonestyWarning(
                            kind=CampaignHonestyWarningKind.BUDGET_BOUND_EXCEEDED,
                            severity="warning",
                            message=(
                                f"trial {trial.trial_artifact_id} assumed concurrency cap "
                                f"{int(evaluated)} but current cap is "
                                f"{current_remote_concurrency_cap}"
                            ),
                            affected_trial_ids=[trial.trial_artifact_id],
                            affected_constraint_ids=[assessment.constraint_id],
                        )
                    )

    return warnings


# ---------------------------------------------------------------------------
# Topology drift checks
# ---------------------------------------------------------------------------


def _check_topology_drift(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    current_worker_inventory: Sequence[BackendInstance] | None,
) -> list[CampaignHonestyWarning]:
    if current_worker_inventory is None:
        return []

    campaign_topology = campaign_artifact.topology_lineage
    if campaign_topology is None:
        return []

    campaign_instance_ids = {
        instance.instance_id
        for instance in campaign_topology.worker_instance_inventory
    }
    current_instance_ids = {
        instance.instance_id for instance in current_worker_inventory
    }

    if not campaign_instance_ids:
        return []

    disappeared = campaign_instance_ids - current_instance_ids
    appeared = current_instance_ids - campaign_instance_ids

    warnings: list[CampaignHonestyWarning] = []

    if disappeared:
        warnings.append(
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.TOPOLOGY_DRIFT,
                severity="warning",
                message=(
                    f"{len(disappeared)} worker(s) from campaign evidence are no "
                    f"longer present in the current inventory"
                ),
                notes=[
                    f"disappeared: {', '.join(sorted(disappeared))}",
                    "recommendations may not be achievable with the current topology",
                ],
            )
        )

    if appeared:
        warnings.append(
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.TOPOLOGY_DRIFT,
                severity="info",
                message=(
                    f"{len(appeared)} new worker(s) appeared since the campaign was "
                    f"executed"
                ),
                notes=[
                    f"appeared: {', '.join(sorted(appeared))}",
                    "campaign results do not account for newly available capacity",
                ],
            )
        )

    return warnings


# ---------------------------------------------------------------------------
# Staleness checks
# ---------------------------------------------------------------------------


def _check_staleness(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    now: datetime,
    staleness_hours: int,
) -> list[CampaignHonestyWarning]:
    threshold = now - timedelta(hours=staleness_hours)

    latest_evidence_window: datetime | None = None
    for record in campaign_artifact.evidence_records:
        if record.window_ended_at is not None:
            if (
                latest_evidence_window is None
                or record.window_ended_at > latest_evidence_window
            ):
                latest_evidence_window = record.window_ended_at

    if latest_evidence_window is not None and latest_evidence_window < threshold:
        age_hours = (now - latest_evidence_window).total_seconds() / 3600.0
        return [
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.STALE_EVIDENCE,
                severity="warning",
                message=(
                    f"newest evidence is {age_hours:.0f}h old "
                    f"(threshold: {staleness_hours}h)"
                ),
                notes=[
                    "recommendations based on stale evidence may not reflect "
                    "current system behavior",
                ],
            )
        ]

    if campaign_artifact.timestamp < threshold:
        age_hours = (now - campaign_artifact.timestamp).total_seconds() / 3600.0
        return [
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.STALE_EVIDENCE,
                severity="info",
                message=(
                    f"campaign artifact is {age_hours:.0f}h old "
                    f"(threshold: {staleness_hours}h)"
                ),
                notes=[
                    "consider re-running the campaign with fresh evidence",
                ],
            )
        ]

    return []


# ---------------------------------------------------------------------------
# Workload coverage / overfit checks
# ---------------------------------------------------------------------------


def _check_workload_coverage(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    min_workload_families: int,
) -> list[CampaignHonestyWarning]:
    """Check whether the campaign's actual evaluated evidence covers enough workload families.

    We look at three sources of actual coverage:
    1. Workload impact summaries from comparison (what was actually measured)
    2. Benefited/regressed workload families from recommendations
    3. Scenario families in the input evidence records' notes
    """

    evaluated_families: set[str] = set()

    # Source 1: workload families from recommendation summaries
    for trial in campaign_artifact.trials:
        if trial.recommendation_summary is not None:
            evaluated_families.update(
                trial.recommendation_summary.benefited_workload_families
            )
            evaluated_families.update(
                trial.recommendation_summary.regressed_workload_families
            )
        # Source 2: evidence record notes may carry scenario_family info
        for record in trial.evidence_records:
            for note in record.notes:
                if note.startswith("scenario_family="):
                    _, _, value = note.partition("=")
                    if value:
                        evaluated_families.add(value)

    # Source 3: campaign-level evidence record notes
    for record in campaign_artifact.evidence_records:
        for note in record.notes:
            if note.startswith("scenario_family="):
                _, _, value = note.partition("=")
                if value:
                    evaluated_families.add(value)

    # If we couldn't determine actual evaluated families, check the benchmark
    # scenario families that were declared in evidence
    if not evaluated_families:
        scenario_families_from_evidence = _scenario_families_from_evidence_notes(
            campaign_artifact
        )
        evaluated_families.update(scenario_families_from_evidence)

    # If we still have no family information, use evidence record count as a proxy:
    # campaigns with evidence from only one run_kind are narrow
    if not evaluated_families and campaign_artifact.trials:
        evidence_source_ids = {
            record.source_artifact_id
            for record in campaign_artifact.evidence_records
            if record.source_type.value == "benchmark_run"
        }
        if len(evidence_source_ids) <= 1:
            evaluated_families.add("single_source")

    if (
        0 < len(evaluated_families) < min_workload_families
        and campaign_artifact.trials
    ):
        return [
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.NARROW_WORKLOAD_COVERAGE,
                severity="warning",
                message=(
                    f"campaign evidence covers only {len(evaluated_families)} workload "
                    f"family/families ({', '.join(sorted(evaluated_families))}); "
                    f"minimum recommended is {min_workload_families}"
                ),
                notes=[
                    "recommendations may overfit to the tested workload shape",
                    "consider adding diverse scenario families before promoting",
                ],
            )
        ]

    return []


def _scenario_families_from_evidence_notes(
    campaign_artifact: OptimizationCampaignArtifact,
) -> set[str]:
    """Extract scenario families from evidence record notes."""

    families: set[str] = set()
    for record in campaign_artifact.evidence_records:
        for note in record.notes:
            if note.startswith("scenario_family="):
                _, _, value = note.partition("=")
                if value:
                    families.add(value)
            elif note.startswith("run_kind="):
                _, _, value = note.partition("=")
                if value:
                    families.add(value)
    return families


# ---------------------------------------------------------------------------
# Evidence consistency checks
# ---------------------------------------------------------------------------


def _check_evidence_consistency(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
) -> list[CampaignHonestyWarning]:
    warnings: list[CampaignHonestyWarning] = []

    evidence_kinds = {
        record.evidence_kind for record in campaign_artifact.evidence_records
    }
    has_observed = OptimizationArtifactEvidenceKind.OBSERVED in evidence_kinds
    has_replayed = OptimizationArtifactEvidenceKind.REPLAYED in evidence_kinds
    has_simulated = OptimizationArtifactEvidenceKind.SIMULATED in evidence_kinds

    promote_trial_ids = [
        trial.trial_artifact_id
        for trial in campaign_artifact.trials
        if trial.recommendation_summary is not None
        and trial.recommendation_summary.disposition
        is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    ]

    if promote_trial_ids and not has_observed:
        severity = "warning"
        if has_simulated and not has_replayed:
            severity = "error"
        warnings.append(
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.OBSERVED_EVIDENCE_MISSING,
                severity=severity,
                message=(
                    "promotion recommendations exist but no observed runtime "
                    "evidence backs the campaign"
                ),
                affected_trial_ids=promote_trial_ids,
                notes=[
                    "offline replay and simulation results alone may not reflect "
                    "real production behavior",
                    "consider requiring observed evidence before acting on promotions",
                ],
            )
        )

    if has_observed and (has_replayed or has_simulated):
        _warnings = _check_observed_vs_replayed_inconsistency(
            campaign_artifact=campaign_artifact,
        )
        warnings.extend(_warnings)

    return warnings


def _check_observed_vs_replayed_inconsistency(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
) -> list[CampaignHonestyWarning]:
    """Check for trials where observed and replayed evidence produce contradictory signals."""

    warnings: list[CampaignHonestyWarning] = []
    inconsistent_trial_ids: list[str] = []

    for trial in campaign_artifact.trials:
        trial_evidence_kinds = {
            record.evidence_kind for record in trial.evidence_records
        }
        has_observed = OptimizationArtifactEvidenceKind.OBSERVED in trial_evidence_kinds
        has_simulated = (
            OptimizationArtifactEvidenceKind.SIMULATED in trial_evidence_kinds
            or OptimizationArtifactEvidenceKind.ESTIMATED in trial_evidence_kinds
        )
        if not (has_observed and has_simulated):
            continue

        recommendation = trial.recommendation_summary
        if recommendation is None:
            continue

        evidence_mix = recommendation.evidence_mix
        if evidence_mix is None:
            continue

        observed_share = evidence_mix.observed_share or 0.0
        estimated_share = evidence_mix.estimated_share or 0.0

        if (
            observed_share > 0.0
            and estimated_share > 0.0
            and estimated_share > observed_share
        ):
            inconsistent_trial_ids.append(trial.trial_artifact_id)

    if inconsistent_trial_ids:
        warnings.append(
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.EVIDENCE_INCONSISTENCY,
                severity="warning",
                message=(
                    f"{len(inconsistent_trial_ids)} trial(s) have estimated evidence "
                    f"share exceeding observed share, indicating mixed-quality inputs"
                ),
                affected_trial_ids=inconsistent_trial_ids,
                notes=[
                    "when estimated evidence outweighs observed evidence, the "
                    "recommendation may not reflect real system behavior",
                    "consider collecting more observed cloud-backed evidence",
                ],
            )
        )

    # Check for contradictory objective assessments across evidence kinds
    contradiction_trial_ids: list[str] = []
    for trial in campaign_artifact.trials:
        if not trial.objective_assessments:
            continue
        observed_objectives = [
            assessment
            for assessment in trial.objective_assessments
            if OptimizationArtifactEvidenceKind.OBSERVED in assessment.evidence_kinds
            and assessment.satisfied is not None
        ]
        simulated_objectives = [
            assessment
            for assessment in trial.objective_assessments
            if (
                OptimizationArtifactEvidenceKind.SIMULATED in assessment.evidence_kinds
                or OptimizationArtifactEvidenceKind.ESTIMATED in assessment.evidence_kinds
            )
            and assessment.satisfied is not None
        ]
        for observed in observed_objectives:
            for simulated in simulated_objectives:
                if (
                    observed.objective_id == simulated.objective_id
                    and observed.satisfied != simulated.satisfied
                ):
                    contradiction_trial_ids.append(trial.trial_artifact_id)
                    break
            else:
                continue
            break

    if contradiction_trial_ids:
        warnings.append(
            CampaignHonestyWarning(
                kind=CampaignHonestyWarningKind.EVIDENCE_INCONSISTENCY,
                severity="error",
                message=(
                    f"{len(contradiction_trial_ids)} trial(s) show contradictory "
                    f"objective outcomes between observed and simulated evidence"
                ),
                affected_trial_ids=contradiction_trial_ids,
                notes=[
                    "observed and simulated evidence disagree on whether objectives "
                    "were satisfied, making the recommendation unreliable",
                    "the campaign should be re-evaluated with consistent evidence",
                ],
            )
        )

    return warnings


# ---------------------------------------------------------------------------
# Cost signal honesty checks
# ---------------------------------------------------------------------------


def _check_cost_signal_honesty(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
) -> list[CampaignHonestyWarning]:
    """Check whether cost-related evidence is observed or merely configured.

    When a campaign's remote budget or spend constraints were evaluated using
    configured values rather than observed runtime cost signals, the
    recommendations may not reflect actual cloud spend behavior.
    """

    warnings: list[CampaignHonestyWarning] = []

    # Identify trials with remote budget/spend constraints that lack observed evidence
    for trial in campaign_artifact.trials:
        remote_constraints = [
            assessment
            for assessment in trial.constraint_assessments
            if assessment.dimension
            in {
                OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE,
                OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                OptimizationConstraintDimension.REMOTE_CONCURRENCY_CAP,
            }
        ]
        if not remote_constraints:
            continue

        # Check if any remote constraint was evaluated without observed evidence
        config_only_constraints = [
            assessment
            for assessment in remote_constraints
            if assessment.evidence_kinds
            and OptimizationArtifactEvidenceKind.OBSERVED not in assessment.evidence_kinds
        ]
        if config_only_constraints:
            constraint_ids = [
                assessment.constraint_id for assessment in config_only_constraints
            ]
            warnings.append(
                CampaignHonestyWarning(
                    kind=CampaignHonestyWarningKind.COST_SIGNAL_MISMATCH,
                    severity="info",
                    message=(
                        f"trial {trial.trial_artifact_id} evaluated "
                        f"{len(config_only_constraints)} remote budget constraint(s) "
                        f"using configured values rather than observed cost signals"
                    ),
                    affected_trial_ids=[trial.trial_artifact_id],
                    affected_constraint_ids=constraint_ids,
                    notes=[
                        "configured cost signals may not reflect actual cloud spend",
                        "consider running the campaign with observed cloud-backed "
                        "cost evidence for higher confidence",
                    ],
                )
            )

    return warnings
