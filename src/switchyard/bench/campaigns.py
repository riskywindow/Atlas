"""Forge Stage A offline campaign execution helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from switchyard.bench.campaign_comparison import compare_optimization_campaign
from switchyard.bench.campaign_honesty import assess_campaign_honesty
from switchyard.bench.candidate_generation import generate_forge_stage_a_candidates
from switchyard.bench.recommendations import build_policy_recommendation_report
from switchyard.bench.simulation import (
    compare_candidate_policies_offline,
    compatibility_policy_spec,
)
from switchyard.config import Settings
from switchyard.optimization import build_optimization_profile
from switchyard.schemas.backend import BackendInstance
from switchyard.schemas.benchmark import (
    BenchmarkRunArtifact,
    BenchmarkRunKind,
    CapturedTraceRecord,
    CounterfactualObjective,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    DeployedTopologyEndpoint,
    PolicyRecommendationReportArtifact,
    RecommendationConfidence,
    RecommendationDisposition,
    RoutingPolicyGuidance,
)
from switchyard.schemas.forge import (
    ForgeCampaignInspectionResponse,
    ForgeCampaignInspectionSummary,
    ForgeCandidateDiffEntry,
    ForgeCandidateProvenanceSummary,
    ForgeHonestyWarningKind,
    ForgeHonestyWarningSummary,
    ForgeTrialInspectionSummary,
)
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactSourceType,
    OptimizationArtifactStatus,
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationCampaignMetadata,
    OptimizationCandidateComparisonArtifact,
    OptimizationCandidateConfigurationArtifact,
    OptimizationCandidateEligibilityRecord,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationMetadata,
    OptimizationCandidateGenerationResult,
    OptimizationComparisonOperator,
    OptimizationConstraint,
    OptimizationConstraintAssessment,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationEvidenceRecord,
    OptimizationGoal,
    OptimizationKnobChange,
    OptimizationObjectiveAssessment,
    OptimizationObjectiveMetric,
    OptimizationObjectiveTarget,
    OptimizationProfile,
    OptimizationPromotionDecision,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationSummary,
    OptimizationTopologyLineage,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import (
    PolicyRolloutMode,
    RoutingPolicy,
    TopologySnapshotReference,
)


@dataclass(frozen=True, slots=True)
class ForgeStageAExecutionResult:
    """Result bundle for one offline Forge Stage A campaign execution."""

    campaign_artifact: OptimizationCampaignArtifact
    campaign_comparison: OptimizationCampaignComparisonArtifact | None
    simulation_comparison: CounterfactualSimulationComparisonArtifact | None
    recommendation_report: PolicyRecommendationReportArtifact | None
    candidate_generation: OptimizationCandidateGenerationResult | None
    skipped_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]


def inspect_forge_stage_a_campaigns(
    *,
    campaign_artifacts: Sequence[OptimizationCampaignArtifact],
    comparison_artifacts: Sequence[OptimizationCampaignComparisonArtifact] = (),
    current_worker_inventory: Sequence[BackendInstance] | None = None,
    current_remote_budget_per_minute: int | None = None,
    current_max_remote_share_percent: float | None = None,
    current_remote_concurrency_cap: int | None = None,
) -> ForgeCampaignInspectionResponse:
    """Build an operator-facing inspection view over one or more campaign artifacts.

    When current environment state is supplied, honesty checks are run against
    each campaign and warnings are attached to the inspection summaries so
    operators can see when results may no longer be trustworthy.
    """

    comparison_by_campaign_id = {
        comparison.campaign_id: comparison for comparison in comparison_artifacts
    }
    campaigns = [
        _campaign_inspection_summary(
            campaign_artifact=campaign_artifact,
            comparison_artifact=comparison_by_campaign_id.get(
                campaign_artifact.campaign.campaign_id
            ),
            current_worker_inventory=current_worker_inventory,
            current_remote_budget_per_minute=current_remote_budget_per_minute,
            current_max_remote_share_percent=current_max_remote_share_percent,
            current_remote_concurrency_cap=current_remote_concurrency_cap,
        )
        for campaign_artifact in campaign_artifacts
    ]
    notes = [
        "inspection summaries are derived from authoritative campaign and comparison artifacts",
        "observed, replayed, simulated, and estimated evidence kinds remain listed explicitly",
        "honesty checks (staleness, workload coverage, evidence consistency, cost signals) "
        "always run; budget and topology checks use environment state when available",
    ]
    if current_worker_inventory is not None:
        notes.append(
            "topology drift checks were run against the current worker inventory"
        )
    if not campaigns:
        notes.append("no campaign artifacts were provided for inspection")
    return ForgeCampaignInspectionResponse(campaigns=campaigns, notes=notes)


def execute_forge_stage_a_campaign(
    *,
    settings: Settings,
    evaluation_artifacts: Sequence[BenchmarkRunArtifact] | None = None,
    evaluation_trace_records: Sequence[CapturedTraceRecord] | None = None,
    history_artifacts: Sequence[BenchmarkRunArtifact] | None = None,
    history_trace_records: Sequence[CapturedTraceRecord] | None = None,
    candidate_generation_config: OptimizationCandidateGenerationConfig | None = None,
    candidate_generation_result: OptimizationCandidateGenerationResult | None = None,
    prior_best_candidate_configurations: Sequence[OptimizationCandidateConfigurationArtifact]
    | None = None,
    timestamp: datetime | None = None,
) -> ForgeStageAExecutionResult:
    """Execute the first bounded Forge Stage A offline campaign slice."""

    resolved_evaluation_artifacts = list(evaluation_artifacts or [])
    resolved_evaluation_traces = list(evaluation_trace_records or [])
    if not resolved_evaluation_artifacts and not resolved_evaluation_traces:
        msg = "offline campaign execution requires at least one benchmark artifact or trace"
        raise ValueError(msg)

    run_timestamp = timestamp or datetime.now(UTC)
    resolved_history_artifacts = (
        list(history_artifacts)
        if history_artifacts is not None
        else list(resolved_evaluation_artifacts)
    )
    resolved_history_traces = (
        list(history_trace_records)
        if history_trace_records is not None
        else list(resolved_evaluation_traces)
    )

    profile = build_optimization_profile(settings)
    if profile.campaign is None or profile.baseline_trial is None:
        msg = "optimization profile must expose campaign and baseline trial metadata"
        raise ValueError(msg)

    campaign = profile.campaign
    generation_result = candidate_generation_result
    if generation_result is None and candidate_generation_config is not None:
        generation_result = generate_forge_stage_a_candidates(
            settings=settings,
            profile=profile,
            config=candidate_generation_config,
            history_artifacts=resolved_history_artifacts,
            prior_best_candidate_configurations=prior_best_candidate_configurations,
            timestamp=run_timestamp,
        )
    topology_lineage = _topology_lineage(
        benchmark_artifacts=[
            *resolved_evaluation_artifacts,
            *resolved_history_artifacts,
        ],
        trace_records=[*resolved_evaluation_traces, *resolved_history_traces],
    )
    shared_input_evidence = _source_evidence_records(
        benchmark_artifacts=[
            *resolved_evaluation_artifacts,
            *resolved_history_artifacts,
        ],
        trace_records=[*resolved_evaluation_traces, *resolved_history_traces],
    )
    expected_input_evidence = _expected_input_evidence_kinds(shared_input_evidence)

    baseline_candidate = _candidate_configuration_artifact(
        campaign_id=campaign.campaign_id,
        baseline_trial=profile.baseline_trial,
        trial=profile.baseline_trial,
        profile=profile,
        topology_lineage=topology_lineage,
        expected_input_evidence=expected_input_evidence,
    )
    generated_candidates = (
        generation_result.eligible_candidates if generation_result is not None else None
    )
    candidate_configurations = (
        [
            _candidate_configuration_artifact(
                campaign_id=campaign.campaign_id,
                baseline_trial=profile.baseline_trial,
                trial=generated.trial,
                profile=profile,
                topology_lineage=topology_lineage,
                expected_input_evidence=expected_input_evidence,
                knob_changes=list(generated.knob_changes),
                generation=generated.generation,
                eligibility=generated.eligibility,
            )
            for generated in generated_candidates
        ]
        if generated_candidates is not None
        else [
            _candidate_configuration_artifact(
                campaign_id=campaign.campaign_id,
                baseline_trial=profile.baseline_trial,
                trial=trial,
                profile=profile,
                topology_lineage=topology_lineage,
                expected_input_evidence=expected_input_evidence,
            )
            for trial in profile.candidate_trials
        ]
    )
    candidate_configuration_by_trial = {
        candidate.candidate.trial_id: candidate for candidate in candidate_configurations
    }

    candidate_trials = [candidate.candidate for candidate in candidate_configurations]
    supported_trials = [
        trial
        for trial in candidate_trials
        if trial.candidate_kind is ForgeCandidateKind.ROUTING_POLICY
        and trial.routing_policy is not None
    ]
    skipped_candidate_ids = tuple(
        trial.candidate_id
        for trial in candidate_trials
        if trial.trial_id not in {supported.trial_id for supported in supported_trials}
    )
    rejected_candidate_ids = (
        tuple(candidate.trial.candidate_id for candidate in generation_result.rejected_candidates)
        if generation_result is not None
        else ()
    )

    comparison: CounterfactualSimulationComparisonArtifact | None = None
    campaign_comparison: OptimizationCampaignComparisonArtifact | None = None
    report: PolicyRecommendationReportArtifact | None = None
    trial_artifacts: list[OptimizationTrialArtifact] = []
    campaign_evidence = list(shared_input_evidence)

    if supported_trials:
        policies = [
            compatibility_policy_spec(profile.active_routing_policy),
            *[
                compatibility_policy_spec(trial.routing_policy)
                for trial in supported_trials
                if trial.routing_policy is not None
            ],
        ]
        comparison = compare_candidate_policies_offline(
            policies=policies,
            evaluation_artifacts=resolved_evaluation_artifacts,
            evaluation_trace_records=resolved_evaluation_traces,
            history_artifacts=resolved_history_artifacts,
            history_trace_records=resolved_history_traces,
            timestamp=run_timestamp,
        )
        report = build_policy_recommendation_report(
            [*resolved_evaluation_artifacts, comparison],
            report_id=_bounded_id(campaign.campaign_id, suffix="recommendations"),
            timestamp=run_timestamp,
        )
        proposed_canary = _proposed_canary_percentage(settings)
        campaign_evidence.extend(_comparison_evidence_records(comparison))
        evaluations = {
            evaluation.policy.policy_id: evaluation for evaluation in comparison.evaluations
        }
        baseline_evaluation = evaluations[profile.active_routing_policy.value]
        for trial in supported_trials:
            if trial.routing_policy is None:
                continue
            candidate_configuration = candidate_configuration_by_trial[trial.trial_id]
            candidate_evaluation = evaluations[trial.routing_policy.value]
            trial_artifacts.append(
                _trial_artifact(
                    campaign_id=campaign.campaign_id,
                    baseline_candidate_configuration=baseline_candidate,
                    candidate_configuration=candidate_configuration,
                    baseline_evaluation=baseline_evaluation,
                    candidate_evaluation=candidate_evaluation,
                    shared_input_evidence=shared_input_evidence,
                    campaign=campaign,
                    report=report,
                    settings=settings,
                    topology_lineage=topology_lineage,
                    timestamp=run_timestamp,
                )
            )
        campaign_snapshot = OptimizationCampaignArtifact(
            campaign_artifact_id=_bounded_id(campaign.campaign_id, suffix="artifact"),
            timestamp=run_timestamp,
            campaign=campaign,
            baseline_candidate_configuration=baseline_candidate,
            candidate_configurations=candidate_configurations,
            trials=trial_artifacts,
            evidence_records=campaign_evidence,
            topology_lineage=topology_lineage,
            recommendation_summaries=[
                trial.recommendation_summary
                for trial in trial_artifacts
                if trial.recommendation_summary is not None
            ],
            promotion_decisions=[
                trial.promotion_decision
                for trial in trial_artifacts
                if trial.promotion_decision is not None
            ],
            result_status=OptimizationArtifactStatus.COMPLETE,
            notes=[],
        )
        campaign_comparison = compare_optimization_campaign(
            campaign_artifact=campaign_snapshot,
            simulation_comparison=comparison,
            evaluation_artifacts=resolved_evaluation_artifacts,
            timestamp=run_timestamp,
        )
        comparison_by_candidate = {
            candidate_comparison.candidate_configuration_id: candidate_comparison
            for candidate_comparison in campaign_comparison.candidate_comparisons
        }
        trial_artifacts = [
            trial.model_copy(
                update={
                    "recommendation_summary": comparison_by_candidate[
                        trial.candidate_configuration.candidate_configuration_id
                    ].recommendation_summary,
                    "promotion_decision": _promotion_decision(
                        candidate_configuration=trial.candidate_configuration,
                        recommendation_summary=comparison_by_candidate[
                            trial.candidate_configuration.candidate_configuration_id
                        ].recommendation_summary,
                        proposed_canary_percentage=proposed_canary,
                        settings=settings,
                        timestamp=run_timestamp,
                    ),
                },
                deep=True,
            )
            for trial in trial_artifacts
        ]

    result_status = (
        OptimizationArtifactStatus.COMPLETE
        if not skipped_candidate_ids
        and all(
            trial.result_status is OptimizationArtifactStatus.COMPLETE for trial in trial_artifacts
        )
        else OptimizationArtifactStatus.PARTIAL
    )
    notes = [
        "offline campaign execution is read-only and does not mutate live rollout state",
        "promotion decisions remain recommendation-only and bounded to explicit canary posture",
        "observed, replayed, simulated, and estimated evidence stay typed separately",
    ]
    if skipped_candidate_ids:
        notes.append(
            "unsupported candidate families were exported as candidate configurations but "
            "not executed as offline trials"
        )
    if rejected_candidate_ids:
        notes.append("candidate generation rejected or pruned unsafe combinations before execution")
    if comparison is None:
        notes.append("no routing-policy candidates were available for offline comparison")
    if campaign_comparison is not None:
        notes.append("campaign comparison ranked executed candidates with Pareto-style posture")

    return ForgeStageAExecutionResult(
        campaign_artifact=OptimizationCampaignArtifact(
            campaign_artifact_id=_bounded_id(campaign.campaign_id, suffix="artifact"),
            timestamp=run_timestamp,
            campaign=campaign,
            baseline_candidate_configuration=baseline_candidate,
            candidate_configurations=candidate_configurations,
            trials=trial_artifacts,
            evidence_records=campaign_evidence,
            topology_lineage=topology_lineage,
            recommendation_summaries=[
                trial.recommendation_summary
                for trial in trial_artifacts
                if trial.recommendation_summary is not None
            ],
            promotion_decisions=[
                trial.promotion_decision
                for trial in trial_artifacts
                if trial.promotion_decision is not None
            ],
            result_status=result_status,
            notes=notes,
        ),
        campaign_comparison=campaign_comparison,
        simulation_comparison=comparison,
        recommendation_report=report,
        candidate_generation=generation_result,
        skipped_candidate_ids=skipped_candidate_ids,
        rejected_candidate_ids=rejected_candidate_ids,
    )


def _candidate_configuration_artifact(
    *,
    campaign_id: str,
    baseline_trial: OptimizationTrialIdentity,
    trial: OptimizationTrialIdentity,
    profile: OptimizationProfile,
    topology_lineage: OptimizationTopologyLineage | None,
    expected_input_evidence: list[OptimizationArtifactEvidenceKind],
    knob_changes: list[OptimizationKnobChange] | None = None,
    generation: OptimizationCandidateGenerationMetadata | None = None,
    eligibility: OptimizationCandidateEligibilityRecord | None = None,
) -> OptimizationCandidateConfigurationArtifact:
    simulated_evidence = (
        [OptimizationArtifactEvidenceKind.SIMULATED, OptimizationArtifactEvidenceKind.ESTIMATED]
        if trial.candidate_kind is ForgeCandidateKind.ROUTING_POLICY
        else []
    )
    return OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=_bounded_id(trial.trial_id, suffix="candidate"),
        timestamp=profile.generated_at,
        campaign_id=campaign_id,
        candidate=trial,
        baseline_config_profile_id=baseline_trial.config_profile_id,
        config_profile_id=trial.config_profile_id,
        knob_changes=(
            knob_changes
            if knob_changes is not None
            else _candidate_knob_changes(
                baseline_trial=baseline_trial,
                trial=trial,
                active_routing_policy=profile.active_routing_policy,
                candidate_policy_id=profile.candidate_policy_id,
            )
        ),
        objectives_in_scope=list(profile.objectives),
        constraints_in_scope=list(profile.constraints),
        workload_sets=list(profile.workload_sets),
        expected_evidence_kinds=[
            *expected_input_evidence,
            *[
                evidence_kind
                for evidence_kind in simulated_evidence
                if evidence_kind not in expected_input_evidence
            ],
        ],
        generation=generation,
        eligibility=eligibility,
        topology_lineage=topology_lineage,
        notes=_candidate_configuration_notes(trial),
    )


def _candidate_knob_changes(
    *,
    baseline_trial: OptimizationTrialIdentity,
    trial: OptimizationTrialIdentity,
    active_routing_policy: RoutingPolicy,
    candidate_policy_id: str | None,
) -> list[OptimizationKnobChange]:
    if trial.trial_id == baseline_trial.trial_id:
        return []
    if (
        trial.candidate_kind is ForgeCandidateKind.ROUTING_POLICY
        and trial.routing_policy is not None
    ):
        return [
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value=active_routing_policy.value,
                candidate_value=trial.routing_policy.value,
                notes=["offline trial changes the active routing policy candidate"],
            )
        ]
    if trial.candidate_kind is ForgeCandidateKind.ROLLOUT_POLICY:
        return [
            OptimizationKnobChange(
                knob_id="candidate_rollout_policy_id",
                config_path="phase4.policy_rollout.candidate_policy_id",
                baseline_value=candidate_policy_id,
                candidate_value=trial.rollout_policy_id,
                notes=["rollout-policy candidate remains bounded behind rollout controls"],
            )
        ]
    if trial.worker_launch_preset is not None:
        return [
            OptimizationKnobChange(
                knob_id="worker_launch_preset",
                config_path="optimization.worker_launch_presets.selection",
                baseline_value=None,
                candidate_value=trial.worker_launch_preset,
                notes=["candidate selects an exported worker launch preset"],
            )
        ]
    return []


def _candidate_configuration_notes(trial: OptimizationTrialIdentity) -> list[str]:
    if trial.candidate_kind is ForgeCandidateKind.ROUTING_POLICY:
        return ["routing-policy candidate is executable by the offline Phase 9 runner"]
    if trial.candidate_kind is ForgeCandidateKind.ROLLOUT_POLICY:
        return [
            "rollout-policy candidate is exported for lineage but not executed by the "
            "first offline runner"
        ]
    if trial.worker_launch_preset is not None:
        return [
            "worker launch preset candidates remain lineage-only until replay-backed "
            "runtime-profile execution is added"
        ]
    return ["candidate configuration is exported for lineage and later campaign slices"]


def _source_evidence_records(
    *,
    benchmark_artifacts: Sequence[BenchmarkRunArtifact],
    trace_records: Sequence[CapturedTraceRecord],
) -> list[OptimizationEvidenceRecord]:
    evidence_by_id: dict[str, OptimizationEvidenceRecord] = {}
    for artifact in benchmark_artifacts:
        for record in _benchmark_evidence_records(artifact):
            evidence_by_id[record.evidence_id] = record
    trace_record = _trace_evidence_record(trace_records)
    if trace_record is not None:
        evidence_by_id[trace_record.evidence_id] = trace_record
    return list(evidence_by_id.values())


def _benchmark_evidence_records(
    artifact: BenchmarkRunArtifact,
) -> list[OptimizationEvidenceRecord]:
    evidence_summary = artifact.summary.evidence_summary
    notes = [f"run_kind={artifact.run_kind.value}"]
    if evidence_summary is not None:
        notes.extend(evidence_summary.confidence_notes)
        notes.extend(evidence_summary.comparability_limitations)
    hybrid_summary = artifact.summary.hybrid_summary
    if hybrid_summary is not None and artifact.summary.request_count > 0:
        remote_share = (
            (hybrid_summary.remote_only_count + hybrid_summary.hybrid_spillover_count)
            / artifact.summary.request_count
        ) * 100.0
        notes.append(f"remote_share_percent={remote_share:.6f}")
    record = OptimizationEvidenceRecord(
        evidence_id=_bounded_id(artifact.run_id, suffix="evidence"),
        evidence_kind=_benchmark_evidence_kind(artifact),
        source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
        source_artifact_id=artifact.run_id,
        source_run_ids=[artifact.run_id],
        window_started_at=min((record.started_at for record in artifact.records), default=None),
        window_ended_at=max((record.completed_at for record in artifact.records), default=None),
        notes=notes,
    )
    return [record]


def _benchmark_evidence_kind(artifact: BenchmarkRunArtifact) -> OptimizationArtifactEvidenceKind:
    evidence_summary = artifact.summary.evidence_summary
    if evidence_summary is not None and (
        evidence_summary.estimated_request_count > 0
        or evidence_summary.configured_cloud_request_count > 0
        or evidence_summary.mock_request_count > 0
    ):
        if (
            evidence_summary.observed_cloud_request_count == 0
            and artifact.run_kind is not BenchmarkRunKind.LOCAL_ONLY
        ):
            return OptimizationArtifactEvidenceKind.ESTIMATED
    return OptimizationArtifactEvidenceKind.OBSERVED


def _trace_evidence_record(
    trace_records: Sequence[CapturedTraceRecord],
) -> OptimizationEvidenceRecord | None:
    if not trace_records:
        return None
    trace_ids = sorted({trace.record_id for trace in trace_records})
    return OptimizationEvidenceRecord(
        evidence_id=_bounded_id(trace_ids[0], suffix="traces"),
        evidence_kind=OptimizationArtifactEvidenceKind.REPLAYED,
        source_type=OptimizationArtifactSourceType.CAPTURED_TRACE,
        source_artifact_id=f"trace-set:{trace_ids[0]}",
        source_trace_ids=trace_ids,
        window_started_at=min(
            (trace.request_timestamp for trace in trace_records),
            default=None,
        ),
        window_ended_at=max(
            (trace.request_timestamp for trace in trace_records),
            default=None,
        ),
        notes=["captured traces are replay-backed evidence, not live observed execution"],
    )


def _comparison_evidence_records(
    comparison: CounterfactualSimulationComparisonArtifact,
) -> list[OptimizationEvidenceRecord]:
    records = [
        OptimizationEvidenceRecord(
            evidence_id=_bounded_id(comparison.simulation_comparison_id, suffix="simulation"),
            evidence_kind=OptimizationArtifactEvidenceKind.SIMULATED,
            source_type=OptimizationArtifactSourceType.SIMULATION_COMPARISON,
            source_artifact_id=comparison.simulation_comparison_id,
            source_run_ids=list(comparison.source_run_ids),
            source_trace_ids=list(comparison.source_trace_ids),
            source_simulation_ids=[
                evaluation.simulation_id for evaluation in comparison.evaluations
            ],
            notes=list(comparison.limitation_notes),
        )
    ]
    if any(_evaluation_uses_estimates(evaluation) for evaluation in comparison.evaluations):
        records.append(
            OptimizationEvidenceRecord(
                evidence_id=_bounded_id(
                    comparison.simulation_comparison_id,
                    suffix="estimates",
                ),
                evidence_kind=OptimizationArtifactEvidenceKind.ESTIMATED,
                source_type=OptimizationArtifactSourceType.ESTIMATE_SUMMARY,
                source_artifact_id=comparison.simulation_comparison_id,
                source_run_ids=list(comparison.source_run_ids),
                source_trace_ids=list(comparison.source_trace_ids),
                source_simulation_ids=[
                    evaluation.simulation_id
                    for evaluation in comparison.evaluations
                    if _evaluation_uses_estimates(evaluation)
                ],
                notes=[
                    "projected outcomes include predictor-derived estimates and stay distinct "
                    "from direct observations"
                ],
            )
        )
    return records


def _trial_artifact(
    *,
    campaign_id: str,
    baseline_candidate_configuration: OptimizationCandidateConfigurationArtifact,
    candidate_configuration: OptimizationCandidateConfigurationArtifact,
    baseline_evaluation: CounterfactualSimulationArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact,
    shared_input_evidence: Sequence[OptimizationEvidenceRecord],
    campaign: OptimizationCampaignMetadata,
    report: PolicyRecommendationReportArtifact,
    settings: Settings,
    topology_lineage: OptimizationTopologyLineage | None,
    timestamp: datetime,
) -> OptimizationTrialArtifact:
    evidence_records = [
        *shared_input_evidence,
        OptimizationEvidenceRecord(
            evidence_id=_bounded_id(candidate_evaluation.simulation_id, suffix="simulation"),
            evidence_kind=OptimizationArtifactEvidenceKind.SIMULATED,
            source_type=OptimizationArtifactSourceType.SIMULATION,
            source_artifact_id=candidate_evaluation.simulation_id,
            source_run_ids=list(candidate_evaluation.source_run_ids),
            source_trace_ids=list(candidate_evaluation.source_trace_ids),
            source_simulation_ids=[candidate_evaluation.simulation_id],
            notes=list(candidate_evaluation.summary.limitation_notes),
        ),
    ]
    if _evaluation_uses_estimates(candidate_evaluation):
        evidence_records.append(
            OptimizationEvidenceRecord(
                evidence_id=_bounded_id(candidate_evaluation.simulation_id, suffix="estimates"),
                evidence_kind=OptimizationArtifactEvidenceKind.ESTIMATED,
                source_type=OptimizationArtifactSourceType.ESTIMATE_SUMMARY,
                source_artifact_id=candidate_evaluation.simulation_id,
                source_run_ids=list(candidate_evaluation.source_run_ids),
                source_trace_ids=list(candidate_evaluation.source_trace_ids),
                source_simulation_ids=[candidate_evaluation.simulation_id],
                notes=["candidate evaluation includes projected estimates"],
            )
        )
    objective_assessments = [
        _objective_assessment(
            objective=objective,
            campaign_objective=campaign.objective,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=candidate_evaluation,
            input_evidence=shared_input_evidence,
        )
        for objective in candidate_configuration.objectives_in_scope
    ]
    proposed_canary = _proposed_canary_percentage(settings)
    constraint_assessments = [
        _constraint_assessment(
            constraint=constraint,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=candidate_evaluation,
            input_evidence=shared_input_evidence,
            settings=settings,
            proposed_canary_percentage=proposed_canary,
        )
        for constraint in candidate_configuration.constraints_in_scope
    ]
    recommendation_summary = _recommendation_summary(
        candidate_configuration=candidate_configuration,
        baseline_evaluation=baseline_evaluation,
        candidate_evaluation=candidate_evaluation,
        constraint_assessments=constraint_assessments,
        report=report,
        campaign_required_sources=campaign.required_evidence_sources,
        evidence_records=evidence_records,
        campaign_objective=campaign.objective,
    )
    promotion_decision = _promotion_decision(
        candidate_configuration=candidate_configuration,
        recommendation_summary=recommendation_summary,
        proposed_canary_percentage=proposed_canary,
        settings=settings,
        timestamp=timestamp,
    )
    result_status = (
        OptimizationArtifactStatus.COMPLETE
        if recommendation_summary.disposition
        is not OptimizationRecommendationDisposition.NEED_MORE_EVIDENCE
        or _required_sources_satisfied(
            required_sources=campaign.required_evidence_sources,
            evidence_records=evidence_records,
        )
        else OptimizationArtifactStatus.PARTIAL
    )
    notes = [
        "trial evaluation is counterfactual and replay-backed; it does not mutate live state",
    ]
    if candidate_evaluation.summary.low_confidence_count > 0:
        notes.append("low-confidence estimates were preserved explicitly in the trial output")
    return OptimizationTrialArtifact(
        trial_artifact_id=_bounded_id(candidate_configuration.candidate.trial_id, suffix="trial"),
        timestamp=timestamp,
        campaign_id=campaign_id,
        baseline_candidate_configuration_id=(
            baseline_candidate_configuration.candidate_configuration_id
        ),
        candidate_configuration=candidate_configuration,
        trial_identity=candidate_configuration.candidate,
        evidence_records=evidence_records,
        topology_lineage=topology_lineage,
        result_status=result_status,
        objective_assessments=objective_assessments,
        constraint_assessments=constraint_assessments,
        recommendation_summary=recommendation_summary,
        promotion_decision=promotion_decision,
        notes=notes,
    )


def _objective_assessment(
    *,
    objective: OptimizationObjectiveTarget,
    campaign_objective: CounterfactualObjective,
    baseline_evaluation: CounterfactualSimulationArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact,
    input_evidence: Sequence[OptimizationEvidenceRecord],
) -> OptimizationObjectiveAssessment:
    measured_value = _objective_measured_value(
        metric=objective.metric,
        candidate_evaluation=candidate_evaluation,
        input_evidence=input_evidence,
    )
    baseline_value = _objective_measured_value(
        metric=objective.metric,
        candidate_evaluation=baseline_evaluation,
        input_evidence=input_evidence,
    )
    satisfied = _objective_satisfied(
        goal=objective.goal,
        measured_value=measured_value,
        baseline_value=baseline_value,
        target_value=objective.target_value,
    )
    notes: list[str] = []
    if measured_value is None:
        notes.append("offline runner does not currently materialize this metric")
    elif objective.metric is _primary_metric_for_campaign(campaign_objective):
        notes.append("assessment follows the campaign primary objective")
    evidence_kinds = _objective_evidence_kinds(
        metric=objective.metric,
        candidate_evaluation=candidate_evaluation,
        input_evidence=input_evidence,
    )
    return OptimizationObjectiveAssessment(
        objective_id=objective.objective_id,
        metric=objective.metric,
        goal=objective.goal,
        measured_value=measured_value,
        target_value=objective.target_value,
        satisfied=satisfied,
        evidence_kinds=evidence_kinds,
        notes=notes,
    )


def _constraint_assessment(
    *,
    constraint: OptimizationConstraint,
    baseline_evaluation: CounterfactualSimulationArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact,
    input_evidence: Sequence[OptimizationEvidenceRecord],
    settings: Settings,
    proposed_canary_percentage: float,
) -> OptimizationConstraintAssessment:
    evaluated_value, evidence_kinds, notes = _constraint_value(
        constraint=constraint,
        baseline_evaluation=baseline_evaluation,
        candidate_evaluation=candidate_evaluation,
        input_evidence=input_evidence,
        settings=settings,
        proposed_canary_percentage=proposed_canary_percentage,
    )
    satisfied = _comparison_satisfied(
        operator=constraint.operator,
        evaluated_value=evaluated_value,
        threshold_value=constraint.threshold_value,
    )
    return OptimizationConstraintAssessment(
        constraint_id=constraint.constraint_id,
        dimension=constraint.dimension,
        strength=constraint.strength,
        operator=constraint.operator,
        threshold_value=constraint.threshold_value,
        evaluated_value=evaluated_value,
        satisfied=satisfied,
        evidence_kinds=evidence_kinds,
        notes=notes,
    )


def _constraint_value(
    *,
    constraint: OptimizationConstraint,
    baseline_evaluation: CounterfactualSimulationArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact,
    input_evidence: Sequence[OptimizationEvidenceRecord],
    settings: Settings,
    proposed_canary_percentage: float,
) -> tuple[bool | int | float | str | None, list[OptimizationArtifactEvidenceKind], list[str]]:
    if constraint.dimension is OptimizationConstraintDimension.PREDICTED_ERROR_RATE:
        return (
            candidate_evaluation.summary.projected_error_rate,
            _simulation_evidence_kinds(candidate_evaluation),
            [],
        )
    if constraint.dimension is OptimizationConstraintDimension.PREDICTED_LATENCY_REGRESSION_MS:
        candidate_latency = candidate_evaluation.summary.projected_avg_latency_ms
        baseline_latency = baseline_evaluation.summary.projected_avg_latency_ms
        if candidate_latency is None or baseline_latency is None:
            return (
                None,
                _simulation_evidence_kinds(candidate_evaluation),
                [
                    "projected latency regression is unavailable for this trial",
                ],
            )
        return (
            max(0.0, candidate_latency - baseline_latency),
            _simulation_evidence_kinds(candidate_evaluation),
            [],
        )
    if constraint.dimension is OptimizationConstraintDimension.CANARY_PERCENTAGE:
        return proposed_canary_percentage, [], []
    if constraint.dimension is OptimizationConstraintDimension.SHADOW_SAMPLING_RATE:
        return settings.phase4.shadow_routing.default_sampling_rate, [], []
    if constraint.dimension is OptimizationConstraintDimension.REMOTE_SHARE_PERCENT:
        remote_share = _remote_share_percent(input_evidence)
        return (
            remote_share,
            _input_evidence_kinds(input_evidence),
            (
                []
                if remote_share is not None
                else ["no observed or replay-backed hybrid summary was available"]
            ),
        )
    if constraint.dimension is OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE:
        return settings.phase7.hybrid_execution.remote_request_budget_per_minute, [], []
    if constraint.dimension is OptimizationConstraintDimension.REMOTE_CONCURRENCY_CAP:
        return settings.phase7.hybrid_execution.remote_concurrency_cap, [], []
    if constraint.dimension is OptimizationConstraintDimension.OPERATOR_REVIEW_REQUIRED:
        return settings.optimization.promotion_requires_operator_review, [], []
    if constraint.dimension is OptimizationConstraintDimension.LOCAL_PREFERENCE_ENABLED:
        return settings.phase7.hybrid_execution.prefer_local, [], []
    return None, [], ["constraint is not yet evaluated by the offline campaign runner"]


def _recommendation_summary(
    *,
    candidate_configuration: OptimizationCandidateConfigurationArtifact,
    baseline_evaluation: CounterfactualSimulationArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact,
    constraint_assessments: Sequence[OptimizationConstraintAssessment],
    report: PolicyRecommendationReportArtifact,
    campaign_required_sources: Sequence[ForgeEvidenceSourceKind],
    evidence_records: Sequence[OptimizationEvidenceRecord],
    campaign_objective: CounterfactualObjective,
) -> OptimizationRecommendationSummary:
    hard_failures = [
        assessment
        for assessment in constraint_assessments
        if assessment.satisfied is False
        and assessment.strength is OptimizationConstraintStrength.HARD
    ]
    policy_id = candidate_configuration.candidate.routing_policy
    supported_scopes = [
        recommendation
        for recommendation in report.recommendations
        if recommendation.recommended_policy_id == (None if policy_id is None else policy_id.value)
    ]
    required_sources_satisfied = _required_sources_satisfied(
        required_sources=campaign_required_sources,
        evidence_records=evidence_records,
    )
    rationale: list[str] = []
    confidence = _candidate_confidence(
        candidate_evaluation=candidate_evaluation,
        supported_scopes=supported_scopes,
    )
    evidence_kinds = sorted(
        {record.evidence_kind for record in evidence_records},
        key=lambda item: item.value,
    )
    if not required_sources_satisfied:
        rationale.append("required evidence sources were not present for this candidate")
        return OptimizationRecommendationSummary(
            recommendation_summary_id=_bounded_id(
                candidate_configuration.candidate.trial_id,
                suffix="recommendation",
            ),
            disposition=OptimizationRecommendationDisposition.NEED_MORE_EVIDENCE,
            confidence=RecommendationConfidence.INSUFFICIENT,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=candidate_configuration.config_profile_id,
            evidence_kinds=evidence_kinds,
            rationale=rationale,
        )
    if candidate_evaluation.summary.unsupported_count == candidate_evaluation.summary.request_count:
        rationale.append("offline comparison could not support any candidate route changes")
        return OptimizationRecommendationSummary(
            recommendation_summary_id=_bounded_id(
                candidate_configuration.candidate.trial_id,
                suffix="recommendation",
            ),
            disposition=OptimizationRecommendationDisposition.NEED_MORE_EVIDENCE,
            confidence=RecommendationConfidence.INSUFFICIENT,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=candidate_configuration.config_profile_id,
            evidence_kinds=evidence_kinds,
            rationale=rationale,
        )
    if hard_failures:
        rationale.append("one or more hard constraints failed under the offline evaluation")
        rationale.extend(assessment.constraint_id for assessment in hard_failures)
        return OptimizationRecommendationSummary(
            recommendation_summary_id=_bounded_id(
                candidate_configuration.candidate.trial_id,
                suffix="recommendation",
            ),
            disposition=OptimizationRecommendationDisposition.KEEP_BASELINE,
            confidence=confidence,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=candidate_configuration.config_profile_id,
            evidence_kinds=evidence_kinds,
            rationale=rationale,
        )
    if _candidate_is_better(
        campaign_objective=campaign_objective,
        baseline_evaluation=baseline_evaluation,
        candidate_evaluation=candidate_evaluation,
    ) and any(
        recommendation.recommendation is RecommendationDisposition.PREFER_POLICY
        for recommendation in supported_scopes
    ):
        rationale.append("scoped offline recommendations support the candidate policy")
        return OptimizationRecommendationSummary(
            recommendation_summary_id=_bounded_id(
                candidate_configuration.candidate.trial_id,
                suffix="recommendation",
            ),
            disposition=OptimizationRecommendationDisposition.PROMOTE_CANDIDATE,
            confidence=confidence,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=candidate_configuration.config_profile_id,
            evidence_kinds=evidence_kinds,
            rationale=rationale,
        )
    rationale.append("offline evidence did not justify changing the active baseline")
    return OptimizationRecommendationSummary(
        recommendation_summary_id=_bounded_id(
            candidate_configuration.candidate.trial_id,
            suffix="recommendation",
        ),
        disposition=OptimizationRecommendationDisposition.NO_CHANGE,
        confidence=confidence,
        candidate_configuration_id=candidate_configuration.candidate_configuration_id,
        config_profile_id=candidate_configuration.config_profile_id,
        evidence_kinds=evidence_kinds,
        rationale=rationale,
    )


def _promotion_decision(
    *,
    candidate_configuration: OptimizationCandidateConfigurationArtifact,
    recommendation_summary: OptimizationRecommendationSummary,
    proposed_canary_percentage: float,
    settings: Settings,
    timestamp: datetime,
) -> OptimizationPromotionDecision:
    disposition = OptimizationPromotionDisposition.NO_ACTION
    rationale = list(recommendation_summary.rationale)
    if (
        recommendation_summary.disposition
        is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    ):
        disposition = OptimizationPromotionDisposition.RECOMMEND_CANARY
        rationale.append("promotion stays recommendation-only and bounded to a reversible canary")
        if settings.optimization.promotion_requires_operator_review:
            rationale.append("explicit operator review is still required before rollout")
    return OptimizationPromotionDecision(
        promotion_decision_id=_bounded_id(
            candidate_configuration.candidate.trial_id,
            suffix="promotion",
        ),
        disposition=disposition,
        candidate_configuration_id=candidate_configuration.candidate_configuration_id,
        config_profile_id=candidate_configuration.config_profile_id,
        rollout_mode=PolicyRolloutMode.CANARY,
        canary_percentage=(
            proposed_canary_percentage
            if disposition is OptimizationPromotionDisposition.RECOMMEND_CANARY
            else 0.0
        ),
        rollback_supported=True,
        decided_at=timestamp,
        rationale=rationale,
    )


def _candidate_confidence(
    *,
    candidate_evaluation: CounterfactualSimulationArtifact,
    supported_scopes: Sequence[RoutingPolicyGuidance],
) -> RecommendationConfidence:
    if candidate_evaluation.summary.request_count == 0:
        return RecommendationConfidence.INSUFFICIENT
    if candidate_evaluation.summary.low_confidence_count > 0:
        return RecommendationConfidence.LOW
    if any(
        recommendation.confidence is RecommendationConfidence.HIGH
        for recommendation in supported_scopes
    ):
        return RecommendationConfidence.HIGH
    if any(
        recommendation.confidence
        in {
            RecommendationConfidence.HIGH,
            RecommendationConfidence.MEDIUM,
        }
        for recommendation in supported_scopes
    ):
        return RecommendationConfidence.MEDIUM
    if candidate_evaluation.summary.direct_observation_count > 0:
        return RecommendationConfidence.MEDIUM
    return RecommendationConfidence.LOW


def _required_sources_satisfied(
    *,
    required_sources: Sequence[ForgeEvidenceSourceKind],
    evidence_records: Sequence[OptimizationEvidenceRecord],
) -> bool:
    present_sources = _present_forge_sources(evidence_records)
    required = set(required_sources)
    if (
        ForgeEvidenceSourceKind.OBSERVED_RUNTIME in required
        and ForgeEvidenceSourceKind.OBSERVED_RUNTIME not in present_sources
    ):
        return False
    if (
        ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION in required
        and ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION not in present_sources
    ):
        return False
    replay_sources = {
        ForgeEvidenceSourceKind.REPLAYED_BENCHMARK,
        ForgeEvidenceSourceKind.REPLAYED_TRACE,
    }
    if required & replay_sources and not present_sources & replay_sources:
        return False
    return True


def _present_forge_sources(
    evidence_records: Sequence[OptimizationEvidenceRecord],
) -> set[ForgeEvidenceSourceKind]:
    present: set[ForgeEvidenceSourceKind] = set()
    for record in evidence_records:
        if record.source_type is OptimizationArtifactSourceType.BENCHMARK_RUN:
            present.add(ForgeEvidenceSourceKind.REPLAYED_BENCHMARK)
        if record.source_type is OptimizationArtifactSourceType.CAPTURED_TRACE:
            present.add(ForgeEvidenceSourceKind.REPLAYED_TRACE)
        if record.evidence_kind is OptimizationArtifactEvidenceKind.OBSERVED:
            present.add(ForgeEvidenceSourceKind.OBSERVED_RUNTIME)
        if record.evidence_kind is OptimizationArtifactEvidenceKind.REPLAYED:
            present.add(ForgeEvidenceSourceKind.REPLAYED_TRACE)
        if record.evidence_kind is OptimizationArtifactEvidenceKind.SIMULATED:
            present.add(ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION)
    return present


def _topology_lineage(
    *,
    benchmark_artifacts: Sequence[BenchmarkRunArtifact],
    trace_records: Sequence[CapturedTraceRecord],
) -> OptimizationTopologyLineage | None:
    topology_references: dict[str, TopologySnapshotReference] = {}
    deployed_topology: dict[str, DeployedTopologyEndpoint] = {}
    worker_inventory: dict[str, BackendInstance] = {}
    for artifact in benchmark_artifacts:
        if artifact.environment.topology_reference is not None:
            topology_references[artifact.environment.topology_reference.topology_snapshot_id] = (
                artifact.environment.topology_reference
            )
        for endpoint in artifact.environment.deployed_topology:
            deployed_topology[endpoint.endpoint_id] = endpoint
        for instance in artifact.environment.worker_instance_inventory:
            worker_inventory[instance.instance_id] = instance
    for trace in trace_records:
        if trace.topology_reference is not None:
            topology_references[trace.topology_reference.topology_snapshot_id] = (
                trace.topology_reference
            )
    if not topology_references and not deployed_topology and not worker_inventory:
        return None
    return OptimizationTopologyLineage(
        topology_references=list(topology_references.values()),
        deployed_topology=list(deployed_topology.values()),
        worker_instance_inventory=list(worker_inventory.values()),
    )


def _expected_input_evidence_kinds(
    evidence_records: Sequence[OptimizationEvidenceRecord],
) -> list[OptimizationArtifactEvidenceKind]:
    return sorted(
        {
            record.evidence_kind
            for record in evidence_records
            if record.evidence_kind
            in {
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.REPLAYED,
                OptimizationArtifactEvidenceKind.ESTIMATED,
            }
        },
        key=lambda item: item.value,
    )


def _simulation_evidence_kinds(
    evaluation: CounterfactualSimulationArtifact,
) -> list[OptimizationArtifactEvidenceKind]:
    evidence_kinds = [OptimizationArtifactEvidenceKind.SIMULATED]
    if _evaluation_uses_estimates(evaluation):
        evidence_kinds.append(OptimizationArtifactEvidenceKind.ESTIMATED)
    return evidence_kinds


def _input_evidence_kinds(
    evidence_records: Sequence[OptimizationEvidenceRecord],
) -> list[OptimizationArtifactEvidenceKind]:
    return sorted(
        {
            record.evidence_kind
            for record in evidence_records
            if record.evidence_kind
            in {
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.REPLAYED,
                OptimizationArtifactEvidenceKind.ESTIMATED,
            }
        },
        key=lambda item: item.value,
    )


def _evaluation_uses_estimates(evaluation: CounterfactualSimulationArtifact) -> bool:
    summary = evaluation.summary
    return (
        summary.predictor_estimate_count > 0
        or summary.low_confidence_count > 0
        or summary.unsupported_count > 0
    )


def _objective_measured_value(
    *,
    metric: OptimizationObjectiveMetric,
    candidate_evaluation: CounterfactualSimulationArtifact,
    input_evidence: Sequence[OptimizationEvidenceRecord],
) -> float | None:
    if metric is OptimizationObjectiveMetric.LATENCY_MS:
        return candidate_evaluation.summary.projected_avg_latency_ms
    if metric is OptimizationObjectiveMetric.ERROR_RATE:
        return candidate_evaluation.summary.projected_error_rate
    if metric is OptimizationObjectiveMetric.TOKENS_PER_SECOND:
        return candidate_evaluation.summary.projected_avg_tokens_per_second
    if metric is OptimizationObjectiveMetric.REMOTE_SHARE_PERCENT:
        return _remote_share_percent(input_evidence)
    return None


def _objective_evidence_kinds(
    *,
    metric: OptimizationObjectiveMetric,
    candidate_evaluation: CounterfactualSimulationArtifact,
    input_evidence: Sequence[OptimizationEvidenceRecord],
) -> list[OptimizationArtifactEvidenceKind]:
    if metric in {
        OptimizationObjectiveMetric.LATENCY_MS,
        OptimizationObjectiveMetric.ERROR_RATE,
        OptimizationObjectiveMetric.TOKENS_PER_SECOND,
    }:
        return _simulation_evidence_kinds(candidate_evaluation)
    if metric is OptimizationObjectiveMetric.REMOTE_SHARE_PERCENT:
        return _input_evidence_kinds(input_evidence)
    return []


def _objective_satisfied(
    *,
    goal: OptimizationGoal,
    measured_value: float | None,
    baseline_value: float | None,
    target_value: float | None,
) -> bool | None:
    if measured_value is None:
        return None
    if goal is OptimizationGoal.AT_MOST and target_value is not None:
        return measured_value <= target_value
    if goal is OptimizationGoal.AT_LEAST and target_value is not None:
        return measured_value >= target_value
    if baseline_value is None:
        return None
    if goal is OptimizationGoal.MINIMIZE:
        return measured_value <= baseline_value
    if goal is OptimizationGoal.MAXIMIZE:
        return measured_value >= baseline_value
    return None


def _comparison_satisfied(
    *,
    operator: OptimizationComparisonOperator,
    evaluated_value: bool | int | float | str | None,
    threshold_value: bool | int | float | str,
) -> bool | None:
    if evaluated_value is None:
        return None
    if isinstance(evaluated_value, bool) and isinstance(threshold_value, bool):
        return (
            evaluated_value == threshold_value
            if operator is OptimizationComparisonOperator.EQ
            else None
        )
    if (
        isinstance(evaluated_value, (int, float))
        and not isinstance(evaluated_value, bool)
        and isinstance(threshold_value, (int, float))
        and not isinstance(threshold_value, bool)
    ):
        if operator is OptimizationComparisonOperator.LTE:
            return float(evaluated_value) <= float(threshold_value)
        if operator is OptimizationComparisonOperator.GTE:
            return float(evaluated_value) >= float(threshold_value)
        return float(evaluated_value) == float(threshold_value)
    if isinstance(evaluated_value, str) and isinstance(threshold_value, str):
        return (
            evaluated_value == threshold_value
            if operator is OptimizationComparisonOperator.EQ
            else None
        )
    return None


def _candidate_is_better(
    *,
    campaign_objective: CounterfactualObjective,
    baseline_evaluation: CounterfactualSimulationArtifact,
    candidate_evaluation: CounterfactualSimulationArtifact,
) -> bool:
    if campaign_objective is CounterfactualObjective.THROUGHPUT:
        baseline_value = baseline_evaluation.summary.projected_avg_tokens_per_second
        candidate_value = candidate_evaluation.summary.projected_avg_tokens_per_second
        return (
            baseline_value is not None
            and candidate_value is not None
            and candidate_value > baseline_value
        )
    if campaign_objective is CounterfactualObjective.RELIABILITY:
        baseline_value = baseline_evaluation.summary.projected_error_rate
        candidate_value = candidate_evaluation.summary.projected_error_rate
        return (
            baseline_value is not None
            and candidate_value is not None
            and candidate_value < baseline_value
        )
    baseline_latency = baseline_evaluation.summary.projected_avg_latency_ms
    candidate_latency = candidate_evaluation.summary.projected_avg_latency_ms
    if baseline_latency is None or candidate_latency is None:
        return False
    if campaign_objective is CounterfactualObjective.BALANCED:
        baseline_error = baseline_evaluation.summary.projected_error_rate
        candidate_error = candidate_evaluation.summary.projected_error_rate
        return candidate_latency < baseline_latency and (
            baseline_error is None or candidate_error is None or candidate_error <= baseline_error
        )
    return candidate_latency < baseline_latency


def _primary_metric_for_campaign(
    objective: CounterfactualObjective,
) -> OptimizationObjectiveMetric:
    if objective is CounterfactualObjective.THROUGHPUT:
        return OptimizationObjectiveMetric.TOKENS_PER_SECOND
    if objective is CounterfactualObjective.RELIABILITY:
        return OptimizationObjectiveMetric.ERROR_RATE
    return OptimizationObjectiveMetric.LATENCY_MS


def _proposed_canary_percentage(settings: Settings) -> float:
    configured = settings.phase4.policy_rollout.canary_percentage
    if configured > 0:
        return min(configured, settings.optimization.max_rollout_canary_percentage)
    return min(10.0, settings.optimization.max_rollout_canary_percentage)


def _remote_share_percent(
    evidence_records: Sequence[OptimizationEvidenceRecord],
) -> float | None:
    for record in evidence_records:
        for note in record.notes:
            if note.startswith("remote_share_percent="):
                _, _, value = note.partition("=")
                return float(value)
    return None


def _campaign_inspection_summary(
    *,
    campaign_artifact: OptimizationCampaignArtifact,
    comparison_artifact: OptimizationCampaignComparisonArtifact | None,
    current_worker_inventory: Sequence[BackendInstance] | None = None,
    current_remote_budget_per_minute: int | None = None,
    current_max_remote_share_percent: float | None = None,
    current_remote_concurrency_cap: int | None = None,
) -> ForgeCampaignInspectionSummary:
    comparison_by_candidate_id = (
        {
            comparison.candidate_configuration_id: comparison
            for comparison in comparison_artifact.candidate_comparisons
        }
        if comparison_artifact is not None
        else {}
    )
    trials = [
        _trial_inspection_summary(
            trial=trial,
            comparison=comparison_by_candidate_id.get(
                trial.candidate_configuration.candidate_configuration_id
            ),
        )
        for trial in campaign_artifact.trials
    ]
    recommendation_status_counts: dict[str, int] = {}
    helped_workload_families = sorted(
        {family for trial in trials for family in trial.helped_workload_families}
    )
    hurt_workload_families = sorted(
        {family for trial in trials for family in trial.hurt_workload_families}
    )
    remote_budget_constraint_ids = sorted(
        {constraint_id for trial in trials for constraint_id in trial.remote_budget_constraint_ids}
    )
    for recommendation in campaign_artifact.recommendation_summaries:
        key = recommendation.disposition.value
        recommendation_status_counts[key] = recommendation_status_counts.get(key, 0) + 1
    evidence_kinds = sorted(
        {record.evidence_kind for record in campaign_artifact.evidence_records},
        key=lambda item: item.value,
    )
    # Always run honesty checks.  Staleness, workload coverage, evidence
    # consistency, and cost-signal checks do not require external environment
    # state.  Budget-bound and topology-drift checks use external state when
    # it is available but degrade gracefully when it is not.
    assessment = assess_campaign_honesty(
        campaign_artifact=campaign_artifact,
        current_worker_inventory=current_worker_inventory,
        current_remote_budget_per_minute=current_remote_budget_per_minute,
        current_max_remote_share_percent=current_max_remote_share_percent,
        current_remote_concurrency_cap=current_remote_concurrency_cap,
    )
    trustworthy = assessment.trustworthy
    honesty_warnings = [
        ForgeHonestyWarningSummary(
            kind=ForgeHonestyWarningKind(warning.kind.value),
            severity=warning.severity,
            message=warning.message,
            affected_trial_ids=list(warning.affected_trial_ids),
            notes=list(warning.notes),
        )
        for warning in assessment.warnings
    ]

    notes = [
        "campaign inspection summary is derived from campaign and comparison artifacts",
        "trial entries keep workload and evidence posture explicit",
        "honesty checks always run; staleness, workload coverage, evidence consistency, "
        "and cost signal checks do not require external environment state",
    ]
    if remote_budget_constraint_ids:
        notes.append("remote budget or remote share constraints were in scope")
    if honesty_warnings:
        notes.append(
            f"{len(honesty_warnings)} honesty warning(s) were raised against the "
            f"current environment state"
        )
    return ForgeCampaignInspectionSummary(
        campaign_artifact_id=campaign_artifact.campaign_artifact_id,
        campaign_id=campaign_artifact.campaign.campaign_id,
        optimization_profile_id=campaign_artifact.campaign.optimization_profile_id,
        result_status=campaign_artifact.result_status,
        objective=campaign_artifact.campaign.objective.value,
        evidence_kinds=evidence_kinds,
        recommendation_status_counts=recommendation_status_counts,
        helped_workload_families=helped_workload_families,
        hurt_workload_families=hurt_workload_families,
        remote_budget_involved=bool(remote_budget_constraint_ids),
        remote_budget_constraint_ids=remote_budget_constraint_ids,
        honesty_warnings=honesty_warnings,
        trustworthy=trustworthy,
        comparison_artifact_id=(
            None if comparison_artifact is None else comparison_artifact.comparison_artifact_id
        ),
        trials=trials,
        notes=notes,
    )


def _trial_inspection_summary(
    *,
    trial: OptimizationTrialArtifact,
    comparison: OptimizationCandidateComparisonArtifact | None,
) -> ForgeTrialInspectionSummary:
    recommendation = (
        trial.recommendation_summary
        if trial.recommendation_summary is not None
        else None
        if comparison is None
        else comparison.recommendation_summary
    )
    evidence_kinds = sorted(
        {
            *(record.evidence_kind for record in trial.evidence_records),
            *([] if recommendation is None else recommendation.evidence_kinds),
        },
        key=lambda item: item.value,
    )
    remote_budget_constraint_ids = [
        assessment.constraint_id
        for assessment in trial.constraint_assessments
        if assessment.dimension
        in {
            OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
            OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE,
        }
    ]
    remote_budget_constraint_outcomes = [
        (f"{assessment.constraint_id}:{'satisfied' if assessment.satisfied else 'violated'}")
        for assessment in trial.constraint_assessments
        if assessment.dimension
        in {
            OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
            OptimizationConstraintDimension.REMOTE_REQUEST_BUDGET_PER_MINUTE,
        }
    ]
    generation_strategy = (
        None
        if trial.candidate_configuration.generation is None
        else trial.candidate_configuration.generation.strategy.value
    )
    eligibility_status = (
        None
        if trial.candidate_configuration.eligibility is None
        else trial.candidate_configuration.eligibility.status.value
    )
    topology_lineage = trial.topology_lineage or trial.candidate_configuration.topology_lineage
    notes = [
        "recommendation and workload-family posture comes from the trial artifact",
        "remote budget involvement is reported only from explicit remote constraints",
    ]
    if comparison is not None:
        notes.append("comparison rank and dominance fields come from the comparison artifact")
    return ForgeTrialInspectionSummary(
        trial_artifact_id=trial.trial_artifact_id,
        candidate_configuration_id=(trial.candidate_configuration.candidate_configuration_id),
        config_profile_id=trial.candidate_configuration.config_profile_id,
        baseline_config_profile_id=trial.candidate_configuration.baseline_config_profile_id,
        trial_status=trial.result_status,
        candidate_kind=trial.trial_identity.candidate_kind,
        routing_policy=(
            None
            if trial.trial_identity.routing_policy is None
            else trial.trial_identity.routing_policy.value
        ),
        recommendation_disposition=(None if recommendation is None else recommendation.disposition),
        recommendation_label=(
            None if recommendation is None else recommendation.recommendation_label
        ),
        evidence_kinds=evidence_kinds,
        helped_workload_families=[]
        if recommendation is None
        else recommendation.benefited_workload_families,
        hurt_workload_families=[]
        if recommendation is None
        else recommendation.regressed_workload_families,
        comparison_rank=None if comparison is None else comparison.rank,
        pareto_optimal=None if comparison is None else comparison.pareto_optimal,
        dominated=None if comparison is None else comparison.dominated,
        remote_budget_involved=bool(remote_budget_constraint_ids),
        remote_budget_constraint_ids=remote_budget_constraint_ids,
        remote_budget_constraint_outcomes=remote_budget_constraint_outcomes,
        diff_entries=[
            ForgeCandidateDiffEntry(
                knob_id=change.knob_id,
                config_path=change.config_path,
                baseline_value=change.baseline_value,
                candidate_value=change.candidate_value,
                notes=list(change.notes),
            )
            for change in trial.candidate_configuration.knob_changes
        ],
        provenance=ForgeCandidateProvenanceSummary(
            campaign_id=trial.campaign_id,
            trial_artifact_id=trial.trial_artifact_id,
            candidate_configuration_id=(trial.candidate_configuration.candidate_configuration_id),
            config_profile_id=trial.candidate_configuration.config_profile_id,
            baseline_config_profile_id=trial.candidate_configuration.baseline_config_profile_id,
            generation_strategy=generation_strategy,
            eligibility_status=eligibility_status,
            topology_reference_count=(
                0 if topology_lineage is None else len(topology_lineage.topology_references)
            ),
            topology_endpoint_count=(
                0 if topology_lineage is None else len(topology_lineage.deployed_topology)
            ),
            notes=[("candidate provenance stays tied to campaign, trial, and topology artifacts")],
        ),
        notes=notes,
    )


def _bounded_id(seed: str, *, suffix: str) -> str:
    candidate = f"{seed}-{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    trimmed = candidate[: 117 - len(suffix)].rstrip("-")
    return f"{trimmed}-{suffix}-{digest}"
