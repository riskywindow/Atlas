from __future__ import annotations

from datetime import UTC, datetime, timedelta

from switchyard.bench.artifacts import summarize_records
from switchyard.bench.campaign_comparison import compare_optimization_campaign
from switchyard.schemas.benchmark import (
    BenchmarkEnvironmentMetadata,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    CandidateRouteEstimateContext,
    CounterfactualCandidateScore,
    CounterfactualObjective,
    CounterfactualSimulationArtifact,
    CounterfactualSimulationComparisonArtifact,
    CounterfactualSimulationRecord,
    CounterfactualSimulationSummary,
    ExplainablePolicySpec,
    HistoricalRouteEstimate,
    PolicyRecommendation,
    RecommendationConfidence,
    SimulationEvidenceKind,
    SimulationSourceKind,
    WorkloadScenarioFamily,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactSourceType,
    OptimizationArtifactStatus,
    OptimizationCampaignArtifact,
    OptimizationCampaignMetadata,
    OptimizationCandidateConfigurationArtifact,
    OptimizationComparisonOperator,
    OptimizationConstraint,
    OptimizationConstraintAssessment,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationEvidenceRecord,
    OptimizationGoal,
    OptimizationObjectiveAssessment,
    OptimizationObjectiveMetric,
    OptimizationObjectiveTarget,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationScope,
    OptimizationScopeKind,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import RequestClass, RoutingPolicy, WorkloadShape


def test_compare_optimization_campaign_ranks_ties_and_dominated_candidates() -> None:
    benchmark_artifact = _benchmark_artifact()
    campaign_artifact = _campaign_artifact(
        objective_targets=[
            _objective_target(
                objective_id="latency-primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                weight=2.0,
            ),
            _objective_target(
                objective_id="error-secondary",
                metric=OptimizationObjectiveMetric.ERROR_RATE,
                goal=OptimizationGoal.MINIMIZE,
                weight=1.0,
            ),
        ],
        trials=[
            _trial_artifact(
                trial_id="trial-a",
                candidate_id="candidate-config-a",
                config_profile_id="phase9-a",
                routing_policy=RoutingPolicy.LATENCY_FIRST,
                objective_values={
                    "latency-primary": 80.0,
                    "error-secondary": 0.02,
                },
                hard_constraint_satisfied=True,
            ),
            _trial_artifact(
                trial_id="trial-b",
                candidate_id="candidate-config-b",
                config_profile_id="phase9-b",
                routing_policy=RoutingPolicy.LOCAL_PREFERRED,
                objective_values={
                    "latency-primary": 80.0,
                    "error-secondary": 0.02,
                },
                hard_constraint_satisfied=True,
            ),
            _trial_artifact(
                trial_id="trial-d",
                candidate_id="candidate-config-d",
                config_profile_id="phase9-d",
                routing_policy=RoutingPolicy.QUALITY_FIRST,
                objective_values={
                    "latency-primary": 110.0,
                    "error-secondary": 0.20,
                },
                hard_constraint_satisfied=True,
            ),
            _trial_artifact(
                trial_id="trial-e",
                candidate_id="candidate-config-e",
                config_profile_id="phase9-e",
                routing_policy=RoutingPolicy.BALANCED,
                objective_values={
                    "latency-primary": 70.0,
                    "error-secondary": 0.15,
                },
                hard_constraint_satisfied=False,
            ),
        ],
    )
    comparison = _simulation_comparison_for_ranking()

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[benchmark_artifact],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    comparisons = {
        item.candidate_configuration_id: item for item in artifact.candidate_comparisons
    }
    candidate_a = comparisons["candidate-config-a"]
    candidate_b = comparisons["candidate-config-b"]
    candidate_d = comparisons["candidate-config-d"]
    candidate_e = comparisons["candidate-config-e"]

    assert candidate_a.rank == 1
    assert candidate_b.rank == 1
    assert candidate_a.tied_candidate_configuration_ids == ["candidate-config-b"]
    assert candidate_b.tied_candidate_configuration_ids == ["candidate-config-a"]
    assert candidate_a.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.PROMOTION_ELIGIBLE
    )
    assert candidate_a.recommendation_summary.disposition is (
        OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    )
    assert (
        OptimizationRecommendationReasonCode.TIED_WITH_PEER
        in candidate_a.recommendation_summary.reason_codes
    )
    assert candidate_d.dominated is True
    assert candidate_d.dominated_by_candidate_configuration_ids == [
        "candidate-config-a",
        "candidate-config-b",
    ]
    assert candidate_d.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.REJECTED
    )
    assert (
        OptimizationRecommendationReasonCode.DOMINATED
        in candidate_d.recommendation_summary.reason_codes
    )
    assert candidate_e.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.REJECTED
    )
    assert (
        OptimizationRecommendationReasonCode.HARD_CONSTRAINT_VIOLATED
        in candidate_e.recommendation_summary.reason_codes
    )
    assert artifact.pareto_summary.frontier_candidate_configuration_ids == [
        "candidate-config-a",
        "candidate-config-b",
    ]


def test_compare_optimization_campaign_marks_mixed_workload_tradeoffs_review_only() -> None:
    benchmark_artifact = _benchmark_artifact()
    campaign_artifact = _campaign_artifact(
        objective_targets=[
            _objective_target(
                objective_id="latency-primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                weight=1.0,
            )
        ],
        trials=[
            _trial_artifact(
                trial_id="trial-mixed",
                candidate_id="candidate-config-mixed",
                config_profile_id="phase9-mixed",
                routing_policy=RoutingPolicy.LATENCY_FIRST,
                objective_values={"latency-primary": 95.0},
                hard_constraint_satisfied=True,
            )
        ],
    )
    comparison = _simulation_comparison_for_mixed_workloads()

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[benchmark_artifact],
        timestamp=datetime(2026, 3, 22, 20, 30, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]

    assert candidate.pareto_optimal is True
    assert candidate.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.REVIEW_ONLY
    )
    assert candidate.recommendation_summary.disposition is (
        OptimizationRecommendationDisposition.NO_CHANGE
    )
    assert candidate.recommendation_summary.benefited_workload_families == ["short_chat"]
    assert candidate.recommendation_summary.regressed_workload_families == [
        "repeated_prefix"
    ]
    assert (
        OptimizationRecommendationReasonCode.MIXED_WORKLOAD_TRADEOFF
        in candidate.recommendation_summary.reason_codes
    )


def _benchmark_artifact() -> BenchmarkRunArtifact:
    records = [
        _benchmark_record(
            request_id="short-1",
            scenario_family=WorkloadScenarioFamily.SHORT_CHAT,
            latency_ms=100.0,
        ),
        _benchmark_record(
            request_id="short-2",
            scenario_family=WorkloadScenarioFamily.SHORT_CHAT,
            latency_ms=100.0,
        ),
        _benchmark_record(
            request_id="repeat-1",
            scenario_family=WorkloadScenarioFamily.REPEATED_PREFIX,
            latency_ms=120.0,
        ),
        _benchmark_record(
            request_id="repeat-2",
            scenario_family=WorkloadScenarioFamily.REPEATED_PREFIX,
            latency_ms=120.0,
        ),
    ]
    return BenchmarkRunArtifact(
        run_id="comparison-run-001",
        timestamp=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
        scenario=BenchmarkScenario(
            name="phase9-comparison",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=len(records),
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["local-observed", "policy-alt"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=len(records),
        summary=summarize_records(records),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="synthetic"),
        records=records,
    )


def _benchmark_record(
    *,
    request_id: str,
    scenario_family: WorkloadScenarioFamily,
    latency_ms: float,
) -> BenchmarkRequestRecord:
    started_at = datetime(2026, 3, 22, 12, 0, tzinfo=UTC)
    return BenchmarkRequestRecord(
        request_id=request_id,
        scenario_family=scenario_family,
        tenant_id="tenant-a",
        request_class=RequestClass.STANDARD,
        backend_name="local-observed",
        backend_type="mock",
        model_alias="chat-shared",
        model_identifier="chat-shared",
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=latency_ms),
        latency_ms=latency_ms,
        output_tokens=32,
        tokens_per_second=50.0,
        success=True,
        status_code=200,
        usage=UsageStats(prompt_tokens=32, completion_tokens=32, total_tokens=64),
    )


def _objective_target(
    *,
    objective_id: str,
    metric: OptimizationObjectiveMetric,
    goal: OptimizationGoal,
    weight: float,
) -> OptimizationObjectiveTarget:
    return OptimizationObjectiveTarget(
        objective_id=objective_id,
        metric=metric,
        goal=goal,
        weight=weight,
        applies_to=[OptimizationScope(kind=OptimizationScopeKind.GLOBAL)],
        evidence_sources=[
            ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
            ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION,
        ],
    )


def _campaign_artifact(
    *,
    objective_targets: list[OptimizationObjectiveTarget],
    trials: list[OptimizationTrialArtifact],
) -> OptimizationCampaignArtifact:
    baseline_candidate = _candidate_configuration(
        trial_identity=OptimizationTrialIdentity(
            trial_id="trial-baseline",
            candidate_id="routing_policy:balanced",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="phase9-baseline",
            routing_policy=RoutingPolicy.BALANCED,
        ),
        candidate_configuration_id="candidate-config-baseline",
        objective_targets=objective_targets,
    )
    return OptimizationCampaignArtifact(
        campaign_artifact_id="campaign-comparison-artifact",
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
        campaign=OptimizationCampaignMetadata(
            campaign_id="campaign-phase9-comparison",
            optimization_profile_id="phase9-stage-a-baseline",
            objective=CounterfactualObjective.BALANCED,
            evidence_sources=[
                ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
                ForgeEvidenceSourceKind.REPLAYED_BENCHMARK,
                ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION,
            ],
            required_evidence_sources=[ForgeEvidenceSourceKind.COUNTERFACTUAL_SIMULATION],
            default_workload_set_ids=["default"],
        ),
        baseline_candidate_configuration=baseline_candidate,
        candidate_configurations=[trial.candidate_configuration for trial in trials],
        trials=trials,
        recommendation_summaries=[
            trial.recommendation_summary
            for trial in trials
            if trial.recommendation_summary is not None
        ],
    )


def _trial_artifact(
    *,
    trial_id: str,
    candidate_id: str,
    config_profile_id: str,
    routing_policy: RoutingPolicy,
    objective_values: dict[str, float],
    hard_constraint_satisfied: bool,
) -> OptimizationTrialArtifact:
    if "error-secondary" in objective_values:
        objective_targets = [
            _objective_target(
                objective_id="latency-primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                weight=2.0,
            ),
            _objective_target(
                objective_id="error-secondary",
                metric=OptimizationObjectiveMetric.ERROR_RATE,
                goal=OptimizationGoal.MINIMIZE,
                weight=1.0,
            ),
        ]
    else:
        objective_targets = [
            _objective_target(
                objective_id="latency-primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                weight=1.0,
            )
        ]
    trial_identity = OptimizationTrialIdentity(
        trial_id=trial_id,
        candidate_id=f"routing_policy:{routing_policy.value}",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )
    candidate_configuration = _candidate_configuration(
        trial_identity=trial_identity,
        candidate_configuration_id=candidate_id,
        objective_targets=objective_targets,
    )
    objective_assessments = [
        OptimizationObjectiveAssessment(
            objective_id=objective.objective_id,
            metric=objective.metric,
            goal=objective.goal,
            measured_value=objective_values[objective.objective_id],
            satisfied=True,
            evidence_kinds=[
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.SIMULATED,
            ],
        )
        for objective in objective_targets
    ]
    return OptimizationTrialArtifact(
        trial_artifact_id=f"{trial_id}-artifact",
        timestamp=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        campaign_id="campaign-phase9-comparison",
        baseline_candidate_configuration_id="candidate-config-baseline",
        candidate_configuration=candidate_configuration,
        trial_identity=trial_identity,
        evidence_records=[
            _evidence_record(
                evidence_id=f"{trial_id}-observed",
                evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
                source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
            ),
            _evidence_record(
                evidence_id=f"{trial_id}-simulated",
                evidence_kind=OptimizationArtifactEvidenceKind.SIMULATED,
                source_type=OptimizationArtifactSourceType.SIMULATION,
            ),
        ],
        result_status=OptimizationArtifactStatus.COMPLETE,
        objective_assessments=objective_assessments,
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="remote-share-cap",
                dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=25.0,
                evaluated_value=15.0 if hard_constraint_satisfied else 40.0,
                satisfied=hard_constraint_satisfied,
                evidence_kinds=[OptimizationArtifactEvidenceKind.OBSERVED],
            )
        ],
        recommendation_summary=OptimizationRecommendationSummary(
            recommendation_summary_id=f"{trial_id}-recommendation",
            disposition=OptimizationRecommendationDisposition.NO_CHANGE,
            confidence=RecommendationConfidence.MEDIUM,
            candidate_configuration_id=candidate_id,
            config_profile_id=config_profile_id,
            evidence_kinds=[
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.SIMULATED,
            ],
        ),
    )


def _candidate_configuration(
    *,
    trial_identity: OptimizationTrialIdentity,
    candidate_configuration_id: str,
    objective_targets: list[OptimizationObjectiveTarget],
) -> OptimizationCandidateConfigurationArtifact:
    return OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=candidate_configuration_id,
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
        campaign_id="campaign-phase9-comparison",
        candidate=trial_identity,
        baseline_config_profile_id="phase9-baseline",
        config_profile_id=trial_identity.config_profile_id,
        objectives_in_scope=objective_targets,
        constraints_in_scope=[
            OptimizationConstraint(
                constraint_id="remote-share-cap",
                dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=25.0,
            )
        ],
    )


def _evidence_record(
    *,
    evidence_id: str,
    evidence_kind: OptimizationArtifactEvidenceKind,
    source_type: OptimizationArtifactSourceType,
) -> OptimizationEvidenceRecord:
    return OptimizationEvidenceRecord(
        evidence_id=evidence_id,
        evidence_kind=evidence_kind,
        source_type=source_type,
        source_artifact_id=evidence_id,
    )


def _simulation_comparison_for_ranking() -> CounterfactualSimulationComparisonArtifact:
    baseline_records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=100.0),
        _simulation_record(request_id="short-2", recommended_latency_ms=100.0),
        _simulation_record(request_id="repeat-1", recommended_latency_ms=120.0),
        _simulation_record(request_id="repeat-2", recommended_latency_ms=120.0),
    ]
    candidate_a_records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=70.0),
        _simulation_record(request_id="short-2", recommended_latency_ms=70.0),
        _simulation_record(request_id="repeat-1", recommended_latency_ms=90.0),
        _simulation_record(request_id="repeat-2", recommended_latency_ms=90.0),
    ]
    candidate_b_records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=70.0),
        _simulation_record(request_id="short-2", recommended_latency_ms=70.0),
        _simulation_record(request_id="repeat-1", recommended_latency_ms=90.0),
        _simulation_record(request_id="repeat-2", recommended_latency_ms=90.0),
    ]
    candidate_d_records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=110.0),
        _simulation_record(request_id="short-2", recommended_latency_ms=110.0),
        _simulation_record(request_id="repeat-1", recommended_latency_ms=130.0),
        _simulation_record(request_id="repeat-2", recommended_latency_ms=130.0),
    ]
    evaluations = [
        _simulation_artifact(
            policy_id=RoutingPolicy.BALANCED.value,
            records=baseline_records,
            projected_avg_latency_ms=110.0,
            projected_error_rate=0.10,
            direct_observation_count=4,
            predictor_estimate_count=0,
        ),
        _simulation_artifact(
            policy_id=RoutingPolicy.LATENCY_FIRST.value,
            records=candidate_a_records,
            projected_avg_latency_ms=80.0,
            projected_error_rate=0.02,
            direct_observation_count=2,
            predictor_estimate_count=2,
        ),
        _simulation_artifact(
            policy_id=RoutingPolicy.LOCAL_PREFERRED.value,
            records=candidate_b_records,
            projected_avg_latency_ms=80.0,
            projected_error_rate=0.02,
            direct_observation_count=2,
            predictor_estimate_count=2,
        ),
        _simulation_artifact(
            policy_id=RoutingPolicy.QUALITY_FIRST.value,
            records=candidate_d_records,
            projected_avg_latency_ms=110.0,
            projected_error_rate=0.20,
            direct_observation_count=2,
            predictor_estimate_count=2,
        ),
    ]
    return CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id="campaign-comparison-ranking",
        timestamp=datetime(2026, 3, 22, 19, 45, tzinfo=UTC),
        source_run_ids=["comparison-run-001"],
        policies=[evaluation.policy for evaluation in evaluations],
        evaluations=evaluations,
    )


def _simulation_comparison_for_mixed_workloads() -> CounterfactualSimulationComparisonArtifact:
    baseline_records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=100.0),
        _simulation_record(request_id="short-2", recommended_latency_ms=100.0),
        _simulation_record(request_id="repeat-1", recommended_latency_ms=120.0),
        _simulation_record(request_id="repeat-2", recommended_latency_ms=120.0),
    ]
    candidate_records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=70.0),
        _simulation_record(request_id="short-2", recommended_latency_ms=70.0),
        _simulation_record(request_id="repeat-1", recommended_latency_ms=140.0),
        _simulation_record(request_id="repeat-2", recommended_latency_ms=140.0),
    ]
    evaluations = [
        _simulation_artifact(
            policy_id=RoutingPolicy.BALANCED.value,
            records=baseline_records,
            projected_avg_latency_ms=110.0,
            projected_error_rate=0.10,
            direct_observation_count=4,
            predictor_estimate_count=0,
        ),
        _simulation_artifact(
            policy_id=RoutingPolicy.LATENCY_FIRST.value,
            records=candidate_records,
            projected_avg_latency_ms=95.0,
            projected_error_rate=0.05,
            direct_observation_count=2,
            predictor_estimate_count=2,
        ),
    ]
    return CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id="campaign-comparison-mixed",
        timestamp=datetime(2026, 3, 22, 20, 15, tzinfo=UTC),
        source_run_ids=["comparison-run-001"],
        policies=[evaluation.policy for evaluation in evaluations],
        evaluations=evaluations,
    )


def _simulation_artifact(
    *,
    policy_id: str,
    records: list[CounterfactualSimulationRecord],
    projected_avg_latency_ms: float,
    projected_error_rate: float,
    direct_observation_count: int,
    predictor_estimate_count: int,
) -> CounterfactualSimulationArtifact:
    return CounterfactualSimulationArtifact(
        simulation_id=f"{policy_id}-simulation",
        timestamp=datetime(2026, 3, 22, 19, 40, tzinfo=UTC),
        source_run_ids=["comparison-run-001"],
        policy=ExplainablePolicySpec(
            policy_id=policy_id,
            objective=CounterfactualObjective.BALANCED,
        ),
        summary=CounterfactualSimulationSummary(
            request_count=len(records),
            changed_count=sum(
                1 for record in records if record.recommendation.recommendation_changed
            ),
            unchanged_count=sum(
                1 for record in records if not record.recommendation.recommendation_changed
            ),
            direct_observation_count=direct_observation_count,
            predictor_estimate_count=predictor_estimate_count,
            low_confidence_count=0,
            unsupported_count=0,
            insufficient_data_count=0,
            guardrail_block_count=0,
            observed_backend_counts={"local-observed": len(records)},
            recommended_backend_counts={
                "policy-alt": sum(
                    1
                    for record in records
                    if record.recommendation.recommended_backend == "policy-alt"
                ),
                "local-observed": sum(
                    1
                    for record in records
                    if record.recommendation.recommended_backend == "local-observed"
                ),
            },
            projected_avg_latency_ms=projected_avg_latency_ms,
            projected_error_rate=projected_error_rate,
            projected_avg_tokens_per_second=50.0,
        ),
        records=records,
    )


def _simulation_record(
    *,
    request_id: str,
    recommended_latency_ms: float,
) -> CounterfactualSimulationRecord:
    changed = recommended_latency_ms < 99.0 or recommended_latency_ms > 121.0
    if changed:
        recommended_backend = "policy-alt"
        evidence_kind = SimulationEvidenceKind.PREDICTOR_ESTIMATE
        candidate_scores = [
            CounterfactualCandidateScore(
                backend_name="local-observed",
                score=0.0,
                eligible=True,
                evidence_kind=SimulationEvidenceKind.DIRECT_OBSERVATION,
                evidence_count=1,
                directly_observed=True,
                observed_latency_ms=100.0 if request_id.startswith("short") else 120.0,
                observed_success=True,
            ),
            CounterfactualCandidateScore(
                backend_name="policy-alt",
                score=1.0,
                eligible=True,
                evidence_kind=SimulationEvidenceKind.PREDICTOR_ESTIMATE,
                evidence_count=8,
                estimate=HistoricalRouteEstimate(
                    context=_estimate_context(request_id),
                    evidence_count=8,
                    sufficient_data=True,
                    expected_latency_ms=recommended_latency_ms,
                    expected_error_rate=0.0,
                    expected_tokens_per_second=55.0,
                ),
            ),
        ]
    else:
        recommended_backend = "local-observed"
        evidence_kind = SimulationEvidenceKind.DIRECT_OBSERVATION
        candidate_scores = [
            CounterfactualCandidateScore(
                backend_name="local-observed",
                score=1.0,
                eligible=True,
                evidence_kind=SimulationEvidenceKind.DIRECT_OBSERVATION,
                evidence_count=1,
                directly_observed=True,
                observed_latency_ms=100.0 if request_id.startswith("short") else 120.0,
                observed_success=True,
            )
        ]
    return CounterfactualSimulationRecord(
        request_id=request_id,
        source_run_id="comparison-run-001",
        source_kind=SimulationSourceKind.BENCHMARK_RUN,
        source_record_id=request_id,
        model_alias="chat-shared",
        tenant_id="tenant-a",
        request_class=RequestClass.STANDARD,
        observed_backend="local-observed",
        observed_latency_ms=100.0 if request_id.startswith("short") else 120.0,
        observed_success=True,
        candidate_scores=candidate_scores,
        recommendation=PolicyRecommendation(
            observed_backend="local-observed",
            recommended_backend=recommended_backend,
            recommendation_changed=changed,
            evidence_kind=evidence_kind,
        ),
    )


def _estimate_context(request_id: str) -> CandidateRouteEstimateContext:
    return CandidateRouteEstimateContext(
        model_alias="chat-shared",
        backend_name=f"policy-alt-{request_id}",
        request_class=RequestClass.STANDARD,
    )


# ---------------------------------------------------------------------------
# Flexible helpers for additional test scenarios
# ---------------------------------------------------------------------------


def _trial_artifact_flexible(
    *,
    trial_id: str,
    candidate_id: str,
    config_profile_id: str,
    routing_policy: RoutingPolicy,
    objective_targets: list[OptimizationObjectiveTarget],
    objective_values: dict[str, float],
    constraint_assessments: list[OptimizationConstraintAssessment] | None = None,
    evidence_records: list[OptimizationEvidenceRecord] | None = None,
) -> OptimizationTrialArtifact:
    trial_identity = OptimizationTrialIdentity(
        trial_id=trial_id,
        candidate_id=f"routing_policy:{routing_policy.value}",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )
    candidate_configuration = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=candidate_id,
        timestamp=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
        campaign_id="campaign-phase9-comparison",
        candidate=trial_identity,
        baseline_config_profile_id="phase9-baseline",
        config_profile_id=config_profile_id,
        objectives_in_scope=objective_targets,
        constraints_in_scope=[
            OptimizationConstraint(
                constraint_id=assessment.constraint_id,
                dimension=assessment.dimension,
                strength=assessment.strength,
                operator=assessment.operator,
                threshold_value=assessment.threshold_value,
            )
            for assessment in (constraint_assessments or [])
        ],
    )
    resolved_evidence = evidence_records or [
        _evidence_record(
            evidence_id=f"{trial_id}-observed",
            evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
            source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
        ),
        _evidence_record(
            evidence_id=f"{trial_id}-simulated",
            evidence_kind=OptimizationArtifactEvidenceKind.SIMULATED,
            source_type=OptimizationArtifactSourceType.SIMULATION,
        ),
    ]
    objective_assessments = [
        OptimizationObjectiveAssessment(
            objective_id=objective.objective_id,
            metric=objective.metric,
            goal=objective.goal,
            measured_value=objective_values.get(objective.objective_id),
            satisfied=True,
            evidence_kinds=[
                record.evidence_kind for record in resolved_evidence
            ],
        )
        for objective in objective_targets
    ]
    return OptimizationTrialArtifact(
        trial_artifact_id=f"{trial_id}-artifact",
        timestamp=datetime(2026, 3, 22, 19, 30, tzinfo=UTC),
        campaign_id="campaign-phase9-comparison",
        baseline_candidate_configuration_id="candidate-config-baseline",
        candidate_configuration=candidate_configuration,
        trial_identity=trial_identity,
        evidence_records=resolved_evidence,
        result_status=OptimizationArtifactStatus.COMPLETE,
        objective_assessments=objective_assessments,
        constraint_assessments=constraint_assessments or [],
        recommendation_summary=OptimizationRecommendationSummary(
            recommendation_summary_id=f"{trial_id}-recommendation",
            disposition=OptimizationRecommendationDisposition.NO_CHANGE,
            confidence=RecommendationConfidence.MEDIUM,
            candidate_configuration_id=candidate_id,
            config_profile_id=config_profile_id,
            evidence_kinds=[record.evidence_kind for record in resolved_evidence],
        ),
    )


def _simple_simulation_comparison(
    *,
    baseline_latency: float,
    baseline_error: float,
    baseline_tps: float = 50.0,
    baseline_direct_obs: int = 4,
    candidates: list[
        tuple[str, float, float, float, int, int]
    ],
) -> CounterfactualSimulationComparisonArtifact:
    """Build a simulation comparison.

    candidates is a list of
    (policy_id, latency, error_rate, tps, direct_obs, predictor_est).
    """
    records = [
        _simulation_record(request_id="short-1", recommended_latency_ms=baseline_latency),
        _simulation_record(request_id="short-2", recommended_latency_ms=baseline_latency),
    ]
    evaluations = [
        _simulation_artifact(
            policy_id=RoutingPolicy.BALANCED.value,
            records=records,
            projected_avg_latency_ms=baseline_latency,
            projected_error_rate=baseline_error,
            direct_observation_count=baseline_direct_obs,
            predictor_estimate_count=0,
        ),
    ]
    for policy_id, latency, error_rate, tps, direct_obs, predictor_est in candidates:
        candidate_records = [
            _simulation_record(request_id="short-1", recommended_latency_ms=latency),
            _simulation_record(request_id="short-2", recommended_latency_ms=latency),
        ]
        evaluations.append(
            CounterfactualSimulationArtifact(
                simulation_id=f"{policy_id}-simulation",
                timestamp=datetime(2026, 3, 22, 19, 40, tzinfo=UTC),
                source_run_ids=["comparison-run-001"],
                policy=ExplainablePolicySpec(
                    policy_id=policy_id,
                    objective=CounterfactualObjective.BALANCED,
                ),
                summary=CounterfactualSimulationSummary(
                    request_count=2,
                    changed_count=2,
                    unchanged_count=0,
                    direct_observation_count=direct_obs,
                    predictor_estimate_count=predictor_est,
                    low_confidence_count=0,
                    unsupported_count=0,
                    insufficient_data_count=0,
                    guardrail_block_count=0,
                    observed_backend_counts={"local-observed": 2},
                    recommended_backend_counts={"policy-alt": 2},
                    projected_avg_latency_ms=latency,
                    projected_error_rate=error_rate,
                    projected_avg_tokens_per_second=tps,
                ),
                records=candidate_records,
            ),
        )
    return CounterfactualSimulationComparisonArtifact(
        simulation_comparison_id="campaign-comparison-test",
        timestamp=datetime(2026, 3, 22, 19, 45, tzinfo=UTC),
        source_run_ids=["comparison-run-001"],
        policies=[evaluation.policy for evaluation in evaluations],
        evaluations=evaluations,
    )


# ---------------------------------------------------------------------------
# Empty campaign
# ---------------------------------------------------------------------------


def test_empty_campaign_produces_no_comparisons() -> None:
    campaign_artifact = _campaign_artifact(
        objective_targets=[
            _objective_target(
                objective_id="latency-primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                weight=1.0,
            ),
        ],
        trials=[],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=None,
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    assert artifact.candidate_comparisons == []
    assert artifact.pareto_summary.frontier_candidate_configuration_ids == []
    assert "did not contain any executed candidate trials" in artifact.notes[0]


# ---------------------------------------------------------------------------
# Single candidate
# ---------------------------------------------------------------------------


def test_single_candidate_is_ranked_first_and_not_tied() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-single",
        candidate_id="candidate-config-single",
        config_profile_id="phase9-single",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    assert len(artifact.candidate_comparisons) == 1
    candidate = artifact.candidate_comparisons[0]
    assert candidate.rank == 1
    assert candidate.tied_candidate_configuration_ids == []
    assert candidate.pareto_optimal is True
    assert candidate.dominated is False


# ---------------------------------------------------------------------------
# Soft constraint violation → review-only
# ---------------------------------------------------------------------------


def test_soft_constraint_violation_keeps_candidate_review_only() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-soft-fail",
        candidate_id="candidate-config-soft-fail",
        config_profile_id="phase9-soft-fail",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="advisory-error-cap",
                dimension=OptimizationConstraintDimension.PREDICTED_ERROR_RATE,
                strength=OptimizationConstraintStrength.SOFT,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=0.05,
                evaluated_value=0.08,
                satisfied=False,
                evidence_kinds=[OptimizationArtifactEvidenceKind.SIMULATED],
            ),
        ],
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.08, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert candidate.soft_constraint_violations == ["advisory-error-cap"]
    assert candidate.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.REVIEW_ONLY
    )
    assert (
        OptimizationRecommendationReasonCode.SOFT_CONSTRAINT_VIOLATED
        in candidate.recommendation_summary.reason_codes
    )
    # Soft constraint violation keeps candidate under review, not rejected
    assert candidate.hard_constraint_violations == []


# ---------------------------------------------------------------------------
# No meaningful delta
# ---------------------------------------------------------------------------


def test_no_meaningful_delta_when_metrics_unavailable() -> None:
    """When simulation cannot measure objectives, no delta is possible."""
    # Use REMOTE_SHARE_PERCENT which the simulation doesn't project
    remote_targets = [
        _objective_target(
            objective_id="remote-share",
            metric=OptimizationObjectiveMetric.REMOTE_SHARE_PERCENT,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-no-delta",
        candidate_id="candidate-config-no-delta",
        config_profile_id="phase9-no-delta",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=remote_targets,
        objective_values={},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=remote_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 100.0, 0.05, 50.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert (
        OptimizationRecommendationReasonCode.NO_MEANINGFUL_DELTA
        in candidate.recommendation_summary.reason_codes
    )
    assert candidate.recommendation_summary.disposition is (
        OptimizationRecommendationDisposition.NO_CHANGE
    )


def test_identical_values_within_tie_tolerance_counts_as_no_improvement() -> None:
    """When candidate matches baseline exactly, it falls below tie tolerance."""
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-same",
        candidate_id="candidate-config-same",
        config_profile_id="phase9-same",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 100.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 100.0, 0.05, 50.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    # Zero delta falls below the tie tolerance, so improved=False (not within tolerance)
    latency_delta = next(
        delta for delta in candidate.objective_deltas
        if delta.metric is OptimizationObjectiveMetric.LATENCY_MS
    )
    assert latency_delta.normalized_tradeoff == 0.0
    assert latency_delta.improved is False


# ---------------------------------------------------------------------------
# Estimated-only evidence → need more evidence
# ---------------------------------------------------------------------------


def test_estimated_only_evidence_triggers_need_more_evidence() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    # No OBSERVED evidence records, only SIMULATED
    trial = _trial_artifact_flexible(
        trial_id="trial-estimated",
        candidate_id="candidate-config-estimated",
        config_profile_id="phase9-estimated",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
        evidence_records=[
            _evidence_record(
                evidence_id="trial-estimated-simulated",
                evidence_kind=OptimizationArtifactEvidenceKind.SIMULATED,
                source_type=OptimizationArtifactSourceType.SIMULATION,
            ),
        ],
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    # Zero direct observations in the simulation evaluation
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        baseline_direct_obs=0,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 0, 2),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert (
        OptimizationRecommendationReasonCode.OBSERVED_EVIDENCE_MISSING
        in candidate.recommendation_summary.reason_codes
    )
    assert candidate.recommendation_summary.disposition is (
        OptimizationRecommendationDisposition.NEED_MORE_EVIDENCE
    )
    assert candidate.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.REVIEW_ONLY
    )


# ---------------------------------------------------------------------------
# Primary improved, secondary regressed reason codes
# ---------------------------------------------------------------------------


def test_primary_improved_and_secondary_regressed_reason_codes() -> None:
    objective_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=2.0,
        ),
        _objective_target(
            objective_id="error-secondary",
            metric=OptimizationObjectiveMetric.ERROR_RATE,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-mixed-obj",
        candidate_id="candidate-config-mixed-obj",
        config_profile_id="phase9-mixed-obj",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=objective_targets,
        objective_values={
            "latency-primary": 70.0,
            "error-secondary": 0.15,
        },
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=objective_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 70.0, 0.15, 50.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    reason_codes = candidate.recommendation_summary.reason_codes
    assert OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED in reason_codes
    assert OptimizationRecommendationReasonCode.SECONDARY_OBJECTIVE_REGRESSED in reason_codes
    assert "latency-primary" in candidate.recommendation_summary.improved_objective_ids
    assert "error-secondary" in candidate.recommendation_summary.regressed_objective_ids


# ---------------------------------------------------------------------------
# Hard constraint violation excludes from Pareto frontier
# ---------------------------------------------------------------------------


def test_hard_constraint_violation_excludes_from_pareto_frontier() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    # This candidate has the best metrics but violates a hard constraint
    trial = _trial_artifact_flexible(
        trial_id="trial-hard-fail",
        candidate_id="candidate-config-hard-fail",
        config_profile_id="phase9-hard-fail",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 60.0},
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="max-remote-share",
                dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=25.0,
                evaluated_value=40.0,
                satisfied=False,
                evidence_kinds=[OptimizationArtifactEvidenceKind.OBSERVED],
            ),
        ],
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 60.0, 0.01, 60.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert candidate.pareto_optimal is False
    assert candidate.hard_constraint_violations == ["max-remote-share"]
    assert candidate.recommendation_summary.recommendation_label is (
        OptimizationRecommendationLabel.REJECTED
    )
    assert artifact.pareto_summary.frontier_candidate_configuration_ids == []


# ---------------------------------------------------------------------------
# Evidence mix tracking
# ---------------------------------------------------------------------------


def test_evidence_mix_tracks_observed_and_estimated_shares() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-evidence",
        candidate_id="candidate-config-evidence",
        config_profile_id="phase9-evidence",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
        evidence_records=[
            _evidence_record(
                evidence_id="trial-evidence-observed",
                evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
                source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
            ),
            _evidence_record(
                evidence_id="trial-evidence-estimated",
                evidence_kind=OptimizationArtifactEvidenceKind.ESTIMATED,
                source_type=OptimizationArtifactSourceType.ESTIMATE_SUMMARY,
            ),
        ],
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    # 1 direct observation, 1 predictor estimate
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 1, 1),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    evidence_mix = candidate.recommendation_summary.evidence_mix
    assert evidence_mix is not None
    assert evidence_mix.direct_observation_count == 1
    assert evidence_mix.estimated_request_count == 1
    assert evidence_mix.observed_share == 0.5
    assert evidence_mix.estimated_share == 0.5
    assert (
        OptimizationRecommendationReasonCode.OBSERVED_EVIDENCE_PRESENT
        in candidate.recommendation_summary.reason_codes
    )
    assert (
        OptimizationRecommendationReasonCode.ESTIMATED_EVIDENCE_PRESENT
        in candidate.recommendation_summary.reason_codes
    )


# ---------------------------------------------------------------------------
# Normalized tradeoff score
# ---------------------------------------------------------------------------


def test_normalized_tradeoff_score_positive_for_improved_latency() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-tradeoff",
        candidate_id="candidate-config-tradeoff",
        config_profile_id="phase9-tradeoff",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    # For MINIMIZE, normalized_tradeoff = (baseline - candidate) / max(|baseline|, 1)
    # = (100 - 80) / 100 = 0.2
    assert candidate.normalized_tradeoff_score > 0.0
    latency_delta = next(
        delta for delta in candidate.objective_deltas
        if delta.metric is OptimizationObjectiveMetric.LATENCY_MS
    )
    assert latency_delta.improved is True
    assert latency_delta.absolute_delta is not None
    assert latency_delta.absolute_delta < 0.0  # candidate - baseline = 80 - 100 = -20


# ---------------------------------------------------------------------------
# Maximize objective (throughput)
# ---------------------------------------------------------------------------


def test_maximize_objective_improvement_produces_positive_tradeoff() -> None:
    tps_targets = [
        _objective_target(
            objective_id="throughput-primary",
            metric=OptimizationObjectiveMetric.TOKENS_PER_SECOND,
            goal=OptimizationGoal.MAXIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-tps",
        candidate_id="candidate-config-tps",
        config_profile_id="phase9-tps",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=tps_targets,
        objective_values={"throughput-primary": 70.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=tps_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        baseline_tps=50.0,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 90.0, 0.03, 70.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    tps_delta = next(
        delta for delta in candidate.objective_deltas
        if delta.metric is OptimizationObjectiveMetric.TOKENS_PER_SECOND
    )
    # For MAXIMIZE, improved means candidate > baseline
    assert tps_delta.improved is True
    assert candidate.normalized_tradeoff_score > 0.0


# ---------------------------------------------------------------------------
# Multi-objective Pareto: two candidates, neither dominates the other
# ---------------------------------------------------------------------------


def test_multi_objective_pareto_two_non_dominated_candidates() -> None:
    objective_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
        _objective_target(
            objective_id="error-secondary",
            metric=OptimizationObjectiveMetric.ERROR_RATE,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    # Candidate F: better latency, worse error
    trial_f = _trial_artifact_flexible(
        trial_id="trial-f",
        candidate_id="candidate-config-f",
        config_profile_id="phase9-f",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=objective_targets,
        objective_values={
            "latency-primary": 70.0,
            "error-secondary": 0.10,
        },
    )
    # Candidate G: worse latency, better error
    trial_g = _trial_artifact_flexible(
        trial_id="trial-g",
        candidate_id="candidate-config-g",
        config_profile_id="phase9-g",
        routing_policy=RoutingPolicy.LOCAL_PREFERRED,
        objective_targets=objective_targets,
        objective_values={
            "latency-primary": 90.0,
            "error-secondary": 0.01,
        },
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=objective_targets,
        trials=[trial_f, trial_g],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 70.0, 0.10, 50.0, 2, 0),
            (RoutingPolicy.LOCAL_PREFERRED.value, 90.0, 0.01, 50.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    comparisons = {
        item.candidate_configuration_id: item for item in artifact.candidate_comparisons
    }
    candidate_f = comparisons["candidate-config-f"]
    candidate_g = comparisons["candidate-config-g"]
    # Neither dominates the other: both should be on the Pareto frontier
    assert candidate_f.dominated is False
    assert candidate_g.dominated is False
    assert candidate_f.pareto_optimal is True
    assert candidate_g.pareto_optimal is True
    assert len(artifact.pareto_summary.frontier_candidate_configuration_ids) == 2
    assert artifact.pareto_summary.dominated_candidate_configuration_ids == []


# ---------------------------------------------------------------------------
# Operator review required reason code
# ---------------------------------------------------------------------------


def test_operator_review_required_reason_code_present() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-review",
        candidate_id="candidate-config-review",
        config_profile_id="phase9-review",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    # The default _campaign_artifact sets promotion_requires_operator_review=True
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert (
        OptimizationRecommendationReasonCode.PROMOTION_REQUIRES_REVIEW
        in candidate.recommendation_summary.reason_codes
    )
    assert any(
        "operator review" in rationale
        for rationale in candidate.recommendation_summary.rationale
    )


# ---------------------------------------------------------------------------
# Ranking orders: promoted > review-only > rejected
# ---------------------------------------------------------------------------


def test_ranking_orders_promoted_above_review_above_rejected() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    # Trial that should get PROMOTION_ELIGIBLE (improved, no regressions, has observed)
    trial_good = _trial_artifact_flexible(
        trial_id="trial-good",
        candidate_id="candidate-config-good",
        config_profile_id="phase9-good",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 60.0},
    )
    # Trial with hard constraint violation → REJECTED
    trial_bad = _trial_artifact_flexible(
        trial_id="trial-bad",
        candidate_id="candidate-config-bad",
        config_profile_id="phase9-bad",
        routing_policy=RoutingPolicy.LOCAL_PREFERRED,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 50.0},
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="error-cap",
                dimension=OptimizationConstraintDimension.PREDICTED_ERROR_RATE,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=0.05,
                evaluated_value=0.20,
                satisfied=False,
                evidence_kinds=[OptimizationArtifactEvidenceKind.SIMULATED],
            ),
        ],
    )
    # Trial with no improvement → REVIEW_ONLY
    trial_neutral = _trial_artifact_flexible(
        trial_id="trial-neutral",
        candidate_id="candidate-config-neutral",
        config_profile_id="phase9-neutral",
        routing_policy=RoutingPolicy.QUALITY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 100.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial_bad, trial_neutral, trial_good],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 60.0, 0.02, 50.0, 2, 0),
            (RoutingPolicy.LOCAL_PREFERRED.value, 50.0, 0.20, 50.0, 2, 0),
            (RoutingPolicy.QUALITY_FIRST.value, 100.0, 0.05, 50.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    comparisons = artifact.candidate_comparisons
    # Verify sort order: promoted candidates before review-only before rejected
    good = next(
        c for c in comparisons if c.candidate_configuration_id == "candidate-config-good"
    )
    bad = next(
        c for c in comparisons if c.candidate_configuration_id == "candidate-config-bad"
    )
    neutral = next(
        c for c in comparisons if c.candidate_configuration_id == "candidate-config-neutral"
    )
    assert good.rank < bad.rank
    assert good.rank <= neutral.rank


# ---------------------------------------------------------------------------
# Objective deltas capture relative change
# ---------------------------------------------------------------------------


def test_objective_delta_captures_relative_change() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-delta",
        candidate_id="candidate-config-delta",
        config_profile_id="phase9-delta",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    latency_delta = next(
        delta for delta in candidate.objective_deltas
        if delta.metric is OptimizationObjectiveMetric.LATENCY_MS
    )
    assert latency_delta.baseline_value is not None
    assert latency_delta.candidate_value is not None
    # absolute_delta = candidate - baseline = 80 - 100 = -20
    assert latency_delta.absolute_delta is not None
    expected_abs = latency_delta.candidate_value - latency_delta.baseline_value
    assert latency_delta.absolute_delta == expected_abs
    # relative_delta = absolute_delta / |baseline|
    assert latency_delta.relative_delta is not None
    expected_relative = latency_delta.absolute_delta / abs(latency_delta.baseline_value)
    assert abs(latency_delta.relative_delta - expected_relative) < 1e-6


# ---------------------------------------------------------------------------
# Satisfied constraint IDs tracked
# ---------------------------------------------------------------------------


def test_satisfied_constraint_ids_tracked_in_recommendation() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-constrained",
        candidate_id="candidate-config-constrained",
        config_profile_id="phase9-constrained",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="error-hard-cap",
                dimension=OptimizationConstraintDimension.PREDICTED_ERROR_RATE,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=0.10,
                evaluated_value=0.02,
                satisfied=True,
                evidence_kinds=[OptimizationArtifactEvidenceKind.SIMULATED],
            ),
            OptimizationConstraintAssessment(
                constraint_id="canary-cap",
                dimension=OptimizationConstraintDimension.CANARY_PERCENTAGE,
                strength=OptimizationConstraintStrength.SOFT,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=25.0,
                evaluated_value=10.0,
                satisfied=True,
                evidence_kinds=[],
            ),
        ],
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert "error-hard-cap" in candidate.recommendation_summary.satisfied_constraint_ids
    assert "canary-cap" in candidate.recommendation_summary.satisfied_constraint_ids
    assert candidate.recommendation_summary.violated_constraint_ids == []


# ---------------------------------------------------------------------------
# Comparison notes contain normalized tradeoff score
# ---------------------------------------------------------------------------


def test_comparison_notes_contain_normalized_tradeoff_score() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-notes",
        candidate_id="candidate-config-notes",
        config_profile_id="phase9-notes",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    candidate = artifact.candidate_comparisons[0]
    assert any("normalized_tradeoff_score=" in note for note in candidate.notes)


# ---------------------------------------------------------------------------
# Campaign-level Pareto summary notes
# ---------------------------------------------------------------------------


def test_pareto_summary_notes_clarify_no_single_global_winner() -> None:
    latency_targets = [
        _objective_target(
            objective_id="latency-primary",
            metric=OptimizationObjectiveMetric.LATENCY_MS,
            goal=OptimizationGoal.MINIMIZE,
            weight=1.0,
        ),
    ]
    trial = _trial_artifact_flexible(
        trial_id="trial-pareto",
        candidate_id="candidate-config-pareto",
        config_profile_id="phase9-pareto",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
        objective_targets=latency_targets,
        objective_values={"latency-primary": 80.0},
    )
    campaign_artifact = _campaign_artifact(
        objective_targets=latency_targets,
        trials=[trial],
    )
    comparison = _simple_simulation_comparison(
        baseline_latency=100.0,
        baseline_error=0.05,
        candidates=[
            (RoutingPolicy.LATENCY_FIRST.value, 80.0, 0.02, 55.0, 2, 0),
        ],
    )

    artifact = compare_optimization_campaign(
        campaign_artifact=campaign_artifact,
        simulation_comparison=comparison,
        evaluation_artifacts=[_benchmark_artifact()],
        timestamp=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
    )

    assert any(
        "not globally ranked as a single winner" in note
        for note in artifact.pareto_summary.notes
    )
