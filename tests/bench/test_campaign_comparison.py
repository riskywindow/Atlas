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
