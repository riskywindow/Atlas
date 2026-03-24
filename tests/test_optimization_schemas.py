from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError

from switchyard.config import Settings
from switchyard.optimization import (
    build_baseline_optimization_config_profile,
    build_candidate_optimization_config_profile,
    build_optimization_profile,
    build_trial_optimization_config_profile,
)
from switchyard.schemas.backend import (
    BackendInstance,
    BackendNetworkEndpoint,
    BackendType,
    DeviceClass,
    ExecutionModeLabel,
    WorkerTransportType,
)
from switchyard.schemas.benchmark import (
    CounterfactualObjective,
    DeployedTopologyEndpoint,
    RecommendationConfidence,
    WorkloadScenarioFamily,
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
    OptimizationCandidateEligibilityStatus,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationMetadata,
    OptimizationCandidateGenerationResult,
    OptimizationCandidateGenerationStrategy,
    OptimizationComparisonOperator,
    OptimizationConfigProfileRole,
    OptimizationConfigProfileValidationIssueKind,
    OptimizationConstraint,
    OptimizationConstraintAssessment,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationDomainKind,
    OptimizationEvidenceMixSummary,
    OptimizationEvidenceRecord,
    OptimizationGeneratedCandidate,
    OptimizationGoal,
    OptimizationKnobChange,
    OptimizationKnobDomain,
    OptimizationObjectiveAssessment,
    OptimizationObjectiveDelta,
    OptimizationObjectiveMetric,
    OptimizationObjectiveTarget,
    OptimizationParetoSummary,
    OptimizationPromotionDecision,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationScope,
    OptimizationScopeKind,
    OptimizationTopologyLineage,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
    OptimizationWorkloadImpactSummary,
    OptimizationWorkloadMetricDelta,
    OptimizationWorkloadSet,
    OptimizationWorkloadSourceKind,
)
from switchyard.schemas.routing import (
    PolicyRolloutMode,
    RoutingPolicy,
    TopologySnapshotReference,
)


def _optimization_workload_set() -> OptimizationWorkloadSet:
    return OptimizationWorkloadSet(
        workload_set_id="phase9-cache-locality",
        source_kind=OptimizationWorkloadSourceKind.BUILT_IN_SCENARIO_FAMILY,
        serving_targets=["chat-shared"],
        scenario_families=[WorkloadScenarioFamily.REPEATED_PREFIX],
    )


def _optimization_topology_lineage() -> OptimizationTopologyLineage:
    return OptimizationTopologyLineage(
        topology_references=[
            TopologySnapshotReference(
                topology_snapshot_id="topology-phase9-001",
                capture_source="runtime-inspection",
                captured_at=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
                artifact_run_id="benchmark-phase9-001",
            )
        ],
        deployed_topology=[
            DeployedTopologyEndpoint(
                endpoint_id="gateway-primary",
                role="gateway",
                address="http://127.0.0.1:8000",
                transport=WorkerTransportType.HTTP,
                execution_mode=ExecutionModeLabel.HOST_NATIVE,
            )
        ],
        worker_instance_inventory=[
            BackendInstance(
                instance_id="worker-local-1",
                endpoint=BackendNetworkEndpoint(
                    base_url="http://127.0.0.1:8001",
                    transport=WorkerTransportType.HTTP,
                ),
                backend_type=BackendType.VLLM_METAL,
                device_class=DeviceClass.APPLE_GPU,
                execution_mode=ExecutionModeLabel.HOST_NATIVE,
            )
        ],
    )


def _baseline_trial_identity() -> OptimizationTrialIdentity:
    return OptimizationTrialIdentity(
        trial_id="trial-baseline",
        candidate_id="routing_policy:balanced",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id="phase9-baseline",
        routing_policy=RoutingPolicy.BALANCED,
    )


def _candidate_trial_identity() -> OptimizationTrialIdentity:
    return OptimizationTrialIdentity(
        trial_id="trial-latency",
        candidate_id="routing_policy:latency-first",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id="phase9-latency",
        routing_policy=RoutingPolicy.LATENCY_FIRST,
    )


def _objective_target() -> OptimizationObjectiveTarget:
    return OptimizationObjectiveTarget(
        objective_id="latency-primary",
        metric=OptimizationObjectiveMetric.LATENCY_MS,
        goal=OptimizationGoal.MINIMIZE,
        applies_to=[OptimizationScope(kind=OptimizationScopeKind.GLOBAL)],
        workload_set_ids=["phase9-cache-locality"],
        evidence_sources=[
            ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
            ForgeEvidenceSourceKind.REPLAYED_TRACE,
        ],
    )


def _constraint() -> OptimizationConstraint:
    return OptimizationConstraint(
        constraint_id="remote-share-cap",
        dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
        strength=OptimizationConstraintStrength.HARD,
        operator=OptimizationComparisonOperator.LTE,
        threshold_value=25.0,
    )


def _candidate_configuration(
    *,
    candidate_configuration_id: str,
    trial_identity: OptimizationTrialIdentity,
    baseline_config_profile_id: str,
    knob_change_value: float | None,
) -> OptimizationCandidateConfigurationArtifact:
    knob_changes = []
    if knob_change_value is not None:
        knob_changes.append(
            OptimizationKnobChange(
                knob_id="shadow_sampling_rate",
                config_path="routing.shadow_policy.sample_rate",
                baseline_value=0.0,
                candidate_value=knob_change_value,
            )
        )
    return OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=candidate_configuration_id,
        campaign_id="campaign-phase9-001",
        candidate=trial_identity,
        baseline_config_profile_id=baseline_config_profile_id,
        config_profile_id=trial_identity.config_profile_id,
        knob_changes=knob_changes,
        objectives_in_scope=[_objective_target()],
        constraints_in_scope=[_constraint()],
        workload_sets=[_optimization_workload_set()],
        expected_evidence_kinds=[
            OptimizationArtifactEvidenceKind.OBSERVED,
            OptimizationArtifactEvidenceKind.REPLAYED,
            OptimizationArtifactEvidenceKind.SIMULATED,
            OptimizationArtifactEvidenceKind.ESTIMATED,
        ],
        topology_lineage=_optimization_topology_lineage(),
    )


def _baseline_candidate_configuration() -> OptimizationCandidateConfigurationArtifact:
    return _candidate_configuration(
        candidate_configuration_id="candidate-config-baseline",
        trial_identity=_baseline_trial_identity(),
        baseline_config_profile_id="phase9-baseline",
        knob_change_value=None,
    )


def _candidate_trial_configuration() -> OptimizationCandidateConfigurationArtifact:
    return _candidate_configuration(
        candidate_configuration_id="candidate-config-latency",
        trial_identity=_candidate_trial_identity(),
        baseline_config_profile_id="phase9-baseline",
        knob_change_value=0.1,
    )


def _evidence_records() -> list[OptimizationEvidenceRecord]:
    return [
        OptimizationEvidenceRecord(
            evidence_id="evidence-observed",
            evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
            source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
            source_artifact_id="benchmark-run-001",
            source_run_ids=["benchmark-run-001", "benchmark-run-001"],
            window_started_at=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
            window_ended_at=datetime(2026, 3, 22, 12, 5, tzinfo=UTC),
        ),
        OptimizationEvidenceRecord(
            evidence_id="evidence-replayed",
            evidence_kind=OptimizationArtifactEvidenceKind.REPLAYED,
            source_type=OptimizationArtifactSourceType.REPLAY_PLAN,
            source_artifact_id="replay-plan-001",
            source_trace_ids=["trace-a"],
        ),
        OptimizationEvidenceRecord(
            evidence_id="evidence-simulated",
            evidence_kind=OptimizationArtifactEvidenceKind.SIMULATED,
            source_type=OptimizationArtifactSourceType.SIMULATION,
            source_artifact_id="simulation-001",
            source_simulation_ids=["simulation-001"],
        ),
        OptimizationEvidenceRecord(
            evidence_id="evidence-estimated",
            evidence_kind=OptimizationArtifactEvidenceKind.ESTIMATED,
            source_type=OptimizationArtifactSourceType.ESTIMATE_SUMMARY,
            source_artifact_id="estimate-001",
        ),
    ]


def _recommendation_summary(
    *,
    candidate_configuration_id: str = "candidate-config-latency",
    config_profile_id: str = "phase9-latency",
) -> OptimizationRecommendationSummary:
    return OptimizationRecommendationSummary(
        recommendation_summary_id="recommendation-001",
        disposition=OptimizationRecommendationDisposition.PROMOTE_CANDIDATE,
        recommendation_label=OptimizationRecommendationLabel.PROMOTION_ELIGIBLE,
        confidence=RecommendationConfidence.MEDIUM,
        candidate_configuration_id=candidate_configuration_id,
        config_profile_id=config_profile_id,
        evidence_kinds=[
            OptimizationArtifactEvidenceKind.OBSERVED,
            OptimizationArtifactEvidenceKind.REPLAYED,
        ],
        reason_codes=[
            OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED,
            OptimizationRecommendationReasonCode.NON_DOMINATED,
        ],
        improved_objective_ids=["latency-primary"],
        satisfied_constraint_ids=["remote-share-cap"],
        benefited_workload_families=["repeated_prefix"],
        evidence_mix=OptimizationEvidenceMixSummary(
            total_request_count=8,
            replay_backed_request_count=8,
            simulated_request_count=8,
            direct_observation_count=5,
            estimated_request_count=3,
            unsupported_request_count=0,
            observed_share=0.625,
            replayed_share=1.0,
            simulated_share=1.0,
            estimated_share=0.375,
        ),
        rationale=["Observed latency improved without violating remote spillover cap."],
    )


def _promotion_decision(
    *,
    candidate_configuration_id: str = "candidate-config-latency",
    config_profile_id: str = "phase9-latency",
) -> OptimizationPromotionDecision:
    return OptimizationPromotionDecision(
        promotion_decision_id="promotion-001",
        disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
        candidate_configuration_id=candidate_configuration_id,
        config_profile_id=config_profile_id,
        rollout_mode=PolicyRolloutMode.CANARY,
        canary_percentage=10.0,
        rollback_supported=True,
        rationale=["Bounded canary is available and rollback controls remain active."],
    )


def _trial_artifact(
    candidate_configuration: OptimizationCandidateConfigurationArtifact | None = None,
) -> OptimizationTrialArtifact:
    chosen_candidate_configuration = (
        _candidate_trial_configuration()
        if candidate_configuration is None
        else candidate_configuration
    )
    return OptimizationTrialArtifact(
        trial_artifact_id="trial-artifact-001",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-config-baseline",
        candidate_configuration=chosen_candidate_configuration,
        trial_identity=chosen_candidate_configuration.candidate,
        evidence_records=_evidence_records(),
        topology_lineage=_optimization_topology_lineage(),
        result_status=OptimizationArtifactStatus.PARTIAL,
        objective_assessments=[
            OptimizationObjectiveAssessment(
                objective_id="latency-primary",
                metric=OptimizationObjectiveMetric.LATENCY_MS,
                goal=OptimizationGoal.MINIMIZE,
                measured_value=91.5,
                target_value=100.0,
                satisfied=True,
                evidence_kinds=[
                    OptimizationArtifactEvidenceKind.OBSERVED,
                    OptimizationArtifactEvidenceKind.REPLAYED,
                ],
            )
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
        recommendation_summary=_recommendation_summary(
            candidate_configuration_id=(
                chosen_candidate_configuration.candidate_configuration_id
            ),
            config_profile_id=chosen_candidate_configuration.config_profile_id,
        ),
        promotion_decision=_promotion_decision(
            candidate_configuration_id=(
                chosen_candidate_configuration.candidate_configuration_id
            ),
            config_profile_id=chosen_candidate_configuration.config_profile_id,
        ),
    )


def _campaign_artifact() -> OptimizationCampaignArtifact:
    baseline_candidate = _baseline_candidate_configuration()
    candidate_configuration = _candidate_trial_configuration()
    return OptimizationCampaignArtifact(
        campaign_artifact_id="campaign-artifact-001",
        campaign=OptimizationCampaignMetadata(
            campaign_id="campaign-phase9-001",
            optimization_profile_id="phase9-stage-a-baseline",
            objective=CounterfactualObjective.BALANCED,
            evidence_sources=[
                ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
                ForgeEvidenceSourceKind.REPLAYED_TRACE,
            ],
            required_evidence_sources=[ForgeEvidenceSourceKind.OBSERVED_RUNTIME],
            default_workload_set_ids=["phase9-cache-locality"],
        ),
        baseline_candidate_configuration=baseline_candidate,
        candidate_configurations=[candidate_configuration],
        trials=[_trial_artifact(candidate_configuration)],
        evidence_records=_evidence_records(),
        topology_lineage=_optimization_topology_lineage(),
        recommendation_summaries=[_recommendation_summary()],
        promotion_decisions=[_promotion_decision()],
        result_status=OptimizationArtifactStatus.PARTIAL,
    )


def test_optimization_knob_domain_rejects_invalid_enum_shape() -> None:
    try:
        OptimizationKnobDomain(domain_kind=OptimizationDomainKind.ENUM)
    except ValidationError as exc:
        assert "allowed_values" in str(exc)
    else:
        raise AssertionError("enum knob domains should require allowed_values")


def test_optimization_scope_rejects_global_target() -> None:
    try:
        OptimizationScope(
            kind=OptimizationScopeKind.GLOBAL,
            target="chat-shared",
        )
    except ValidationError as exc:
        assert "global scopes must not set target" in str(exc)
    else:
        raise AssertionError("global optimization scope should reject target")


def test_optimization_workload_set_requires_selector() -> None:
    try:
        OptimizationWorkloadSet(
            workload_set_id="empty",
            source_kind=OptimizationWorkloadSourceKind.BUILT_IN_SCENARIO_FAMILY,
        )
    except ValidationError as exc:
        assert "workload sets must select at least one" in str(exc)
    else:
        raise AssertionError("workload sets should require at least one selector")


def test_optimization_trial_identity_validates_candidate_kind() -> None:
    try:
        OptimizationTrialIdentity(
            trial_id="trial-1",
            candidate_id="routing-policy:balanced",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="phase9-balanced",
        )
    except ValidationError as exc:
        assert "routing_policy candidates require routing_policy" in str(exc)
    else:
        raise AssertionError("routing-policy trial identities should require routing_policy")


def test_build_optimization_profile_exposes_phase9_contract() -> None:
    profile = build_optimization_profile(Settings())

    assert profile.profile_id == "phase9-stage-a-baseline"
    assert profile.campaign is not None
    assert profile.campaign.campaign_id == "phase9-stage-a-baseline-forge-stage-a"
    assert any(knob.knob_id == "session_affinity_ttl_seconds" for knob in profile.knobs)
    assert any(
        knob.knob_id == "hybrid_remote_concurrency_cap"
        and knob.domain.domain_kind is OptimizationDomainKind.INTEGER_RANGE
        and knob.domain.nullable is True
        for knob in profile.knobs
    )
    assert any(
        objective.objective_id == "cache_locality_latency"
        and objective.applies_to[0].kind is OptimizationScopeKind.SCENARIO_FAMILY
        for objective in profile.objectives
    )
    assert any(constraint.strength.value == "soft" for constraint in profile.constraints)
    assert any(
        workload_set.workload_set_id == "phase9-cache-locality"
        for workload_set in profile.workload_sets
    )
    assert any(
        candidate.worker_launch_preset == "host_native_config"
        and candidate.applies_to[0].kind is OptimizationScopeKind.WORKER_CLASS
        for candidate in profile.candidate_trials
    )
    assert any(
        excluded.dimension_id == "prefix_locality_tracker_limits"
        for excluded in profile.excluded_dimensions
    )
    assert any(
        excluded.dimension_id == "per_backend_admission_overrides"
        for excluded in profile.excluded_dimensions
    )
    admission_knobs = [
        knob for knob in profile.knobs
        if knob.group.value == "admission_control"
    ]
    assert len(admission_knobs) >= 3
    assert any(knob.knob_id == "admission_global_queue_size" for knob in admission_knobs)
    assert any(knob.knob_id == "admission_queue_timeout_seconds" for knob in admission_knobs)
    backend_protection_knobs = [
        knob for knob in profile.knobs
        if knob.group.value == "backend_protection"
    ]
    assert len(backend_protection_knobs) >= 2
    assert any(knob.knob_id == "circuit_failure_threshold" for knob in backend_protection_knobs)
    assert any(
        knob.knob_id == "circuit_failure_threshold"
        and knob.domain.domain_kind is OptimizationDomainKind.INTEGER_RANGE
        and knob.domain.min_value == 1
        for knob in backend_protection_knobs
    )

    round_tripped = type(profile).model_validate_json(profile.model_dump_json())
    assert round_tripped == profile


def test_candidate_generation_result_round_trips_with_metadata() -> None:
    result = OptimizationCandidateGenerationResult(
        profile_id="phase9-stage-a-baseline",
        baseline_trial=_baseline_trial_identity(),
        baseline_generation=OptimizationCandidateGenerationMetadata(
            strategy=OptimizationCandidateGenerationStrategy.FIXED_BASELINE,
            rationale=["baseline captured from active profile"],
        ),
        generation_config=OptimizationCandidateGenerationConfig(),
        eligible_candidates=[
            OptimizationGeneratedCandidate(
                trial=_candidate_trial_identity(),
                knob_changes=[
                    OptimizationKnobChange(
                        knob_id="default_routing_policy",
                        config_path="default_routing_policy",
                        baseline_value="balanced",
                        candidate_value="latency_first",
                    )
                ],
                generation=OptimizationCandidateGenerationMetadata(
                    strategy=OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
                    strategy_index=1,
                    seed=17,
                    varied_knob_ids=["default_routing_policy"],
                    rationale=["single-knob search"],
                ),
                eligibility=OptimizationCandidateEligibilityRecord(
                    status=OptimizationCandidateEligibilityStatus.ELIGIBLE,
                ),
            )
        ],
        rejected_candidates=[
            OptimizationGeneratedCandidate(
                trial=OptimizationTrialIdentity(
                    trial_id="trial-rejected",
                    candidate_id="generated-config:rejected",
                    candidate_kind=ForgeCandidateKind.CONFIG_PROFILE,
                    config_profile_id="phase9-rejected",
                ),
                generation=OptimizationCandidateGenerationMetadata(
                    strategy=OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH,
                    strategy_index=2,
                    varied_knob_ids=["policy_rollout_mode"],
                    rationale=["grid search candidate"],
                ),
                eligibility=OptimizationCandidateEligibilityRecord(
                    status=OptimizationCandidateEligibilityStatus.REJECTED,
                    rejection_reasons=["canary rollout mode requires a positive canary percentage"],
                ),
            )
        ],
    )

    round_tripped = OptimizationCandidateGenerationResult.model_validate_json(
        result.model_dump_json()
    )

    assert round_tripped == result
    assert round_tripped.eligible_candidates[0].eligibility.eligible is True
    assert round_tripped.rejected_candidates[0].eligibility.eligible is False


def test_optimization_campaign_artifact_round_trips_with_lineage() -> None:
    artifact = _campaign_artifact()

    round_tripped = OptimizationCampaignArtifact.model_validate_json(
        artifact.model_dump_json()
    )
    evidence_payload = round_tripped.model_dump(mode="json")["evidence_records"]

    assert round_tripped == artifact
    assert round_tripped.trials[0].candidate_configuration.workload_sets[0].workload_set_id == (
        "phase9-cache-locality"
    )
    assert round_tripped.topology_lineage is not None
    assert round_tripped.topology_lineage.worker_instance_inventory[0].instance_id == (
        "worker-local-1"
    )
    assert {
        record["evidence_kind"] for record in evidence_payload
    } == {"observed", "replayed", "simulated", "estimated"}
    assert round_tripped.evidence_records[0].source_run_ids == ["benchmark-run-001"]


def test_optimization_campaign_comparison_artifact_round_trips() -> None:
    comparison = OptimizationCampaignComparisonArtifact(
        comparison_artifact_id="comparison-001",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-config-baseline",
        candidate_comparisons=[
            OptimizationCandidateComparisonArtifact(
                candidate_configuration_id="candidate-config-latency",
                trial_artifact_id="trial-artifact-001",
                config_profile_id="phase9-latency",
                rank=1,
                tied_candidate_configuration_ids=["candidate-config-quality"],
                objective_deltas=[
                    OptimizationObjectiveDelta(
                        objective_id="latency-primary",
                        metric=OptimizationObjectiveMetric.LATENCY_MS,
                        goal=OptimizationGoal.MINIMIZE,
                        baseline_value=100.0,
                        candidate_value=91.5,
                        absolute_delta=-8.5,
                        relative_delta=-0.085,
                        normalized_tradeoff=0.085,
                        improved=True,
                    )
                ],
                normalized_tradeoff_score=0.085,
                pareto_optimal=True,
                dominated=False,
                workload_impacts=[
                    OptimizationWorkloadImpactSummary(
                        workload_family="repeated_prefix",
                        request_count=4,
                        metric_deltas=[
                            OptimizationWorkloadMetricDelta(
                                metric=OptimizationObjectiveMetric.LATENCY_MS,
                                baseline_value=110.0,
                                candidate_value=95.0,
                                absolute_delta=-15.0,
                                improved=True,
                            )
                        ],
                        improved_metrics=[OptimizationObjectiveMetric.LATENCY_MS],
                    )
                ],
                recommendation_summary=_recommendation_summary(),
            )
        ],
        pareto_summary=OptimizationParetoSummary(
            frontier_candidate_configuration_ids=["candidate-config-latency"],
        ),
    )

    round_tripped = OptimizationCampaignComparisonArtifact.model_validate_json(
        comparison.model_dump_json()
    )

    assert round_tripped == comparison
    assert round_tripped.candidate_comparisons[0].recommendation_summary.evidence_mix is not None


def test_optimization_campaign_artifact_rejects_trial_with_unknown_candidate() -> None:
    unknown_candidate_configuration = _candidate_configuration(
        candidate_configuration_id="candidate-config-unknown",
        trial_identity=OptimizationTrialIdentity(
            trial_id="trial-unknown",
            candidate_id="routing_policy:quality-first",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="phase9-quality",
            routing_policy=RoutingPolicy.QUALITY_FIRST,
        ),
        baseline_config_profile_id="phase9-baseline",
        knob_change_value=0.2,
    )

    try:
        OptimizationCampaignArtifact(
            campaign_artifact_id="campaign-artifact-invalid",
            campaign=OptimizationCampaignMetadata(
                campaign_id="campaign-phase9-001",
                optimization_profile_id="phase9-stage-a-baseline",
            ),
            baseline_candidate_configuration=_baseline_candidate_configuration(),
            candidate_configurations=[_candidate_trial_configuration()],
            trials=[_trial_artifact(unknown_candidate_configuration)],
        )
    except ValidationError as exc:
        assert "present in the campaign artifact" in str(exc)
    else:
        raise AssertionError("campaign artifacts should reject unknown trial candidates")


def test_optimization_trial_artifact_requires_status_reasons() -> None:
    candidate_configuration = _candidate_trial_configuration()
    trial_payload = _trial_artifact(candidate_configuration).model_dump(mode="python")

    try:
        OptimizationTrialArtifact(
            **(
                trial_payload
                | {
                    "result_status": OptimizationArtifactStatus.STALE,
                    "stale_reason": None,
                }
            )
        )
    except ValidationError as exc:
        assert "stale artifacts require stale_reason" in str(exc)
    else:
        raise AssertionError("stale trial artifacts should require stale_reason")

    try:
        OptimizationTrialArtifact(
            **(
                trial_payload
                | {
                    "result_status": OptimizationArtifactStatus.INVALIDATED,
                    "invalidation_reason": None,
                }
            )
        )
    except ValidationError as exc:
        assert "invalidated artifacts require invalidation_reason" in str(exc)
    else:
        raise AssertionError(
            "invalidated trial artifacts should require invalidation_reason"
        )


def test_optimization_evidence_record_validates_time_windows() -> None:
    try:
        OptimizationEvidenceRecord(
            evidence_id="evidence-invalid-window",
            evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
            source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
            source_artifact_id="benchmark-run-invalid",
            window_started_at=datetime(2026, 3, 22, 12, 5, tzinfo=UTC),
            window_ended_at=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
        )
    except ValidationError as exc:
        assert "window_started_at must be <= window_ended_at" in str(exc)
    else:
        raise AssertionError("evidence records should reject inverted time windows")


def test_build_baseline_optimization_config_profile_exposes_identity_and_scope() -> None:
    profile = build_baseline_optimization_config_profile(Settings(), profile_version=3)

    assert profile.profile_role is OptimizationConfigProfileRole.BASELINE
    assert profile.profile_version == 3
    assert profile.config_profile_id == "phase9-stage-a-baseline-baseline"
    assert profile.baseline_config_profile_id == profile.config_profile_id
    assert profile.applies_to[0].kind is OptimizationScopeKind.GLOBAL
    assert profile.diff.changed_knob_ids == []
    assert profile.validation.compatible is True


def test_build_trial_optimization_config_profile_carries_provenance_and_diff() -> None:
    trial = _trial_artifact()

    profile = build_trial_optimization_config_profile(
        settings=Settings(),
        trial_artifact=trial,
        campaign_artifact_id="campaign-artifact-001",
        profile_version=2,
    )

    assert profile.profile_role is OptimizationConfigProfileRole.PROMOTED
    assert profile.profile_version == 2
    assert profile.provenance.campaign_artifact_id == "campaign-artifact-001"
    assert profile.provenance.trial_artifact_id == "trial-artifact-001"
    assert profile.provenance.candidate_configuration_id == "candidate-config-latency"
    assert profile.provenance.recommendation_summary_id == "recommendation-001"
    assert profile.diff.changed_knob_ids == ["shadow_sampling_rate"]
    assert profile.validation.compatible is False
    assert {issue.issue_kind for issue in profile.validation.issues} == {
        OptimizationConfigProfileValidationIssueKind.CONFIG_PATH_MISMATCH,
        OptimizationConfigProfileValidationIssueKind.PROVENANCE_MISMATCH,
    }


def test_candidate_optimization_config_profile_rejects_scope_mismatch() -> None:
    candidate = _candidate_configuration(
        candidate_configuration_id="candidate-config-scoped",
        trial_identity=OptimizationTrialIdentity(
            trial_id="trial-scoped",
            candidate_id="routing_policy:balanced-scoped",
            candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
            config_profile_id="phase9-scoped",
            routing_policy=RoutingPolicy.BALANCED,
            applies_to=[
                OptimizationScope(
                    kind=OptimizationScopeKind.SCENARIO_FAMILY,
                    target=WorkloadScenarioFamily.REPEATED_PREFIX.value,
                )
            ],
        ),
        baseline_config_profile_id="phase9-stage-a-baseline-baseline",
        knob_change_value=None,
    ).model_copy(
        update={
            "knob_changes": [
                OptimizationKnobChange(
                    knob_id="default_routing_policy",
                    config_path="default_routing_policy",
                    applies_to=[
                        OptimizationScope(
                            kind=OptimizationScopeKind.SCENARIO_FAMILY,
                            target=WorkloadScenarioFamily.REPEATED_PREFIX.value,
                        )
                    ],
                    baseline_value=RoutingPolicy.BALANCED.value,
                    candidate_value=RoutingPolicy.LOCAL_PREFERRED.value,
                )
            ]
        },
        deep=True,
    )

    profile = build_candidate_optimization_config_profile(
        settings=Settings(),
        candidate_configuration=candidate,
    )

    assert profile.applies_to[0].kind is OptimizationScopeKind.SCENARIO_FAMILY
    assert profile.validation.compatible is False
    assert {
        issue.issue_kind for issue in profile.validation.issues
    } == {OptimizationConfigProfileValidationIssueKind.SCOPE_NOT_ALLOWED}


def test_candidate_optimization_config_profile_tracks_declared_supported_diffs() -> None:
    settings = Settings()
    candidate = _candidate_configuration(
        candidate_configuration_id="candidate-config-supported",
        trial_identity=OptimizationTrialIdentity(
            trial_id="trial-supported",
            candidate_id="config_profile:supported",
            candidate_kind=ForgeCandidateKind.CONFIG_PROFILE,
            config_profile_id="phase9-supported",
        ),
        baseline_config_profile_id="phase9-stage-a-baseline-baseline",
        knob_change_value=None,
    ).model_copy(
        update={
            "knob_changes": [
                OptimizationKnobChange(
                    knob_id="hybrid_max_remote_share_percent",
                    config_path="phase7.hybrid_execution.max_remote_share_percent",
                    baseline_value=10.0,
                    candidate_value=20.0,
                )
            ]
        },
        deep=True,
    )

    profile = build_candidate_optimization_config_profile(
        settings=settings,
        candidate_configuration=candidate,
    )

    assert profile.validation.compatible is True
    assert profile.diff.changed_knob_ids == ["hybrid_max_remote_share_percent"]
    assert profile.diff.mutable_runtime_knob_ids == ["hybrid_max_remote_share_percent"]
    assert profile.changes[0].candidate_value == 20.0
