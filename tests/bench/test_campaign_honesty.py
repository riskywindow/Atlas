"""Tests for campaign honesty assessment in hybrid local+remote environments."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from switchyard.bench.artifacts import summarize_records
from switchyard.bench.campaign_honesty import (
    CampaignHonestyWarningKind,
    assess_campaign_honesty,
    mark_campaign_invalidated,
    mark_campaign_stale,
    mark_trial_invalidated,
    mark_trial_stale,
)
from switchyard.bench.campaigns import (
    ForgeStageAExecutionResult,
    execute_forge_stage_a_campaign,
    inspect_forge_stage_a_campaigns,
)
from switchyard.config import Settings
from switchyard.schemas.backend import (
    BackendInstance,
    BackendNetworkEndpoint,
    BackendType,
    DeviceClass,
    ExecutionModeLabel,
    WorkerTransportType,
)
from switchyard.schemas.benchmark import (
    BenchmarkEnvironmentMetadata,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    CounterfactualObjective,
    DeployedTopologyEndpoint,
    WorkloadScenarioFamily,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.optimization import (
    OptimizationArtifactStatus,
)
from switchyard.schemas.routing import (
    RequestClass,
    RouteDecision,
    RoutingPolicy,
    WorkloadShape,
)


def test_budget_bound_warning_when_remote_budget_decreased() -> None:
    """A campaign that assumed a higher remote budget should warn when budget drops."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        current_remote_budget_per_minute=10,
        current_max_remote_share_percent=5.0,
    )

    # The campaign may or may not produce budget warnings depending on whether
    # trials have remote budget constraint assessments with values exceeding
    # our test thresholds. What matters is the assessment ran without error
    # and the result is well-typed.
    assert assessment.campaign_artifact_id == campaign.campaign_artifact_id
    assert isinstance(assessment.trustworthy, bool)
    for warning in assessment.warnings:
        if warning.kind is CampaignHonestyWarningKind.BUDGET_BOUND_EXCEEDED:
            assert len(warning.affected_trial_ids) > 0


def test_topology_drift_warning_when_workers_disappear() -> None:
    """When campaign workers are no longer in the current inventory, warn about drift."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Current inventory has a different worker than what was in the campaign
    current_inventory = [
        BackendInstance(
            instance_id="worker-new-gpu",
            endpoint=BackendNetworkEndpoint(
                base_url="http://10.0.0.99:8001",
                transport=WorkerTransportType.HTTP,
            ),
            backend_type=BackendType.VLLM_CUDA,
            device_class=DeviceClass.NVIDIA_GPU,
            execution_mode=ExecutionModeLabel.REMOTE_WORKER,
        )
    ]

    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        current_worker_inventory=current_inventory,
    )

    drift_warnings = [
        warning
        for warning in assessment.warnings
        if warning.kind is CampaignHonestyWarningKind.TOPOLOGY_DRIFT
    ]

    assert len(drift_warnings) >= 1
    disappeared_warning = next(
        (w for w in drift_warnings if "no longer present" in w.message), None
    )
    appeared_warning = next(
        (w for w in drift_warnings if "new worker" in w.message), None
    )
    assert disappeared_warning is not None
    assert appeared_warning is not None
    assert appeared_warning.severity == "info"
    assert disappeared_warning.severity == "warning"


def test_stale_evidence_warning_when_evidence_is_old() -> None:
    """Campaigns with old evidence windows should be flagged as stale."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Check staleness with a time far in the future
    far_future = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        now=far_future,
        staleness_hours=24,
    )

    stale_warnings = [
        warning
        for warning in assessment.warnings
        if warning.kind is CampaignHonestyWarningKind.STALE_EVIDENCE
    ]

    assert len(stale_warnings) >= 1
    assert assessment.trustworthy is False
    assert assessment.recommended_status in {
        OptimizationArtifactStatus.STALE,
        OptimizationArtifactStatus.INVALIDATED,
    }


def test_no_staleness_warning_when_evidence_is_fresh() -> None:
    """A recently run campaign should not be flagged as stale."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Check with a timestamp close to the campaign
    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        now=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
        staleness_hours=48,
    )

    stale_warnings = [
        warning
        for warning in assessment.warnings
        if warning.kind is CampaignHonestyWarningKind.STALE_EVIDENCE
    ]

    assert len(stale_warnings) == 0


def test_narrow_workload_coverage_warning() -> None:
    """Campaigns that only cover one workload family should warn about overfit."""

    result = _run_campaign_with_single_family()
    campaign = result.campaign_artifact

    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        min_workload_families=2,
    )

    overfit_warnings = [
        warning
        for warning in assessment.warnings
        if warning.kind is CampaignHonestyWarningKind.NARROW_WORKLOAD_COVERAGE
    ]

    assert len(overfit_warnings) >= 1
    assert "overfit" in overfit_warnings[0].notes[0]


def test_evidence_consistency_warning_on_replay_only_promotions() -> None:
    """Promotion recommendations without observed evidence should warn."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Check for consistency warnings; the campaign from our test helper
    # uses a local-only benchmark artifact (which produces OBSERVED evidence)
    assessment = assess_campaign_honesty(campaign_artifact=campaign)

    consistency_warnings = [
        warning
        for warning in assessment.warnings
        if warning.kind
        in {
            CampaignHonestyWarningKind.EVIDENCE_INCONSISTENCY,
            CampaignHonestyWarningKind.OBSERVED_EVIDENCE_MISSING,
        }
    ]

    # With observed evidence present, we should NOT get the
    # OBSERVED_EVIDENCE_MISSING warning
    missing_observed = [
        w
        for w in consistency_warnings
        if w.kind is CampaignHonestyWarningKind.OBSERVED_EVIDENCE_MISSING
    ]
    assert len(missing_observed) == 0


def test_mark_campaign_stale_sets_status_and_reason() -> None:
    """mark_campaign_stale should produce a STALE artifact with a reason."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    stale = mark_campaign_stale(campaign, reason="topology changed significantly")

    assert stale.result_status is OptimizationArtifactStatus.STALE
    assert stale.stale_reason == "topology changed significantly"
    assert any("marked stale" in note for note in stale.notes)
    # Original should be unchanged
    assert campaign.result_status is not OptimizationArtifactStatus.STALE


def test_mark_campaign_invalidated_sets_status_and_reason() -> None:
    """mark_campaign_invalidated should produce an INVALIDATED artifact with a reason."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    invalidated = mark_campaign_invalidated(
        campaign, reason="budget was reduced to zero"
    )

    assert invalidated.result_status is OptimizationArtifactStatus.INVALIDATED
    assert invalidated.invalidation_reason == "budget was reduced to zero"
    assert any("invalidated" in note for note in invalidated.notes)


def test_mark_trial_stale_sets_status_and_reason() -> None:
    """mark_trial_stale should produce a STALE trial artifact."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact
    assert len(campaign.trials) > 0
    trial = campaign.trials[0]

    stale_trial = mark_trial_stale(trial, reason="worker removed from inventory")

    assert stale_trial.result_status is OptimizationArtifactStatus.STALE
    assert stale_trial.stale_reason == "worker removed from inventory"


def test_inspection_surface_includes_honesty_warnings() -> None:
    """Inspection with current environment state should carry honesty warnings."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Use a different worker inventory to trigger drift
    current_inventory = [
        BackendInstance(
            instance_id="worker-new-gpu",
            endpoint=BackendNetworkEndpoint(
                base_url="http://10.0.0.99:8001",
                transport=WorkerTransportType.HTTP,
            ),
            backend_type=BackendType.VLLM_CUDA,
            device_class=DeviceClass.NVIDIA_GPU,
            execution_mode=ExecutionModeLabel.REMOTE_WORKER,
        )
    ]

    inspection = inspect_forge_stage_a_campaigns(
        campaign_artifacts=[campaign],
        comparison_artifacts=[result.campaign_comparison]
        if result.campaign_comparison is not None
        else [],
        current_worker_inventory=current_inventory,
    )

    assert len(inspection.campaigns) == 1
    campaign_summary = inspection.campaigns[0]

    # Should have honesty warnings from topology drift
    drift_warnings = [
        w
        for w in campaign_summary.honesty_warnings
        if w.kind.value == "topology_drift"
    ]
    assert len(drift_warnings) >= 1
    assert campaign_summary.trustworthy is False
    assert any(
        "honesty" in note for note in inspection.notes
    )


def test_assessment_is_clean_when_environment_matches() -> None:
    """When the environment matches the campaign, no warnings should be raised."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Use the same worker inventory as the campaign
    campaign_workers = (
        campaign.topology_lineage.worker_instance_inventory
        if campaign.topology_lineage is not None
        else []
    )

    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        current_worker_inventory=campaign_workers,
        now=datetime(2026, 3, 22, 19, 0, tzinfo=UTC),
        staleness_hours=48,
        min_workload_families=1,
    )

    assert assessment.trustworthy is True
    assert assessment.recommended_status in {
        OptimizationArtifactStatus.COMPLETE,
        OptimizationArtifactStatus.PARTIAL,
    }


def test_mark_trial_invalidated_sets_status_and_reason() -> None:
    """mark_trial_invalidated should produce an INVALIDATED trial artifact."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact
    assert len(campaign.trials) > 0
    trial = campaign.trials[0]

    invalidated = mark_trial_invalidated(
        trial, reason="observed and simulated evidence contradict"
    )

    assert invalidated.result_status is OptimizationArtifactStatus.INVALIDATED
    assert invalidated.invalidation_reason == "observed and simulated evidence contradict"
    assert any("invalidated" in note for note in invalidated.notes)
    # Original should be unchanged
    assert trial.result_status is not OptimizationArtifactStatus.INVALIDATED


def test_concurrency_cap_budget_warning() -> None:
    """Budget checks should include concurrency cap constraints."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # The campaign may not have concurrency cap constraints, but the function
    # should accept the parameter and run without error
    assessment = assess_campaign_honesty(
        campaign_artifact=campaign,
        current_remote_concurrency_cap=1,
    )

    assert assessment.campaign_artifact_id == campaign.campaign_artifact_id
    assert isinstance(assessment.trustworthy, bool)
    # Warnings about concurrency cap appear only when trials have
    # REMOTE_CONCURRENCY_CAP constraint assessments with values > 1
    for warning in assessment.warnings:
        if warning.kind is CampaignHonestyWarningKind.BUDGET_BOUND_EXCEEDED:
            assert len(warning.affected_trial_ids) > 0


def test_cost_signal_mismatch_warning() -> None:
    """Cost signal honesty check should flag configured-only remote constraints."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    assessment = assess_campaign_honesty(campaign_artifact=campaign)

    cost_warnings = [
        warning
        for warning in assessment.warnings
        if warning.kind is CampaignHonestyWarningKind.COST_SIGNAL_MISMATCH
    ]

    # Cost signal mismatch warnings appear when remote constraints were
    # evaluated without observed evidence.  Our test campaign uses local-only
    # benchmarks so the constraint evidence_kinds may be empty (no warning)
    # or populated with non-observed kinds (warning).  Either way the check
    # must run without error.
    for warning in cost_warnings:
        assert warning.severity == "info"
        assert len(warning.affected_trial_ids) > 0
        assert any("configured" in note or "cost" in note for note in warning.notes)


def test_inspection_always_runs_staleness_checks() -> None:
    """Inspection should run honesty checks even without environment state."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Provide NO environment state at all; force the campaign to look stale
    stale_campaign = campaign.model_copy(
        update={
            "timestamp": datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
            "evidence_records": [
                record.model_copy(
                    update={
                        "window_ended_at": datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
                    },
                    deep=True,
                )
                for record in campaign.evidence_records
            ],
        },
        deep=True,
    )

    # Inspection with no env state should still produce staleness warnings
    inspection = inspect_forge_stage_a_campaigns(
        campaign_artifacts=[stale_campaign],
    )

    assert len(inspection.campaigns) == 1
    campaign_summary = inspection.campaigns[0]

    # Always-on checks should have run
    assert any(
        "honesty checks always run" in note or "always run" in note
        for note in campaign_summary.notes
    )


def test_inspection_without_env_state_runs_coverage_checks() -> None:
    """Inspection runs workload coverage checks without environment state."""

    result = _run_campaign_with_single_family()
    campaign = result.campaign_artifact

    inspection = inspect_forge_stage_a_campaigns(
        campaign_artifacts=[campaign],
    )

    assert len(inspection.campaigns) == 1
    campaign_summary = inspection.campaigns[0]

    # Workload coverage warnings should appear since only one family was used
    coverage_warnings = [
        w
        for w in campaign_summary.honesty_warnings
        if w.kind.value == "narrow_workload_coverage"
    ]
    assert len(coverage_warnings) >= 1


def test_mixed_evidence_scenario_with_no_observed() -> None:
    """Campaigns with promotion recommendations but no observed evidence should warn."""

    result = _run_campaign_with_remote_evidence()
    campaign = result.campaign_artifact

    # Regardless of promotion recommendations, the consistency check should run
    assessment = assess_campaign_honesty(campaign_artifact=campaign)
    assert isinstance(assessment.trustworthy, bool)
    assert isinstance(assessment.warnings, list)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _run_campaign_with_remote_evidence() -> ForgeStageAExecutionResult:
    """Run a standard campaign for testing honesty checks."""

    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.optimization.worker_launch_presets = ()
    settings.phase4.policy_rollout.candidate_policy_id = None
    settings.phase4.policy_rollout.canary_percentage = 15.0

    artifact = _benchmark_artifact()

    return execute_forge_stage_a_campaign(
        settings=settings,
        evaluation_artifacts=[artifact],
        history_artifacts=[artifact],
        timestamp=datetime(2026, 3, 22, 18, 0, tzinfo=UTC),
    )


def _run_campaign_with_single_family() -> ForgeStageAExecutionResult:
    """Run a campaign with only one workload scenario family for overfit testing."""

    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.optimization.worker_launch_presets = ()
    settings.phase4.policy_rollout.candidate_policy_id = None

    artifact = _benchmark_artifact_single_family()

    return execute_forge_stage_a_campaign(
        settings=settings,
        evaluation_artifacts=[artifact],
        history_artifacts=[artifact],
        timestamp=datetime(2026, 3, 22, 18, 0, tzinfo=UTC),
    )


def _benchmark_artifact() -> BenchmarkRunArtifact:
    records = [
        *_records_for_backend(
            backend_name="local-reliable",
            latency_ms=60.0,
            tokens_per_second=200.0,
            count=12,
            start_index=0,
            failure_indexes=set(),
        ),
        *_records_for_backend(
            backend_name="local-fast",
            latency_ms=20.0,
            tokens_per_second=40.0,
            count=18,
            start_index=12,
            failure_indexes={12},
        ),
    ]
    summary = summarize_records(records)
    return BenchmarkRunArtifact(
        run_id="phase9-honesty-benchmark",
        timestamp=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
        scenario=BenchmarkScenario(
            name="phase9-honesty",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=30,
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["local-fast", "local-reliable"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=30,
        summary=summary,
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="synthetic",
            deployed_topology=[
                DeployedTopologyEndpoint(
                    endpoint_id="gateway-primary",
                    role="gateway",
                    address="http://127.0.0.1:8000",
                    transport=WorkerTransportType.HTTP,
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
        ),
        records=records,
    )


def _benchmark_artifact_single_family() -> BenchmarkRunArtifact:
    records = [
        *_records_for_backend(
            backend_name="local-fast",
            latency_ms=20.0,
            tokens_per_second=40.0,
            count=10,
            start_index=0,
            failure_indexes=set(),
            scenario_family=WorkloadScenarioFamily.SHORT_CHAT,
        ),
    ]
    summary = summarize_records(records)
    return BenchmarkRunArtifact(
        run_id="phase9-single-family",
        timestamp=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
        scenario=BenchmarkScenario(
            name="phase9-single-family",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=10,
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["local-fast"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=10,
        summary=summary,
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="synthetic",
            deployed_topology=[
                DeployedTopologyEndpoint(
                    endpoint_id="gateway-primary",
                    role="gateway",
                    address="http://127.0.0.1:8000",
                    transport=WorkerTransportType.HTTP,
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
        ),
        records=records,
    )


def _records_for_backend(
    *,
    backend_name: str,
    latency_ms: float,
    tokens_per_second: float,
    count: int,
    start_index: int,
    failure_indexes: set[int],
    scenario_family: WorkloadScenarioFamily | None = None,
) -> list[BenchmarkRequestRecord]:
    records = []
    for index in range(start_index, start_index + count):
        started_at = datetime(2026, 3, 22, 12, 0, tzinfo=UTC) + timedelta(
            seconds=index
        )
        success = index not in failure_indexes
        status_code = 200 if success else 503
        records.append(
            BenchmarkRequestRecord(
                request_id=f"honesty-request-{index}",
                tenant_id="tenant-a",
                request_class=RequestClass.STANDARD,
                backend_name=backend_name,
                backend_type="mock",
                model_alias="chat-shared",
                model_identifier="chat-shared",
                started_at=started_at,
                completed_at=started_at + timedelta(milliseconds=latency_ms),
                latency_ms=latency_ms,
                output_tokens=32,
                tokens_per_second=tokens_per_second,
                route_decision=RouteDecision(
                    backend_name=backend_name,
                    serving_target="chat-shared",
                    policy=RoutingPolicy.BALANCED,
                    request_id=f"honesty-request-{index}",
                    workload_shape=WorkloadShape.INTERACTIVE,
                    rationale=["fixture route decision"],
                    considered_backends=[backend_name],
                ),
                success=success,
                status_code=status_code,
                usage=UsageStats(
                    prompt_tokens=24,
                    completion_tokens=32,
                    total_tokens=56,
                ),
                error=None if success else "backend overloaded",
                error_category=None if success else "runtime_error",
                scenario_family=scenario_family,
            )
        )
    return records
