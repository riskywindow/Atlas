from __future__ import annotations

from datetime import UTC, datetime, timedelta

from switchyard.bench.artifacts import summarize_records
from switchyard.bench.campaigns import execute_forge_stage_a_campaign
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
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.optimization import (
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactStatus,
    OptimizationCandidateGenerationConfig,
    OptimizationCandidateGenerationStrategy,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
)
from switchyard.schemas.routing import (
    RequestClass,
    RouteDecision,
    RoutingPolicy,
    WorkloadShape,
)


def test_execute_forge_stage_a_campaign_materializes_supported_policy_trials() -> None:
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

    result = execute_forge_stage_a_campaign(
        settings=settings,
        evaluation_artifacts=[artifact],
        history_artifacts=[artifact],
        timestamp=datetime(2026, 3, 22, 18, 0, tzinfo=UTC),
    )

    campaign = result.campaign_artifact

    assert result.simulation_comparison is not None
    assert result.campaign_comparison is not None
    assert result.recommendation_report is not None
    assert result.skipped_candidate_ids == ()
    assert campaign.result_status is OptimizationArtifactStatus.COMPLETE
    assert campaign.topology_lineage is not None
    assert campaign.topology_lineage.worker_instance_inventory[0].instance_id == "worker-local-1"
    assert len(campaign.trials) == 1

    trial = campaign.trials[0]
    assert trial.trial_identity.routing_policy is RoutingPolicy.LATENCY_FIRST
    assert trial.recommendation_summary is not None
    assert (
        trial.recommendation_summary.disposition
        is OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
    )
    assert trial.promotion_decision is not None
    assert (
        trial.promotion_decision.disposition
        is OptimizationPromotionDisposition.RECOMMEND_CANARY
    )
    assert trial.promotion_decision.canary_percentage == 15.0
    assert {
        record.evidence_kind for record in trial.evidence_records
    } == {
        OptimizationArtifactEvidenceKind.OBSERVED,
        OptimizationArtifactEvidenceKind.SIMULATED,
        OptimizationArtifactEvidenceKind.ESTIMATED,
    }


def test_execute_forge_stage_a_campaign_marks_unsupported_candidates_partial() -> None:
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LATENCY_FIRST,
    )
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.optimization.worker_launch_presets = (
        settings.optimization.worker_launch_presets[0],
    )
    settings.phase4.policy_rollout.candidate_policy_id = None

    artifact = _benchmark_artifact()

    result = execute_forge_stage_a_campaign(
        settings=settings,
        evaluation_artifacts=[artifact],
        history_artifacts=[artifact],
        timestamp=datetime(2026, 3, 22, 18, 30, tzinfo=UTC),
    )

    campaign = result.campaign_artifact

    assert campaign.result_status is OptimizationArtifactStatus.PARTIAL
    assert result.skipped_candidate_ids == ("runtime-profile:host_native_config",)
    assert len(campaign.candidate_configurations) == 2
    assert len(campaign.trials) == 1
    assert any(
        candidate.candidate.worker_launch_preset == "host_native_config"
        for candidate in campaign.candidate_configurations
    )


def test_execute_forge_stage_a_campaign_carries_generation_metadata() -> None:
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.BALANCED
    settings.optimization.allowlisted_routing_policies = (
        RoutingPolicy.BALANCED,
        RoutingPolicy.LOCAL_PREFERRED,
    )
    settings.optimization.objective = CounterfactualObjective.LATENCY
    settings.phase4.policy_rollout.candidate_policy_id = None

    artifact = _benchmark_artifact()

    result = execute_forge_stage_a_campaign(
        settings=settings,
        evaluation_artifacts=[artifact],
        history_artifacts=[artifact],
        candidate_generation_config=OptimizationCandidateGenerationConfig(
            strategies=[
                OptimizationCandidateGenerationStrategy.ONE_FACTOR_AT_A_TIME,
                OptimizationCandidateGenerationStrategy.BOUNDED_GRID_SEARCH,
            ],
            allowed_knob_ids=[
                "default_routing_policy",
                "policy_rollout_mode",
                "policy_rollout_canary_percentage",
            ],
            seed=23,
        ),
        timestamp=datetime(2026, 3, 22, 18, 45, tzinfo=UTC),
    )

    assert result.candidate_generation is not None
    assert result.rejected_candidate_ids
    assert result.candidate_generation.rejected_candidates[0].eligibility.eligible is False
    assert len(result.campaign_artifact.trials) == 1
    candidate_configuration = result.campaign_artifact.trials[0].candidate_configuration
    assert candidate_configuration.generation is not None
    assert candidate_configuration.generation.seed == 23
    assert candidate_configuration.eligibility is not None
    assert candidate_configuration.eligibility.eligible is True


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
        run_id="phase9-campaign-benchmark",
        timestamp=datetime(2026, 3, 22, 12, 0, tzinfo=UTC),
        scenario=BenchmarkScenario(
            name="phase9-campaign",
            model="chat-shared",
            model_alias="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=len(records),
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["local-fast", "local-reliable"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=len(records),
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
) -> list[BenchmarkRequestRecord]:
    records = []
    for index in range(start_index, start_index + count):
        started_at = datetime(2026, 3, 22, 12, 0, tzinfo=UTC) + timedelta(seconds=index)
        success = index not in failure_indexes
        status_code = 200 if success else 503
        records.append(
            BenchmarkRequestRecord(
                request_id=f"request-{index}",
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
                    request_id=f"request-{index}",
                    workload_shape=WorkloadShape.INTERACTIVE,
                    rationale=["fixture route decision"],
                    considered_backends=["local-reliable", "local-fast"],
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
            )
        )
    return records
