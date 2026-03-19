from __future__ import annotations

from switchyard.control.operator import HybridOperatorService
from switchyard.schemas.backend import (
    BackendDeployment,
    BackendInstance,
    BackendNetworkEndpoint,
    BackendType,
    CloudPlacementMetadata,
    CostBudgetProfile,
    CostProfileClass,
    DeploymentProfile,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
    WorkerTransportType,
)
from switchyard.schemas.routing import (
    RouteDecision,
    RouteExecutionObservation,
    RoutingPolicy,
    WorkloadShape,
)


def test_operator_runtime_reports_estimated_cloud_evidence_from_selected_deployment() -> None:
    service = HybridOperatorService()
    decision = RouteDecision(
        backend_name="remote-worker:cuda-a",
        serving_target="chat-shared",
        policy=RoutingPolicy.BURST_TO_REMOTE,
        request_id="req-cloud-1",
        workload_shape=WorkloadShape.INTERACTIVE,
        rationale=["remote spillover selected"],
        considered_backends=["remote-worker:cuda-a"],
        selected_deployment=BackendDeployment(
            name="remote-worker:cuda-a",
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM,
            model_identifier="meta-llama/Llama-3.1-8B-Instruct",
            serving_targets=["chat-shared"],
            deployment_profile=DeploymentProfile.REMOTE,
            execution_mode=ExecutionModeLabel.REMOTE_WORKER,
            environment="staging",
            placement=CloudPlacementMetadata(
                provider="aws",
                region="us-east-1",
                zone="us-east-1a",
            ),
            cost_profile=CostBudgetProfile(
                profile=CostProfileClass.STANDARD,
                budget_bucket="gpu-staging",
                relative_cost_index=0.42,
                currency="usd",
            ),
            instances=[
                BackendInstance(
                    instance_id="cuda-a-1",
                    endpoint=BackendNetworkEndpoint(
                        base_url="https://cuda-a.internal",
                        transport=WorkerTransportType.HTTPS,
                    ),
                    backend_type=BackendType.VLLM_CUDA,
                    device_class=DeviceClass.NVIDIA_GPU,
                    execution_mode=ExecutionModeLabel.REMOTE_WORKER,
                    placement=CloudPlacementMetadata(
                        provider="aws",
                        region="us-east-1",
                        zone="us-east-1a",
                    ),
                )
            ],
        ),
    )

    service.record_route_example(
        decision=decision,
        executed_backend="remote-worker:cuda-a",
        remote_candidate_count=1,
        final_outcome="succeeded",
        fallback_used=False,
    )

    summary = service.inspect_state(remote_effectively_enabled=True)

    assert summary.recent_placement_distribution.remote_count == 1
    assert summary.recent_cloud_evidence.estimated_placement_count == 1
    assert summary.recent_cloud_evidence.estimated_cost_count == 1
    assert summary.recent_cloud_evidence.remote_provider_counts == {"aws": 1}
    assert summary.recent_cloud_evidence.total_estimated_relative_cost_index == 0.42
    assert summary.recent_route_examples[0].placement_provider == "aws"
    assert summary.recent_route_examples[0].placement_region == "us-east-1"
    assert summary.recent_route_examples[0].placement_evidence_source == (
        "deployment_metadata_estimate"
    )
    assert summary.recent_route_examples[0].cost_evidence_source == (
        "deployment_metadata_estimate"
    )


def test_operator_runtime_prefers_observed_instance_cloud_evidence() -> None:
    service = HybridOperatorService()
    decision = RouteDecision(
        backend_name="remote-worker:cuda-a",
        serving_target="chat-shared",
        policy=RoutingPolicy.BURST_TO_REMOTE,
        request_id="req-cloud-2",
        workload_shape=WorkloadShape.INTERACTIVE,
        rationale=["remote spillover selected"],
        considered_backends=["remote-worker:cuda-a"],
        selected_deployment=BackendDeployment(
            name="remote-worker:cuda-a",
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM,
            model_identifier="meta-llama/Llama-3.1-8B-Instruct",
            serving_targets=["chat-shared"],
            deployment_profile=DeploymentProfile.REMOTE,
            execution_mode=ExecutionModeLabel.REMOTE_WORKER,
            environment="staging",
            placement=CloudPlacementMetadata(provider="aws", region="us-east-1"),
            cost_profile=CostBudgetProfile(
                profile=CostProfileClass.STANDARD,
                budget_bucket="gpu-staging",
                relative_cost_index=0.42,
                currency="usd",
            ),
            instances=[
                BackendInstance(
                    instance_id="cuda-a-1",
                    endpoint=BackendNetworkEndpoint(
                        base_url="https://cuda-a.internal",
                        transport=WorkerTransportType.HTTPS,
                    ),
                    backend_type=BackendType.VLLM_CUDA,
                    device_class=DeviceClass.NVIDIA_GPU,
                    execution_mode=ExecutionModeLabel.REMOTE_WORKER,
                    placement=CloudPlacementMetadata(
                        provider="aws",
                        region="us-east-1",
                        zone="us-east-1b",
                    ),
                    cost_profile=CostBudgetProfile(
                        profile=CostProfileClass.PREMIUM,
                        budget_bucket="gpu-canary",
                        relative_cost_index=0.73,
                        currency="usd",
                    ),
                )
            ],
        ),
        execution_observation=RouteExecutionObservation(
            executed_backend="remote-worker:cuda-a",
            backend_instance_id="cuda-a-1",
            status_code=200,
            final_outcome="succeeded",
        ),
    )

    service.record_route_example(
        decision=decision,
        executed_backend="remote-worker:cuda-a",
        remote_candidate_count=1,
        final_outcome="succeeded",
        fallback_used=False,
    )

    summary = service.inspect_state(remote_effectively_enabled=True)

    assert summary.recent_cloud_evidence.observed_placement_count == 1
    assert summary.recent_cloud_evidence.observed_cost_count == 1
    assert summary.recent_cloud_evidence.estimated_placement_count == 0
    assert summary.recent_cloud_evidence.estimated_cost_count == 0
    assert summary.recent_route_examples[0].placement_zone == "us-east-1b"
    assert summary.recent_route_examples[0].relative_cost_index == 0.73
    assert summary.recent_route_examples[0].placement_evidence_source == "observed_runtime"
    assert summary.recent_route_examples[0].cost_evidence_source == "observed_runtime"
