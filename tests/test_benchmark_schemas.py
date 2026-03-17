from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from switchyard.schemas.backend import (
    BackendHealth,
    BackendHealthState,
    BackendImageMetadata,
    BackendInstance,
    BackendNetworkEndpoint,
    BackendType,
    DeploymentProfile,
    DeviceClass,
)
from switchyard.schemas.benchmark import (
    BenchmarkArtifactSchemaVersion,
    BenchmarkComparisonArtifact,
    BenchmarkComparisonDeltaSummary,
    BenchmarkComparisonSideSummary,
    BenchmarkDeploymentTarget,
    BenchmarkEnvironmentMetadata,
    BenchmarkPolicyComparison,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkRunConfig,
    BenchmarkScenario,
    BenchmarkSummary,
    BenchmarkTargetComparisonArtifact,
    BenchmarkWarmupConfig,
    CacheObservation,
    CapturedTraceRecord,
    ComparisonRunSummary,
    ComparisonSourceKind,
    ControlPlaneReportMetadata,
    DeployedTopologyEndpoint,
    ExecutionTarget,
    ExecutionTargetType,
    FamilyBenchmarkSummary,
    Phase4SchemaCompatibility,
    ReplayMode,
    ReplayPlan,
    ReplayRequest,
    ReportFormat,
    ReportMetadata,
    ReportSourceOfTruth,
    ScenarioDelta,
    TraceCaptureMode,
    WorkloadGenerationConfig,
    WorkloadItem,
    WorkloadPattern,
    WorkloadScenarioFamily,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import (
    AdmissionDecision,
    AdmissionDecisionState,
    CanaryPolicy,
    RouteCandidateExplanation,
    RouteDecision,
    RouteEligibilityState,
    RouteExplanation,
    RouteTelemetryMetadata,
    RoutingPolicy,
    ShadowPolicy,
    WorkloadShape,
)


def test_benchmark_artifact_serializes_with_phase3_defaults() -> None:
    started_at = datetime(2026, 3, 15, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=25)
    scenario = BenchmarkScenario(
        name="smoke",
        model="mock-chat",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        request_count=1,
        workload_generation=WorkloadGenerationConfig(
            pattern=WorkloadPattern.REPEATED_PREFIX,
            seed=7,
            shared_prefix="shared-prefix",
        ),
    )
    route_decision = RouteDecision(
        backend_name="mock-a",
        serving_target="mock-chat",
        policy=RoutingPolicy.BALANCED,
        request_id="req_1",
        workload_shape=WorkloadShape.INTERACTIVE,
        rationale=["lowest latency"],
        considered_backends=["mock-a", "mock-b"],
        fallback_backends=["mock-b"],
        explanation=RouteExplanation(
            serving_target="mock-chat",
            candidates=[
                RouteCandidateExplanation(
                    backend_name="mock-a",
                    serving_target="mock-chat",
                    eligibility_state=RouteEligibilityState.ELIGIBLE,
                    score=10.0,
                    rationale=["lowest latency"],
                )
            ],
            selected_backend="mock-a",
            selected_reason=["lowest latency"],
        ),
    )
    record = BenchmarkRequestRecord(
        request_id="req_1",
        scenario_family=WorkloadScenarioFamily.REPEATED_PREFIX,
        backend_name="mock-a",
        backend_type="mock",
        model_alias="mock-chat",
        model_identifier="mock-chat",
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=25.0,
        output_tokens=2,
        tokens_per_second=80.0,
        route_decision=route_decision,
        cache_observation=CacheObservation(supports_prefix_cache=True),
        success=True,
        status_code=200,
        usage=UsageStats(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    artifact = BenchmarkRunArtifact(
        run_id="run_1",
        git_sha="abcdef123456",
        scenario=scenario,
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-a"],
        backend_types_involved=["mock"],
        model_aliases_involved=["mock-chat"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=25.0,
            p50_latency_ms=25.0,
            p95_latency_ms=25.0,
            avg_ttft_ms=None,
            p50_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=2,
            avg_output_tokens=2.0,
            avg_tokens_per_second=80.0,
            p95_tokens_per_second=80.0,
            fallback_count=0,
            chosen_backend_counts={"mock-a": 1},
            family_summaries={
                WorkloadScenarioFamily.REPEATED_PREFIX: FamilyBenchmarkSummary(
                    family=WorkloadScenarioFamily.REPEATED_PREFIX,
                    request_count=1,
                    success_count=1,
                    failure_count=0,
                    avg_latency_ms=25.0,
                    p50_latency_ms=25.0,
                    p95_latency_ms=25.0,
                    avg_ttft_ms=None,
                    p50_ttft_ms=None,
                    p95_ttft_ms=None,
                    total_output_tokens=2,
                    avg_output_tokens=2.0,
                    avg_tokens_per_second=80.0,
                    p95_tokens_per_second=80.0,
                    fallback_count=0,
                    chosen_backend_counts={"mock-a": 1},
                )
            },
        ),
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="synthetic",
            deployment_target=BenchmarkDeploymentTarget.COMPOSE,
            deployment_profile=DeploymentProfile.COMPOSE,
            config_profile_name="compose-smoke",
            stream=False,
            timeout_seconds=30.0,
            canary_percentage=5.0,
            shadow_sampling_rate=0.1,
            deployed_topology=[
                DeployedTopologyEndpoint(
                    endpoint_id="gateway-local",
                    role="control_plane",
                    address="http://127.0.0.1:8000",
                )
            ],
            worker_instance_inventory=[
                BackendInstance(
                    instance_id="worker-a",
                    endpoint=BackendNetworkEndpoint(base_url="http://host.docker.internal:8101"),
                    backend_type=BackendType.MLX_LM,
                    device_class=DeviceClass.APPLE_GPU,
                    health=BackendHealth(state=BackendHealthState.HEALTHY),
                )
            ],
            control_plane_image=BackendImageMetadata(
                image_tag="switchyard/control-plane:dev",
                git_sha="abcdef123456",
            ),
            topology_capture_source="gateway_admin_runtime",
        ),
        records=[record],
    )

    payload = artifact.model_dump(mode="json")

    assert payload["schema_version"] == BenchmarkArtifactSchemaVersion.V2.value
    assert payload["git_revision"] == "abcdef123456"
    assert payload["model_alias"] == "mock-chat"
    assert payload["run_config"]["concurrency"] == 1
    assert payload["run_config"]["warmup"]["enabled"] is False
    assert payload["run_config"]["canary_percentage"] == 0.0
    assert payload["scenario"]["scenario_seed"] == 7
    assert payload["records"][0]["routing_policy"] == "balanced"
    assert payload["records"][0]["scenario_family"] == "repeated_prefix"
    assert payload["records"][0]["cache_observation"]["supports_prefix_cache"] is True
    assert payload["records"][0]["route_candidate_count"] == 2
    assert payload["records"][0]["route_reason"] == (
        "target=mock-chat | selected=mock-a | reason=lowest latency"
    )
    assert payload["environment"]["deployment_target"] == "compose"
    assert payload["environment"]["deployment_profile"] == "compose"
    assert payload["environment"]["config_profile_name"] == "compose-smoke"
    assert payload["environment"]["deployed_topology"][0]["address"] == "http://127.0.0.1:8000"
    assert payload["environment"]["worker_instance_inventory"][0]["instance_id"] == "worker-a"
    assert (
        payload["environment"]["control_plane_image"]["image_tag"]
        == "switchyard/control-plane:dev"
    )
    assert payload["environment_snapshot"]["platform"] == payload["environment"]["platform"]
    assert payload["summary"]["family_summaries"]["repeated_prefix"]["request_count"] == 1


def test_control_plane_report_metadata_serializes_with_v2_compatibility() -> None:
    metadata = ControlPlaneReportMetadata(
        tenant_id="tenant-a",
        session_id="session-1",
        admission_decision=AdmissionDecision(
            state=AdmissionDecisionState.ADMITTED,
            limiter_key="tenant-a",
        ),
        canary_policy=CanaryPolicy(
            policy_name="rollout",
            serving_target="chat-shared",
            enabled=True,
        ),
        shadow_policy=ShadowPolicy(
            policy_name="shadow",
            enabled=True,
            serving_target="chat-shared",
            target_backend="mock-b",
            sampling_rate=0.2,
        ),
        telemetry_metadata=RouteTelemetryMetadata(tenant_id="tenant-a"),
    )

    payload = metadata.model_dump(mode="json")

    assert payload["compatibility"] == Phase4SchemaCompatibility.V2_EXTENDED.value
    assert payload["admission_decision"]["state"] == "admitted"


def test_benchmark_request_record_carries_phase4_control_plane_metadata() -> None:
    started_at = datetime(2026, 3, 15, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=10)
    record = BenchmarkRequestRecord(
        request_id="req-phase4",
        backend_name="mock-a",
        backend_type="mock",
        model_alias="mock-chat",
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=10.0,
        success=True,
        status_code=200,
        control_plane_metadata=ControlPlaneReportMetadata(
            tenant_id="tenant-a",
            admission_decision=AdmissionDecision(
                state=AdmissionDecisionState.ADMITTED,
                limiter_key="tenant-a",
            ),
        ),
    )

    payload = record.model_dump(mode="json")

    assert payload["control_plane_metadata"]["tenant_id"] == "tenant-a"


def test_phase4_workload_family_serializes_cleanly() -> None:
    item = WorkloadItem(
        item_id="tenant-contention-1",
        family=WorkloadScenarioFamily.TENANT_CONTENTION,
        prompt="Summarize the contention window.",
        metadata={
            "tenant_id": "tenant-priority",
            "request_class": "latency_sensitive",
        },
    )

    payload = item.model_dump(mode="json")

    assert payload["family"] == "tenant_contention"


def test_benchmark_record_rejects_invalid_failure_shape() -> None:
    started_at = datetime(2026, 3, 15, tzinfo=UTC)
    completed_at = started_at + timedelta(milliseconds=10)

    try:
        BenchmarkRequestRecord(
            request_id="req_2",
            backend_name="mock-a",
            backend_type="mock",
            model_alias="mock-chat",
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=10.0,
            success=False,
            status_code=500,
        )
    except ValidationError as exc:
        assert "error message" in str(exc)
    else:
        raise AssertionError("BenchmarkRequestRecord should require an error for failures")


def test_workload_generation_rejects_invalid_repeated_prefix_shape() -> None:
    try:
        WorkloadGenerationConfig(pattern=WorkloadPattern.REPEATED_PREFIX)
    except ValidationError as exc:
        assert "shared_prefix" in str(exc)
    else:
        raise AssertionError("Repeated-prefix workloads should require a shared prefix")


def test_workload_scenario_rejects_mismatched_item_count() -> None:
    try:
        BenchmarkScenario(
            name="items",
            model="mock-chat",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=2,
            items=[
                WorkloadItem(
                    item_id="1",
                    family=WorkloadScenarioFamily.SHORT_CHAT,
                    prompt="hello",
                )
            ],
        )
    except ValidationError as exc:
        assert "items length" in str(exc)
    else:
        raise AssertionError("Scenario should reject item counts that do not match request_count")


def test_run_config_rejects_disabled_warmup_with_requests() -> None:
    try:
        BenchmarkRunConfig(
            warmup=BenchmarkWarmupConfig(enabled=False, request_count=1),
        )
    except ValidationError as exc:
        assert "warmup.request_count" in str(exc)
    else:
        raise AssertionError("Disabled warmup should not permit warmup requests")


def test_trace_record_and_replay_plan_serialize() -> None:
    target = ExecutionTarget(
        target_type=ExecutionTargetType.PINNED_BACKEND,
        model_alias="chat-shared",
        pinned_backend="mlx-lm:chat-mlx",
    )
    trace = CapturedTraceRecord(
        record_id="trace-1",
        request_id="req-1",
        execution_target=target,
        fallback_used=False,
        status_code=200,
        latency_ms=12.0,
        capture_mode=TraceCaptureMode.METADATA_ONLY,
    )
    replay = ReplayPlan(
        plan_id="replay-1",
        source_run_id="run-1",
        execution_target=target,
        replay_mode=ReplayMode.FIXED_CONCURRENCY,
        concurrency=2,
        warmup=BenchmarkWarmupConfig(enabled=True, request_count=1),
        requests=[
            ReplayRequest(
                replay_request_id="replay_req_1",
                source_request_id="req-1",
                source_trace_record_id="trace-1",
                order_index=0,
                stream=False,
            )
        ],
    )

    trace_payload = trace.model_dump(mode="json")
    replay_payload = replay.model_dump(mode="json")

    assert trace_payload["execution_target"]["target_type"] == "pinned_backend"
    assert replay_payload["warmup"]["request_count"] == 1
    assert replay_payload["concurrency"] == 2
    assert replay_payload["replay_mode"] == "fixed_concurrency"
    assert replay_payload["trace_record_ids"] == ["trace-1"]


def test_comparison_summary_and_report_metadata_serialize() -> None:
    comparison = BenchmarkComparisonArtifact(
        run_id="comparison-1",
        scenario_name="phase3-compare",
        model="chat-shared",
        workload_shape=WorkloadShape.INTERACTIVE,
        request_count=1,
        results=[
            BenchmarkPolicyComparison(
                comparison_label="balanced",
                policy=RoutingPolicy.BALANCED,
                run_id="run-balanced",
                backends_involved=["mock-a"],
                summary=BenchmarkSummary(
                    request_count=1,
                    success_count=1,
                    failure_count=0,
                    avg_latency_ms=5.0,
                    p50_latency_ms=5.0,
                    p95_latency_ms=5.0,
                    avg_ttft_ms=None,
                    p95_ttft_ms=None,
                    total_output_tokens=4,
                    avg_output_tokens=4.0,
                    avg_tokens_per_second=10.0,
                    p95_tokens_per_second=10.0,
                    fallback_count=0,
                    chosen_backend_counts={"mock-a": 1},
                ),
            )
        ],
        best_policy_by_latency=RoutingPolicy.BALANCED,
        best_result_by_latency="balanced",
    )
    report = ReportMetadata(
        report_id="report-1",
        report_format=ReportFormat.MARKDOWN,
        source_of_truth=ReportSourceOfTruth.BENCHMARK_ARTIFACT,
        source_run_ids=["run-balanced"],
        source_schema_versions=[BenchmarkArtifactSchemaVersion.V2],
    )

    payload = comparison.model_dump(mode="json")
    report_payload = report.model_dump(mode="json")

    assert payload["schema_version"] == "switchyard.benchmark.v2"
    assert payload["comparison_summary"]["result_count"] == 1
    assert report_payload["source_of_truth"] == "benchmark_artifact"


def test_two_target_comparison_artifact_serializes() -> None:
    left_target = ExecutionTarget(
        target_type=ExecutionTargetType.ROUTING_POLICY,
        model_alias="chat-shared",
        routing_policy=RoutingPolicy.BALANCED,
    )
    right_target = ExecutionTarget(
        target_type=ExecutionTargetType.PINNED_BACKEND,
        model_alias="chat-shared",
        pinned_backend="mock-fast",
    )
    artifact = BenchmarkTargetComparisonArtifact(
        comparison_id="compare-1",
        source_kind=ComparisonSourceKind.WORKLOAD_MANIFEST,
        source_name="phase3-manifest",
        request_count=2,
        left=BenchmarkComparisonSideSummary(
            run_id="run-left",
            execution_target=left_target,
            request_count=2,
            success_rate=1.0,
            error_rate=0.0,
            fallback_rate=0.0,
            p50_latency_ms=10.0,
            p95_latency_ms=12.0,
            p50_ttft_ms=3.0,
            p95_ttft_ms=4.0,
            avg_tokens_per_second=20.0,
            p95_tokens_per_second=18.0,
            route_distribution={"mock-a": 2},
            backend_distribution={"mock-a": 2},
        ),
        right=BenchmarkComparisonSideSummary(
            run_id="run-right",
            execution_target=right_target,
            request_count=2,
            success_rate=0.5,
            error_rate=0.5,
            fallback_rate=0.5,
            p50_latency_ms=15.0,
            p95_latency_ms=18.0,
            p50_ttft_ms=5.0,
            p95_ttft_ms=6.0,
            avg_tokens_per_second=16.0,
            p95_tokens_per_second=14.0,
            route_distribution={"mock-fast": 2},
            backend_distribution={"mock-fast": 2},
        ),
        delta=BenchmarkComparisonDeltaSummary(
            request_count_delta=0,
            success_rate_delta=-0.5,
            error_rate_delta=0.5,
            fallback_rate_delta=0.5,
            p50_latency_delta_ms=5.0,
            p95_latency_delta_ms=6.0,
            p50_ttft_delta_ms=2.0,
            p95_ttft_delta_ms=2.0,
            avg_tokens_per_second_delta=-4.0,
            p95_tokens_per_second_delta=-4.0,
            route_distribution_delta={"mock-fast": 2, "mock-a": -2},
            backend_distribution_delta={"mock-fast": 2, "mock-a": -2},
            notable_scenario_deltas=[
                ScenarioDelta(
                    key="workload:item-1",
                    left_request_id="left-1",
                    right_request_id="right-1",
                    latency_delta_ms=5.0,
                    success_changed=True,
                    backend_changed=True,
                    route_changed=True,
                )
            ],
        ),
    )

    payload = artifact.model_dump(mode="json")

    assert payload["source_kind"] == "workload_manifest"
    assert payload["comparison_summary"]["result_count"] == 2
    assert payload["delta"]["notable_scenario_deltas"][0]["key"] == "workload:item-1"


def test_phase2_style_artifact_input_still_derives_phase3_fields() -> None:
    started_at = datetime(2026, 3, 15, tzinfo=UTC)
    artifact = BenchmarkRunArtifact(
        run_id="legacy-shape",
        scenario=BenchmarkScenario(
            name="legacy",
            model="mock-chat",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=1,
        ),
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-a"],
        backend_types_involved=["mock"],
        model_aliases_involved=["mock-chat"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=1.0,
            p50_latency_ms=1.0,
            p95_latency_ms=1.0,
            avg_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=1,
            avg_output_tokens=1.0,
            avg_tokens_per_second=1.0,
            p95_tokens_per_second=1.0,
            fallback_count=0,
            chosen_backend_counts={"mock-a": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="synthetic",
            stream=False,
            timeout_seconds=30.0,
        ),
        records=[
            BenchmarkRequestRecord(
                request_id="req-1",
                backend_name="mock-a",
                backend_type="mock",
                model_alias="mock-chat",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=0.0,
                success=True,
                status_code=200,
            )
        ],
    )

    assert artifact.schema_version is BenchmarkArtifactSchemaVersion.V2
    assert artifact.execution_target is not None
    assert artifact.run_config.execution_target is not None
    assert artifact.environment_snapshot is not None
    assert artifact.model_alias == "mock-chat"


def test_comparison_summary_rejects_length_mismatch() -> None:
    try:
        ComparisonRunSummary(
            compared_run_ids=["run-1"],
            compared_targets=[
                ExecutionTarget(
                    target_type=ExecutionTargetType.ROUTING_POLICY,
                    model_alias="chat-shared",
                    routing_policy=RoutingPolicy.BALANCED,
                ),
                ExecutionTarget(
                    target_type=ExecutionTargetType.ROUTING_POLICY,
                    model_alias="chat-shared",
                    routing_policy=RoutingPolicy.QUALITY_FIRST,
                ),
            ],
            result_count=1,
            best_result_by_latency="run-1",
        )
    except ValidationError as exc:
        assert "compared_targets" in str(exc)
    else:
        raise AssertionError("ComparisonRunSummary should enforce length consistency")
