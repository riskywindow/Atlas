from pydantic import ValidationError

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendDeployment,
    BackendHealth,
    BackendHealthState,
    BackendImageMetadata,
    BackendInstance,
    BackendInstanceSource,
    BackendLoadState,
    BackendNetworkEndpoint,
    BackendRegistrationMetadata,
    BackendStatusSnapshot,
    BackendType,
    CacheCapabilityFlags,
    CloudPlacementMetadata,
    CostBudgetProfile,
    CostProfileClass,
    DeploymentProfile,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
    LogicalModelTarget,
    NetworkProfile,
    PerformanceHint,
    QualityHint,
    ReadinessHints,
    TopologySchemaVersion,
    TrustMetadata,
    WorkerAuthState,
    WorkerLocalityClass,
    WorkerRegistrationState,
    WorkerTransportType,
    WorkerTrustState,
)


def test_backend_snapshot_serializes() -> None:
    snapshot = BackendStatusSnapshot(
        name="mock-a",
        capabilities=BackendCapabilities(
            backend_type=BackendType.MOCK,
            engine_type=EngineType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["chat-default"],
            max_context_tokens=8192,
            concurrency_limit=2,
            quality_hint=QualityHint.BALANCED,
            performance_hint=PerformanceHint.LATENCY_OPTIMIZED,
            readiness_hints=ReadinessHints(cold_start_likely=False),
            model_aliases={"chat-default": "mock-chat"},
            default_model="mock-chat",
            cache_capabilities=CacheCapabilityFlags(supports_prefix_cache=True),
        ),
        deployment=BackendDeployment(
            name="mock-a",
            backend_type=BackendType.MOCK,
            engine_type=EngineType.MOCK,
            model_identifier="mock-chat",
            serving_targets=["chat-default"],
            configured_priority=50,
            configured_weight=2.0,
            execution_mode=ExecutionModeLabel.HOST_NATIVE,
            instances=[
                BackendInstance(
                    instance_id="mock-a-1",
                    endpoint=BackendNetworkEndpoint(
                        base_url="http://127.0.0.1:8101",
                        transport=WorkerTransportType.HTTP,
                    ),
                    source_of_truth=BackendInstanceSource.REGISTERED,
                    backend_type=BackendType.MOCK,
                    device_class=DeviceClass.CPU,
                    model_identifier="mock-chat",
                    locality_class=WorkerLocalityClass.LOCAL_NETWORK,
                    execution_mode=ExecutionModeLabel.HOST_NATIVE,
                    placement=CloudPlacementMetadata(provider="local-lab", region="dev"),
                    cost_profile=CostBudgetProfile(profile=CostProfileClass.LOCAL),
                    readiness_hints=ReadinessHints(warm_pool_enabled=True),
                    trust=TrustMetadata(
                        auth_state=WorkerAuthState.STATIC_TOKEN,
                        trust_state=WorkerTrustState.TRUSTED,
                    ),
                    tags=["local", "canary"],
                    registration=BackendRegistrationMetadata(
                        state=WorkerRegistrationState.REGISTERED,
                    ),
                    health=BackendHealth(
                        state=BackendHealthState.HEALTHY,
                        load_state=BackendLoadState.READY,
                    ),
                    image_metadata=BackendImageMetadata(image_tag="switchyard/mock:dev"),
                )
            ],
            deployment_profile=DeploymentProfile.COMPOSE,
            environment="dev",
            placement=CloudPlacementMetadata(provider="local-lab", region="dev"),
            cost_profile=CostBudgetProfile(profile=CostProfileClass.LOCAL),
            readiness_hints=ReadinessHints(warm_pool_enabled=True),
            build_metadata=BackendImageMetadata(image_tag="switchyard/control-plane:dev"),
        ),
        logical_targets=[LogicalModelTarget(alias="chat-default", deployments=["mock-a"])],
        health=BackendHealth(
            state=BackendHealthState.HEALTHY,
            latency_ms=12.5,
            error_rate=0.0,
            load_state=BackendLoadState.READY,
            warmed_models=["mock-chat"],
        ),
        active_requests=1,
    )

    payload = snapshot.model_dump(mode="json")

    assert payload["capabilities"]["backend_type"] == "mock"
    assert payload["capabilities"]["topology_schema_version"] == TopologySchemaVersion.V1.value
    assert payload["capabilities"]["engine_type"] == "mock"
    assert payload["capabilities"]["readiness_hints"]["cold_start_likely"] is False
    assert payload["capabilities"]["serving_targets"] == ["chat-default"]
    assert payload["health"]["state"] == "healthy"
    assert payload["health"]["load_state"] == "ready"
    assert payload["health"]["circuit_open"] is False
    assert payload["deployment"]["configured_priority"] == 50
    assert payload["deployment"]["deployment_profile"] == "compose"
    assert payload["deployment"]["execution_mode"] == "host_native"
    assert payload["deployment"]["placement"]["provider"] == "local-lab"
    assert payload["deployment"]["environment"] == "dev"
    assert payload["deployment"]["instances"][0]["endpoint"]["base_url"] == "http://127.0.0.1:8101"
    assert payload["deployment"]["instances"][0]["endpoint"]["transport"] == "http"
    assert payload["deployment"]["instances"][0]["source_of_truth"] == "registered"
    assert payload["deployment"]["instances"][0]["locality_class"] == "local_network"
    assert payload["deployment"]["instances"][0]["execution_mode"] == "host_native"
    assert payload["deployment"]["instances"][0]["placement"]["provider"] == "local-lab"
    assert payload["deployment"]["instances"][0]["cost_profile"]["profile"] == "local"
    assert payload["deployment"]["instances"][0]["trust"]["auth_state"] == "static_token"
    assert payload["deployment"]["instances"][0]["tags"] == ["local", "canary"]
    assert payload["deployment"]["instances"][0]["registration"]["state"] == "registered"
    assert payload["deployment"]["instances"][0]["health"]["state"] == "healthy"
    assert payload["instance_inventory"][0]["instance_id"] == "mock-a-1"
    assert payload["logical_targets"][0]["alias"] == "chat-default"


def test_backend_instance_infers_remote_defaults_from_cloud_metadata() -> None:
    instance = BackendInstance(
        instance_id="cuda-remote-1",
        endpoint=BackendNetworkEndpoint(
            base_url="https://worker.example.com",
            transport=WorkerTransportType.HTTPS,
        ),
        backend_type=BackendType.VLLM_CUDA,
        device_class=DeviceClass.NVIDIA_GPU,
        placement=CloudPlacementMetadata(
            provider="aws",
            region="us-east-1",
            zone="us-east-1a",
        ),
    )

    assert instance.topology_schema_version is TopologySchemaVersion.V1
    assert instance.locality_class is WorkerLocalityClass.REMOTE_CLOUD
    assert instance.execution_mode is ExecutionModeLabel.REMOTE_WORKER
    assert instance.network_characteristics.profile is NetworkProfile.WAN


def test_backend_instance_old_payload_remains_compatible() -> None:
    instance = BackendInstance.model_validate(
        {
            "instance_id": "legacy-worker",
            "endpoint": {"base_url": "http://127.0.0.1:8101", "transport": "http"},
            "device_class": "cpu",
        }
    )

    assert instance.topology_schema_version is TopologySchemaVersion.V1
    assert instance.locality_class is WorkerLocalityClass.LOCAL_NETWORK
    assert instance.execution_mode is ExecutionModeLabel.HOST_NATIVE


def test_backend_health_rejects_invalid_error_rate() -> None:
    try:
        BackendHealth(state=BackendHealthState.DEGRADED, error_rate=1.5)
    except ValidationError as exc:
        assert "error_rate" in str(exc)
    else:
        raise AssertionError("BackendHealth should reject error_rate values above 1.0")


def test_backend_capabilities_support_logical_serving_targets() -> None:
    capabilities = BackendCapabilities(
        backend_type=BackendType.MLX_LM,
        engine_type=EngineType.MLX,
        device_class=DeviceClass.APPLE_GPU,
        model_ids=["mlx-chat", "mlx-community/Qwen"],
        serving_targets=["chat-default"],
        max_context_tokens=32768,
        model_aliases={"chat-default": "mlx-community/Qwen"},
        default_model="chat-default",
    )

    assert capabilities.supports_model_target("chat-default") is True
    assert capabilities.supports_model_target("mlx-community/Qwen") is True
    assert capabilities.supports_model_target("missing-model") is False


def test_backend_network_endpoint_rejects_relative_paths() -> None:
    try:
        BackendNetworkEndpoint(
            base_url="http://127.0.0.1:8101",
            health_path="healthz",
        )
    except ValidationError as exc:
        assert "health_path" in str(exc)
    else:
        raise AssertionError("BackendNetworkEndpoint should reject paths without a leading slash")
