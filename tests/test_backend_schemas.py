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
    DeploymentProfile,
    DeviceClass,
    EngineType,
    LogicalModelTarget,
    PerformanceHint,
    QualityHint,
    WorkerRegistrationState,
    WorkerTransportType,
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
    assert payload["capabilities"]["engine_type"] == "mock"
    assert payload["capabilities"]["serving_targets"] == ["chat-default"]
    assert payload["health"]["state"] == "healthy"
    assert payload["health"]["load_state"] == "ready"
    assert payload["health"]["circuit_open"] is False
    assert payload["deployment"]["configured_priority"] == 50
    assert payload["deployment"]["deployment_profile"] == "compose"
    assert payload["deployment"]["environment"] == "dev"
    assert payload["deployment"]["instances"][0]["endpoint"]["base_url"] == "http://127.0.0.1:8101"
    assert payload["deployment"]["instances"][0]["endpoint"]["transport"] == "http"
    assert payload["deployment"]["instances"][0]["source_of_truth"] == "registered"
    assert payload["deployment"]["instances"][0]["tags"] == ["local", "canary"]
    assert payload["deployment"]["instances"][0]["registration"]["state"] == "registered"
    assert payload["deployment"]["instances"][0]["health"]["state"] == "healthy"
    assert payload["instance_inventory"][0]["instance_id"] == "mock-a-1"
    assert payload["logical_targets"][0]["alias"] == "chat-default"


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
