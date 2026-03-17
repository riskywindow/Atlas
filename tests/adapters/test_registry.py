from typing import Any, cast

import pytest

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import BackendInstanceConfig, LocalModelConfig
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendInstanceSource,
    BackendType,
    DeviceClass,
)


def test_registry_registers_and_lists_adapters() -> None:
    registry = AdapterRegistry()
    first = MockBackendAdapter(name="mock-a")
    second = MockBackendAdapter(name="mock-b")

    registry.register(first)
    registry.register(second)

    assert registry.get("mock-a") is first
    assert registry.names() == ["mock-a", "mock-b"]
    assert registry.list() == [first, second]


def test_registry_rejects_duplicate_names() -> None:
    registry = AdapterRegistry()
    registry.register(MockBackendAdapter(name="mock-a"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(MockBackendAdapter(name="mock-a"))


def test_registry_raises_for_missing_adapter() -> None:
    registry = AdapterRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")


def test_registry_resolves_multiple_backends_for_one_logical_target() -> None:
    registry = AdapterRegistry()
    mlx_adapter = MockBackendAdapter(
        name="mlx-lm:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MLX_LM,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["mlx-lm:chat-shared", "mlx-community/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=False,
            configured_priority=10,
        ),
    )
    vllm_adapter = MockBackendAdapter(
        name="vllm-metal:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.VLLM_METAL,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["vllm-metal:chat-shared", "NousResearch/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=True,
            configured_priority=20,
        ),
    )

    registry.register(mlx_adapter)
    registry.register(vllm_adapter)

    assert registry.serving_targets() == ["chat-shared"]
    assert registry.names_for_target("chat-shared") == [
        "mlx-lm:chat-shared",
        "vllm-metal:chat-shared",
    ]
    assert registry.get_for_target("chat-shared") == [mlx_adapter, vllm_adapter]


@pytest.mark.asyncio
async def test_registry_target_snapshot_answers_health_streaming_warm_and_preference_questions(
) -> None:
    registry = AdapterRegistry()
    cold_adapter = MockBackendAdapter(
        name="mlx-lm:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MLX_LM,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["mlx-lm:chat-shared", "mlx-community/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=False,
            configured_priority=5,
            configured_weight=3.0,
        ),
    )
    warm_streaming_adapter = MockBackendAdapter(
        name="vllm-metal:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.VLLM_METAL,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["vllm-metal:chat-shared", "NousResearch/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=True,
            configured_priority=10,
            configured_weight=1.0,
        ),
    )
    unavailable_adapter = MockBackendAdapter(
        name="mock:chat-shared-remote",
        health_state=BackendHealthState.UNAVAILABLE,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.REMOTE,
            model_ids=["mock:chat-shared-remote", "remote/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=True,
            configured_priority=100,
        ),
    )

    registry.register(cold_adapter)
    registry.register(warm_streaming_adapter)
    registry.register(unavailable_adapter)
    await warm_streaming_adapter.warmup("chat-shared")

    snapshot = await registry.snapshots_for_target("chat-shared")

    assert [deployment.name for deployment in snapshot.deployments] == [
        "mlx-lm:chat-shared",
        "vllm-metal:chat-shared",
        "mock:chat-shared-remote",
    ]
    assert [deployment.name for deployment in snapshot.healthy] == [
        "mlx-lm:chat-shared",
        "vllm-metal:chat-shared",
    ]
    assert [deployment.name for deployment in snapshot.supports_streaming] == [
        "vllm-metal:chat-shared",
        "mock:chat-shared-remote",
    ]
    assert [deployment.name for deployment in snapshot.warm] == ["vllm-metal:chat-shared"]
    assert [deployment.name for deployment in snapshot.preferred] == [
        "mlx-lm:chat-shared",
        "vllm-metal:chat-shared",
        "mock:chat-shared-remote",
    ]


def test_registry_allows_internal_backend_pinning_for_target_resolution() -> None:
    registry = AdapterRegistry()
    first = MockBackendAdapter(
        name="mlx-lm:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MLX_LM,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["mlx-lm:chat-shared", "mlx-community/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
        ),
    )
    second = MockBackendAdapter(
        name="vllm-metal:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.VLLM_METAL,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["vllm-metal:chat-shared", "NousResearch/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
        ),
    )

    registry.register(first)
    registry.register(second)

    assert registry.get_for_target(
        "chat-shared",
        pinned_backend_name="vllm-metal:chat-shared",
    ) == [second]

    with pytest.raises(KeyError, match="not registered"):
        registry.get_for_target("chat-shared", pinned_backend_name="missing-backend")

    other = MockBackendAdapter(
        name="mlx-lm:other-target",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MLX_LM,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["mlx-lm:other-target", "mlx-community/other-model"],
            serving_targets=["other-target"],
            max_context_tokens=8192,
        ),
    )
    registry.register(other)

    with pytest.raises(KeyError, match="not registered for serving target"):
        registry.get_for_target("chat-shared", pinned_backend_name="mlx-lm:other-target")


@pytest.mark.asyncio
async def test_registry_hydrates_instance_inventory_from_model_config() -> None:
    registry = AdapterRegistry()
    adapter = MockBackendAdapter(
        name="mlx-lm:chat-shared",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MLX_LM,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=["mlx-lm:chat-shared", "mlx-community/test-model"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
        ),
    )
    cast(Any, adapter).model_config = LocalModelConfig(
        alias="mlx-chat",
        serving_target="chat-shared",
        model_identifier="mlx-community/test-model",
        backend_type=BackendType.MLX_LM,
        instances=(
            BackendInstanceConfig(
                instance_id="mlx-local-1",
                base_url="http://127.0.0.1:9001",
                locality="compose",
                tags=("local", "experimental"),
            ),
        ),
    )

    registry.register(adapter)

    snapshot = await registry.snapshots_for_target("chat-shared")

    assert snapshot.deployments[0].instance_inventory[0].instance_id == "mlx-local-1"
    assert (
        snapshot.deployments[0].instance_inventory[0].source_of_truth
        is BackendInstanceSource.STATIC_CONFIG
    )
    assert snapshot.deployments[0].instance_inventory[0].tags == ["local", "experimental"]
    assert (
        snapshot.deployments[0].instance_inventory[0].endpoint.chat_completions_path
        == "/internal/worker/generate"
    )
    assert snapshot.deployments[0].deployment is not None
    assert snapshot.deployments[0].deployment.instances[0].locality == "compose"


@pytest.mark.asyncio
async def test_registry_flattens_and_filters_instances_for_target() -> None:
    registry = AdapterRegistry()
    first = MockBackendAdapter(
        name="mock-a",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-a"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
        ),
    )
    second = MockBackendAdapter(
        name="mock-b",
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-b"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
        ),
        health_state=BackendHealthState.UNAVAILABLE,
    )
    cast(Any, first).model_config = LocalModelConfig(
        alias="mock-a",
        serving_target="chat-shared",
        model_identifier="mock-a",
        backend_type=BackendType.MOCK,
        instances=(
            BackendInstanceConfig(
                instance_id="mock-a-1",
                base_url="http://127.0.0.1:8101",
            ),
        ),
    )
    cast(Any, second).model_config = LocalModelConfig(
        alias="mock-b",
        serving_target="chat-shared",
        model_identifier="mock-b",
        backend_type=BackendType.MOCK,
        instances=(
            BackendInstanceConfig(
                instance_id="mock-b-1",
                base_url="http://127.0.0.1:8102",
            ),
        ),
    )

    registry.register(first)
    registry.register(second)

    snapshot = await registry.snapshots_for_target("chat-shared")
    inventory = await registry.instance_inventory_for_target("chat-shared")

    assert [instance.instance_id for instance in snapshot.instance_inventory] == [
        "mock-a-1",
        "mock-b-1",
    ]
    assert [instance.instance_id for instance in snapshot.preferred_instances] == [
        "mock-a-1",
        "mock-b-1",
    ]
    assert [instance.instance_id for instance in snapshot.healthy_instances] == []
    assert [instance.instance_id for instance in inventory] == ["mock-a-1", "mock-b-1"]
