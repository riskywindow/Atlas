from __future__ import annotations

from collections.abc import Iterator
from typing import cast

from switchyard.adapters.factory import build_registry_from_settings
from switchyard.config import (
    BackendInstanceConfig,
    GenerationDefaults,
    LocalModelConfig,
    Settings,
    WarmupSettings,
)
from switchyard.runtime import RuntimeGenerationResult, RuntimeHealthSnapshot, RuntimeStreamChunk
from switchyard.runtime.base import ChatModelRuntime
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendType,
    WorkerTransportType,
)
from switchyard.schemas.chat import ChatCompletionRequest, FinishReason


class FakeRuntime:
    backend_type = BackendType.MLX_LM

    def load_model(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def health(self) -> RuntimeHealthSnapshot:
        return RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.COLD,
            detail="fake runtime available",
        )

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        return RuntimeGenerationResult(
            text=f"handled {request.model}",
            finish_reason=FinishReason.STOP,
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[RuntimeStreamChunk]:
        return iter(())


class FakeVLLMRuntime(FakeRuntime):
    backend_type = BackendType.VLLM_METAL


def test_build_registry_from_settings_registers_only_mlx_models() -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        )
    )

    registry = build_registry_from_settings(
        settings,
        mlx_runtime_factory=lambda _config: cast("ChatModelRuntime", FakeRuntime()),
    )

    assert registry.names() == ["mlx-lm:mlx-chat"]
    adapter = registry.get("mlx-lm:mlx-chat")
    assert adapter.backend_type is BackendType.MLX_LM


def test_build_registry_from_settings_registers_multiple_real_backends() -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                serving_target="chat-shared",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
            LocalModelConfig(
                alias="metal-chat",
                serving_target="chat-shared",
                model_identifier="NousResearch/Meta-Llama-3",
                backend_type=BackendType.VLLM_METAL,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        )
    )

    registry = build_registry_from_settings(
        settings,
        mlx_runtime_factory=lambda _config: cast("ChatModelRuntime", FakeRuntime()),
        vllm_runtime_factory=lambda _config: cast("ChatModelRuntime", FakeVLLMRuntime()),
    )

    assert registry.names() == ["mlx-lm:mlx-chat", "vllm-metal:metal-chat"]
    assert registry.get("vllm-metal:metal-chat").backend_type is BackendType.VLLM_METAL
    assert registry.serving_targets() == ["chat-shared"]
    assert registry.names_for_target("chat-shared") == ["mlx-lm:mlx-chat", "vllm-metal:metal-chat"]


def test_build_registry_from_settings_rejects_unsupported_local_backend_types() -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="remote-chat",
                model_identifier="https://example.invalid/v1",
                backend_type=BackendType.REMOTE_OPENAI_LIKE,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        )
    )

    try:
        build_registry_from_settings(settings)
    except ValueError as exc:
        assert "unsupported local backend_type" in str(exc)
        assert "remote-chat" in str(exc)
    else:
        raise AssertionError("unsupported local backend types should fail fast")


def test_build_registry_from_settings_uses_remote_worker_adapter_for_network_transport() -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="remote-chat",
                model_identifier="mock-chat",
                backend_type=BackendType.MOCK,
                worker_transport=WorkerTransportType.HTTP,
                instances=(
                    BackendInstanceConfig(
                        instance_id="worker-1",
                        base_url="http://worker.internal:8100",
                        transport=WorkerTransportType.HTTP,
                    ),
                ),
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(),
            ),
        )
    )

    registry = build_registry_from_settings(settings)

    adapter = registry.get("remote-worker:remote-chat")
    assert adapter.__class__.__name__ == "RemoteWorkerAdapter"
