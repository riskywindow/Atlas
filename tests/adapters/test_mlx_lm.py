from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

from switchyard.adapters.mlx_lm import MLXLMAdapter
from switchyard.config import GenerationDefaults, LocalModelConfig, WarmupSettings
from switchyard.runtime import RuntimeGenerationResult, RuntimeHealthSnapshot, RuntimeStreamChunk
from switchyard.runtime.base import ChatModelRuntime
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendType,
    DeviceClass,
    EngineType,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason
from switchyard.schemas.routing import RequestContext


class FakeRuntime:
    backend_type = BackendType.MLX_LM

    def __init__(self) -> None:
        self.health_snapshot = RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            detail="fake mlx runtime ready",
        )
        self.generate_result = RuntimeGenerationResult(
            text="mlx says hello",
            finish_reason=FinishReason.STOP,
            prompt_tokens=4,
            completion_tokens=3,
        )
        self.stream_chunks = [
            RuntimeStreamChunk(text="mlx "),
            RuntimeStreamChunk(text="stream"),
            RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP),
        ]
        self.load_calls = 0
        self.warmup_calls = 0
        self.generate_requests: list[ChatCompletionRequest] = []
        self.stream_requests: list[ChatCompletionRequest] = []

    def load_model(self) -> None:
        self.load_calls += 1

    def warmup(self) -> None:
        self.warmup_calls += 1

    def health(self) -> RuntimeHealthSnapshot:
        return self.health_snapshot

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        self.generate_requests.append(request)
        return self.generate_result

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[RuntimeStreamChunk]:
        self.stream_requests.append(request)
        return iter(self.stream_chunks)


def build_model_config() -> LocalModelConfig:
    return LocalModelConfig(
        alias="mlx-chat",
        model_identifier="mlx-community/Qwen",
        backend_type=BackendType.MLX_LM,
        generation_defaults=GenerationDefaults(max_output_tokens=128),
        warmup=WarmupSettings(enabled=True),
    )


def build_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mlx-chat",
        messages=[ChatMessage(role=ChatRole.USER, content="hello adapter")],
    )


@pytest.mark.asyncio
async def test_mlx_adapter_exposes_honest_capabilities_and_health() -> None:
    runtime = FakeRuntime()
    adapter = MLXLMAdapter(
        build_model_config(),
        runtime=cast("ChatModelRuntime", runtime),
    )

    capabilities = await adapter.capabilities()
    health = await adapter.health()
    status = await adapter.status()

    assert adapter.name == "mlx-lm:mlx-chat"
    assert adapter.backend_type is BackendType.MLX_LM
    assert capabilities.device_class is DeviceClass.APPLE_GPU
    assert capabilities.engine_type is EngineType.MLX
    assert capabilities.serving_targets == ["mlx-chat"]
    assert capabilities.model_ids == ["mlx-chat", "mlx-community/Qwen"]
    assert capabilities.model_aliases == {"mlx-chat": "mlx-community/Qwen"}
    assert capabilities.default_model == "mlx-chat"
    assert health.state is BackendHealthState.HEALTHY
    assert health.load_state is BackendLoadState.READY
    assert health.warmed_models == ["mlx-chat"]
    assert status.deployment is not None
    assert status.deployment.engine_type is EngineType.MLX
    assert status.metadata["model_alias"] == "mlx-chat"


@pytest.mark.asyncio
async def test_mlx_adapter_warmup_and_generate_translate_runtime_output() -> None:
    runtime = FakeRuntime()
    adapter = MLXLMAdapter(
        build_model_config(),
        runtime=cast("ChatModelRuntime", runtime),
    )
    request = build_request()
    context = RequestContext(request_id="req-mlx")

    await adapter.warmup()
    response = await adapter.generate(request, context)

    assert runtime.warmup_calls == 1
    assert runtime.generate_requests == [request]
    assert response.backend_name == "mlx-lm:mlx-chat"
    assert response.model == "mlx-chat"
    assert response.choices[0].message.content == "mlx says hello"
    assert response.usage.prompt_tokens == 4
    assert response.usage.completion_tokens == 3


@pytest.mark.asyncio
async def test_mlx_adapter_stream_generate_emits_chunks() -> None:
    runtime = FakeRuntime()
    adapter = MLXLMAdapter(
        build_model_config(),
        runtime=cast("ChatModelRuntime", runtime),
    )

    chunks = [
        chunk
        async for chunk in adapter.stream_generate(
            build_request(),
            RequestContext(request_id="req-stream"),
        )
    ]

    assert chunks[0].choices[0].delta.role is ChatRole.ASSISTANT
    assert [chunk.choices[0].delta.content for chunk in chunks[1:]] == [
        "mlx ",
        "stream",
        "",
    ]
    assert chunks[-1].choices[0].finish_reason is FinishReason.STOP
