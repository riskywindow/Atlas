from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from switchyard.adapters.vllm_cuda import VLLMCUDAAdapter
from switchyard.config import GenerationDefaults, LocalModelConfig, WarmupSettings
from switchyard.runtime import RuntimeGenerationResult, RuntimeHealthSnapshot, RuntimeStreamChunk
from switchyard.runtime.base import UnsupportedRequestError
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendType,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
    GPUDeviceMetadata,
    PerformanceHint,
    QualityHint,
    RequestFeatureSupport,
    RuntimeIdentity,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason
from switchyard.schemas.routing import RequestContext


class FakeRuntime:
    backend_type = BackendType.VLLM_CUDA

    def __init__(self, *, supports_system_messages: bool = True) -> None:
        self.health_snapshot = RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            detail="fake vllm cuda runtime ready",
        )
        self.generate_result = RuntimeGenerationResult(
            text="vllm cuda says hello",
            finish_reason=FinishReason.STOP,
            prompt_tokens=6,
            completion_tokens=4,
        )
        self.stream_chunks = [
            RuntimeStreamChunk(text="vllm "),
            RuntimeStreamChunk(text="cuda"),
            RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP),
        ]
        self.warmup_calls = 0
        self.generate_requests: list[ChatCompletionRequest] = []
        self.stream_requests: list[ChatCompletionRequest] = []
        self._supports_system_messages = supports_system_messages

    def load_model(self) -> None:
        return None

    def warmup(self) -> None:
        self.warmup_calls += 1

    def health(self) -> RuntimeHealthSnapshot:
        return self.health_snapshot

    def capabilities(self) -> object:
        return type(
            "Capabilities",
            (),
            {
                "runtime": RuntimeIdentity(
                    runtime_family="vllm_cuda",
                    runtime_label="vllm_cuda",
                    runtime_version="0.6.1",
                    engine_type=EngineType.VLLM_CUDA,
                    backend_type=BackendType.VLLM_CUDA,
                ),
                "gpu": GPUDeviceMetadata(
                    vendor="nvidia",
                    model="L4",
                    count=1,
                    memory_per_device_gib=24.0,
                    cuda_version="12.4",
                ),
                "request_features": RequestFeatureSupport(
                    supports_streaming=True,
                    supports_native_streaming=True,
                    supports_system_messages=self._supports_system_messages,
                ),
                "max_context_tokens": 65536,
                "concurrency_limit": 16,
                "quality_hint": QualityHint.PREMIUM,
                "performance_hint": PerformanceHint.THROUGHPUT_OPTIMIZED,
                "supports_prefix_cache": True,
                "supports_kv_cache_reuse": True,
            },
        )()

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
        alias="cuda-chat",
        serving_target="chat-shared",
        model_identifier="meta-llama/Llama-3.1-8B-Instruct",
        backend_type=BackendType.VLLM_CUDA,
        execution_mode=ExecutionModeLabel.REMOTE_WORKER,
        generation_defaults=GenerationDefaults(max_output_tokens=128),
        warmup=WarmupSettings(enabled=True),
    )


def build_request(*, with_system_message: bool = False) -> ChatCompletionRequest:
    messages = [ChatMessage(role=ChatRole.USER, content="hello adapter")]
    if with_system_message:
        messages.insert(0, ChatMessage(role=ChatRole.SYSTEM, content="system policy"))
    return ChatCompletionRequest(model="chat-shared", messages=messages)


@pytest.mark.asyncio
async def test_vllm_cuda_adapter_exposes_remote_capabilities_and_status() -> None:
    runtime = FakeRuntime()
    adapter = VLLMCUDAAdapter(build_model_config(), runtime=cast("Any", runtime))

    capabilities = await adapter.capabilities()
    health = await adapter.health()
    status = await adapter.status()

    assert adapter.name == "vllm-cuda:cuda-chat"
    assert adapter.backend_type is BackendType.VLLM_CUDA
    assert capabilities.device_class is DeviceClass.NVIDIA_GPU
    assert capabilities.engine_type is EngineType.VLLM_CUDA
    assert capabilities.runtime is not None
    assert capabilities.gpu is not None
    assert capabilities.runtime.runtime_version == "0.6.1"
    assert capabilities.gpu.model == "L4"
    assert capabilities.request_features.supports_native_streaming is True
    assert capabilities.serving_targets == ["chat-shared"]
    assert health.state is BackendHealthState.HEALTHY
    assert status.deployment is not None
    assert status.deployment.engine_type is EngineType.VLLM_CUDA
    assert status.deployment.request_features.supports_streaming is True


@pytest.mark.asyncio
async def test_vllm_cuda_adapter_warmup_generate_and_stream_translate_runtime_output() -> None:
    runtime = FakeRuntime()
    adapter = VLLMCUDAAdapter(build_model_config(), runtime=cast("Any", runtime))
    request = build_request()
    context = RequestContext(request_id="req-vllm-cuda")

    await adapter.warmup()
    response = await adapter.generate(request, context)
    chunks = [chunk async for chunk in adapter.stream_generate(request, context)]

    assert runtime.warmup_calls == 1
    assert runtime.generate_requests == [request]
    assert runtime.stream_requests == [request]
    assert response.backend_name == "vllm-cuda:cuda-chat"
    assert response.choices[0].message.content == "vllm cuda says hello"
    assert chunks[0].choices[0].delta.role is ChatRole.ASSISTANT
    assert [chunk.choices[0].delta.content for chunk in chunks[1:]] == ["vllm ", "cuda", ""]
    assert chunks[-1].choices[0].finish_reason is FinishReason.STOP


@pytest.mark.asyncio
async def test_vllm_cuda_adapter_rejects_unsupported_system_messages() -> None:
    runtime = FakeRuntime(supports_system_messages=False)
    adapter = VLLMCUDAAdapter(build_model_config(), runtime=cast("Any", runtime))

    with pytest.raises(UnsupportedRequestError, match="system messages"):
        await adapter.generate(
            build_request(with_system_message=True),
            RequestContext(request_id="req-unsupported"),
        )
