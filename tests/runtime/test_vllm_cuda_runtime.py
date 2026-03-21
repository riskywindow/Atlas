from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import pytest

from switchyard.config import GenerationDefaults, LocalModelConfig, WarmupSettings
from switchyard.runtime.base import RuntimeSamplingParams
from switchyard.runtime.vllm_cuda import (
    LoadedEngine,
    VLLMCUDAChatRuntime,
    VLLMCUDADependencyError,
    VLLMCUDAProvider,
    VLLMCUDARuntimeCapabilities,
)
from switchyard.schemas.backend import (
    BackendType,
    EngineType,
    GPUDeviceMetadata,
    PerformanceHint,
    QualityHint,
    RequestFeatureSupport,
    RuntimeIdentity,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole


class FakeProvider:
    def __init__(self) -> None:
        self.loaded = False
        self.generated_requests: list[tuple[str, int | None]] = []
        self.load_kwargs: dict[str, object] = {}

    def ensure_available(self) -> None:
        return None

    def load(
        self,
        model_identifier: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
    ) -> LoadedEngine:
        self.loaded = True
        self.load_kwargs = {
            "model_identifier": model_identifier,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
        }
        return LoadedEngine(engine={"model_identifier": model_identifier})

    def declare_capabilities(self) -> VLLMCUDARuntimeCapabilities:
        return VLLMCUDARuntimeCapabilities(
            runtime=RuntimeIdentity(
                runtime_family="vllm_cuda",
                runtime_label="vllm_cuda",
                runtime_version="0.6.0",
                engine_type=EngineType.VLLM_CUDA,
                backend_type=BackendType.VLLM_CUDA,
            ),
            gpu=GPUDeviceMetadata(vendor="nvidia", model="L4", count=1, cuda_version="12.4"),
            request_features=RequestFeatureSupport(
                supports_streaming=True,
                supports_native_streaming=True,
                supports_system_messages=False,
            ),
            max_context_tokens=65536,
            concurrency_limit=12,
            quality_hint=QualityHint.PREMIUM,
            performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
            supports_prefix_cache=True,
            supports_kv_cache_reuse=True,
        )

    def generate_text(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> str:
        self.generated_requests.append(
            (messages[-1].content, params.max_output_tokens)
        )
        return f"cuda handled {messages[-1].content}"

    def stream_text(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> Iterator[str]:
        yield "cuda "
        yield "stream"


class MissingDependencyProvider(FakeProvider):
    def ensure_available(self) -> None:
        raise VLLMCUDADependencyError("missing vllm")

    def declare_capabilities(self) -> VLLMCUDARuntimeCapabilities:
        raise VLLMCUDADependencyError("missing vllm")


class FailingCudaProvider(FakeProvider):
    def load(
        self,
        model_identifier: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float | None = None,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
    ) -> LoadedEngine:
        raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")


def _build_model_config() -> LocalModelConfig:
    return LocalModelConfig(
        alias="chat-shared",
        serving_target="chat-shared",
        model_identifier="meta-llama/Llama-3.1-8B-Instruct",
        backend_type=BackendType.VLLM_CUDA,
        runtime=RuntimeIdentity(
            runtime_family="vllm_cuda",
            runtime_label="vllm_cuda",
            runtime_version="worker-1.2.3",
            engine_type=EngineType.VLLM_CUDA,
            backend_type=BackendType.VLLM_CUDA,
        ),
        gpu=GPUDeviceMetadata(vendor="nvidia", model="L40S", count=2, cuda_version="12.4"),
        generation_defaults=GenerationDefaults(max_output_tokens=64),
        warmup=WarmupSettings(enabled=True),
    )


def test_vllm_cuda_runtime_reports_configured_runtime_identity_and_gpu() -> None:
    runtime = VLLMCUDAChatRuntime(
        _build_model_config(),
        provider=cast("VLLMCUDAProvider", FakeProvider()),
    )

    capabilities = runtime.capabilities()
    health_before = runtime.health()

    assert capabilities.runtime.runtime_version == "worker-1.2.3"
    assert capabilities.gpu.model == "L40S"
    assert capabilities.request_features.supports_system_messages is False
    assert capabilities.concurrency_limit == 12
    assert health_before.load_state.value == "cold"


def test_vllm_cuda_runtime_generates_and_streams_without_cuda_dependency() -> None:
    provider = FakeProvider()
    runtime = VLLMCUDAChatRuntime(
        _build_model_config(),
        provider=cast("VLLMCUDAProvider", provider),
        tensor_parallel_size=2,
        gpu_memory_utilization=0.92,
        max_model_len=8192,
        trust_remote_code=True,
    )
    request = ChatCompletionRequest(
        model="chat-shared",
        messages=[ChatMessage(role=ChatRole.USER, content="hello runtime")],
    )

    result = runtime.generate(request)
    chunks = list(runtime.stream_generate(request))

    assert result.text == "cuda handled hello runtime"
    assert [chunk.text for chunk in chunks[:-1]] == ["cuda ", "stream"]
    assert chunks[-1].finish_reason is not None
    assert provider.load_kwargs == {
        "model_identifier": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 8192,
        "trust_remote_code": True,
    }


def test_vllm_cuda_runtime_health_falls_back_cleanly_when_dependency_missing() -> None:
    runtime = VLLMCUDAChatRuntime(
        _build_model_config(),
        provider=cast("VLLMCUDAProvider", MissingDependencyProvider()),
    )

    health = runtime.health()

    assert health.state.value == "unavailable"
    assert health.load_state.value == "failed"
    assert "unavailable" in (health.detail or "")


def test_vllm_cuda_runtime_surfaces_clear_cuda_startup_failures() -> None:
    runtime = VLLMCUDAChatRuntime(
        _build_model_config(),
        provider=cast("VLLMCUDAProvider", FailingCudaProvider()),
    )

    with pytest.raises(Exception, match="verify NVIDIA drivers, GPU visibility"):
        runtime.load_model()
