from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from switchyard.runtime import RuntimeGenerationResult, RuntimeHealthSnapshot, RuntimeStreamChunk
from switchyard.runtime.vllm_cuda import VLLMCUDARuntimeCapabilities
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendType,
    EngineType,
    GPUDeviceMetadata,
    PerformanceHint,
    QualityHint,
    RequestFeatureSupport,
    RuntimeIdentity,
)
from switchyard.schemas.chat import ChatCompletionRequest, FinishReason
from switchyard.worker.config import RemoteWorkerRuntimeSettings
from switchyard.worker.vllm_cuda import create_vllm_cuda_worker_app


class FakeRuntime:
    backend_type = BackendType.VLLM_CUDA

    def health(self) -> RuntimeHealthSnapshot:
        return RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            detail="fake worker runtime ready",
        )

    def capabilities(self) -> VLLMCUDARuntimeCapabilities:
        return VLLMCUDARuntimeCapabilities(
            runtime=RuntimeIdentity(
                runtime_family="vllm_cuda",
                runtime_label="vllm_cuda",
                runtime_version="0.6.2",
                engine_type=EngineType.VLLM_CUDA,
                backend_type=BackendType.VLLM_CUDA,
            ),
            gpu=GPUDeviceMetadata(
                vendor="nvidia",
                model="L4",
                count=1,
                memory_per_device_gib=24.0,
                cuda_version="12.4",
            ),
            request_features=RequestFeatureSupport(
                supports_streaming=True,
                supports_native_streaming=True,
            ),
            max_context_tokens=32768,
            concurrency_limit=6,
            quality_hint=QualityHint.PREMIUM,
            performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
            supports_prefix_cache=True,
            supports_kv_cache_reuse=True,
        )

    def load_model(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        return RuntimeGenerationResult(
            text=f"worker handled {request.messages[-1].content}",
            finish_reason=FinishReason.STOP,
        )

    def stream_generate(self, request: ChatCompletionRequest) -> Iterator[RuntimeStreamChunk]:
        yield RuntimeStreamChunk(text="worker ")
        yield RuntimeStreamChunk(text="stream")
        yield RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP)


def _settings() -> RemoteWorkerRuntimeSettings:
    return RemoteWorkerRuntimeSettings(
        worker_name="cuda-worker-1",
        serving_target="chat-shared",
        model_identifier="meta-llama/Llama-3.1-8B-Instruct",
        backend_type=BackendType.VLLM_CUDA,
    )


@pytest.mark.asyncio
async def test_vllm_cuda_worker_app_serves_capabilities_and_protocol_metadata() -> None:
    app = create_vllm_cuda_worker_app(_settings(), runtime=FakeRuntime())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    try:
        capabilities = await client.get("/internal/worker/capabilities")
        readiness = await client.get(
            "/internal/worker/ready",
            headers={"x-request-id": "req-worker-ready", "x-trace-id": "trace-123"},
        )
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "hello worker"}],
            },
            headers={"x-request-id": "req-worker-generate"},
        )
    finally:
        await client.aclose()

    assert capabilities.status_code == 200
    assert capabilities.json()["runtime"]["runtime_label"] == "vllm_cuda"
    assert capabilities.json()["capabilities"]["gpu"]["model"] == "L4"
    assert readiness.status_code == 200
    assert readiness.json()["ready"] is True
    assert readiness.json()["transport_metadata"]["request_id"] == "req-worker-ready"
    assert readiness.json()["transport_metadata"]["trace_id"] == "trace-123"
    assert response.status_code == 200
    assert response.json()["backend_name"] == "vllm-cuda:chat-shared"


@pytest.mark.asyncio
async def test_vllm_cuda_worker_app_rejects_new_requests_while_draining() -> None:
    app = create_vllm_cuda_worker_app(_settings(), runtime=FakeRuntime())
    await app.state.worker.begin_drain(reason="rolling restart")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    try:
        readiness = await client.get("/internal/worker/ready")
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "hello worker"}],
            },
        )
    finally:
        await client.aclose()

    assert readiness.status_code == 200
    assert readiness.json()["ready"] is False
    assert "draining" in readiness.json()["health"]["detail"]
    assert response.status_code == 503
    assert response.json()["detail"] == "rolling restart"
