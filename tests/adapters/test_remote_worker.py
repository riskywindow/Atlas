from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from switchyard.adapters.remote_worker import (
    RemoteWorkerAdapter,
    RemoteWorkerClient,
    RemoteWorkerErrorKind,
    RemoteWorkerResponseError,
    RemoteWorkerTransportError,
)
from switchyard.config import (
    BackendInstanceConfig,
    GenerationDefaults,
    LocalModelConfig,
    WarmupSettings,
)
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendType,
    CapacitySnapshot,
    DeviceClass,
    EngineType,
    GPUDeviceMetadata,
    RequestFeatureSupport,
    RuntimeIdentity,
    WorkerTransportType,
)
from switchyard.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    FinishReason,
    UsageStats,
)
from switchyard.schemas.routing import RequestContext
from switchyard.schemas.worker import (
    WorkerCapabilitiesResponse,
    WorkerGenerateResponse,
    WorkerHealthResponse,
    WorkerReadinessResponse,
    WorkerRequestMetadata,
    WorkerStreamChunkResponse,
    WorkerWarmupResponse,
)
from switchyard.worker.fake import FakeRemoteWorkerConfig, create_fake_remote_worker_app


def _build_model_config() -> LocalModelConfig:
    return LocalModelConfig(
        alias="remote-chat",
        serving_target="chat-shared",
        model_identifier="mock-chat",
        backend_type=BackendType.MOCK,
        worker_transport=WorkerTransportType.HTTP,
        instances=(
            BackendInstanceConfig(
                instance_id="worker-1",
                base_url="http://worker.internal",
                transport=WorkerTransportType.HTTP,
                request_timeout_seconds=5.0,
                connect_timeout_seconds=1.0,
            ),
        ),
        generation_defaults=GenerationDefaults(),
        warmup=WarmupSettings(enabled=True),
    )


def _build_multi_instance_model_config() -> LocalModelConfig:
    return LocalModelConfig(
        alias="remote-chat",
        serving_target="chat-shared",
        model_identifier="mock-chat",
        backend_type=BackendType.MOCK,
        worker_transport=WorkerTransportType.HTTP,
        instances=(
            BackendInstanceConfig(
                instance_id="worker-1",
                base_url="http://worker-1.internal",
                transport=WorkerTransportType.HTTP,
                tags=("local", "canary"),
            ),
            BackendInstanceConfig(
                instance_id="worker-2",
                base_url="http://worker-2.internal",
                transport=WorkerTransportType.HTTP,
                tags=("local",),
            ),
        ),
        generation_defaults=GenerationDefaults(),
        warmup=WarmupSettings(enabled=True),
    )


def _build_request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="chat-shared",
        stream=stream,
        messages=[ChatMessage(role=ChatRole.USER, content="hello remote worker")],
    )


def _build_context() -> RequestContext:
    return RequestContext(request_id="req-remote")


def _build_worker_app() -> FastAPI:
    app = FastAPI()

    health = BackendHealth(
        state=BackendHealthState.HEALTHY,
        load_state=BackendLoadState.READY,
        warmed_models=["chat-shared"],
        latency_ms=2.0,
    )
    capabilities = BackendCapabilities(
        backend_type=BackendType.MOCK,
        engine_type=EngineType.MOCK,
        device_class=DeviceClass.REMOTE,
        runtime=RuntimeIdentity(runtime_family="mock", runtime_label="fake-remote"),
        gpu=GPUDeviceMetadata(
            vendor="nvidia",
            model="L4",
            count=1,
            memory_per_device_gib=24.0,
        ),
        model_ids=["chat-shared", "mock-chat"],
        serving_targets=["chat-shared"],
        max_context_tokens=8192,
        supports_streaming=True,
        concurrency_limit=4,
        request_features=RequestFeatureSupport(supports_streaming=True),
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return WorkerHealthResponse(worker_name="worker-1", health=health).model_dump(mode="json")

    @app.get("/internal/worker/ready")
    async def ready() -> dict[str, object]:
        return WorkerReadinessResponse(
            worker_name="worker-1",
            runtime=RuntimeIdentity(runtime_family="mock", runtime_label="fake-remote"),
            gpu=GPUDeviceMetadata(
                vendor="nvidia",
                model="L4",
                count=1,
                memory_per_device_gib=24.0,
            ),
            ready=True,
            active_requests=1,
            queue_depth=0,
            observed_capacity=CapacitySnapshot(
                concurrency_limit=4,
                active_requests=1,
                queue_depth=0,
                tokens_per_second=88.0,
            ),
            health=health,
        ).model_dump(mode="json")

    @app.get("/internal/worker/capabilities")
    async def worker_capabilities() -> dict[str, object]:
        return WorkerCapabilitiesResponse(
            worker_name="worker-1",
            backend_type=BackendType.MOCK,
            capabilities=capabilities,
        ).model_dump(mode="json")

    @app.post("/internal/worker/warmup")
    async def warmup() -> dict[str, object]:
        return WorkerWarmupResponse(worker_name="worker-1", health=health).model_dump(mode="json")

    @app.post("/internal/worker/generate")
    async def generate() -> dict[str, object]:
        response = ChatCompletionResponse(
            id="remote-response-1",
            created_at=datetime(2026, 3, 17, tzinfo=UTC),
            model="chat-shared",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=ChatRole.ASSISTANT, content="hello from remote"),
                    finish_reason=FinishReason.STOP,
                )
            ],
            usage=UsageStats(prompt_tokens=3, completion_tokens=3, total_tokens=6),
            backend_name="worker-1",
        )
        return WorkerGenerateResponse(
            worker_name="worker-1",
            response=response,
        ).model_dump(mode="json")

    @app.post("/internal/worker/generate/stream")
    async def stream() -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            chunks = [
                ChatCompletionChunk(
                    id="remote-stream-1",
                    created_at=datetime(2026, 3, 17, tzinfo=UTC),
                    model="chat-shared",
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role=ChatRole.ASSISTANT),
                        )
                    ],
                    backend_name="worker-1",
                ),
                ChatCompletionChunk(
                    id="remote-stream-1",
                    created_at=datetime(2026, 3, 17, tzinfo=UTC),
                    model="chat-shared",
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content="hello "),
                        )
                    ],
                    backend_name="worker-1",
                ),
                ChatCompletionChunk(
                    id="remote-stream-1",
                    created_at=datetime(2026, 3, 17, tzinfo=UTC),
                    model="chat-shared",
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content="stream"),
                            finish_reason=FinishReason.STOP,
                        )
                    ],
                    backend_name="worker-1",
                ),
            ]
            for chunk in chunks:
                payload = WorkerStreamChunkResponse(
                    worker_name="worker-1",
                    chunk=chunk,
                ).model_dump_json()
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


@pytest.mark.asyncio
async def test_remote_worker_client_round_trips_with_fake_worker_app() -> None:
    app = create_fake_remote_worker_app(
        FakeRemoteWorkerConfig(
            worker_name="fake-remote",
            simulated_active_requests=2,
            simulated_queue_depth=1,
        )
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://worker.internal",
    )
    remote_client = RemoteWorkerClient(client=client)
    instance = _build_model_config().instances[0]
    metadata = WorkerRequestMetadata(
        request_id="req-transport-001",
        trace_id="trace-transport-001",
        timeout_ms=5000,
    )

    try:
        health = await remote_client.health(instance, metadata=metadata)
        readiness = await remote_client.readiness(instance, metadata=metadata)
        capabilities = await remote_client.capabilities(instance, metadata=metadata)
        response = await remote_client.generate(
            instance,
            request=_build_request(),
            context=_build_context(),
            metadata=metadata,
        )
        chunks = [
            item
            async for item in remote_client.stream_generate(
                instance,
                request=_build_request(stream=True),
                context=_build_context(),
                metadata=metadata,
            )
        ]
    finally:
        await client.aclose()

    assert health.transport_metadata is not None
    assert health.transport_metadata.request_id == "req-transport-001"
    assert readiness.ready is False
    assert readiness.transport_metadata is not None
    assert capabilities.execution_mode is not None
    assert response.transport_metadata is not None
    assert response.transport_metadata.trace_id == "trace-transport-001"
    assert chunks[0].transport_metadata is not None
    assert chunks[0].transport_metadata.request_id == "req-transport-001"


@pytest.mark.asyncio
async def test_remote_worker_client_classifies_timeout_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://worker.internal",
    )
    remote_client = RemoteWorkerClient(client=client)
    instance = _build_model_config().instances[0]

    try:
        with pytest.raises(RemoteWorkerTransportError) as exc_info:
            await remote_client.generate(
                instance,
                request=_build_request(),
                context=_build_context(),
                metadata=WorkerRequestMetadata(request_id="req-timeout"),
            )
    finally:
        await client.aclose()

    assert exc_info.value.kind is RemoteWorkerErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_remote_worker_adapter_round_trips_over_http() -> None:
    app = _build_worker_app()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://worker.internal",
    )
    adapter = RemoteWorkerAdapter(_build_model_config(), client=client)

    try:
        health = await adapter.health()
        capabilities = await adapter.capabilities()
        status = await adapter.status()
        await adapter.warmup()
        response = await adapter.generate(_build_request(), _build_context())
        chunks = [
            chunk
            async for chunk in adapter.stream_generate(
                _build_request(stream=True),
                _build_context(),
            )
        ]
    finally:
        await client.aclose()

    assert health.state is BackendHealthState.HEALTHY
    assert capabilities.supports_streaming is True
    assert capabilities.runtime is not None
    assert capabilities.runtime.runtime_label == "fake-remote"
    assert capabilities.gpu is not None
    assert capabilities.gpu.vendor == "nvidia"
    assert status.active_requests == 1
    assert status.metadata["execution_mode"] == "remote_worker"
    assert status.deployment is not None
    assert status.deployment.request_features.supports_streaming is True
    assert status.instance_inventory[0].runtime is not None
    assert status.instance_inventory[0].runtime.runtime_label == "fake-remote"
    assert status.instance_inventory[0].observed_capacity is not None
    assert status.instance_inventory[0].observed_capacity.tokens_per_second == 88.0
    assert response.backend_name == "remote-worker:remote-chat"
    assert response.choices[0].message.content == "hello from remote"
    assert chunks[0].backend_name == "remote-worker:remote-chat"
    assert chunks[-1].choices[0].finish_reason is FinishReason.STOP


@pytest.mark.asyncio
async def test_remote_worker_adapter_raises_explicit_transport_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://worker.internal",
    )
    adapter = RemoteWorkerAdapter(_build_model_config(), client=client)

    try:
        with pytest.raises(RemoteWorkerTransportError, match="connection failed"):
            await adapter.generate(_build_request(), _build_context())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_remote_worker_adapter_selects_healthy_instance_and_reports_inventory() -> None:
    health_unavailable = BackendHealth(
        state=BackendHealthState.UNAVAILABLE,
        load_state=BackendLoadState.FAILED,
        detail="instance unavailable",
    )
    health_ready = BackendHealth(
        state=BackendHealthState.HEALTHY,
        load_state=BackendLoadState.READY,
        warmed_models=["chat-shared"],
        latency_ms=3.0,
    )
    capabilities = BackendCapabilities(
        backend_type=BackendType.MOCK,
        engine_type=EngineType.MOCK,
        device_class=DeviceClass.REMOTE,
        model_ids=["chat-shared", "mock-chat"],
        serving_targets=["chat-shared"],
        max_context_tokens=8192,
        supports_streaming=False,
        concurrency_limit=4,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "worker-1.internal":
            if request.url.path == "/healthz":
                payload = WorkerHealthResponse(
                    worker_name="worker-1",
                    health=health_unavailable,
                ).model_dump(mode="json")
                return httpx.Response(200, json=payload)
            if request.url.path == "/internal/worker/ready":
                return httpx.Response(503, json={"detail": "unavailable"})
            if request.url.path == "/internal/worker/capabilities":
                payload = WorkerCapabilitiesResponse(
                    worker_name="worker-1",
                    backend_type=BackendType.MOCK,
                    capabilities=capabilities,
                ).model_dump(mode="json")
                return httpx.Response(200, json=payload)
        if request.url.host == "worker-2.internal":
            if request.url.path == "/healthz":
                payload = WorkerHealthResponse(
                    worker_name="worker-2",
                    health=health_ready,
                ).model_dump(mode="json")
                return httpx.Response(200, json=payload)
            if request.url.path == "/internal/worker/ready":
                payload = WorkerReadinessResponse(
                    worker_name="worker-2",
                    ready=True,
                    active_requests=1,
                    queue_depth=0,
                    health=health_ready,
                ).model_dump(mode="json")
                return httpx.Response(200, json=payload)
            if request.url.path == "/internal/worker/capabilities":
                payload = WorkerCapabilitiesResponse(
                    worker_name="worker-2",
                    backend_type=BackendType.MOCK,
                    capabilities=capabilities,
                ).model_dump(mode="json")
                return httpx.Response(200, json=payload)
            if request.url.path == "/internal/worker/generate":
                payload = WorkerGenerateResponse(
                    worker_name="worker-2",
                    response=ChatCompletionResponse(
                        id="remote-response-2",
                        created_at=datetime(2026, 3, 17, tzinfo=UTC),
                        model="chat-shared",
                        choices=[
                            ChatCompletionChoice(
                                index=0,
                                message=ChatMessage(
                                    role=ChatRole.ASSISTANT,
                                    content="hello from healthy worker",
                                ),
                                finish_reason=FinishReason.STOP,
                            )
                        ],
                        usage=UsageStats(prompt_tokens=3, completion_tokens=4, total_tokens=7),
                        backend_name="worker-2",
                    ),
                ).model_dump(mode="json")
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"detail": "unexpected"})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://worker-1.internal",
    )
    adapter = RemoteWorkerAdapter(_build_multi_instance_model_config(), client=client)

    try:
        status = await adapter.status()
        response = await adapter.generate(_build_request(), _build_context())
    finally:
        await client.aclose()

    assert status.metadata["preferred_instance_id"] == "worker-2"
    assert [instance.instance_id for instance in status.instance_inventory] == [
        "worker-1",
        "worker-2",
    ]
    assert status.instance_inventory[0].health is not None
    assert status.instance_inventory[0].health.state is BackendHealthState.UNAVAILABLE
    assert status.instance_inventory[1].health is not None
    assert status.instance_inventory[1].health.state is BackendHealthState.HEALTHY
    assert status.instance_inventory[1].tags == ["local"]
    assert response.choices[0].message.content == "hello from healthy worker"


@pytest.mark.asyncio
async def test_remote_worker_adapter_raises_explicit_malformed_response_error() -> None:
    app = FastAPI()

    @app.post("/internal/worker/generate")
    async def malformed() -> dict[str, object]:
        return {"worker_name": "worker-1", "protocol_version": "switchyard.worker.v1"}

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://worker.internal",
    )
    adapter = RemoteWorkerAdapter(_build_model_config(), client=client)

    try:
        with pytest.raises(RemoteWorkerResponseError, match="malformed generate payload"):
            await adapter.generate(_build_request(), _build_context())
    finally:
        await client.aclose()
