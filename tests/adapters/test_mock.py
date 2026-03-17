from __future__ import annotations

from time import perf_counter

import pytest

from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendLoadState,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason
from switchyard.schemas.routing import RequestContext, RoutingPolicy, WorkloadShape


def build_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock-chat",
        messages=[ChatMessage(role=ChatRole.USER, content="Tell me who handled this")],
        max_output_tokens=32,
    )


def build_context() -> RequestContext:
    return RequestContext(
        request_id="req_backend_1",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
    )


@pytest.mark.asyncio
async def test_mock_backend_generates_deterministic_response() -> None:
    adapter = MockBackendAdapter(
        name="mock-a",
        response_template=MockResponseTemplate(
            content="backend={backend_name} request={request_id} said={user_message}"
        ),
    )
    request = build_request()
    context = build_context()

    first = await adapter.generate(request, context)
    second = await adapter.generate(request, context)

    assert first.id == second.id
    assert first.backend_name == "mock-a"
    assert first.choices[0].message.content == (
        "backend=mock-a request=req_backend_1 said=Tell me who handled this"
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


@pytest.mark.asyncio
async def test_mock_backend_reports_configured_health_and_capabilities() -> None:
    capabilities = BackendCapabilities(
        backend_type=BackendType.MOCK,
        device_class=DeviceClass.CPU,
        model_ids=["mock-chat", "mock-reasoner"],
        max_context_tokens=4096,
        supports_streaming=True,
        concurrency_limit=2,
    )
    adapter = MockBackendAdapter(
        name="mock-health",
        simulated_latency_ms=15.0,
        health_state=BackendHealthState.DEGRADED,
        capability_metadata=capabilities,
        health_detail="synthetic degradation",
        error_rate=0.25,
    )

    health = await adapter.health()
    reported_capabilities = await adapter.capabilities()

    assert health.state is BackendHealthState.DEGRADED
    assert health.latency_ms == 15.0
    assert health.error_rate == 0.25
    assert health.detail == "synthetic degradation"
    assert health.load_state is BackendLoadState.COLD
    assert reported_capabilities == capabilities


@pytest.mark.asyncio
async def test_mock_backend_applies_simulated_latency() -> None:
    adapter = MockBackendAdapter(name="mock-latent", simulated_latency_ms=20.0)
    start = perf_counter()
    await adapter.generate(build_request(), build_context())
    elapsed_ms = (perf_counter() - start) * 1000

    assert elapsed_ms >= 15.0


@pytest.mark.asyncio
async def test_mock_backend_supports_status_and_warmup() -> None:
    adapter = MockBackendAdapter(name="mock-status")

    before = await adapter.status()
    await adapter.warmup("mock-chat")
    after = await adapter.status()

    assert before.health.load_state is BackendLoadState.COLD
    assert after.health.load_state is BackendLoadState.READY
    assert after.health.warmed_models == ["mock-chat"]
    assert after.last_warmup_at is not None
    assert after.metadata["adapter_kind"] == "mock"


@pytest.mark.asyncio
async def test_mock_backend_streams_deterministic_chunks() -> None:
    adapter = MockBackendAdapter(
        name="mock-stream",
        stream_chunk_size=2,
        response_template=MockResponseTemplate(
            content="backend={backend_name} request={request_id} said={user_message}"
        ),
    )

    chunks = [chunk async for chunk in adapter.stream_generate(build_request(), build_context())]

    assert chunks[0].choices[0].delta.role is ChatRole.ASSISTANT
    assert [chunk.choices[0].delta.content for chunk in chunks[1:-1]] == [
        "backend=mock-stream request=req_backend_1",
        "said=Tell me",
        "who handled",
        "this",
    ]
    assert chunks[-1].choices[0].finish_reason is FinishReason.STOP
