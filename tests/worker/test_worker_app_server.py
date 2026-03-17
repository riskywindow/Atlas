from __future__ import annotations

import json

import httpx
import pytest

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.schemas.backend import BackendHealthState
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import RequestContext, RoutingPolicy, WorkloadShape
from switchyard.worker.app import create_worker_app


def _build_request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock-chat",
        stream=stream,
        messages=[ChatMessage(role=ChatRole.USER, content="hello worker")],
    )


@pytest.mark.asyncio
async def test_worker_app_reports_real_readiness_and_supports_warmup() -> None:
    adapter = MockBackendAdapter(name="mock-worker")
    app = create_worker_app(adapter, worker_name="mock-worker")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    try:
        readiness_before = await client.get("/internal/worker/ready")
        warmup = await client.post("/internal/worker/warmup", json={})
        readiness_after = await client.get("/internal/worker/ready")
        health = await client.get("/healthz")
    finally:
        await client.aclose()

    assert readiness_before.status_code == 200
    assert readiness_before.json()["ready"] is False
    assert readiness_before.json()["health"]["load_state"] == "cold"
    assert warmup.status_code == 200
    assert readiness_after.json()["ready"] is True
    assert readiness_after.json()["health"]["load_state"] == "ready"
    assert health.json()["health"]["state"] == BackendHealthState.HEALTHY.value


@pytest.mark.asyncio
async def test_worker_app_serves_internal_and_public_generate_paths() -> None:
    adapter = MockBackendAdapter(name="mock-worker")
    await adapter.warmup()
    app = create_worker_app(adapter, worker_name="mock-worker")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    try:
        internal_response = await client.post(
            "/internal/worker/generate",
            json={
                "request": _build_request().model_dump(mode="json"),
                "context": RequestContext(
                    request_id="req-worker-001",
                    policy=RoutingPolicy.LOCAL_ONLY,
                    workload_shape=WorkloadShape.INTERACTIVE,
                ).model_dump(mode="json"),
            },
        )
        public_response = await client.post(
            "/v1/chat/completions",
            json=_build_request().model_dump(mode="json"),
            headers={"x-request-id": "req-worker-002"},
        )
    finally:
        await client.aclose()

    assert internal_response.status_code == 200
    assert internal_response.json()["worker_name"] == "mock-worker"
    assert internal_response.json()["response"]["backend_name"] == "mock-worker"
    assert public_response.status_code == 200
    assert public_response.json()["backend_name"] == "mock-worker"


@pytest.mark.asyncio
async def test_worker_app_streams_internal_and_public_sse() -> None:
    adapter = MockBackendAdapter(name="mock-worker")
    await adapter.warmup()
    app = create_worker_app(adapter, worker_name="mock-worker")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    try:
        internal_response = await client.post(
            "/internal/worker/generate/stream",
            json={
                "request": _build_request(stream=True).model_dump(mode="json"),
                "context": RequestContext(
                    request_id="req-worker-stream-001",
                    policy=RoutingPolicy.LOCAL_ONLY,
                    workload_shape=WorkloadShape.INTERACTIVE,
                ).model_dump(mode="json"),
            },
        )
        public_response = await client.post(
            "/v1/chat/completions",
            json=_build_request(stream=True).model_dump(mode="json"),
            headers={"x-request-id": "req-worker-stream-002"},
        )
    finally:
        await client.aclose()

    internal_events = [
        line[6:]
        for line in internal_response.text.splitlines()
        if line.startswith("data: ")
    ]
    public_events = [
        line[6:] for line in public_response.text.splitlines() if line.startswith("data: ")
    ]

    assert json.loads(internal_events[0])["chunk"]["backend_name"] == "mock-worker"
    assert internal_events[-1] == "[DONE]"
    assert json.loads(public_events[0])["backend_name"] == "mock-worker"
    assert public_events[-1] == "[DONE]"
