"""Fake remote-worker app for CI and local contract testing."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI

from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendType,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
    QualityHint,
)
from switchyard.worker.app import create_worker_app


@dataclass(frozen=True, slots=True)
class FakeRemoteWorkerConfig:
    """Deterministic fake worker settings for remote transport tests."""

    worker_name: str = "fake-remote-worker"
    serving_target: str = "chat-shared"
    model_identifier: str = "mock-chat"
    backend_type: BackendType = BackendType.MOCK
    device_class: DeviceClass = DeviceClass.REMOTE
    engine_type: EngineType = EngineType.MOCK
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.REMOTE_WORKER
    simulated_latency_ms: float = 0.0
    health_state: BackendHealthState = BackendHealthState.HEALTHY
    supports_streaming: bool = True
    concurrency_limit: int = 4
    response_template: str = "fake remote response from {backend_name} for {request_id}"
    stream_chunk_size: int = 3
    simulated_active_requests: int = 0
    simulated_queue_depth: int = 0


def create_fake_remote_worker_app(
    config: FakeRemoteWorkerConfig | None = None,
) -> FastAPI:
    """Build a fake but realistic network worker around the shared worker protocol."""

    resolved = config or FakeRemoteWorkerConfig()
    adapter = MockBackendAdapter(
        name=resolved.worker_name,
        simulated_latency_ms=resolved.simulated_latency_ms,
        health_state=resolved.health_state,
        capability_metadata=BackendCapabilities(
            backend_type=resolved.backend_type,
            engine_type=resolved.engine_type,
            device_class=resolved.device_class,
            execution_mode=resolved.execution_mode,
            model_ids=[resolved.serving_target, resolved.model_identifier],
            serving_targets=[resolved.serving_target],
            max_context_tokens=8192,
            supports_streaming=resolved.supports_streaming,
            concurrency_limit=resolved.concurrency_limit,
            quality_hint=QualityHint.BALANCED,
        ),
        response_template=MockResponseTemplate(content=resolved.response_template),
        stream_chunk_size=resolved.stream_chunk_size,
        simulated_active_requests=resolved.simulated_active_requests,
        simulated_queue_depth=resolved.simulated_queue_depth,
    )
    return create_worker_app(adapter, worker_name=resolved.worker_name)
