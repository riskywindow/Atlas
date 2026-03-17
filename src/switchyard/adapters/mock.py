"""Deterministic mock backend adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendDeployment,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
    EngineType,
    PerformanceHint,
    QualityHint,
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


@dataclass(frozen=True, slots=True)
class MockResponseTemplate:
    """Template used to build deterministic mock responses."""

    content: str = "mock response from {backend_name} for request {request_id}"


class MockBackendAdapter:
    """Deterministic backend adapter for Phase 0 tests and integration slices."""

    _DEFAULT_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)

    def __init__(
        self,
        *,
        name: str = "mock-backend",
        simulated_latency_ms: float = 0.0,
        health_state: BackendHealthState = BackendHealthState.HEALTHY,
        capability_metadata: BackendCapabilities | None = None,
        response_template: MockResponseTemplate | None = None,
        health_detail: str | None = None,
        error_rate: float | None = None,
        stream_chunk_size: int = 3,
        simulated_active_requests: int = 0,
        simulated_queue_depth: int = 0,
        circuit_open: bool = False,
        circuit_reason: str | None = None,
    ) -> None:
        self.name = name
        self.backend_type = BackendType.MOCK
        self._simulated_latency_ms = simulated_latency_ms
        self._health_state = health_state
        self._capability_metadata = capability_metadata or BackendCapabilities(
            backend_type=BackendType.MOCK,
            engine_type=EngineType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            supports_streaming=True,
            concurrency_limit=1,
            configured_priority=100,
            configured_weight=1.0,
            quality_hint=QualityHint.BALANCED,
            performance_hint=PerformanceHint.BALANCED,
            model_aliases={"mock-chat": "mock-chat"},
            default_model="mock-chat",
        )
        self._response_template = response_template or MockResponseTemplate()
        self._health_detail = health_detail
        self._error_rate = error_rate
        self._stream_chunk_size = stream_chunk_size
        self._simulated_active_requests = simulated_active_requests
        self._simulated_queue_depth = simulated_queue_depth
        self._circuit_open = circuit_open
        self._circuit_reason = circuit_reason
        self._warmed_models: set[str] = set()
        self._last_warmup_at: datetime | None = None

    async def health(self) -> BackendHealth:
        """Return the configured health state."""

        return BackendHealth(
            state=self._health_state,
            latency_ms=self._simulated_latency_ms,
            error_rate=self._error_rate,
            detail=self._health_detail,
            load_state=self._load_state(),
            warmed_models=sorted(self._warmed_models),
            circuit_open=self._circuit_open,
            circuit_reason=self._circuit_reason,
        )

    async def capabilities(self) -> BackendCapabilities:
        """Return the configured capabilities."""

        return self._capability_metadata

    async def status(self) -> BackendStatusSnapshot:
        """Return a full point-in-time status snapshot."""

        return BackendStatusSnapshot(
            name=self.name,
            deployment=BackendDeployment(
                name=self.name,
                backend_type=self.backend_type,
                engine_type=self._capability_metadata.engine_type,
                model_identifier=self._capability_metadata.model_ids[-1],
                serving_targets=self._capability_metadata.serving_targets,
                configured_priority=self._capability_metadata.configured_priority,
                configured_weight=self._capability_metadata.configured_weight,
            ),
            capabilities=await self.capabilities(),
            health=await self.health(),
            active_requests=self._simulated_active_requests,
            queue_depth=self._simulated_queue_depth,
            last_warmup_at=self._last_warmup_at,
            metadata={"adapter_kind": "mock"},
        )

    async def warmup(self, model_id: str | None = None) -> None:
        """Warmup is a no-op for the mock adapter."""

        warmed_model = (
            model_id
            or self._capability_metadata.default_model
            or self._capability_metadata.model_ids[0]
        )
        self._warmed_models.add(warmed_model)
        self._last_warmup_at = self._DEFAULT_CREATED_AT

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        """Generate a deterministic response for a valid request."""

        if self._simulated_latency_ms > 0:
            await asyncio.sleep(self._simulated_latency_ms / 1000)

        content = self._build_response_content(request=request, context=context)
        completion_tokens = len(content.split())
        prompt_tokens = sum(len(message.content.split()) for message in request.messages)
        response_id = self._build_response_id(request=request, context=context)

        return ChatCompletionResponse(
            id=response_id,
            created_at=self._DEFAULT_CREATED_AT,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=ChatRole.ASSISTANT, content=content),
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            backend_name=self.name,
        )

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Generate deterministic streamed chunks for a valid request."""

        if self._simulated_latency_ms > 0:
            await asyncio.sleep(self._simulated_latency_ms / 1000)

        content = self._build_response_content(request=request, context=context)
        response_id = self._build_response_id(request=request, context=context)

        yield ChatCompletionChunk(
            id=response_id,
            created_at=self._DEFAULT_CREATED_AT,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role=ChatRole.ASSISTANT),
                )
            ],
            backend_name=self.name,
        )

        for chunk_text in self._chunk_content(content):
            yield ChatCompletionChunk(
                id=response_id,
                created_at=self._DEFAULT_CREATED_AT,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=chunk_text),
                    )
                ],
                backend_name=self.name,
            )

        yield ChatCompletionChunk(
            id=response_id,
            created_at=self._DEFAULT_CREATED_AT,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=""),
                    finish_reason=FinishReason.STOP,
                )
            ],
            backend_name=self.name,
        )

    def _build_response_content(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> str:
        last_message = request.messages[-1]
        return self._response_template.content.format(
            backend_name=self.name,
            request_id=context.request_id,
            model=request.model,
            user_message=last_message.content,
            policy=context.policy.value,
            workload_shape=context.workload_shape.value,
        )

    def _build_response_id(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> str:
        digest = sha256(
            f"{self.name}:{request.model}:{context.request_id}:{len(request.messages)}".encode()
        ).hexdigest()
        return f"mockcmpl_{digest[:16]}"

    def _chunk_content(self, content: str) -> list[str]:
        words = content.split()
        if not words:
            return [""]

        chunk_size = max(1, self._stream_chunk_size)
        return [
            " ".join(words[index : index + chunk_size])
            for index in range(0, len(words), chunk_size)
        ]

    def _load_state(self) -> BackendLoadState:
        if self._health_state is BackendHealthState.UNAVAILABLE:
            return BackendLoadState.FAILED
        if self._warmed_models:
            return BackendLoadState.READY
        return BackendLoadState.COLD
