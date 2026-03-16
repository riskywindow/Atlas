"""Deterministic mock backend adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
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
    ) -> None:
        self.name = name
        self.backend_type = BackendType.MOCK
        self._simulated_latency_ms = simulated_latency_ms
        self._health_state = health_state
        self._capability_metadata = capability_metadata or BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            max_context_tokens=8192,
            supports_streaming=False,
            concurrency_limit=1,
        )
        self._response_template = response_template or MockResponseTemplate()
        self._health_detail = health_detail
        self._error_rate = error_rate

    async def health(self) -> BackendHealth:
        """Return the configured health state."""

        return BackendHealth(
            state=self._health_state,
            latency_ms=self._simulated_latency_ms,
            error_rate=self._error_rate,
            detail=self._health_detail,
        )

    async def capabilities(self) -> BackendCapabilities:
        """Return the configured capabilities."""

        return self._capability_metadata

    async def warmup(self, model_id: str | None = None) -> None:
        """Warmup is a no-op for the mock adapter."""

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
