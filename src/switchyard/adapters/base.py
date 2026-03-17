"""Base adapter contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendStatusSnapshot,
    BackendType,
)
from switchyard.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from switchyard.schemas.routing import RequestContext


class BackendAdapter(Protocol):
    """Async contract implemented by all backend adapters."""

    name: str
    backend_type: BackendType

    async def health(self) -> BackendHealth:
        """Return the backend's current health state."""

    async def capabilities(self) -> BackendCapabilities:
        """Return the backend's declared capabilities."""

    async def status(self) -> BackendStatusSnapshot:
        """Return a point-in-time backend status snapshot."""

    async def warmup(self, model_id: str | None = None) -> None:
        """Perform adapter-specific warmup."""

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        """Generate a chat completion response."""

    def stream_generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield streamed chat completion chunks."""
