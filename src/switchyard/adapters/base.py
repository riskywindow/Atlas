"""Base adapter contracts."""

from __future__ import annotations

from typing import Protocol

from switchyard.schemas.backend import BackendCapabilities, BackendHealth, BackendType
from switchyard.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from switchyard.schemas.routing import RequestContext


class BackendAdapter(Protocol):
    """Async contract implemented by all backend adapters."""

    name: str
    backend_type: BackendType

    async def health(self) -> BackendHealth:
        """Return the backend's current health state."""

    async def capabilities(self) -> BackendCapabilities:
        """Return the backend's declared capabilities."""

    async def warmup(self, model_id: str | None = None) -> None:
        """Perform adapter-specific warmup."""

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        """Generate a chat completion response."""
