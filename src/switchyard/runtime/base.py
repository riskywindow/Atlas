"""Internal runtime contracts for real backend integrations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from switchyard.schemas.backend import BackendHealthState, BackendLoadState, BackendType
from switchyard.schemas.chat import ChatCompletionRequest, FinishReason


class UnsupportedRequestError(ValueError):
    """Raised when a backend runtime cannot honor a request feature."""

    def __init__(self, *, message: str, field_name: str | None = None) -> None:
        super().__init__(message)
        self.field_name = field_name


@dataclass(frozen=True, slots=True)
class RuntimeSamplingParams:
    """Normalized generation parameters passed to a runtime provider."""

    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True, slots=True)
class RuntimeGenerationResult:
    """Non-streaming text emitted by a backend runtime."""

    text: str
    finish_reason: FinishReason = FinishReason.STOP
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStreamChunk:
    """Incremental text emitted by a backend runtime."""

    text: str
    finish_reason: FinishReason | None = None


@dataclass(frozen=True, slots=True)
class RuntimeHealthSnapshot:
    """Health view exposed by a backend runtime before adapter translation."""

    state: BackendHealthState
    load_state: BackendLoadState
    detail: str | None = None
    last_error: str | None = None


class ChatModelRuntime(Protocol):
    """Small runtime contract for backend-specific model execution."""

    backend_type: BackendType

    def load_model(self) -> None:
        """Load the configured model into memory if needed."""

    def warmup(self) -> None:
        """Warm the runtime using the configured model."""

    def health(self) -> RuntimeHealthSnapshot:
        """Return the runtime's current health state."""

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        """Produce a full text completion for a chat request."""

    def stream_generate(self, request: ChatCompletionRequest) -> Iterator[RuntimeStreamChunk]:
        """Yield streamed completion chunks for a chat request."""
