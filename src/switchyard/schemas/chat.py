"""Chat request, response, and streaming schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatRole(StrEnum):
    """Supported chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Why a choice stopped generating output."""

    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool"
    ERROR = "error"


class ChatMessage(BaseModel):
    """A single message in a chat exchange."""

    role: ChatRole
    content: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1, max_length=64)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatCompletionRequest(BaseModel):
    """OpenAI-like chat completion input for the Switchyard gateway."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1)
    max_output_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    user: str | None = Field(default=None, min_length=1, max_length=128)


class UsageStats(BaseModel):
    """Token accounting for a response."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total_tokens(self) -> UsageStats:
        expected_total = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected_total:
            msg = "total_tokens must equal prompt_tokens + completion_tokens"
            raise ValueError(msg)
        return self


class ChatCompletionChoice(BaseModel):
    """One candidate response produced by a backend."""

    index: int = Field(ge=0)
    message: ChatMessage
    finish_reason: FinishReason = FinishReason.STOP

    @model_validator(mode="after")
    def validate_assistant_message(self) -> ChatCompletionChoice:
        if self.message.role is not ChatRole.ASSISTANT:
            msg = "completion choices must contain assistant messages"
            raise ValueError(msg)
        return self


class ChatCompletionChunkDelta(BaseModel):
    """Incremental content emitted during streamed generation."""

    role: ChatRole | None = None
    content: str | None = None

    @model_validator(mode="after")
    def validate_delta(self) -> ChatCompletionChunkDelta:
        if self.role is None and self.content is None:
            msg = "chunk deltas must include role and/or content"
            raise ValueError(msg)
        return self


class ChatCompletionChunkChoice(BaseModel):
    """One streamed chunk for a response choice."""

    index: int = Field(ge=0)
    delta: ChatCompletionChunkDelta
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-like streamed chat completion chunk emitted by Switchyard."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = Field(min_length=1, max_length=128)
    choices: list[ChatCompletionChunkChoice] = Field(min_length=1)
    backend_name: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_finish_reason_usage(self) -> ChatCompletionChunk:
        finish_count = sum(choice.finish_reason is not None for choice in self.choices)
        if finish_count > 1:
            msg = "at most one chunk choice may include a finish_reason"
            raise ValueError(msg)
        return self


class ChatCompletionResponse(BaseModel):
    """Chat completion response emitted by Switchyard."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = Field(min_length=1, max_length=128)
    choices: list[ChatCompletionChoice] = Field(min_length=1)
    usage: UsageStats
    backend_name: str = Field(min_length=1, max_length=128)


class ErrorResponse(BaseModel):
    """Typed error payload returned by the gateway."""

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=512)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
