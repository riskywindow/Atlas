"""Typed schema package for Switchyard."""

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.benchmark import (
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
)
from switchyard.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    ErrorResponse,
    FinishReason,
    UsageStats,
)
from switchyard.schemas.routing import RequestContext, RouteDecision, RoutingPolicy, WorkloadShape

__all__ = [
    "BackendCapabilities",
    "BackendHealth",
    "BackendHealthState",
    "BackendStatusSnapshot",
    "BackendType",
    "BenchmarkRequestRecord",
    "BenchmarkRunArtifact",
    "BenchmarkScenario",
    "BenchmarkSummary",
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "ChatRole",
    "DeviceClass",
    "ErrorResponse",
    "FinishReason",
    "RequestContext",
    "RouteDecision",
    "RoutingPolicy",
    "UsageStats",
    "WorkloadShape",
]
