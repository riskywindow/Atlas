"""Routing-related schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RoutingPolicy(StrEnum):
    """Routing policies available in Phase 0."""

    LATENCY_FIRST = "latency_first"
    BALANCED = "balanced"
    QUALITY_FIRST = "quality_first"
    LOCAL_ONLY = "local_only"


class WorkloadShape(StrEnum):
    """Broad request shapes the router can reason about."""

    INTERACTIVE = "interactive"
    BATCH = "batch"
    EVALUATION = "evaluation"


class RequestContext(BaseModel):
    """Routing-relevant metadata attached to a request."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    policy: RoutingPolicy = RoutingPolicy.BALANCED
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE
    max_latency_ms: int | None = Field(default=None, ge=1)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)


class RouteDecision(BaseModel):
    """Result of a routing decision."""

    backend_name: str = Field(min_length=1, max_length=128)
    policy: RoutingPolicy
    request_id: str = Field(min_length=1, max_length=128)
    workload_shape: WorkloadShape
    rationale: list[str] = Field(min_length=1)
    score: float | None = None
    considered_backends: list[str] = Field(min_length=1)
    rejected_backends: dict[str, str] = Field(default_factory=dict)
    fallback_backends: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fallback_backends(self) -> RouteDecision:
        if self.backend_name in self.fallback_backends:
            msg = "fallback_backends must not include the chosen backend"
            raise ValueError(msg)
        if self.backend_name not in self.considered_backends:
            msg = "considered_backends must include the chosen backend"
            raise ValueError(msg)
        return self
