"""Internal Switchyard worker protocol schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendType,
    ExecutionModeLabel,
    TopologySchemaVersion,
)
from switchyard.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from switchyard.schemas.routing import RequestContext


class WorkerProtocolVersion(StrEnum):
    """Explicit version for the internal worker HTTP protocol."""

    V1 = "switchyard.worker.v1"
    V2 = "switchyard.worker.v2"


class WorkerRequestMetadata(BaseModel):
    """Transport metadata sent alongside internal worker requests."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    timeout_ms: int | None = Field(default=None, ge=1)


class WorkerResponseMetadata(BaseModel):
    """Transport metadata returned by worker responses for correlation."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    worker_request_id: str = Field(min_length=1, max_length=128)


class WorkerProtocolEnvelope(BaseModel):
    """Common envelope metadata for worker protocol messages."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: WorkerProtocolVersion = WorkerProtocolVersion.V2
    topology_schema_version: TopologySchemaVersion = TopologySchemaVersion.V1
    worker_name: str = Field(min_length=1, max_length=128)
    transport_metadata: WorkerResponseMetadata | None = None


class WorkerHealthResponse(WorkerProtocolEnvelope):
    """Worker health response."""

    health: BackendHealth


class WorkerReadinessResponse(WorkerProtocolEnvelope):
    """Worker readiness response."""

    ready: bool
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    health: BackendHealth


class WorkerCapabilitiesResponse(WorkerProtocolEnvelope):
    """Worker capabilities response."""

    backend_type: BackendType | None = None
    execution_mode: ExecutionModeLabel | None = None
    capabilities: BackendCapabilities


class WorkerWarmupRequest(BaseModel):
    """Worker warmup request."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = Field(default=None, min_length=1, max_length=512)
    transport_metadata: WorkerRequestMetadata | None = None


class WorkerWarmupResponse(WorkerProtocolEnvelope):
    """Worker warmup response."""

    warmed: bool = True
    health: BackendHealth


class WorkerGenerateRequest(BaseModel):
    """Worker generate request."""

    model_config = ConfigDict(extra="forbid")

    request: ChatCompletionRequest
    context: RequestContext
    transport_metadata: WorkerRequestMetadata | None = None


class WorkerGenerateResponse(WorkerProtocolEnvelope):
    """Worker generate response."""

    response: ChatCompletionResponse


class WorkerStreamChunkResponse(WorkerProtocolEnvelope):
    """One streamed worker chunk."""

    chunk: ChatCompletionChunk
