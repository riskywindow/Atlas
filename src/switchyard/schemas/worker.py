"""Internal Switchyard worker protocol schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendDeployment,
    BackendHealth,
    BackendImageMetadata,
    BackendInstance,
    BackendInstanceSource,
    BackendNetworkEndpoint,
    BackendType,
    CapacitySnapshot,
    CloudPlacementMetadata,
    CostBudgetProfile,
    DeviceClass,
    ExecutionModeLabel,
    GPUDeviceMetadata,
    NetworkCharacteristics,
    ReadinessHints,
    RuntimeIdentity,
    TopologySchemaVersion,
    TrustMetadata,
    WorkerLifecycleState,
    WorkerLocalityClass,
    WorkerRegistrationState,
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


class RemoteWorkerAuthMode(StrEnum):
    """Authentication mode for remote worker enrollment."""

    NONE = "none"
    STATIC_TOKEN = "static_token"
    SIGNED_ENROLLMENT = "signed_enrollment"


class RemoteWorkerLifecycleEventType(StrEnum):
    """Bounded event types for remote worker lifecycle inspection."""

    REGISTERED = "registered"
    HEARTBEAT = "heartbeat"
    DEREGISTERED = "deregistered"
    EVICTED = "evicted"
    AUTH_REJECTED = "auth_rejected"
    STATE_CHANGED = "state_changed"


class RemoteWorkerRegistrationRequest(BaseModel):
    """Control-plane registration payload for one remote worker instance."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(min_length=1, max_length=128)
    worker_name: str = Field(min_length=1, max_length=128)
    backend_type: BackendType
    model_identifier: str = Field(min_length=1, max_length=512)
    serving_targets: list[str] = Field(min_length=1)
    endpoint: BackendNetworkEndpoint
    capabilities: BackendCapabilities
    device_class: DeviceClass = DeviceClass.REMOTE
    runtime: RuntimeIdentity | None = None
    gpu: GPUDeviceMetadata | None = None
    environment: str = Field(default="remote", min_length=1, max_length=64)
    locality: str = Field(default="remote", min_length=1, max_length=64)
    locality_class: WorkerLocalityClass = WorkerLocalityClass.REMOTE_PRIVATE
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.REMOTE_WORKER
    source_of_truth: BackendInstanceSource = BackendInstanceSource.REGISTERED
    placement: CloudPlacementMetadata = Field(default_factory=CloudPlacementMetadata)
    cost_profile: CostBudgetProfile = Field(default_factory=CostBudgetProfile)
    readiness_hints: ReadinessHints = Field(default_factory=ReadinessHints)
    trust: TrustMetadata = Field(default_factory=TrustMetadata)
    network_characteristics: NetworkCharacteristics = Field(
        default_factory=NetworkCharacteristics
    )
    image_metadata: BackendImageMetadata | None = None
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.REGISTERING
    ready: bool = False
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    observed_capacity: CapacitySnapshot | None = None
    health: BackendHealth | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class RemoteWorkerHeartbeatRequest(BaseModel):
    """Heartbeat payload for a previously registered remote worker instance."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(min_length=1, max_length=128)
    lifecycle_state: WorkerLifecycleState | None = None
    ready: bool | None = None
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    observed_capacity: CapacitySnapshot | None = None
    health: BackendHealth | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RemoteWorkerDeregisterRequest(BaseModel):
    """Graceful worker de-registration payload."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, min_length=1, max_length=256)
    retire: bool = True


class RemoteWorkerCleanupResponse(BaseModel):
    """Result of stale-worker cleanup."""

    model_config = ConfigDict(extra="forbid")

    evicted_worker_ids: list[str] = Field(default_factory=list)
    remaining_worker_count: int = Field(default=0, ge=0)


class RemoteWorkerRegistrationResponse(BaseModel):
    """Acknowledgement for worker registration or heartbeat updates."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(min_length=1, max_length=128)
    lifecycle_state: WorkerLifecycleState
    registration_state: WorkerRegistrationState
    registered_at: datetime
    last_heartbeat_at: datetime
    expires_at: datetime
    ready: bool = False
    live: bool = True
    secure_registration_required: bool = False
    auth_mode: RemoteWorkerAuthMode = RemoteWorkerAuthMode.NONE
    token_verified: bool = False
    lease_token: str | None = Field(default=None, min_length=1, max_length=256)


class RegisteredRemoteWorkerRecord(BaseModel):
    """Operator-facing snapshot of one registered remote worker."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(min_length=1, max_length=128)
    worker_name: str = Field(min_length=1, max_length=128)
    environment: str = Field(min_length=1, max_length=64)
    serving_targets: list[str] = Field(default_factory=list)
    backend_name: str = Field(min_length=1, max_length=128)
    backend_type: BackendType
    lifecycle_state: WorkerLifecycleState
    registration_state: WorkerRegistrationState
    registered_at: datetime
    last_heartbeat_at: datetime
    expires_at: datetime
    deregistered_at: datetime | None = None
    stale: bool = False
    live: bool = True
    ready: bool = False
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    heartbeat_count: int = Field(default=0, ge=0)
    capabilities: BackendCapabilities
    runtime: RuntimeIdentity | None = None
    gpu: GPUDeviceMetadata | None = None
    deployment: BackendDeployment | None = None
    observed_capacity: CapacitySnapshot | None = None
    token_verified: bool = False
    instance: BackendInstance
    metadata: dict[str, str] = Field(default_factory=dict)


class RemoteWorkerLifecycleEvent(BaseModel):
    """One bounded lifecycle event for operator inspection and artifacts."""

    model_config = ConfigDict(extra="forbid")

    event_type: RemoteWorkerLifecycleEventType
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    worker_id: str = Field(min_length=1, max_length=128)
    lifecycle_state: WorkerLifecycleState | None = None
    detail: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: dict[str, str] = Field(default_factory=dict)


class RegisteredRemoteWorkerSnapshot(BaseModel):
    """Bounded operator view of the current registered worker set."""

    model_config = ConfigDict(extra="forbid")

    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    secure_registration_required: bool = False
    auth_mode: RemoteWorkerAuthMode = RemoteWorkerAuthMode.NONE
    dynamic_registration_enabled: bool = False
    heartbeat_timeout_seconds: float = Field(gt=0.0, le=3600.0)
    stale_eviction_seconds: float = Field(gt=0.0, le=86_400.0)
    registration_token_name: str | None = Field(default=None, min_length=1, max_length=128)
    worker_count: int = Field(default=0, ge=0)
    stale_worker_count: int = Field(default=0, ge=0)
    ready_worker_count: int = Field(default=0, ge=0)
    live_worker_count: int = Field(default=0, ge=0)
    draining_worker_count: int = Field(default=0, ge=0)
    unhealthy_worker_count: int = Field(default=0, ge=0)
    lost_worker_count: int = Field(default=0, ge=0)
    retired_worker_count: int = Field(default=0, ge=0)
    workers: list[RegisteredRemoteWorkerRecord] = Field(default_factory=list)
    recent_events: list[RemoteWorkerLifecycleEvent] = Field(default_factory=list)


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
    runtime: RuntimeIdentity | None = None
    gpu: GPUDeviceMetadata | None = None
    transport_metadata: WorkerResponseMetadata | None = None


class WorkerHealthResponse(WorkerProtocolEnvelope):
    """Worker health response."""

    health: BackendHealth


class WorkerReadinessResponse(WorkerProtocolEnvelope):
    """Worker readiness response."""

    ready: bool
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    observed_capacity: CapacitySnapshot | None = None
    health: BackendHealth


class WorkerCapabilitiesResponse(WorkerProtocolEnvelope):
    """Worker capabilities response."""

    backend_type: BackendType | None = None
    execution_mode: ExecutionModeLabel | None = None
    capabilities: BackendCapabilities
    deployment: BackendDeployment | None = None


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
