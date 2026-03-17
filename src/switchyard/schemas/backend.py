"""Backend capability and health schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class BackendLoadState(StrEnum):
    """Lifecycle state for backend runtime readiness."""

    COLD = "cold"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"


class BackendType(StrEnum):
    """Supported backend families."""

    MOCK = "mock"
    MLX_LM = "mlx_lm"
    VLLM_METAL = "vllm_metal"
    VLLM_CUDA = "vllm_cuda"
    REMOTE_OPENAI_LIKE = "remote_openai_like"


class EngineType(StrEnum):
    """Inference engine family exposed by a backend deployment."""

    MOCK = "mock"
    MLX = "mlx"
    VLLM = "vllm"
    REMOTE_OPENAI = "remote_openai"


class DeviceClass(StrEnum):
    """Logical device classes that a backend can run on."""

    CPU = "cpu"
    APPLE_GPU = "apple_gpu"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    REMOTE = "remote"


class DeploymentProfile(StrEnum):
    """Portable deployment profile for the control plane or a backend deployment."""

    HOST_NATIVE = "host_native"
    CONTROL_PLANE_CONTAINER = "control_plane_container"
    COMPOSE = "compose"
    KIND = "kind"
    REMOTE = "remote"


class WorkerTransportType(StrEnum):
    """Transport used to reach a worker instance."""

    IN_PROCESS = "in_process"
    HTTP = "http"
    HTTPS = "https"


class BackendHealthState(StrEnum):
    """High-level health states for a backend."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class QualityHint(StrEnum):
    """Coarse quality guidance for route selection and benchmarking."""

    ECONOMY = "economy"
    BALANCED = "balanced"
    PREMIUM = "premium"


class PerformanceHint(StrEnum):
    """Coarse performance guidance for route selection and benchmarking."""

    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    BALANCED = "balanced"


class CacheCapabilityFlags(BaseModel):
    """Cache-related features exposed by a backend deployment."""

    supports_prefix_cache: bool = False
    supports_prompt_cache_read: bool = False
    supports_prompt_cache_write: bool = False
    supports_kv_cache_reuse: bool = False


class BackendImageMetadata(BaseModel):
    """Build and image metadata for a deployment or worker."""

    image_repository: str | None = Field(default=None, min_length=1, max_length=256)
    image_tag: str | None = Field(default=None, min_length=1, max_length=128)
    image_digest: str | None = Field(default=None, min_length=1, max_length=256)
    git_sha: str | None = Field(default=None, min_length=7, max_length=40)
    build_metadata: dict[str, str] = Field(default_factory=dict)


class WorkerRegistrationState(StrEnum):
    """Registration lifecycle state for an explicit worker instance."""

    STATIC = "static"
    REGISTERED = "registered"
    DISCOVERED = "discovered"
    STALE = "stale"


class BackendInstanceSource(StrEnum):
    """Source of truth for a backend instance record."""

    STATIC_CONFIG = "static_config"
    REGISTERED = "registered"
    DISCOVERED = "discovered"


class BackendRegistrationMetadata(BaseModel):
    """Worker registration and heartbeat metadata."""

    state: WorkerRegistrationState = WorkerRegistrationState.STATIC
    registered_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    expires_at: datetime | None = None
    source: str | None = Field(default=None, min_length=1, max_length=128)
    detail: str | None = Field(default=None, max_length=256)


class BackendNetworkEndpoint(BaseModel):
    """Network-addressable endpoint for a backend worker instance."""

    base_url: str = Field(min_length=1, max_length=512)
    transport: WorkerTransportType = WorkerTransportType.HTTP
    health_path: str = Field(default="/healthz", min_length=1, max_length=128)
    readiness_path: str = Field(
        default="/internal/worker/ready",
        min_length=1,
        max_length=128,
    )
    capabilities_path: str = Field(
        default="/internal/worker/capabilities",
        min_length=1,
        max_length=128,
    )
    warmup_path: str = Field(
        default="/internal/worker/warmup",
        min_length=1,
        max_length=128,
    )
    chat_completions_path: str = Field(
        default="/internal/worker/generate",
        min_length=1,
        max_length=128,
    )
    stream_chat_completions_path: str = Field(
        default="/internal/worker/generate/stream",
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_paths(self) -> BackendNetworkEndpoint:
        for field_name in (
            "health_path",
            "readiness_path",
            "capabilities_path",
            "warmup_path",
            "chat_completions_path",
            "stream_chat_completions_path",
        ):
            value = getattr(self, field_name)
            if not value.startswith("/"):
                msg = f"{field_name} must start with '/'"
                raise ValueError(msg)
        return self


class BackendInstance(BaseModel):
    """Explicit worker-instance inventory for one backend deployment."""

    instance_id: str = Field(min_length=1, max_length=128)
    endpoint: BackendNetworkEndpoint
    source_of_truth: BackendInstanceSource = BackendInstanceSource.STATIC_CONFIG
    backend_type: BackendType | None = None
    device_class: DeviceClass
    model_identifier: str | None = Field(default=None, min_length=1, max_length=512)
    locality: str = Field(default="local", min_length=1, max_length=64)
    tags: list[str] = Field(default_factory=list)
    registration: BackendRegistrationMetadata = Field(
        default_factory=BackendRegistrationMetadata
    )
    health: BackendHealth | None = None
    last_seen_at: datetime | None = None
    image_metadata: BackendImageMetadata | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class LogicalModelTarget(BaseModel):
    """Client-visible logical serving target that may map to several deployments."""

    alias: str = Field(min_length=1, max_length=128)
    model_identifier: str | None = Field(default=None, min_length=1, max_length=512)
    deployments: list[str] = Field(default_factory=list)


class BackendDeployment(BaseModel):
    """Concrete backend deployment behind a logical serving target."""

    name: str = Field(min_length=1, max_length=128)
    backend_type: BackendType
    engine_type: EngineType
    model_identifier: str = Field(min_length=1, max_length=512)
    serving_targets: list[str] = Field(min_length=1)
    configured_priority: int = Field(default=100, ge=0, le=1000)
    configured_weight: float = Field(default=1.0, gt=0.0, le=1000.0)
    deployment_profile: DeploymentProfile = DeploymentProfile.HOST_NATIVE
    environment: str = Field(default="local", min_length=1, max_length=64)
    build_metadata: BackendImageMetadata | None = None
    instances: list[BackendInstance] = Field(default_factory=list)


class BackendCapabilities(BaseModel):
    """Declared backend capabilities used by routing."""

    backend_type: BackendType
    engine_type: EngineType = EngineType.MOCK
    device_class: DeviceClass
    model_ids: list[str] = Field(min_length=1)
    serving_targets: list[str] = Field(default_factory=list)
    max_context_tokens: int = Field(ge=1)
    supports_streaming: bool = False
    concurrency_limit: int = Field(default=1, ge=1)
    configured_priority: int = Field(default=100, ge=0, le=1000)
    configured_weight: float = Field(default=1.0, gt=0.0, le=1000.0)
    quality_tier: int = Field(default=1, ge=1, le=5)
    quality_hint: QualityHint = QualityHint.BALANCED
    performance_hint: PerformanceHint = PerformanceHint.BALANCED
    model_aliases: dict[str, str] = Field(default_factory=dict)
    default_model: str | None = Field(default=None, min_length=1, max_length=256)
    supports_tools: bool = False
    warmup_required: bool = False
    cache_capabilities: CacheCapabilityFlags = Field(default_factory=CacheCapabilityFlags)

    @model_validator(mode="after")
    def validate_serving_targets(self) -> BackendCapabilities:
        if not self.serving_targets:
            default_targets: list[str] = []
            if self.default_model is not None:
                default_targets.append(self.default_model)
            default_targets.extend(
                alias for alias in self.model_aliases if alias not in default_targets
            )
            if not default_targets:
                default_targets.append(self.model_ids[0])
            self.serving_targets = default_targets
        return self

    def supports_model_target(self, target: str) -> bool:
        """Return whether the capability set can serve the requested logical target."""

        return (
            target in self.serving_targets
            or target in self.model_ids
            or target in self.model_aliases
        )


class BackendHealth(BaseModel):
    """Health signal for a backend adapter."""

    state: BackendHealthState
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str | None = Field(default=None, max_length=256)
    load_state: BackendLoadState = BackendLoadState.COLD
    warmed_models: list[str] = Field(default_factory=list)
    last_error: str | None = Field(default=None, max_length=256)
    circuit_open: bool = False
    circuit_reason: str | None = Field(default=None, max_length=256)


class BackendStatusSnapshot(BaseModel):
    """Point-in-time backend status combining capabilities and health."""

    name: str = Field(min_length=1, max_length=128)
    deployment: BackendDeployment | None = None
    logical_targets: list[LogicalModelTarget] = Field(default_factory=list)
    instance_inventory: list[BackendInstance] = Field(default_factory=list)
    capabilities: BackendCapabilities
    health: BackendHealth
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
    last_warmup_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_deployment_targets(self) -> BackendStatusSnapshot:
        if self.deployment is None:
            model_identifier = (
                self.metadata.get("model_identifier") or self.capabilities.model_ids[-1]
            )
            self.deployment = BackendDeployment(
                name=self.name,
                backend_type=self.capabilities.backend_type,
                engine_type=self.capabilities.engine_type,
                model_identifier=model_identifier,
                serving_targets=self.capabilities.serving_targets,
                configured_priority=self.capabilities.configured_priority,
                configured_weight=self.capabilities.configured_weight,
            )
        if not self.instance_inventory and self.deployment.instances:
            self.instance_inventory = list(self.deployment.instances)
        if self.instance_inventory and not self.deployment.instances:
            self.deployment.instances = list(self.instance_inventory)
        if not self.logical_targets:
            targets: list[LogicalModelTarget] = []
            for target in self.capabilities.serving_targets:
                targets.append(
                    LogicalModelTarget(
                        alias=target,
                        model_identifier=self.capabilities.model_aliases.get(target),
                        deployments=[self.name],
                    )
                )
            self.logical_targets = targets
        return self
