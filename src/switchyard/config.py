"""Application configuration for Switchyard."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from switchyard.schemas.backend import (
    BackendImageMetadata,
    BackendInstance,
    BackendInstanceSource,
    BackendNetworkEndpoint,
    BackendRegistrationMetadata,
    BackendType,
    CloudPlacementMetadata,
    CostBudgetProfile,
    DeploymentProfile,
    DeviceClass,
    ExecutionModeLabel,
    NetworkCharacteristics,
    ReadinessHints,
    TrustMetadata,
    WorkerLocalityClass,
    WorkerTransportType,
)
from switchyard.schemas.benchmark import TraceCaptureMode
from switchyard.schemas.routing import (
    CanaryPolicy,
    PolicyRolloutMode,
    RequestClass,
    RoutingPolicy,
    ShadowPolicy,
)


class AppEnvironment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class GenerationDefaults(BaseModel):
    """Optional generation defaults for a configured local model."""

    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class WarmupSettings(BaseModel):
    """Optional warmup settings for a configured local model."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    eager: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0.0, le=3600.0)


class BackendInstanceConfig(BaseModel):
    """Static worker-instance inventory for a configured deployment."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=128)
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
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0, le=3600.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    device_class: DeviceClass | None = None
    locality: str = Field(default="local", min_length=1, max_length=64)
    locality_class: WorkerLocalityClass = WorkerLocalityClass.UNKNOWN
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.HOST_NATIVE
    placement: CloudPlacementMetadata = Field(default_factory=CloudPlacementMetadata)
    cost_profile: CostBudgetProfile = Field(default_factory=CostBudgetProfile)
    readiness_hints: ReadinessHints = Field(default_factory=ReadinessHints)
    trust: TrustMetadata = Field(default_factory=TrustMetadata)
    network_characteristics: NetworkCharacteristics = Field(
        default_factory=NetworkCharacteristics
    )
    source_of_truth: BackendInstanceSource = BackendInstanceSource.STATIC_CONFIG
    tags: tuple[str, ...] = ()
    registration: BackendRegistrationMetadata = Field(
        default_factory=BackendRegistrationMetadata
    )
    image_metadata: BackendImageMetadata | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_paths(self) -> BackendInstanceConfig:
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

    def to_backend_instance(
        self,
        *,
        backend_type: BackendType,
        default_device_class: DeviceClass,
        model_identifier: str,
    ) -> BackendInstance:
        """Convert config-time inventory into the shared backend schema."""

        return BackendInstance(
            instance_id=self.instance_id,
            endpoint=BackendNetworkEndpoint(
                base_url=self.base_url,
                transport=self.transport,
                health_path=self.health_path,
                readiness_path=self.readiness_path,
                capabilities_path=self.capabilities_path,
                warmup_path=self.warmup_path,
                chat_completions_path=self.chat_completions_path,
                stream_chat_completions_path=self.stream_chat_completions_path,
            ),
            source_of_truth=self.source_of_truth,
            backend_type=backend_type,
            device_class=self.device_class or default_device_class,
            model_identifier=model_identifier,
            locality=self.locality,
            locality_class=self.locality_class,
            execution_mode=self.execution_mode,
            placement=self.placement.model_copy(deep=True),
            cost_profile=self.cost_profile.model_copy(deep=True),
            readiness_hints=self.readiness_hints.model_copy(deep=True),
            trust=self.trust.model_copy(deep=True),
            network_characteristics=self.network_characteristics.model_copy(deep=True),
            tags=list(self.tags),
            registration=self.registration.model_copy(deep=True),
            image_metadata=self.image_metadata.model_copy(deep=True)
            if self.image_metadata is not None
            else None,
            metadata={
                **dict(self.metadata),
                "connect_timeout_seconds": str(self.connect_timeout_seconds),
                "request_timeout_seconds": str(self.request_timeout_seconds),
            },
        )


class DeploymentLayerConfig(BaseModel):
    """Environment-specific deployment layer metadata for the control plane."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    deployment_profile: DeploymentProfile
    default_transport: WorkerTransportType = WorkerTransportType.HTTP
    gateway_base_url: str | None = Field(default=None, min_length=1, max_length=256)
    image_metadata: BackendImageMetadata | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DeploymentTopologySettings(BaseModel):
    """Portable control-plane topology configuration."""

    model_config = ConfigDict(extra="forbid")

    active_environment: str = Field(default="local", min_length=1, max_length=64)
    deployment_profile: DeploymentProfile = DeploymentProfile.HOST_NATIVE
    default_transport: WorkerTransportType = WorkerTransportType.IN_PROCESS
    control_plane_image: BackendImageMetadata | None = None
    layers: tuple[DeploymentLayerConfig, ...] = ()

    @model_validator(mode="after")
    def validate_layers(self) -> DeploymentTopologySettings:
        names = [layer.name for layer in self.layers]
        if len(names) != len(set(names)):
            msg = "topology.layers must not contain duplicate layer names"
            raise ValueError(msg)
        return self


class LocalModelConfig(BaseModel):
    """Backend-agnostic configuration for a locally hosted model."""

    model_config = ConfigDict(extra="forbid")

    alias: str = Field(min_length=1, max_length=128)
    serving_target: str | None = Field(default=None, min_length=1, max_length=128)
    environment: str = Field(default="local", min_length=1, max_length=64)
    deployment_profile: DeploymentProfile = DeploymentProfile.HOST_NATIVE
    model_identifier: str = Field(min_length=1, max_length=512)
    backend_type: BackendType
    configured_priority: int = Field(default=100, ge=0, le=1000)
    configured_weight: float = Field(default=1.0, gt=0.0, le=1000.0)
    worker_transport: WorkerTransportType = WorkerTransportType.IN_PROCESS
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.HOST_NATIVE
    image_tag: str | None = Field(default=None, min_length=1, max_length=128)
    placement: CloudPlacementMetadata = Field(default_factory=CloudPlacementMetadata)
    cost_profile: CostBudgetProfile = Field(default_factory=CostBudgetProfile)
    readiness_hints: ReadinessHints = Field(default_factory=ReadinessHints)
    trust: TrustMetadata = Field(default_factory=TrustMetadata)
    network_characteristics: NetworkCharacteristics = Field(
        default_factory=NetworkCharacteristics
    )
    build_metadata: dict[str, str] = Field(default_factory=dict)
    instances: tuple[BackendInstanceConfig, ...] = ()
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    warmup: WarmupSettings = Field(default_factory=WarmupSettings)


class Phase4FeatureToggles(BaseModel):
    """Explicit switches for advanced Phase 4 behavior."""

    model_config = ConfigDict(extra="forbid")

    admission_control_enabled: bool = False
    circuit_breakers_enabled: bool = False
    session_affinity_enabled: bool = False
    canary_routing_enabled: bool = False
    shadow_routing_enabled: bool = False


class TenantLimitConfig(BaseModel):
    """Per-tenant concurrency and queue caps."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=128)
    request_class: RequestClass | None = None
    concurrency_cap: int = Field(ge=1, le=100_000)
    queue_size: int = Field(default=0, ge=0, le=100_000)


class AdmissionControlSettings(BaseModel):
    """Admission-control defaults and per-tenant overrides."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    global_concurrency_cap: int = Field(default=64, ge=1, le=100_000)
    global_queue_size: int = Field(default=128, ge=0, le=100_000)
    default_concurrency_cap: int = Field(default=1, ge=1, le=100_000)
    default_queue_size: int = Field(default=0, ge=0, le=100_000)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    queue_timeout_seconds: float = Field(default=5.0, gt=0.0, le=3600.0)
    per_tenant_limits: tuple[TenantLimitConfig, ...] = ()

    @model_validator(mode="after")
    def validate_tenants(self) -> AdmissionControlSettings:
        tenant_keys = [
            (
                limit.tenant_id,
                limit.request_class.value if limit.request_class is not None else None,
            )
            for limit in self.per_tenant_limits
        ]
        if len(tenant_keys) != len(set(tenant_keys)):
            msg = "per_tenant_limits must not contain duplicate tenant_id/request_class pairs"
            raise ValueError(msg)
        return self


class CircuitBreakerSettings(BaseModel):
    """Portable backend-protection defaults."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    failure_threshold: int = Field(default=5, ge=1, le=100_000)
    recovery_success_threshold: int = Field(default=1, ge=1, le=100_000)
    open_cooldown_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)


class SessionAffinitySettings(BaseModel):
    """Sticky-route defaults for multi-turn chat."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ttl_seconds: float = Field(default=300.0, gt=0.0, le=86_400.0)
    max_sessions: int = Field(default=10_000, ge=1, le=10_000_000)


class CanaryRoutingSettings(BaseModel):
    """Canary-routing defaults and explicit weighted policies."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    policies: tuple[CanaryPolicy, ...] = ()


class ShadowRoutingSettings(BaseModel):
    """Shadow-traffic defaults and explicit policies."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    policies: tuple[ShadowPolicy, ...] = ()


class PolicyRolloutSettings(BaseModel):
    """Local-first rollout controls for intelligent routing policies."""

    model_config = ConfigDict(extra="forbid")

    mode: PolicyRolloutMode = PolicyRolloutMode.DISABLED
    candidate_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    shadow_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    canary_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    kill_switch_enabled: bool = False
    learning_frozen: bool = False
    max_recent_decisions: int = Field(default=25, ge=1, le=200)


class HybridExecutionSettings(BaseModel):
    """Phase 7 hybrid local/remote execution guardrails and operator budgets."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    prefer_local: bool = True
    spillover_enabled: bool = False
    require_healthy_local_backends: bool = True
    max_remote_share_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    remote_request_budget_per_minute: int | None = Field(default=None, ge=1, le=1_000_000)
    allowed_remote_environments: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_remote_envs(self) -> HybridExecutionSettings:
        if len(self.allowed_remote_environments) != len(set(self.allowed_remote_environments)):
            msg = "allowed_remote_environments must not contain duplicate entries"
            raise ValueError(msg)
        return self


class RemoteWorkerLifecycleSettings(BaseModel):
    """Phase 7 worker registration and heartbeat posture."""

    model_config = ConfigDict(extra="forbid")

    secure_registration_required: bool = False
    dynamic_registration_enabled: bool = False
    heartbeat_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    registration_token_name: str | None = Field(default=None, min_length=1, max_length=128)
    allow_static_instances: bool = True


class Phase7ControlPlaneSettings(BaseModel):
    """Phase 7 hybrid execution and remote worker controls."""

    model_config = ConfigDict(extra="forbid")

    hybrid_execution: HybridExecutionSettings = Field(default_factory=HybridExecutionSettings)
    remote_workers: RemoteWorkerLifecycleSettings = Field(
        default_factory=RemoteWorkerLifecycleSettings
    )


class Phase4ControlPlaneSettings(BaseModel):
    """Phase 4 control-plane configuration."""

    model_config = ConfigDict(extra="forbid")

    feature_toggles: Phase4FeatureToggles = Field(default_factory=Phase4FeatureToggles)
    admission_control: AdmissionControlSettings = Field(
        default_factory=AdmissionControlSettings
    )
    circuit_breakers: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    session_affinity: SessionAffinitySettings = Field(default_factory=SessionAffinitySettings)
    canary_routing: CanaryRoutingSettings = Field(default_factory=CanaryRoutingSettings)
    shadow_routing: ShadowRoutingSettings = Field(default_factory=ShadowRoutingSettings)
    policy_rollout: PolicyRolloutSettings = Field(default_factory=PolicyRolloutSettings)

    @model_validator(mode="after")
    def sync_feature_toggles(self) -> Phase4ControlPlaneSettings:
        self.feature_toggles.admission_control_enabled = self.admission_control.enabled
        self.feature_toggles.circuit_breakers_enabled = self.circuit_breakers.enabled
        self.feature_toggles.session_affinity_enabled = self.session_affinity.enabled
        self.feature_toggles.canary_routing_enabled = self.canary_routing.enabled
        self.feature_toggles.shadow_routing_enabled = self.shadow_routing.enabled
        return self


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    env: AppEnvironment = AppEnvironment.DEVELOPMENT
    log_level: str = "INFO"
    service_name: str = "switchyard-gateway"
    otel_enabled: bool = False
    metrics_enabled: bool = False
    metrics_path: str = "/metrics"
    gateway_host: str = "127.0.0.1"
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    default_routing_policy: RoutingPolicy = RoutingPolicy.BALANCED
    benchmark_output_dir: Path = Path("artifacts/benchmarks")
    trace_capture_mode: TraceCaptureMode = TraceCaptureMode.OFF
    trace_capture_output_path: Path = Path("artifacts/traces/gateway-traces.jsonl")
    local_models: tuple[LocalModelConfig, ...] = ()
    default_model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    topology: DeploymentTopologySettings = Field(default_factory=DeploymentTopologySettings)
    phase4: Phase4ControlPlaneSettings = Field(default_factory=Phase4ControlPlaneSettings)
    phase7: Phase7ControlPlaneSettings = Field(default_factory=Phase7ControlPlaneSettings)

    model_config = SettingsConfigDict(
        env_prefix="SWITCHYARD_",
        env_file=".env",
        extra="ignore",
    )
