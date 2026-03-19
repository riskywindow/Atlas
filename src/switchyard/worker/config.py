"""Remote-worker runtime settings for Linux/container packaging."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from switchyard.config import GenerationDefaults, LocalModelConfig, WarmupSettings
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendType,
    CloudPlacementMetadata,
    CostBudgetProfile,
    CostProfileClass,
    DeploymentProfile,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
    GPUDeviceMetadata,
    NetworkCharacteristics,
    NetworkProfile,
    PerformanceHint,
    QualityHint,
    ReadinessHints,
    RequestFeatureSupport,
    RuntimeIdentity,
    TrustMetadata,
    WorkerAuthState,
    WorkerTransportType,
    WorkerTrustState,
)
from switchyard.schemas.worker import RemoteWorkerAuthMode
from switchyard.worker.fake import FakeRemoteWorkerConfig


class RemoteWorkerRuntimeSettings(BaseSettings):
    """Environment contract for a generic remote worker container/runtime."""

    model_config = SettingsConfigDict(
        env_prefix="SWITCHYARD_REMOTE_WORKER_",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8090, ge=1, le=65535)
    log_level: str = Field(default="INFO", min_length=1, max_length=32)
    worker_name: str = Field(default="stub-vllm-cuda-worker", min_length=1, max_length=128)
    worker_id: str = Field(default="stub-vllm-cuda-worker-1", min_length=1, max_length=128)
    serving_target: str = Field(default="chat-shared", min_length=1, max_length=128)
    model_identifier: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        min_length=1,
        max_length=512,
    )
    backend_type: BackendType = BackendType.VLLM_CUDA
    device_class: DeviceClass = DeviceClass.NVIDIA_GPU
    engine_type: EngineType = EngineType.VLLM_CUDA
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.REMOTE_WORKER
    health_state: BackendHealthState = BackendHealthState.HEALTHY
    supports_streaming: bool = True
    supports_native_streaming: bool = True
    supports_system_messages: bool = True
    supports_response_format_json: bool = False
    supports_tools: bool = False
    max_context_tokens: int = Field(default=32768, ge=1, le=10_000_000)
    concurrency_limit: int = Field(default=4, ge=1, le=4096)
    simulated_latency_ms: float = Field(default=0.0, ge=0.0, le=60000.0)
    simulated_active_requests: int = Field(default=0, ge=0, le=100000)
    simulated_queue_depth: int = Field(default=0, ge=0, le=100000)
    stream_chunk_size: int = Field(default=3, ge=1, le=4096)
    warmup_on_start: bool = False
    drain_timeout_seconds: float = Field(default=15.0, gt=0.0, le=3600.0)
    response_template: str = Field(
        default="stub remote response from {backend_name} for {request_id}",
        min_length=1,
        max_length=512,
    )
    runtime_version: str | None = Field(default=None, min_length=1, max_length=128)
    api_compatibility: str = Field(
        default="openai_chat_completions",
        min_length=1,
        max_length=128,
    )
    gpu_vendor: str = Field(default="nvidia", min_length=1, max_length=128)
    gpu_model: str | None = Field(default=None, min_length=1, max_length=128)
    gpu_count: int = Field(default=1, ge=0, le=4096)
    gpu_memory_per_device_gib: float | None = Field(default=None, ge=0.0)
    gpu_total_memory_gib: float | None = Field(default=None, ge=0.0)
    gpu_compute_capability: str | None = Field(default=None, min_length=1, max_length=64)
    gpu_interconnect: str | None = Field(default=None, min_length=1, max_length=64)
    driver_version: str | None = Field(default=None, min_length=1, max_length=128)
    cuda_version: str | None = Field(default=None, min_length=1, max_length=64)
    quality_hint: QualityHint = QualityHint.PREMIUM
    performance_hint: PerformanceHint = PerformanceHint.THROUGHPUT_OPTIMIZED
    supports_prefix_cache: bool = True
    supports_kv_cache_reuse: bool = True
    cold_start_likely: bool = False
    warm_pool_enabled: bool = False
    estimated_cold_start_ms: float | None = Field(default=None, ge=0.0)
    estimated_warm_ttl_seconds: float | None = Field(default=None, ge=0.0)
    cost_profile_class: CostProfileClass = CostProfileClass.PREMIUM
    relative_cost_index: float | None = Field(default=None, ge=0.0)
    environment: str = Field(default="staging", min_length=1, max_length=64)
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    region: str | None = Field(default=None, min_length=1, max_length=128)
    zone: str | None = Field(default=None, min_length=1, max_length=128)
    image_tag: str | None = Field(default=None, min_length=1, max_length=256)
    control_plane_url: str | None = Field(default=None, min_length=1, max_length=256)
    auth_mode: RemoteWorkerAuthMode = RemoteWorkerAuthMode.NONE
    registration_token: str | None = Field(default=None, min_length=1, max_length=256)
    enrollment_token: str | None = Field(default=None, min_length=1, max_length=2048)
    heartbeat_interval_seconds: float = Field(default=15.0, gt=0.0, le=3600.0)

    @model_validator(mode="after")
    def validate_auth_contract(self) -> RemoteWorkerRuntimeSettings:
        if self.auth_mode is RemoteWorkerAuthMode.STATIC_TOKEN and self.registration_token is None:
            msg = "registration_token is required when auth_mode=static_token"
            raise ValueError(msg)
        if (
            self.auth_mode is RemoteWorkerAuthMode.SIGNED_ENROLLMENT
            and self.enrollment_token is None
        ):
            msg = "enrollment_token is required when auth_mode=signed_enrollment"
            raise ValueError(msg)
        return self

    def to_fake_worker_config(self) -> FakeRemoteWorkerConfig:
        """Project the environment contract into the deterministic fake worker app."""

        return FakeRemoteWorkerConfig(
            worker_name=self.worker_name,
            serving_target=self.serving_target,
            model_identifier=self.model_identifier,
            backend_type=self.backend_type,
            device_class=self.device_class,
            engine_type=self.engine_type,
            execution_mode=self.execution_mode,
            runtime=self.runtime_identity(),
            gpu=self.gpu_metadata(),
            simulated_latency_ms=self.simulated_latency_ms,
            health_state=self.health_state,
            supports_streaming=self.supports_streaming,
            concurrency_limit=self.concurrency_limit,
            response_template=self.response_template,
            stream_chunk_size=self.stream_chunk_size,
            simulated_active_requests=self.simulated_active_requests,
            simulated_queue_depth=self.simulated_queue_depth,
            request_features=self.request_feature_support(),
        )

    def runtime_identity(self) -> RuntimeIdentity:
        """Return the typed runtime identity for this worker process."""

        return RuntimeIdentity(
            runtime_family=self.backend_type.value,
            runtime_label=self.engine_type.value,
            runtime_version=self.runtime_version,
            engine_type=self.engine_type,
            backend_type=self.backend_type,
            api_compatibility=self.api_compatibility,
            metadata={"dependency_mode": "optional_import"},
        )

    def gpu_metadata(self) -> GPUDeviceMetadata:
        """Return typed GPU metadata for Linux/NVIDIA worker hosts."""

        return GPUDeviceMetadata(
            accelerator_type="cuda",
            vendor=self.gpu_vendor,
            model=self.gpu_model,
            count=self.gpu_count,
            memory_per_device_gib=self.gpu_memory_per_device_gib,
            total_memory_gib=self.gpu_total_memory_gib,
            compute_capability=self.gpu_compute_capability,
            interconnect=self.gpu_interconnect,
            driver_version=self.driver_version,
            cuda_version=self.cuda_version,
        )

    def request_feature_support(self) -> RequestFeatureSupport:
        """Return supported request features for the worker runtime."""

        limitations: list[str] = []
        if not self.supports_tools:
            limitations.append(
                "Tool calling is intentionally disabled for the first "
                "vLLM-CUDA worker path."
            )
        if not self.supports_response_format_json:
            limitations.append(
                "Structured JSON response formatting remains feature-gated "
                "until later Phase 8 slices."
            )
        return RequestFeatureSupport(
            supports_streaming=self.supports_streaming,
            supports_native_streaming=self.supports_native_streaming,
            supports_tools=self.supports_tools,
            supports_system_messages=self.supports_system_messages,
            supports_response_format_json=self.supports_response_format_json,
            limitations=limitations,
        )

    def to_local_model_config(self) -> LocalModelConfig:
        """Project the worker runtime settings into the shared backend config schema."""

        return LocalModelConfig(
            alias=self.serving_target,
            serving_target=self.serving_target,
            environment=self.environment,
            deployment_profile=DeploymentProfile.REMOTE,
            model_identifier=self.model_identifier,
            backend_type=self.backend_type,
            runtime=self.runtime_identity(),
            gpu=self.gpu_metadata(),
            worker_transport=WorkerTransportType.IN_PROCESS,
            execution_mode=self.execution_mode,
            image_tag=self.image_tag,
            placement=CloudPlacementMetadata(
                provider=self.provider,
                region=self.region,
                zone=self.zone,
            ),
            cost_profile=CostBudgetProfile(
                profile=self.cost_profile_class,
                relative_cost_index=self.relative_cost_index,
            ),
            readiness_hints=ReadinessHints(
                cold_start_likely=self.cold_start_likely,
                warm_pool_enabled=self.warm_pool_enabled,
                estimated_cold_start_ms=self.estimated_cold_start_ms,
                estimated_warm_ttl_seconds=self.estimated_warm_ttl_seconds,
            ),
            trust=TrustMetadata(
                auth_state=(
                    WorkerAuthState.NONE
                    if self.auth_mode is RemoteWorkerAuthMode.NONE
                    else WorkerAuthState.STATIC_TOKEN
                ),
                trust_state=WorkerTrustState.UNKNOWN,
            ),
            network_characteristics=NetworkCharacteristics(
                profile=(
                    NetworkProfile.LAN
                    if self.control_plane_url is None
                    else NetworkProfile.WAN
                ),
            ),
            build_metadata={
                "worker_runtime": "vllm_cuda_remote_worker",
                "dependency_mode": "optional_import",
            },
            generation_defaults=GenerationDefaults(),
            warmup=WarmupSettings(enabled=True, eager=self.warmup_on_start),
        )
