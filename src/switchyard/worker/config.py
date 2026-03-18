"""Remote-worker runtime settings for Linux/container packaging."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from switchyard.schemas.backend import (
    BackendHealthState,
    BackendType,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
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
    engine_type: EngineType = EngineType.VLLM
    execution_mode: ExecutionModeLabel = ExecutionModeLabel.REMOTE_WORKER
    health_state: BackendHealthState = BackendHealthState.HEALTHY
    supports_streaming: bool = True
    concurrency_limit: int = Field(default=4, ge=1, le=4096)
    simulated_latency_ms: float = Field(default=0.0, ge=0.0, le=60000.0)
    simulated_active_requests: int = Field(default=0, ge=0, le=100000)
    simulated_queue_depth: int = Field(default=0, ge=0, le=100000)
    stream_chunk_size: int = Field(default=3, ge=1, le=4096)
    response_template: str = Field(
        default="stub remote response from {backend_name} for {request_id}",
        min_length=1,
        max_length=512,
    )
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
            simulated_latency_ms=self.simulated_latency_ms,
            health_state=self.health_state,
            supports_streaming=self.supports_streaming,
            concurrency_limit=self.concurrency_limit,
            response_template=self.response_template,
            stream_chunk_size=self.stream_chunk_size,
            simulated_active_requests=self.simulated_active_requests,
            simulated_queue_depth=self.simulated_queue_depth,
        )
