"""Backend capability and health schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class BackendType(StrEnum):
    """Supported backend families."""

    MOCK = "mock"
    MLX_LM = "mlx_lm"
    VLLM_METAL = "vllm_metal"
    VLLM_CUDA = "vllm_cuda"
    REMOTE_OPENAI_LIKE = "remote_openai_like"


class DeviceClass(StrEnum):
    """Logical device classes that a backend can run on."""

    CPU = "cpu"
    APPLE_GPU = "apple_gpu"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    REMOTE = "remote"


class BackendHealthState(StrEnum):
    """High-level health states for a backend."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class BackendCapabilities(BaseModel):
    """Declared backend capabilities used by routing."""

    backend_type: BackendType
    device_class: DeviceClass
    model_ids: list[str] = Field(min_length=1)
    max_context_tokens: int = Field(ge=1)
    supports_streaming: bool = False
    concurrency_limit: int = Field(default=1, ge=1)
    quality_tier: int = Field(default=1, ge=1, le=5)


class BackendHealth(BaseModel):
    """Health signal for a backend adapter."""

    state: BackendHealthState
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str | None = Field(default=None, max_length=256)


class BackendStatusSnapshot(BaseModel):
    """Point-in-time backend status combining capabilities and health."""

    name: str = Field(min_length=1, max_length=128)
    capabilities: BackendCapabilities
    health: BackendHealth
    active_requests: int = Field(default=0, ge=0)
    queue_depth: int = Field(default=0, ge=0)
