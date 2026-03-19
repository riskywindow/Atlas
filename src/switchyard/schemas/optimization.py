"""Typed optimization-ready control-plane surfaces."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from switchyard.schemas.benchmark import CounterfactualObjective
from switchyard.schemas.routing import PolicyRolloutMode, RoutingPolicy


class OptimizationKnobType(StrEnum):
    """Portable type label for an exported optimization knob."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    ENUM = "enum"
    STRING_LIST = "string_list"


class OptimizationKnobSurface(BaseModel):
    """One exported optimization-ready knob with current value and bounds."""

    model_config = ConfigDict(extra="forbid")

    knob_id: str = Field(min_length=1, max_length=128)
    config_path: str = Field(min_length=1, max_length=256)
    knob_type: OptimizationKnobType
    current_value: bool | int | float | str | list[str] | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    allowed_values: list[str] = Field(default_factory=list)
    mutable_at_runtime: bool = False
    notes: list[str] = Field(default_factory=list)


class WorkerLaunchPresetScope(StrEnum):
    """Scope for one worker launch preset."""

    HOST_NATIVE = "host_native"
    REMOTE_WORKER = "remote_worker"


class WorkerLaunchPreset(BaseModel):
    """Typed worker launch preset that later benchmark loops may select from."""

    model_config = ConfigDict(extra="forbid")

    preset_name: str = Field(min_length=1, max_length=128)
    scope: WorkerLaunchPresetScope
    warmup_mode: str | None = Field(default=None, min_length=1, max_length=32)
    concurrency_limit: int | None = Field(default=None, ge=1, le=100_000)
    supports_streaming: bool | None = None
    stream_chunk_size: int | None = Field(default=None, ge=1, le=4096)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class OptimizationEvidenceProfile(BaseModel):
    """Offline evidence posture for later Stage A policy search."""

    model_config = ConfigDict(extra="forbid")

    objective: CounterfactualObjective = CounterfactualObjective.BALANCED
    min_evidence_count: int = Field(default=3, ge=1, le=100_000)
    max_predicted_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_predicted_latency_regression_ms: float | None = Field(default=None, ge=0.0)
    require_observed_backend_evidence: bool = False
    promotion_requires_operator_review: bool = True
    notes: list[str] = Field(default_factory=list)


class OptimizationProfile(BaseModel):
    """Exportable optimization surface for later Forge-style tuning."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    active_routing_policy: RoutingPolicy
    active_rollout_mode: PolicyRolloutMode
    allowlisted_routing_policies: list[RoutingPolicy] = Field(default_factory=list)
    allowlisted_rollout_modes: list[PolicyRolloutMode] = Field(default_factory=list)
    candidate_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    shadow_policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    hybrid_remote_enabled: bool = False
    worker_launch_presets: list[WorkerLaunchPreset] = Field(default_factory=list)
    evidence: OptimizationEvidenceProfile = Field(default_factory=OptimizationEvidenceProfile)
    knobs: list[OptimizationKnobSurface] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
