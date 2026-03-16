"""Benchmark artifact schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import RoutingPolicy, WorkloadShape


class BenchmarkScenario(BaseModel):
    """Configuration for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    policy: RoutingPolicy
    workload_shape: WorkloadShape
    request_count: int = Field(ge=1)
    input_messages_per_request: int = Field(default=1, ge=1)


class BenchmarkRequestRecord(BaseModel):
    """Per-request benchmark result."""

    request_id: str = Field(min_length=1, max_length=128)
    backend_name: str = Field(min_length=1, max_length=128)
    started_at: datetime
    completed_at: datetime
    latency_ms: float = Field(ge=0.0)
    success: bool
    status_code: int = Field(ge=100, le=599)
    usage: UsageStats | None = None
    error: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def validate_outcome_fields(self) -> BenchmarkRequestRecord:
        if self.completed_at < self.started_at:
            msg = "completed_at must be greater than or equal to started_at"
            raise ValueError(msg)
        if self.success and self.error is not None:
            msg = "successful benchmark records cannot include an error message"
            raise ValueError(msg)
        if not self.success and self.error is None:
            msg = "failed benchmark records must include an error message"
            raise ValueError(msg)
        return self


class BenchmarkSummary(BaseModel):
    """Summary statistics for a benchmark run."""

    request_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    avg_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_counts(self) -> BenchmarkSummary:
        if self.success_count + self.failure_count != self.request_count:
            msg = "success_count + failure_count must equal request_count"
            raise ValueError(msg)
        return self


class BenchmarkRunArtifact(BaseModel):
    """Serializable benchmark artifact for reproducible analysis."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str | None = Field(default=None, min_length=7, max_length=40)
    scenario: BenchmarkScenario
    policy: RoutingPolicy
    backends_involved: list[str] = Field(min_length=1)
    request_count: int = Field(ge=0)
    summary: BenchmarkSummary
    records: list[BenchmarkRequestRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> BenchmarkRunArtifact:
        if self.request_count != len(self.records):
            msg = "request_count must match the number of records"
            raise ValueError(msg)
        if self.summary.request_count != self.request_count:
            msg = "summary.request_count must match request_count"
            raise ValueError(msg)
        return self
