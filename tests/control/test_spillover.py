from __future__ import annotations

from datetime import UTC, datetime, timedelta

from switchyard.config import HybridExecutionSettings, RemoteTenantSpilloverRule
from switchyard.control.spillover import RemoteSpilloverControlService
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendDeployment,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
    EngineType,
    ExecutionModeLabel,
)
from switchyard.schemas.routing import (
    RequestClass,
    RequestContext,
    RoutingPolicy,
    TenantTier,
    WorkloadShape,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def monotonic(self) -> float:
        return 0.0

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def build_context() -> RequestContext:
    return RequestContext(
        request_id="req-spill",
        policy=RoutingPolicy.BURST_TO_REMOTE,
        workload_shape=WorkloadShape.INTERACTIVE,
        tenant_id="tenant-a",
    )


def build_remote_snapshot() -> BackendStatusSnapshot:
    return BackendStatusSnapshot(
        name="remote-a",
        deployment=BackendDeployment(
            name="remote-a",
            backend_type=BackendType.MOCK,
            engine_type=EngineType.MOCK,
            model_identifier="mock-chat",
            serving_targets=["mock-chat"],
            execution_mode=ExecutionModeLabel.REMOTE_WORKER,
        ),
        capabilities=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.REMOTE,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            execution_mode=ExecutionModeLabel.REMOTE_WORKER,
            concurrency_limit=4,
        ),
        health=BackendHealth(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            latency_ms=12.0,
        ),
    )


def test_spillover_priority_bypass_allows_denied_tenant_when_enabled() -> None:
    service = RemoteSpilloverControlService(
        HybridExecutionSettings(
            enabled=True,
            spillover_enabled=True,
            per_tenant_remote_spillover=(
                RemoteTenantSpilloverRule(
                    tenant_id="tenant-a",
                    remote_enabled=False,
                    allow_high_priority_bypass=True,
                ),
            ),
        )
    )
    context = build_context().model_copy(
        update={
            "tenant_tier": TenantTier.PRIORITY,
            "request_class": RequestClass.LATENCY_SENSITIVE,
        }
    )

    decision = service.evaluate_local_admission_failure(
        context=context,
        remote_candidates=[build_remote_snapshot()],
    )

    assert decision.allowed is True


def test_spillover_budget_window_recovers_after_one_minute() -> None:
    clock = FakeClock()
    service = RemoteSpilloverControlService(
        HybridExecutionSettings(
            enabled=True,
            spillover_enabled=True,
            remote_request_budget_per_minute=1,
        ),
        clock=clock,
    )
    context = build_context()
    remote_candidates = [build_remote_snapshot()]

    permit = service.acquire_remote_permit(
        context=context,
        backend_name="remote-a",
        remote_candidates=remote_candidates,
    )
    service.release_remote_permit(permit)

    blocked = service.evaluate_local_admission_failure(
        context=context,
        remote_candidates=remote_candidates,
    )
    assert blocked.allowed is False

    clock.advance(61.0)
    recovered = service.evaluate_local_admission_failure(
        context=context,
        remote_candidates=remote_candidates,
    )
    assert recovered.allowed is True
