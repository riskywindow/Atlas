from __future__ import annotations

import asyncio

import pytest

from switchyard.config import AdmissionControlSettings, TenantLimitConfig
from switchyard.control.admission import (
    AdmissionControlService,
    AdmissionRejectedError,
)
from switchyard.schemas.routing import (
    AdmissionDecisionState,
    AdmissionReasonCode,
    RequestClass,
    RequestContext,
    RoutingPolicy,
    WorkloadShape,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _context(
    request_id: str,
    *,
    tenant_id: str = "default",
    request_class: RequestClass = RequestClass.STANDARD,
) -> RequestContext:
    return RequestContext(
        request_id=request_id,
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        tenant_id=tenant_id,
        request_class=request_class,
    )


@pytest.mark.asyncio
async def test_admission_rejects_when_global_capacity_and_queue_are_full() -> None:
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=1,
            global_queue_size=0,
            default_concurrency_cap=1,
        )
    )

    lease = await service.acquire(_context("req-1"), serving_target="chat")

    with pytest.raises(AdmissionRejectedError) as exc_info:
        await service.acquire(_context("req-2"), serving_target="chat")

    assert lease.decision.state is AdmissionDecisionState.ADMITTED
    assert exc_info.value.decision.reason_code is AdmissionReasonCode.GLOBAL_CONCURRENCY_LIMIT


@pytest.mark.asyncio
async def test_admission_enforces_per_tenant_caps() -> None:
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=3,
            global_queue_size=0,
            default_concurrency_cap=2,
            per_tenant_limits=(
                TenantLimitConfig(
                    tenant_id="tenant-a",
                    concurrency_cap=1,
                    queue_size=0,
                ),
            ),
        )
    )

    lease_a = await service.acquire(_context("req-a1", tenant_id="tenant-a"), serving_target="chat")
    lease_b = await service.acquire(_context("req-b1", tenant_id="tenant-b"), serving_target="chat")

    with pytest.raises(AdmissionRejectedError) as exc_info:
        await service.acquire(_context("req-a2", tenant_id="tenant-a"), serving_target="chat")

    assert lease_a.decision.state is AdmissionDecisionState.ADMITTED
    assert lease_b.decision.state is AdmissionDecisionState.ADMITTED
    assert exc_info.value.decision.reason_code is AdmissionReasonCode.TENANT_CONCURRENCY_LIMIT


@pytest.mark.asyncio
async def test_admission_queue_is_fifo_fair() -> None:
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=1,
            global_queue_size=2,
            default_concurrency_cap=1,
            queue_timeout_seconds=30.0,
        )
    )
    first = await service.acquire(_context("req-1"), serving_target="chat")
    second_task = asyncio.create_task(service.acquire(_context("req-2"), serving_target="chat"))
    third_task = asyncio.create_task(service.acquire(_context("req-3"), serving_target="chat"))
    await asyncio.sleep(0)

    await service.release(first)
    second = await asyncio.wait_for(second_task, timeout=0.1)

    assert second.decision.state is AdmissionDecisionState.QUEUED
    assert second.decision.queue_position == 1
    assert third_task.done() is False

    await service.release(second)
    third = await asyncio.wait_for(third_task, timeout=0.1)
    assert third.request_id == "req-3"


@pytest.mark.asyncio
async def test_admission_preserves_serving_target_in_queued_decision_snapshot() -> None:
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=1,
            global_queue_size=1,
            default_concurrency_cap=1,
            queue_timeout_seconds=30.0,
        )
    )

    first = await service.acquire(_context("req-1"), serving_target="chat-shared")
    queued_task = asyncio.create_task(
        service.acquire(_context("req-2"), serving_target="chat-shared")
    )
    await asyncio.sleep(0)
    await service.release(first)
    queued = await asyncio.wait_for(queued_task, timeout=0.1)

    assert queued.decision.state is AdmissionDecisionState.QUEUED
    assert queued.decision.queue_snapshot is not None
    assert queued.decision.queue_snapshot.queue_name == "chat-shared"


@pytest.mark.asyncio
async def test_admission_expires_stale_queued_requests_with_fake_clock() -> None:
    clock = FakeClock()
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=1,
            global_queue_size=1,
            default_concurrency_cap=1,
            queue_timeout_seconds=1.0,
        ),
        clock=clock,
    )
    first = await service.acquire(_context("req-1"), serving_target="chat")
    queued_task = asyncio.create_task(service.acquire(_context("req-2"), serving_target="chat"))
    await asyncio.sleep(0)

    clock.advance(2.0)
    await service.expire_stale_requests()

    with pytest.raises(AdmissionRejectedError) as exc_info:
        await queued_task

    assert exc_info.value.decision.reason_code is AdmissionReasonCode.QUEUE_TIMEOUT
    await service.release(first)


@pytest.mark.asyncio
async def test_admission_inspect_state_reports_live_tenant_limiter_summary() -> None:
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=3,
            global_queue_size=2,
            default_concurrency_cap=2,
            per_tenant_limits=(
                TenantLimitConfig(
                    tenant_id="tenant-a",
                    request_class=RequestClass.BULK,
                    concurrency_cap=1,
                    queue_size=0,
                ),
            ),
        )
    )

    lease = await service.acquire(
        _context("req-inspect", tenant_id="tenant-a", request_class=RequestClass.BULK),
        serving_target="chat",
    )
    snapshot = await service.inspect_state()

    assert snapshot.enabled is True
    assert snapshot.in_flight_total == 1
    assert snapshot.queued_requests == 0
    assert snapshot.tenant_limiters[0].tenant_id == "tenant-a"
    assert snapshot.tenant_limiters[0].request_class == "bulk"
    assert snapshot.tenant_limiters[0].in_flight_requests == 1
    assert snapshot.tenant_limiters[0].concurrency_cap == 1

    await service.release(lease)


@pytest.mark.asyncio
async def test_admission_inspect_state_includes_idle_configured_tenant_limiters() -> None:
    service = AdmissionControlService(
        AdmissionControlSettings(
            enabled=True,
            global_concurrency_cap=3,
            global_queue_size=2,
            default_concurrency_cap=2,
            per_tenant_limits=(
                TenantLimitConfig(
                    tenant_id="tenant-a",
                    request_class=RequestClass.BULK,
                    concurrency_cap=1,
                    queue_size=0,
                ),
                TenantLimitConfig(
                    tenant_id="tenant-b",
                    concurrency_cap=1,
                    queue_size=0,
                ),
            ),
        )
    )

    snapshot = await service.inspect_state()

    limiter_rows = [
        (item.tenant_id, item.request_class, item.in_flight_requests)
        for item in snapshot.tenant_limiters
    ]

    assert limiter_rows == [
        ("tenant-a", "bulk", 0),
        ("tenant-b", None, 0),
    ]
