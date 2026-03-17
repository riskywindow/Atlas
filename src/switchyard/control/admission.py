"""In-process admission control and bounded queueing."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from switchyard.config import AdmissionControlSettings
from switchyard.schemas.admin import AdmissionRuntimeSummary, TenantLimiterRuntimeSummary
from switchyard.schemas.routing import (
    AdmissionDecision,
    AdmissionDecisionState,
    AdmissionReasonCode,
    LimiterMode,
    LimiterState,
    QueueSnapshot,
    RequestContext,
)


class AdmissionClock(Protocol):
    """Clock abstraction for deterministic admission testing."""

    def now(self) -> float: ...


@dataclass(frozen=True, slots=True)
class MonotonicAdmissionClock:
    """Default monotonic clock."""

    def now(self) -> float:
        return monotonic()


@dataclass(frozen=True, slots=True)
class AdmissionLease:
    """Active admission lease held for one in-flight request."""

    request_id: str
    tenant_id: str
    request_class: str
    decision: AdmissionDecision


class AdmissionRejectedError(RuntimeError):
    """Raised when admission control rejects a request."""

    def __init__(self, decision: AdmissionDecision, *, status_code: int = 429) -> None:
        message = decision.reason or "request rejected by admission control"
        super().__init__(message)
        self.decision = decision
        self.status_code = status_code


@dataclass(slots=True)
class _Waiter:
    request_id: str
    context: RequestContext
    serving_target: str
    enqueued_at: float
    expires_at: float
    queue_position: int
    future: asyncio.Future[AdmissionLease]


class AdmissionControlService:
    """Small in-process admission controller with a bounded FIFO queue."""

    def __init__(
        self,
        settings: AdmissionControlSettings,
        *,
        clock: AdmissionClock | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or MonotonicAdmissionClock()
        self._lock = asyncio.Lock()
        self._in_flight_total = 0
        self._in_flight_by_tenant: dict[tuple[str, str | None], int] = {}
        self._queue: deque[_Waiter] = deque()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    async def acquire(
        self,
        context: RequestContext,
        *,
        serving_target: str,
    ) -> AdmissionLease:
        """Acquire a lease or raise an explicit rejection."""

        if not self.enabled:
            return AdmissionLease(
                request_id=context.request_id,
                tenant_id=context.tenant_id,
                request_class=context.request_class.value,
                decision=AdmissionDecision(
                    state=AdmissionDecisionState.BYPASSED,
                    reason_code=AdmissionReasonCode.DISABLED,
                    reason="admission control disabled",
                    limiter_key=self._limiter_key(context),
                    queue_snapshot=self._queue_snapshot(serving_target=serving_target),
                ),
            )

        loop = asyncio.get_running_loop()
        async with self._lock:
            self._expire_waiters_locked(serving_target=serving_target)
            immediate_reason = self._blocking_reason_locked(context)
            if immediate_reason is None:
                decision = self._admit_locked(
                    context,
                    serving_target=serving_target,
                    queued=False,
                    queue_position=None,
                    enqueued_at=None,
                )
                return AdmissionLease(
                    request_id=context.request_id,
                    tenant_id=context.tenant_id,
                    request_class=context.request_class.value,
                    decision=decision,
                )
            if len(self._queue) >= self._settings.global_queue_size:
                raise AdmissionRejectedError(
                    self._rejection_decision(
                        context=context,
                        serving_target=serving_target,
                        reason_code=(
                            immediate_reason
                            if self._settings.global_queue_size == 0
                            else AdmissionReasonCode.QUEUE_FULL
                        ),
                        reason=(
                            "global concurrency cap reached"
                            if immediate_reason is AdmissionReasonCode.GLOBAL_CONCURRENCY_LIMIT
                            else (
                                "per-tenant concurrency cap reached"
                                if immediate_reason
                                is AdmissionReasonCode.TENANT_CONCURRENCY_LIMIT
                                else "admission queue is full"
                            )
                        ),
                    )
                )

            timeout_seconds = self._settings.queue_timeout_seconds
            now = self._clock.now()
            waiter = _Waiter(
                request_id=context.request_id,
                context=context,
                serving_target=serving_target,
                enqueued_at=now,
                expires_at=now + timeout_seconds,
                queue_position=len(self._queue) + 1,
                future=loop.create_future(),
            )
            self._queue.append(waiter)

        try:
            return await waiter.future
        finally:
            if waiter.future.cancelled():
                await self._remove_waiter(waiter)

    async def release(self, lease: AdmissionLease) -> None:
        """Release a previously admitted lease."""

        if not self.enabled:
            return
        async with self._lock:
            tenant_key = (lease.tenant_id, lease.request_class)
            self._in_flight_total = max(0, self._in_flight_total - 1)
            remaining = max(0, self._in_flight_by_tenant.get(tenant_key, 0) - 1)
            if remaining == 0:
                self._in_flight_by_tenant.pop(tenant_key, None)
            else:
                self._in_flight_by_tenant[tenant_key] = remaining
            self._expire_waiters_locked(serving_target=None)
            self._promote_waiters_locked()

    async def expire_stale_requests(self) -> None:
        """Expire queued requests using the injected clock."""

        async with self._lock:
            self._expire_waiters_locked(serving_target=None)

    async def snapshot(
        self,
        *,
        context: RequestContext,
        serving_target: str,
    ) -> LimiterState:
        """Return a limiter snapshot for logs, tests, and artifacts."""

        async with self._lock:
            return LimiterState(
                limiter_key=self._limiter_key(context),
                mode=LimiterMode.ENFORCING if self.enabled else LimiterMode.DISABLED,
                in_flight_requests=self._in_flight_by_tenant.get(self._tenant_key(context), 0),
                concurrency_limit=self._tenant_concurrency_cap(context),
                queue_snapshot=self._queue_snapshot(serving_target=serving_target),
            )

    def queue_depth(self) -> int:
        """Return the number of queued requests."""

        return len(self._queue)

    async def inspect_state(self) -> AdmissionRuntimeSummary:
        """Return a consistent runtime summary for local admin inspection."""

        async with self._lock:
            oldest_queue_age_ms = None
            if self._queue:
                oldest_queue_age_ms = (self._clock.now() - self._queue[0].enqueued_at) * 1000
            configured_limiters = {
                (
                    limit.tenant_id,
                    limit.request_class.value if limit.request_class is not None else None,
                ): limit.concurrency_cap
                for limit in self._settings.per_tenant_limits
            }
            active_keys = set(self._in_flight_by_tenant)
            limiter_keys = sorted(active_keys | set(configured_limiters))
            tenant_limiters = [
                TenantLimiterRuntimeSummary(
                    tenant_id=tenant_id,
                    request_class=request_class,
                    in_flight_requests=self._in_flight_by_tenant.get(
                        (tenant_id, request_class),
                        0,
                    ),
                    concurrency_cap=self._tenant_concurrency_cap_for(
                        tenant_id=tenant_id,
                        request_class=request_class,
                    ),
                )
                for tenant_id, request_class in limiter_keys
            ]
            return AdmissionRuntimeSummary(
                enabled=self.enabled,
                global_concurrency_cap=self._settings.global_concurrency_cap,
                global_queue_size=self._settings.global_queue_size,
                in_flight_total=self._in_flight_total,
                queued_requests=len(self._queue),
                oldest_queue_age_ms=oldest_queue_age_ms,
                tenant_limiters=tenant_limiters,
            )

    def _promote_waiters_locked(self) -> None:
        while self._queue:
            waiter = self._queue[0]
            if self._clock.now() >= waiter.expires_at:
                self._queue.popleft()
                if not waiter.future.done():
                    waiter.future.set_exception(
                        AdmissionRejectedError(
                            self._rejection_decision(
                                context=waiter.context,
                                serving_target=waiter.serving_target,
                                reason_code=AdmissionReasonCode.QUEUE_TIMEOUT,
                                reason="request expired while waiting in the admission queue",
                            )
                        )
                    )
                continue
            if self._blocking_reason_locked(waiter.context) is not None:
                break
            self._queue.popleft()
            decision = self._admit_locked(
                waiter.context,
                serving_target=waiter.serving_target,
                queued=True,
                queue_position=waiter.queue_position,
                enqueued_at=waiter.enqueued_at,
            )
            if not waiter.future.done():
                waiter.future.set_result(
                    AdmissionLease(
                        request_id=waiter.request_id,
                        tenant_id=waiter.context.tenant_id,
                        request_class=waiter.context.request_class.value,
                        decision=decision,
                    )
                )

    async def _remove_waiter(self, waiter: _Waiter) -> None:
        async with self._lock:
            try:
                self._queue.remove(waiter)
            except ValueError:
                return

    def _admit_locked(
        self,
        context: RequestContext,
        *,
        serving_target: str | None,
        queued: bool,
        queue_position: int | None,
        enqueued_at: float | None,
    ) -> AdmissionDecision:
        self._in_flight_total += 1
        tenant_key = self._tenant_key(context)
        self._in_flight_by_tenant[tenant_key] = self._in_flight_by_tenant.get(tenant_key, 0) + 1
        queue_wait_ms = None
        if enqueued_at is not None:
            queue_wait_ms = round((self._clock.now() - enqueued_at) * 1000, 3)
        return AdmissionDecision(
            state=AdmissionDecisionState.QUEUED if queued else AdmissionDecisionState.ADMITTED,
            limiter_key=self._limiter_key(context),
            queue_snapshot=self._queue_snapshot(serving_target=serving_target),
            queue_position=queue_position,
            request_timeout_ms=round(self._settings.request_timeout_seconds * 1000),
            queue_wait_ms=queue_wait_ms,
        )

    def _rejection_decision(
        self,
        *,
        context: RequestContext,
        serving_target: str | None,
        reason_code: AdmissionReasonCode,
        reason: str,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            state=AdmissionDecisionState.REJECTED,
            reason_code=reason_code,
            reason=reason,
            limiter_key=self._limiter_key(context),
            queue_snapshot=self._queue_snapshot(serving_target=serving_target),
            request_timeout_ms=round(self._settings.request_timeout_seconds * 1000),
        )

    def _blocking_reason_locked(self, context: RequestContext) -> AdmissionReasonCode | None:
        if self._in_flight_total >= self._settings.global_concurrency_cap:
            return AdmissionReasonCode.GLOBAL_CONCURRENCY_LIMIT
        if (
            self._in_flight_by_tenant.get(self._tenant_key(context), 0)
            >= self._tenant_concurrency_cap(context)
        ):
            return AdmissionReasonCode.TENANT_CONCURRENCY_LIMIT
        return None

    def _expire_waiters_locked(self, *, serving_target: str | None) -> None:
        now = self._clock.now()
        survivors: deque[_Waiter] = deque()
        while self._queue:
            waiter = self._queue.popleft()
            if now < waiter.expires_at:
                survivors.append(waiter)
                continue
            if not waiter.future.done():
                waiter.future.set_exception(
                    AdmissionRejectedError(
                        self._rejection_decision(
                            context=waiter.context,
                            serving_target=(
                                waiter.serving_target
                                if serving_target is None
                                else serving_target
                            ),
                            reason_code=AdmissionReasonCode.QUEUE_TIMEOUT,
                            reason="request expired while waiting in the admission queue",
                        )
                    )
                )
        self._queue = survivors

    def _tenant_concurrency_cap(self, context: RequestContext) -> int:
        return self._tenant_concurrency_cap_for(
            tenant_id=context.tenant_id,
            request_class=context.request_class.value,
        )

    def _tenant_concurrency_cap_for(self, *, tenant_id: str, request_class: str | None) -> int:
        for limit in self._settings.per_tenant_limits:
            if limit.tenant_id != tenant_id:
                continue
            if limit.request_class is None or limit.request_class.value == request_class:
                return limit.concurrency_cap
        return self._settings.default_concurrency_cap

    def _queue_snapshot(self, *, serving_target: str | None) -> QueueSnapshot:
        return QueueSnapshot(
            queue_name=serving_target or "admission",
            current_depth=len(self._queue),
            max_depth=self._settings.global_queue_size,
            queued_requests=len(self._queue),
            queue_timeout_ms=round(self._settings.queue_timeout_seconds * 1000),
        )

    def _limiter_key(self, context: RequestContext) -> str:
        return f"{context.tenant_id}:{context.request_class.value}"

    def _tenant_key(self, context: RequestContext) -> tuple[str, str | None]:
        return (context.tenant_id, context.request_class.value)
