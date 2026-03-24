"""Remote spillover guardrails for hybrid local/remote execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Protocol, cast

from switchyard.config import HybridExecutionSettings
from switchyard.schemas.admin import HybridExecutionRuntimeSummary
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    DeploymentProfile,
    DeviceClass,
    ExecutionModeLabel,
    WorkerLocalityClass,
)
from switchyard.schemas.routing import (
    AdmissionDecision,
    AdmissionDecisionState,
    AdmissionReasonCode,
    RequestClass,
    RequestContext,
    RouteSelectionReasonCode,
    RoutingPolicy,
    TenantTier,
)


class SpilloverClock(Protocol):
    """Clock abstraction for deterministic spillover tests."""

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


@dataclass(frozen=True, slots=True)
class SystemSpilloverClock:
    """Default clock implementation."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return monotonic()


@dataclass(frozen=True, slots=True)
class RemotePermit:
    """One remote concurrency/budget permit held during request execution."""

    request_id: str
    backend_name: str


_UNSET = object()


@dataclass(frozen=True, slots=True)
class HybridMutableRuntimeSettings:
    """Mutable hybrid guardrails that may be overridden at runtime."""

    spillover_enabled: bool
    max_remote_share_percent: float
    remote_request_budget_per_minute: int | None
    remote_concurrency_cap: int | None
    remote_cooldown_seconds: float


class SpilloverPermitRejectedError(RuntimeError):
    """Raised when execution-time remote spillover permit acquisition fails."""

    def __init__(self, decision: SpilloverGuardrailDecision) -> None:
        super().__init__(decision.reason or "remote spillover rejected")
        self.decision = decision


@dataclass(frozen=True, slots=True)
class SpilloverGuardrailDecision:
    """Result of evaluating a remote candidate or spillover opportunity."""

    allowed: bool
    reason_code: AdmissionReasonCode | None = None
    reason: str | None = None
    route_reason_code: RouteSelectionReasonCode | None = None
    guardrail_trigger: str | None = None


class RemoteSpilloverControlService:
    """Bounded remote spillover controls with simple local-first defaults."""

    def __init__(
        self,
        settings: HybridExecutionSettings,
        *,
        clock: SpilloverClock | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or SystemSpilloverClock()
        self._budget_window_started_at = self._clock.now()
        self._budget_requests_used = 0
        self._remote_in_flight_requests = 0
        self._cooldown_until: datetime | None = None
        self._remote_enabled_override: bool | None = None

    @property
    def enabled(self) -> bool:
        if self._remote_enabled_override is None:
            return self._settings.enabled
        return self._remote_enabled_override

    @property
    def remote_enabled_override(self) -> bool | None:
        return self._remote_enabled_override

    def set_remote_enabled(self, enabled: bool, *, reason: str | None = None) -> None:
        del reason
        self._remote_enabled_override = enabled

    def reset_budget_window(self) -> HybridExecutionRuntimeSummary:
        self._budget_window_started_at = self._clock.now()
        self._budget_requests_used = 0
        return self.inspect_state()

    def export_mutable_settings(self) -> HybridMutableRuntimeSettings:
        """Return mutable hybrid guardrails for later rollback."""

        return HybridMutableRuntimeSettings(
            spillover_enabled=self._settings.spillover_enabled,
            max_remote_share_percent=self._settings.max_remote_share_percent,
            remote_request_budget_per_minute=(
                self._settings.remote_request_budget_per_minute
            ),
            remote_concurrency_cap=self._settings.remote_concurrency_cap,
            remote_cooldown_seconds=self._settings.remote_cooldown_seconds,
        )

    def restore_mutable_settings(self, snapshot: HybridMutableRuntimeSettings) -> None:
        """Restore a previous mutable hybrid guardrail snapshot."""

        self.apply_mutable_settings(
            spillover_enabled=snapshot.spillover_enabled,
            max_remote_share_percent=snapshot.max_remote_share_percent,
            remote_request_budget_per_minute=snapshot.remote_request_budget_per_minute,
            remote_concurrency_cap=snapshot.remote_concurrency_cap,
            remote_cooldown_seconds=snapshot.remote_cooldown_seconds,
        )

    def apply_mutable_settings(
        self,
        *,
        spillover_enabled: bool | None = None,
        max_remote_share_percent: float | None = None,
        remote_request_budget_per_minute: int | None | object = _UNSET,
        remote_concurrency_cap: int | None | object = _UNSET,
        remote_cooldown_seconds: float | None = None,
    ) -> None:
        """Update the bounded runtime-mutable hybrid guardrails."""

        if spillover_enabled is not None:
            self._settings.spillover_enabled = spillover_enabled
        if max_remote_share_percent is not None:
            self._settings.max_remote_share_percent = max_remote_share_percent
        if remote_request_budget_per_minute is not _UNSET:
            self._settings.remote_request_budget_per_minute = cast(
                int | None,
                remote_request_budget_per_minute,
            )
        if remote_concurrency_cap is not _UNSET:
            self._settings.remote_concurrency_cap = cast(
                int | None,
                remote_concurrency_cap,
            )
        if remote_cooldown_seconds is not None:
            self._settings.remote_cooldown_seconds = remote_cooldown_seconds

    def remote_policy_enabled(self, policy: RoutingPolicy) -> bool:
        if policy in {RoutingPolicy.LOCAL_ONLY, RoutingPolicy.REMOTE_DISABLED}:
            return False
        return policy in {
            RoutingPolicy.LOCAL_PREFERRED,
            RoutingPolicy.BURST_TO_REMOTE,
            RoutingPolicy.LATENCY_SLO,
            RoutingPolicy.QUALITY_ON_DEMAND,
            RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY,
        }

    def evaluate_local_admission_failure(
        self,
        *,
        context: RequestContext,
        remote_candidates: list[BackendStatusSnapshot],
    ) -> SpilloverGuardrailDecision:
        if not self.enabled or not self._settings.spillover_enabled:
            return self._deny(
                AdmissionReasonCode.REMOTE_SPILLOVER_NOT_PERMITTED,
                "remote spillover is disabled",
                RouteSelectionReasonCode.REMOTE_TENANT_RESTRICTION,
                "spillover_disabled",
            )
        if not self.remote_policy_enabled(context.policy):
            return self._deny(
                AdmissionReasonCode.REMOTE_SPILLOVER_NOT_PERMITTED,
                "routing policy does not permit remote spillover",
                RouteSelectionReasonCode.REMOTE_TENANT_RESTRICTION,
                "policy_remote_disabled",
            )
        if not remote_candidates:
            return self._deny(
                AdmissionReasonCode.REMOTE_HEALTH_FAILURE,
                "no remote-capable backends are registered",
                RouteSelectionReasonCode.REMOTE_HEALTH_GUARDRAIL,
                "remote_absent",
            )
        return self._evaluate_common_guardrails(
            context=context,
            remote_candidates=remote_candidates,
            for_candidate=False,
        )

    def evaluate_remote_candidate(
        self,
        *,
        context: RequestContext,
        snapshot: BackendStatusSnapshot,
        remote_candidates: list[BackendStatusSnapshot],
    ) -> SpilloverGuardrailDecision:
        if not self.enabled or not self._is_remote_snapshot(snapshot):
            return SpilloverGuardrailDecision(allowed=True)
        if context.force_remote_candidates_only:
            decision = self._evaluate_common_guardrails(
                context=context,
                remote_candidates=remote_candidates,
                for_candidate=True,
            )
            if decision.allowed and self._is_escalated(context):
                return SpilloverGuardrailDecision(
                    allowed=True,
                    route_reason_code=RouteSelectionReasonCode.REMOTE_ESCALATION,
                )
            return decision
        if not self._settings.spillover_enabled or not self.remote_policy_enabled(context.policy):
            return self._deny(
                AdmissionReasonCode.REMOTE_SPILLOVER_NOT_PERMITTED,
                "remote spillover is not enabled for this policy",
                RouteSelectionReasonCode.REMOTE_TENANT_RESTRICTION,
                "policy_remote_disabled",
            )
        return self._evaluate_common_guardrails(
            context=context,
            remote_candidates=remote_candidates,
            for_candidate=True,
        )

    def acquire_remote_permit(
        self,
        *,
        context: RequestContext,
        backend_name: str,
        remote_candidates: list[BackendStatusSnapshot],
    ) -> RemotePermit:
        decision = self._evaluate_common_guardrails(
            context=context,
            remote_candidates=remote_candidates,
            for_candidate=False,
        )
        if not decision.allowed:
            raise SpilloverPermitRejectedError(decision)
        self._roll_budget_window()
        self._remote_in_flight_requests += 1
        self._budget_requests_used += 1
        return RemotePermit(request_id=context.request_id, backend_name=backend_name)

    def release_remote_permit(self, permit: RemotePermit | None) -> None:
        del permit
        self._remote_in_flight_requests = max(0, self._remote_in_flight_requests - 1)

    def record_remote_instability(self) -> None:
        if self._settings.remote_cooldown_seconds <= 0:
            return
        self._cooldown_until = self._clock.now() + timedelta(
            seconds=self._settings.remote_cooldown_seconds
        )

    def inspect_state(
        self,
        *,
        runtime_backends: list[BackendStatusSnapshot] | None = None,
    ) -> HybridExecutionRuntimeSummary:
        del runtime_backends
        self._roll_budget_window()
        budget_remaining = None
        if self._settings.remote_request_budget_per_minute is not None:
            budget_remaining = max(
                self._settings.remote_request_budget_per_minute - self._budget_requests_used,
                0,
            )
        cooldown_active = (
            self._cooldown_until is not None and self._cooldown_until > self._clock.now()
        )
        notes: list[str] = []
        if self._settings.remote_kill_switch_enabled:
            notes.append("remote spillover kill switch is active")
        if self._remote_enabled_override is False:
            notes.append("remote spillover is operator-disabled")
        elif self._remote_enabled_override is True and not self._settings.enabled:
            notes.append("remote spillover is operator-enabled despite config default")
        if cooldown_active:
            notes.append("remote spillover cooldown is active after recent instability")
        return HybridExecutionRuntimeSummary(
            enabled=self.enabled,
            prefer_local=self._settings.prefer_local,
            spillover_enabled=self._settings.spillover_enabled,
            require_healthy_local_backends=self._settings.require_healthy_local_backends,
            max_remote_share_percent=self._settings.max_remote_share_percent,
            remote_request_budget_per_minute=self._settings.remote_request_budget_per_minute,
            remote_concurrency_cap=self._settings.remote_concurrency_cap,
            remote_kill_switch_enabled=self._settings.remote_kill_switch_enabled,
            remote_cooldown_seconds=self._settings.remote_cooldown_seconds,
            allow_high_priority_remote_escalation=(
                self._settings.allow_high_priority_remote_escalation
            ),
            allowed_remote_environments=list(self._settings.allowed_remote_environments),
            tenant_remote_policy_count=len(self._settings.per_tenant_remote_spillover),
            remote_budget_window_started_at=self._budget_window_started_at,
            remote_budget_requests_used=self._budget_requests_used,
            remote_budget_requests_remaining=budget_remaining,
            remote_in_flight_requests=self._remote_in_flight_requests,
            cooldown_active=cooldown_active,
            cooldown_until=self._cooldown_until if cooldown_active else None,
            remote_policy_eligible=sorted(
                policy.value
                for policy in RoutingPolicy
                if self.remote_policy_enabled(policy)
            ),
            remote_policy_ineligible=sorted(
                policy.value
                for policy in RoutingPolicy
                if not self.remote_policy_enabled(policy)
            ),
            notes=notes,
        )

    def _evaluate_common_guardrails(
        self,
        *,
        context: RequestContext,
        remote_candidates: list[BackendStatusSnapshot],
        for_candidate: bool,
    ) -> SpilloverGuardrailDecision:
        self._roll_budget_window()
        escalated = self._is_escalated(context)
        if self._settings.remote_kill_switch_enabled:
            return self._deny(
                AdmissionReasonCode.REMOTE_KILL_SWITCH,
                "remote spillover kill switch is active",
                RouteSelectionReasonCode.REMOTE_KILL_SWITCH,
                "remote_kill_switch",
            )
        if not self._tenant_allows_remote(context=context, escalated=escalated):
            return self._deny(
                AdmissionReasonCode.REMOTE_SPILLOVER_NOT_PERMITTED,
                "tenant is not permitted to use remote spillover",
                RouteSelectionReasonCode.REMOTE_TENANT_RESTRICTION,
                "tenant_remote_denied",
            )
        healthy_candidates = [
            snapshot
            for snapshot in remote_candidates
            if self._remote_candidate_is_healthy(snapshot)
        ]
        if not healthy_candidates:
            return self._deny(
                AdmissionReasonCode.REMOTE_HEALTH_FAILURE,
                "no healthy remote spillover candidates are available",
                RouteSelectionReasonCode.REMOTE_HEALTH_GUARDRAIL,
                "remote_unhealthy",
            )
        if not escalated and self._cooldown_active():
            return self._deny(
                AdmissionReasonCode.REMOTE_COOLDOWN_ACTIVE,
                "remote spillover is cooling down after recent instability",
                RouteSelectionReasonCode.REMOTE_COOLDOWN,
                "remote_cooldown_active",
            )
        if (
            self._settings.remote_request_budget_per_minute is not None
            and self._budget_requests_used >= self._settings.remote_request_budget_per_minute
        ):
            return self._deny(
                AdmissionReasonCode.REMOTE_BUDGET_EXHAUSTED,
                "remote spillover budget is exhausted for the current minute",
                RouteSelectionReasonCode.REMOTE_BUDGET_GUARDRAIL,
                "remote_budget_exhausted",
            )
        if (
            self._settings.remote_concurrency_cap is not None
            and self._remote_in_flight_requests >= self._settings.remote_concurrency_cap
        ):
            return self._deny(
                AdmissionReasonCode.REMOTE_CONCURRENCY_LIMIT,
                "remote spillover concurrency cap is reached",
                RouteSelectionReasonCode.REMOTE_CONCURRENCY_GUARDRAIL,
                "remote_concurrency_exhausted",
            )
        if for_candidate and escalated:
            return SpilloverGuardrailDecision(
                allowed=True,
                route_reason_code=RouteSelectionReasonCode.REMOTE_ESCALATION,
            )
        return SpilloverGuardrailDecision(allowed=True)

    def _tenant_allows_remote(self, *, context: RequestContext, escalated: bool) -> bool:
        rule = next(
            (
                item
                for item in self._settings.per_tenant_remote_spillover
                if item.tenant_id == context.tenant_id
            ),
            None,
        )
        if rule is None:
            return True
        if rule.remote_enabled:
            return True
        return escalated and rule.allow_high_priority_bypass

    def _is_escalated(self, context: RequestContext) -> bool:
        if not self._settings.allow_high_priority_remote_escalation:
            return False
        return (
            context.tenant_tier is TenantTier.PRIORITY
            or context.request_class is RequestClass.LATENCY_SENSITIVE
        )

    def _cooldown_active(self) -> bool:
        return self._cooldown_until is not None and self._cooldown_until > self._clock.now()

    def _roll_budget_window(self) -> None:
        now = self._clock.now()
        if now - self._budget_window_started_at >= timedelta(minutes=1):
            self._budget_window_started_at = now
            self._budget_requests_used = 0

    def _remote_candidate_is_healthy(self, snapshot: BackendStatusSnapshot) -> bool:
        if not self._is_remote_snapshot(snapshot):
            return False
        if (
            self._settings.allowed_remote_environments
            and snapshot.deployment is not None
            and snapshot.deployment.environment not in self._settings.allowed_remote_environments
        ):
            return False
        return (
            snapshot.health.state is not BackendHealthState.UNAVAILABLE
            and snapshot.health.load_state is not BackendLoadState.FAILED
        )

    def _deny(
        self,
        reason_code: AdmissionReasonCode,
        reason: str,
        route_reason_code: RouteSelectionReasonCode,
        guardrail_trigger: str,
    ) -> SpilloverGuardrailDecision:
        return SpilloverGuardrailDecision(
            allowed=False,
            reason_code=reason_code,
            reason=reason,
            route_reason_code=route_reason_code,
            guardrail_trigger=guardrail_trigger,
        )

    @staticmethod
    def _is_remote_snapshot(snapshot: BackendStatusSnapshot) -> bool:
        if snapshot.capabilities.execution_mode in {
            ExecutionModeLabel.REMOTE_WORKER,
            ExecutionModeLabel.EXTERNAL_SERVICE,
        }:
            return True
        if snapshot.deployment is not None and snapshot.deployment.execution_mode in {
            ExecutionModeLabel.REMOTE_WORKER,
            ExecutionModeLabel.EXTERNAL_SERVICE,
        }:
            return True
        if (
            snapshot.deployment is not None
            and snapshot.deployment.deployment_profile is DeploymentProfile.REMOTE
        ):
            return True
        if snapshot.capabilities.device_class is DeviceClass.REMOTE:
            return True
        return any(
            instance.locality_class
            in {
                WorkerLocalityClass.REMOTE_PRIVATE,
                WorkerLocalityClass.REMOTE_CLOUD,
                WorkerLocalityClass.EXTERNAL_SERVICE,
            }
            for instance in snapshot.instance_inventory
        )


def spillover_bypass_decision(
    *,
    limiter_key: str,
    reason: str,
) -> AdmissionDecision:
    """Return the explicit admission decision used for remote spillover bypass."""

    return AdmissionDecision(
        state=AdmissionDecisionState.BYPASSED,
        reason_code=AdmissionReasonCode.LOCAL_ADMISSION_SPILLOVER_ELIGIBLE,
        reason=reason,
        limiter_key=limiter_key,
    )
