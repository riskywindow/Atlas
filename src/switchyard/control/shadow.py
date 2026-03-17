"""Best-effort local shadow traffic helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING

from switchyard.adapters.base import BackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import ShadowRoutingSettings
from switchyard.logging import get_logger
from switchyard.schemas.admin import ShadowRoutingRuntimeSummary
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import (
    RequestContext,
    RouteAnnotations,
    RouteDecision,
    RouteSelectionReasonCode,
    ShadowDisposition,
    ShadowPolicy,
    ShadowRouteEvidence,
)
from switchyard.telemetry import BackendLabels, Telemetry

if TYPE_CHECKING:
    from switchyard.router.service import RouterService

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ShadowLaunchPlan:
    """Planned best-effort shadow dispatch for one primary request."""

    primary_request_id: str
    shadow_request_id: str
    policy: ShadowPolicy
    requested_model: str
    primary_backend_name: str


class ShadowTrafficService:
    """Lightweight local shadow traffic orchestration."""

    def __init__(self, settings: ShadowRoutingSettings) -> None:
        self._settings = settings
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def plan(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        decision: RouteDecision,
        primary_backend_name: str,
    ) -> ShadowLaunchPlan | None:
        """Evaluate whether a primary request should launch a shadow copy."""

        annotations = _ensure_shadow_annotations(decision)
        if not self.enabled:
            annotations.shadow_disposition = ShadowDisposition.DISABLED
            decision.shadow_decision = None
            return None
        if context.internal_backend_pin is not None:
            annotations.shadow_disposition = ShadowDisposition.SKIPPED
            annotations.notes.append("shadow traffic skipped for internally pinned requests")
            decision.shadow_decision = None
            return None

        policy = self._matching_policy(request=request, context=context)
        if policy is None:
            annotations.shadow_disposition = ShadowDisposition.SKIPPED
            decision.shadow_decision = None
            return None

        decision.shadow_policy = policy
        if not _sampled_in(request_id=context.request_id, policy=policy):
            annotations.shadow_disposition = ShadowDisposition.SKIPPED
            annotations.notes.append(
                f"shadow policy '{policy.policy_name}' skipped by sampling"
            )
            decision.shadow_decision = ShadowRouteEvidence(
                policy_name=policy.policy_name,
                disposition=ShadowDisposition.SKIPPED,
                target_backend=policy.target_backend,
                target_alias=policy.target_alias,
                sampling_rate=policy.sampling_rate,
                decision_reason="sampling_skipped",
            )
            return None

        if policy.target_backend == primary_backend_name:
            annotations.shadow_disposition = ShadowDisposition.SKIPPED
            annotations.notes.append(
                "shadow policy "
                f"'{policy.policy_name}' resolved to primary backend '{primary_backend_name}'"
            )
            decision.shadow_decision = ShadowRouteEvidence(
                policy_name=policy.policy_name,
                disposition=ShadowDisposition.SKIPPED,
                target_backend=policy.target_backend,
                target_alias=policy.target_alias,
                sampling_rate=policy.sampling_rate,
                decision_reason="resolved_to_primary_backend",
            )
            return None

        annotations.shadow_disposition = ShadowDisposition.SHADOWED
        annotations.notes.append(f"shadow policy '{policy.policy_name}' launched")
        if decision.telemetry_metadata is not None:
            decision.telemetry_metadata.shadow_enabled = True
        if (
            decision.explanation is not None
            and RouteSelectionReasonCode.SHADOW_LAUNCHED
            not in decision.explanation.selection_reason_codes
        ):
            decision.explanation.selection_reason_codes.append(
                RouteSelectionReasonCode.SHADOW_LAUNCHED
            )
        decision.shadow_decision = ShadowRouteEvidence(
            policy_name=policy.policy_name,
            disposition=ShadowDisposition.SHADOWED,
            target_backend=policy.target_backend,
            target_alias=policy.target_alias,
            sampling_rate=policy.sampling_rate,
            decision_reason="launched",
        )
        return ShadowLaunchPlan(
            primary_request_id=context.request_id,
            shadow_request_id=f"{context.request_id}:shadow:{policy.policy_name}",
            policy=policy,
            requested_model=request.model,
            primary_backend_name=primary_backend_name,
        )

    def launch(
        self,
        *,
        plan: ShadowLaunchPlan,
        request: ChatCompletionRequest,
        context: RequestContext,
        registry: AdapterRegistry,
        router: RouterService,
        telemetry: Telemetry,
    ) -> None:
        """Launch a best-effort shadow execution in the background."""

        if not self.enabled:
            return
        task = asyncio.create_task(
            self._run_shadow(
                plan=plan,
                request=request,
                context=context,
                registry=registry,
                router=router,
                telemetry=telemetry,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def execute_plan(
        self,
        *,
        plan: ShadowLaunchPlan,
        request: ChatCompletionRequest,
        context: RequestContext,
        registry: AdapterRegistry,
        router: RouterService,
        telemetry: Telemetry,
    ) -> None:
        """Execute one shadow plan inline for background-task integrations."""

        if not self.enabled:
            return
        await self._run_shadow(
            plan=plan,
            request=request,
            context=context,
            registry=registry,
            router=router,
            telemetry=telemetry,
        )

    async def wait_for_idle(self) -> None:
        """Wait for currently scheduled shadow executions."""

        if not self._tasks:
            return
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def inspect_state(self) -> ShadowRoutingRuntimeSummary:
        """Return active shadow configuration and current task count."""

        return ShadowRoutingRuntimeSummary(
            enabled=self.enabled,
            default_sampling_rate=self._settings.default_sampling_rate,
            active_tasks=len(self._tasks),
            policies=list(self._settings.policies),
        )

    def _matching_policy(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ShadowPolicy | None:
        for policy in self._settings.policies:
            if not policy.enabled:
                continue
            if policy.serving_target is not None and policy.serving_target != request.model:
                continue
            if policy.tenant_id is not None and policy.tenant_id != context.tenant_id:
                continue
            if (
                policy.request_class is not None
                and policy.request_class is not context.request_class
            ):
                continue
            return policy
        return None

    async def _run_shadow(
        self,
        *,
        plan: ShadowLaunchPlan,
        request: ChatCompletionRequest,
        context: RequestContext,
        registry: AdapterRegistry,
        router: RouterService,
        telemetry: Telemetry,
    ) -> None:
        launched_at = datetime.now(UTC)
        started = asyncio.get_running_loop().time()
        shadow_context = context.model_copy(
            update={
                "request_id": plan.shadow_request_id,
                "session_id": None,
            }
        )
        shadow_request = request.model_copy(
            update={
                "stream": False,
                "model": plan.policy.target_alias or request.model,
            }
        )
        shadow_target = plan.policy.target_backend or plan.policy.target_alias or "unknown"
        try:
            if plan.policy.target_backend is not None:
                adapter = registry.get(plan.policy.target_backend)
                backend_labels = await _resolve_shadow_labels(
                    adapter=adapter,
                    requested_model=shadow_request.model,
                )
                if backend_labels.backend_name == plan.primary_backend_name:
                    telemetry.record_shadow_execution(
                        primary_request_id=plan.primary_request_id,
                        shadow_request_id=plan.shadow_request_id,
                        policy_name=plan.policy.policy_name,
                        target_kind="backend",
                        configured_target=shadow_target,
                        resolved_backend_name=backend_labels.backend_name,
                        requested_model=shadow_request.model,
                        launched_at=launched_at,
                        success=False,
                        latency_ms=0.0,
                        error="shadow target resolved to primary backend",
                    )
                    logger.info(
                        "shadow_execution_skipped",
                        primary_request_id=plan.primary_request_id,
                        shadow_request_id=plan.shadow_request_id,
                        policy_name=plan.policy.policy_name,
                        shadow_target=shadow_target,
                        reason="shadow target resolved to primary backend",
                    )
                    return
                await adapter.generate(shadow_request, shadow_context)
                latency_ms = (asyncio.get_running_loop().time() - started) * 1000
                telemetry.record_shadow_execution(
                    primary_request_id=plan.primary_request_id,
                    shadow_request_id=plan.shadow_request_id,
                    policy_name=plan.policy.policy_name,
                    target_kind="backend",
                    configured_target=shadow_target,
                    resolved_backend_name=backend_labels.backend_name,
                    requested_model=shadow_request.model,
                    launched_at=launched_at,
                    success=True,
                    latency_ms=latency_ms,
                    error=None,
                )
                logger.info(
                    "shadow_execution_completed",
                    primary_request_id=plan.primary_request_id,
                    shadow_request_id=plan.shadow_request_id,
                    policy_name=plan.policy.policy_name,
                    shadow_target=shadow_target,
                    backend_name=backend_labels.backend_name,
                    latency_ms=round(latency_ms, 3),
                )
                return

            shadow_decision = await router.route(shadow_request, shadow_context)
            if shadow_decision.backend_name == plan.primary_backend_name:
                telemetry.record_shadow_execution(
                    primary_request_id=plan.primary_request_id,
                    shadow_request_id=plan.shadow_request_id,
                    policy_name=plan.policy.policy_name,
                    target_kind="alias",
                    configured_target=shadow_target,
                    resolved_backend_name=shadow_decision.backend_name,
                    requested_model=shadow_request.model,
                    launched_at=launched_at,
                    success=False,
                    latency_ms=0.0,
                    error="shadow target resolved to primary backend",
                )
                logger.info(
                    "shadow_execution_skipped",
                    primary_request_id=plan.primary_request_id,
                    shadow_request_id=plan.shadow_request_id,
                    policy_name=plan.policy.policy_name,
                    shadow_target=shadow_target,
                    reason="shadow target resolved to primary backend",
                )
                return
            adapter = registry.get(shadow_decision.backend_name)
            await adapter.generate(shadow_request, shadow_context)
            latency_ms = (asyncio.get_running_loop().time() - started) * 1000
            telemetry.record_shadow_execution(
                primary_request_id=plan.primary_request_id,
                shadow_request_id=plan.shadow_request_id,
                policy_name=plan.policy.policy_name,
                target_kind="alias",
                configured_target=shadow_target,
                resolved_backend_name=shadow_decision.backend_name,
                requested_model=shadow_request.model,
                launched_at=launched_at,
                success=True,
                latency_ms=latency_ms,
                error=None,
            )
            logger.info(
                "shadow_execution_completed",
                primary_request_id=plan.primary_request_id,
                shadow_request_id=plan.shadow_request_id,
                policy_name=plan.policy.policy_name,
                shadow_target=shadow_target,
                backend_name=shadow_decision.backend_name,
                latency_ms=round(latency_ms, 3),
            )
        except Exception as exc:
            latency_ms = (asyncio.get_running_loop().time() - started) * 1000
            telemetry.record_shadow_execution(
                primary_request_id=plan.primary_request_id,
                shadow_request_id=plan.shadow_request_id,
                policy_name=plan.policy.policy_name,
                target_kind="backend" if plan.policy.target_backend is not None else "alias",
                configured_target=shadow_target,
                resolved_backend_name=None,
                requested_model=shadow_request.model,
                launched_at=launched_at,
                success=False,
                latency_ms=latency_ms,
                error=str(exc),
            )
            logger.warning(
                "shadow_execution_failed",
                primary_request_id=plan.primary_request_id,
                shadow_request_id=plan.shadow_request_id,
                policy_name=plan.policy.policy_name,
                shadow_target=shadow_target,
                error=str(exc),
                latency_ms=round(latency_ms, 3),
            )


def _sampled_in(*, request_id: str, policy: ShadowPolicy) -> bool:
    if policy.sampling_rate >= 1.0:
        return True
    if policy.sampling_rate <= 0.0:
        return False
    digest = sha256(f"{policy.policy_name}:{request_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < policy.sampling_rate


def _ensure_shadow_annotations(decision: RouteDecision) -> RouteAnnotations:
    if decision.annotations is None:
        decision.annotations = RouteAnnotations()
    return decision.annotations


async def _resolve_shadow_labels(
    *,
    adapter: BackendAdapter,
    requested_model: str,
) -> BackendLabels:
    status = await adapter.status()
    default_model = status.capabilities.default_model or requested_model
    model_identifier = (
        status.capabilities.model_aliases.get(requested_model)
        or status.capabilities.model_aliases.get(default_model)
        or status.metadata.get("model_identifier")
        or default_model
    )
    return BackendLabels(
        backend_name=adapter.name,
        backend_type=status.capabilities.backend_type.value,
        model=requested_model,
        model_identifier=model_identifier,
        execution_mode=status.metadata.get("execution_mode", "in_process"),
        worker_transport=status.metadata.get("worker_transport", "in_process"),
    )
