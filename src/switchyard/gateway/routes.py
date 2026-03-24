"""Gateway routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol, cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import StreamingResponse

from switchyard.adapters.base import BackendAdapter
from switchyard.bench.campaigns import inspect_forge_stage_a_campaigns
from switchyard.control.admission import AdmissionLease, AdmissionRejectedError
from switchyard.control.shadow import ShadowLaunchPlan
from switchyard.control.spillover import (
    RemotePermit,
    SpilloverPermitRejectedError,
    spillover_bypass_decision,
)
from switchyard.diagnostics import (
    collect_deployment_diagnostics,
    collect_runtime_backend_summaries,
    summarize_hybrid_execution,
    summarize_remote_worker_lifecycle,
)
from switchyard.gateway.dependencies import GatewayServices, get_services
from switchyard.logging import bind_request_context, get_logger
from switchyard.optimization import build_forge_stage_a_campaign
from switchyard.router.features import routing_feature_runtime_summary
from switchyard.schemas.admin import (
    AliasCompatibilityRuntimeEntry,
    AliasRoutingOverrideRequest,
    AliasRoutingOverrideState,
    BackendRuntimeSummary,
    CloudRolloutRuntimeSummary,
    CloudRolloutUpdateRequest,
    DeploymentDiagnosticsResponse,
    HybridBudgetResetResponse,
    HybridBudgetRuntimeSummary,
    HybridOperatorRuntimeSummary,
    HybridRemoteToggleRequest,
    HybridRouteExample,
    PolicyRolloutRuntimeSummary,
    PolicyRolloutStateSnapshot,
    PolicyRolloutUpdateRequest,
    RemoteWorkerOperatorRequest,
    RuntimeInspectionResponse,
)
from switchyard.schemas.backend import BackendHealthState, BackendStatusSnapshot
from switchyard.schemas.benchmark import ExecutionTarget, ExecutionTargetType
from switchyard.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from switchyard.schemas.forge import (
    ForgeCampaignInspectionRequest,
    ForgeCampaignInspectionResponse,
    ForgePromotionApplyRequest,
    ForgePromotionCompareRequest,
    ForgePromotionDecisionRequest,
    ForgePromotionProposeRequest,
    ForgePromotionRuntimeSummary,
)
from switchyard.schemas.optimization import ForgeStageACampaign
from switchyard.schemas.routing import (
    AdmissionDecision,
    AdmissionDecisionState,
    AffinityDisposition,
    RequestClass,
    RequestContext,
    RolloutDisposition,
    RouteAnnotations,
    RouteDecision,
    RouteExecutionObservation,
    RouteSelectionReasonCode,
    RoutingPolicy,
    SessionAffinityKey,
    TenantTier,
    WorkloadShape,
)
from switchyard.schemas.worker import (
    RegisteredRemoteWorkerSnapshot,
    RemoteWorkerCleanupResponse,
    RemoteWorkerDeregisterRequest,
    RemoteWorkerHeartbeatRequest,
    RemoteWorkerRegistrationRequest,
    RemoteWorkerRegistrationResponse,
)
from switchyard.telemetry import BackendLabels, estimate_token_count

router = APIRouter()
logger = get_logger(__name__)


class InvalidRequestContextError(ValueError):
    """Raised when request-scoped routing metadata cannot be parsed."""


class BackendExecutionExhaustedError(RuntimeError):
    """Raised when all routed backend candidates fail execution."""


_FALLBACK_RETRY_BUDGET = 1


class _ObservedInstanceGeneratingAdapter(Protocol):
    async def generate_with_instance(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> tuple[ChatCompletionResponse, str | None]: ...


class _ObservedInstanceStreamingAdapter(Protocol):
    async def stream_generate_with_instance(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> tuple[str | None, AsyncIterator[ChatCompletionChunk]]: ...


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Basic liveness endpoint."""

    return {"status": "ok"}


@router.get("/readyz")
async def readyz(
    response: Response,
    services: GatewayServices = Depends(get_services),
) -> dict[str, object]:
    """Basic readiness endpoint."""

    if not services.registry.names():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "no adapters registered"}

    unhealthy: list[str] = []
    for adapter in services.registry.list():
        health = await adapter.health()
        services.telemetry.record_backend_health_snapshot(
            backend_name=adapter.name,
            health_state=health.state.value,
            latency_ms=health.latency_ms,
        )
        if health.state is BackendHealthState.UNAVAILABLE:
            unhealthy.append(adapter.name)

    if len(unhealthy) == len(services.registry.names()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "reason": "no healthy adapters", "adapters": unhealthy}

    return {"status": "ready", "adapters": services.registry.names()}


@router.get("/admin/runtime", response_model=RuntimeInspectionResponse)
async def admin_runtime(
    services: GatewayServices = Depends(get_services),
) -> RuntimeInspectionResponse:
    """Return a lightweight read-only snapshot of runtime control-plane state."""

    runtime_backends = await collect_runtime_backend_summaries(services.registry)
    remote_worker_snapshot = services.remote_workers.snapshot()
    hybrid_operator = _hybrid_operator_summary(
        services=services,
        runtime_backends=runtime_backends,
        remote_worker_snapshot=remote_worker_snapshot,
    )
    return RuntimeInspectionResponse(
        backends=runtime_backends,
        admission=await services.admission.inspect_state(),
        circuit_breakers=services.circuit_breaker.inspect_state(),
        canary_routing=services.canary.inspect_state(),
        shadow_routing=services.shadow.inspect_state(),
        policy_rollout=services.policy_rollout.inspect_state(),
        session_affinity=services.session_affinity.inspect_state(),
        hybrid_execution=summarize_hybrid_execution(
            settings=services.settings,
            runtime_backends=runtime_backends,
            spillover_runtime=services.spillover.inspect_state(),
        ),
        hybrid_operator=hybrid_operator,
        remote_workers=summarize_remote_worker_lifecycle(
            settings=services.settings,
            runtime_backends=runtime_backends,
            remote_worker_registry=remote_worker_snapshot,
        ),
        remote_worker_registry=remote_worker_snapshot,
        routing_features=routing_feature_runtime_summary(),
        prefix_locality=services.prefix_locality.inspect_state(),
    )


@router.get("/admin/policy-rollout", response_model=PolicyRolloutRuntimeSummary)
async def admin_policy_rollout(
    services: GatewayServices = Depends(get_services),
) -> PolicyRolloutRuntimeSummary:
    """Return current intelligent-policy rollout controls and recent decisions."""

    return services.policy_rollout.inspect_state()


@router.get("/admin/forge/stage-a", response_model=ForgeStageACampaign)
async def admin_forge_stage_a(
    services: GatewayServices = Depends(get_services),
) -> ForgeStageACampaign:
    """Return the typed Forge Stage A campaign inspection snapshot."""

    return build_forge_stage_a_campaign(services.settings)


@router.post(
    "/admin/forge/stage-a/campaigns/inspect",
    response_model=ForgeCampaignInspectionResponse,
)
async def inspect_admin_forge_stage_a_campaigns(
    request: ForgeCampaignInspectionRequest,
) -> ForgeCampaignInspectionResponse:
    """Summarize one or more Forge campaign artifacts for operator inspection."""

    return inspect_forge_stage_a_campaigns(
        campaign_artifacts=request.campaign_artifacts,
        comparison_artifacts=request.comparison_artifacts,
    )


@router.get(
    "/admin/forge/stage-a/promotion",
    response_model=ForgePromotionRuntimeSummary,
)
async def admin_forge_stage_a_promotion(
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Return the live Forge Stage A promotion and rollback posture."""

    return services.forge_promotion.inspect_state()


@router.post(
    "/admin/forge/stage-a/promotion/propose",
    response_model=ForgePromotionRuntimeSummary,
)
async def propose_admin_forge_stage_a_promotion(
    request: ForgePromotionProposeRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Propose one reviewed Forge Stage A trial as a promotion-ready profile."""

    try:
        return services.forge_promotion.propose(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/admin/forge/stage-a/promotion/approve",
    response_model=ForgePromotionRuntimeSummary,
)
async def approve_admin_forge_stage_a_promotion(
    request: ForgePromotionDecisionRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Approve one proposed Forge Stage A rollout without activating it yet."""

    try:
        return services.forge_promotion.approve(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/admin/forge/stage-a/promotion/apply",
    response_model=ForgePromotionRuntimeSummary,
)
async def apply_admin_forge_stage_a_promotion(
    request: ForgePromotionApplyRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Activate one approved Forge Stage A profile as a bounded canary."""

    try:
        return services.forge_promotion.apply(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/admin/forge/stage-a/promotion/compare",
    response_model=ForgePromotionRuntimeSummary,
)
async def compare_admin_forge_stage_a_promotion(
    request: ForgePromotionCompareRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Attach artifact-backed canary comparison evidence to the active rollout."""

    try:
        return services.forge_promotion.compare(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/admin/forge/stage-a/promotion/promote-default",
    response_model=ForgePromotionRuntimeSummary,
)
async def promote_default_admin_forge_stage_a_promotion(
    request: ForgePromotionDecisionRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Promote a compared canary profile to the runtime default."""

    try:
        return services.forge_promotion.promote_default(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/admin/forge/stage-a/promotion/reset",
    response_model=ForgePromotionRuntimeSummary,
)
async def reset_admin_forge_stage_a_promotion(
    request: ForgePromotionDecisionRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Reset the live Forge Stage A promotion controller to the baseline profile."""

    try:
        return services.forge_promotion.reset(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/admin/forge/stage-a/promotion/reject",
    response_model=ForgePromotionRuntimeSummary,
)
async def reject_admin_forge_stage_a_promotion(
    request: ForgePromotionDecisionRequest,
    services: GatewayServices = Depends(get_services),
) -> ForgePromotionRuntimeSummary:
    """Reject the current Forge Stage A rollout proposal or active canary."""

    try:
        return services.forge_promotion.reject(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/admin/policy-rollout", response_model=PolicyRolloutRuntimeSummary)
async def update_admin_policy_rollout(
    update: PolicyRolloutUpdateRequest,
    services: GatewayServices = Depends(get_services),
) -> PolicyRolloutRuntimeSummary:
    """Mutate local policy-rollout controls."""

    return services.policy_rollout.update(update)


@router.post("/admin/policy-rollout/reset", response_model=PolicyRolloutRuntimeSummary)
async def reset_admin_policy_rollout(
    services: GatewayServices = Depends(get_services),
) -> PolicyRolloutRuntimeSummary:
    """Reset local rollout state to its configured defaults."""

    return services.policy_rollout.reset_state()


@router.get("/admin/policy-rollout/export", response_model=PolicyRolloutStateSnapshot)
async def export_admin_policy_rollout(
    services: GatewayServices = Depends(get_services),
) -> PolicyRolloutStateSnapshot:
    """Export local rollout state for operator backup or transfer."""

    return services.policy_rollout.export_state()


@router.post("/admin/policy-rollout/import", response_model=PolicyRolloutRuntimeSummary)
async def import_admin_policy_rollout(
    snapshot: PolicyRolloutStateSnapshot,
    services: GatewayServices = Depends(get_services),
) -> PolicyRolloutRuntimeSummary:
    """Import a previously exported local rollout state."""

    return services.policy_rollout.import_state(snapshot)


@router.get("/admin/deployment", response_model=DeploymentDiagnosticsResponse)
async def admin_deployment(
    services: GatewayServices = Depends(get_services),
) -> DeploymentDiagnosticsResponse:
    """Return deployment-aware diagnostics rooted in current runtime truth."""

    return await collect_deployment_diagnostics(
        services.settings,
        registry=services.registry,
        remote_worker_registry=services.remote_workers.snapshot(),
    )


@router.get("/admin/remote-workers", response_model=RegisteredRemoteWorkerSnapshot)
async def admin_remote_workers(
    services: GatewayServices = Depends(get_services),
) -> RegisteredRemoteWorkerSnapshot:
    """Return the dynamic remote worker registration snapshot."""

    return services.remote_workers.snapshot()


@router.get("/admin/hybrid", response_model=HybridOperatorRuntimeSummary)
async def admin_hybrid(
    services: GatewayServices = Depends(get_services),
) -> HybridOperatorRuntimeSummary:
    """Return recent operator-facing hybrid routing state."""

    return _hybrid_operator_summary(services=services)


@router.get("/admin/hybrid/cloud-rollout", response_model=CloudRolloutRuntimeSummary)
async def admin_hybrid_cloud_rollout(
    services: GatewayServices = Depends(get_services),
) -> CloudRolloutRuntimeSummary:
    """Return runtime controls for canary-only cloud backends."""

    return services.cloud_rollout.inspect_state()


@router.post("/admin/hybrid/cloud-rollout", response_model=CloudRolloutRuntimeSummary)
async def update_hybrid_cloud_rollout(
    update: CloudRolloutUpdateRequest,
    services: GatewayServices = Depends(get_services),
) -> CloudRolloutRuntimeSummary:
    """Mutate runtime rollout posture for canary-only cloud backends."""

    return services.cloud_rollout.update(update)


@router.post("/admin/hybrid/remote-enabled", response_model=HybridOperatorRuntimeSummary)
async def update_hybrid_remote_enabled(
    update: HybridRemoteToggleRequest,
    services: GatewayServices = Depends(get_services),
) -> HybridOperatorRuntimeSummary:
    """Enable or disable remote hybrid routing at runtime."""

    services.spillover.set_remote_enabled(update.enabled, reason=update.reason)
    services.operator.set_remote_enabled_override(update.enabled)
    return _hybrid_operator_summary(services=services)


@router.get("/admin/hybrid/budget", response_model=HybridBudgetRuntimeSummary)
async def admin_hybrid_budget(
    services: GatewayServices = Depends(get_services),
) -> HybridBudgetRuntimeSummary:
    """Return remote budget, spend bucket, and cooldown posture."""

    return _hybrid_operator_summary(services=services).budget_state


@router.get("/admin/hybrid/aliases", response_model=list[AliasCompatibilityRuntimeEntry])
async def admin_hybrid_aliases(
    services: GatewayServices = Depends(get_services),
) -> list[AliasCompatibilityRuntimeEntry]:
    """Return alias compatibility and operator override posture for remote routing."""

    return _hybrid_operator_summary(services=services).alias_compatibility


@router.post(
    "/admin/hybrid/aliases/{serving_target}/override",
    response_model=AliasRoutingOverrideState,
)
async def update_hybrid_alias_override(
    serving_target: str,
    update: AliasRoutingOverrideRequest,
    services: GatewayServices = Depends(get_services),
) -> AliasRoutingOverrideState:
    """Pin or disable remote backends for one logical serving target."""

    try:
        return services.alias_overrides.update(
            serving_target=serving_target,
            pinned_backend=update.pinned_backend,
            disabled_backends=update.disabled_backends,
            reason=update.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/admin/hybrid/aliases/{serving_target}/override/reset",
    response_model=HybridOperatorRuntimeSummary,
)
async def reset_hybrid_alias_override(
    serving_target: str,
    services: GatewayServices = Depends(get_services),
) -> HybridOperatorRuntimeSummary:
    """Clear the alias-scoped routing override for one serving target."""

    services.alias_overrides.clear(serving_target)
    return _hybrid_operator_summary(services=services)


@router.post("/admin/hybrid/budget/reset", response_model=HybridBudgetResetResponse)
async def reset_hybrid_budget(
    services: GatewayServices = Depends(get_services),
) -> HybridBudgetResetResponse:
    """Reset the active remote budget window."""

    state = services.spillover.reset_budget_window()
    return HybridBudgetResetResponse(
        budget_window_started_at=state.remote_budget_window_started_at or datetime.now(UTC),
        remote_budget_requests_used=state.remote_budget_requests_used,
        remote_budget_requests_remaining=state.remote_budget_requests_remaining,
        notes=state.notes,
    )


@router.get("/admin/hybrid/routes", response_model=list[HybridRouteExample])
async def admin_hybrid_routes(
    services: GatewayServices = Depends(get_services),
) -> list[HybridRouteExample]:
    """Return recent operator-facing hybrid route examples wrapped for consistency."""

    return _hybrid_operator_summary(services=services).recent_route_examples


@router.post(
    "/admin/remote-workers/{worker_id}/drain",
    response_model=RemoteWorkerRegistrationResponse,
)
async def drain_remote_worker(
    worker_id: str,
    update: RemoteWorkerOperatorRequest,
    services: GatewayServices = Depends(get_services),
) -> RemoteWorkerRegistrationResponse:
    """Mark a registered remote worker as draining."""

    return services.remote_workers.mark_draining(worker_id, reason=update.reason)


@router.post(
    "/admin/remote-workers/{worker_id}/quarantine",
    response_model=RemoteWorkerRegistrationResponse,
)
async def quarantine_remote_worker(
    worker_id: str,
    update: RemoteWorkerOperatorRequest,
    services: GatewayServices = Depends(get_services),
) -> RemoteWorkerRegistrationResponse:
    """Mark or clear quarantine for a registered remote worker."""

    return services.remote_workers.set_quarantined(
        worker_id,
        enabled=update.enabled,
        reason=update.reason,
    )


@router.post(
    "/admin/remote-workers/{worker_id}/canary-only",
    response_model=RemoteWorkerRegistrationResponse,
)
async def canary_only_remote_worker(
    worker_id: str,
    update: RemoteWorkerOperatorRequest,
    services: GatewayServices = Depends(get_services),
) -> RemoteWorkerRegistrationResponse:
    """Mark or clear canary-only posture for a registered remote worker."""

    return services.remote_workers.set_canary_only(
        worker_id,
        enabled=update.enabled,
        reason=update.reason,
    )


@router.post(
    "/internal/control-plane/remote-workers/register",
    response_model=RemoteWorkerRegistrationResponse,
)
async def register_remote_worker(
    registration: RemoteWorkerRegistrationRequest,
    services: GatewayServices = Depends(get_services),
    x_switchyard_registration_token: str | None = Header(default=None),
) -> RemoteWorkerRegistrationResponse:
    """Register a remote worker instance with the control plane."""

    return services.remote_workers.register(
        registration,
        token=x_switchyard_registration_token,
    )


@router.post(
    "/internal/control-plane/remote-workers/heartbeat",
    response_model=RemoteWorkerRegistrationResponse,
)
async def heartbeat_remote_worker(
    heartbeat: RemoteWorkerHeartbeatRequest,
    services: GatewayServices = Depends(get_services),
    x_switchyard_lease_token: str | None = Header(default=None),
) -> RemoteWorkerRegistrationResponse:
    """Record a heartbeat for a previously registered remote worker."""

    return services.remote_workers.heartbeat(
        heartbeat,
        lease_token=x_switchyard_lease_token,
    )


@router.post(
    "/internal/control-plane/remote-workers/deregister",
    response_model=RemoteWorkerRegistrationResponse,
)
async def deregister_remote_worker(
    deregistration: RemoteWorkerDeregisterRequest,
    services: GatewayServices = Depends(get_services),
    x_switchyard_lease_token: str | None = Header(default=None),
) -> RemoteWorkerRegistrationResponse:
    """Gracefully de-register a remote worker."""

    return services.remote_workers.deregister(
        deregistration,
        lease_token=x_switchyard_lease_token,
    )


@router.post(
    "/admin/remote-workers/cleanup",
    response_model=RemoteWorkerCleanupResponse,
)
async def cleanup_remote_workers(
    services: GatewayServices = Depends(get_services),
) -> RemoteWorkerCleanupResponse:
    """Evict remote workers that have exceeded the stale retention window."""

    return services.remote_workers.cleanup_stale_workers()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    services: GatewayServices = Depends(get_services),
) -> ChatCompletionResponse | StreamingResponse:
    """Route a chat completion request and invoke the selected backend."""

    request_id = request.state.request_id
    request_timestamp = datetime.now(UTC)
    explicit_backend_pin = _resolve_internal_backend_pin(request)
    alias_override = (
        None
        if explicit_backend_pin is not None
        else services.alias_overrides.get(chat_request.model)
    )
    route_context = RequestContext(
        request_id=request_id,
        policy=_resolve_request_policy(
            request=request,
            default_policy=services.settings.default_routing_policy,
        ),
        workload_shape=_resolve_workload_shape(request),
        request_class=_resolve_request_class(request),
        internal_backend_pin=(
            explicit_backend_pin
            if explicit_backend_pin is not None
            else (None if alias_override is None else alias_override.pinned_backend)
        ),
        tenant_id=_resolve_tenant_id(request),
        tenant_tier=_resolve_tenant_tier(request),
        session_id=_resolve_session_id(request),
    )
    (
        blocked_backend_reasons,
        registered_cloud_rollout_decisions,
    ) = _blocked_backend_reasons(
        services=services,
        context=route_context,
        serving_target=chat_request.model,
        alias_override=alias_override,
    )
    route_context = route_context.model_copy(
        update={
            "blocked_backends": sorted(blocked_backend_reasons),
            "blocked_backend_reasons": blocked_backend_reasons,
        }
    )
    execution_target = _execution_target_from_context(
        model_alias=chat_request.model,
        context=route_context,
    )
    admission_lease = await _admit_request(
        chat_request=chat_request,
        context=route_context,
        services=services,
        request_timestamp=request_timestamp,
        execution_target=execution_target,
    )
    route_started = perf_counter()
    try:
        decision = await services.router.route(chat_request, route_context)
    except Exception as exc:
        await services.trace_capture.capture_chat_completion(
            request_timestamp=request_timestamp,
            request_id=request_id,
            logical_alias=chat_request.model,
            execution_target=execution_target,
            request_context=route_context,
            route_decision=None,
            chosen_backend=None,
            stream=chat_request.stream,
            status_code=503,
            latency_ms=(perf_counter() - route_started) * 1000,
            ttft_ms=None,
            output_tokens=None,
            fallback_used=False,
            error=str(exc),
            error_category="route_error",
            request_payload=chat_request,
            response_payload=None,
            admission_decision=admission_lease.decision,
        )
        await services.admission.release(admission_lease)
        raise
    decision.admission_decision = admission_lease.decision
    if decision.telemetry_metadata is not None:
        decision.telemetry_metadata.admission_control_enabled = services.admission.enabled
    _annotate_operator_override(
        decision=decision,
        alias_override=alias_override,
    )
    _annotate_registered_cloud_rollout(
        decision=decision,
        rollout_decisions=registered_cloud_rollout_decisions,
    )
    route_reason = (
        decision.explanation.compact_reason()
        if decision.explanation is not None
        else "; ".join(decision.rationale)
    )
    route_record = services.telemetry.record_route_decision(
        request_id=request_id,
        tenant_id=route_context.tenant_id,
        session_id=route_context.session_id,
        requested_model=chat_request.model,
        serving_target=decision.serving_target,
        policy=decision.policy.value,
        backend_name=decision.backend_name,
        considered_backends=decision.considered_backends,
        fallback_backends=decision.fallback_backends,
        rejected_backends=decision.rejected_backends,
        admission_limited_backends=decision.admission_limited_backends,
        protected_backends=decision.protected_backends,
        degraded_backends=decision.degraded_backends,
        route_reason=route_reason,
        route_latency_ms=(perf_counter() - route_started) * 1000,
    )
    logger.info(
        "route_decision",
        serving_target=decision.serving_target,
        chosen_backend=decision.backend_name,
        route_policy=decision.policy.value,
        tenant_id=route_context.tenant_id,
        tenant_tier=route_context.tenant_tier.value,
        request_class=route_context.request_class.value,
        session_id=route_context.session_id,
        requested_model=chat_request.model,
        candidate_backend_count=route_record.candidate_backend_count,
        considered_backends=decision.considered_backends,
        fallback_backends=decision.fallback_backends,
        admission_limited_backends=decision.admission_limited_backends,
        protected_backends=decision.protected_backends,
        degraded_backends=decision.degraded_backends,
        affinity_disposition=(
            decision.annotations.affinity_disposition.value
            if decision.annotations is not None
            else AffinityDisposition.NOT_REQUESTED.value
        ),
        sticky_backend=(
            decision.sticky_route.backend_name if decision.sticky_route is not None else None
        ),
        affinity_notes=decision.annotations.notes if decision.annotations is not None else [],
        rollout_disposition=(
            decision.annotations.rollout_disposition.value
            if decision.annotations is not None
            else RolloutDisposition.NONE.value
        ),
        canary_policy=(
            decision.canary_policy.policy_name if decision.canary_policy is not None else None
        ),
        fallback_occurred=route_record.fallback_occurred,
        route_reason=route_record.route_reason,
        route_latency_ms=route_record.route_latency_ms,
    )
    if decision.request_features is not None:
        services.prefix_locality.observe_request(
            serving_target=decision.serving_target,
            request_features=decision.request_features,
        )
    if chat_request.stream:
        return StreamingResponse(
            _stream_chat_completion(
                chat_request=chat_request,
                context=route_context,
                request=request,
                services=services,
                decision=decision,
                request_timestamp=request_timestamp,
                execution_target=execution_target,
                admission_lease=admission_lease,
            ),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "x-switchyard-admission-decision": admission_lease.decision.model_dump_json(
                    exclude_none=True
                ),
                **_route_metadata_headers(decision),
            },
        )

    try:
        chat_response, final_backend_name, shadow_plan = await _generate_with_fallback(
            chat_request=chat_request,
            context=route_context,
            services=services,
            decision=decision,
            route=request.url.path,
            method=request.method,
        )
    except Exception as exc:
        await services.trace_capture.capture_chat_completion(
            request_timestamp=request_timestamp,
            request_id=request_id,
            logical_alias=chat_request.model,
            execution_target=execution_target,
            request_context=route_context,
            route_decision=decision,
            chosen_backend=decision.explanation.executed_backend if decision.explanation else None,
            stream=False,
            status_code=503,
            latency_ms=(perf_counter() - route_started) * 1000,
            ttft_ms=None,
            output_tokens=None,
            fallback_used=decision.explanation.fallback_used if decision.explanation else False,
            error=str(exc),
            error_category="execution_error",
            request_payload=chat_request,
            response_payload=None,
        )
        await services.admission.release(admission_lease)
        raise
    await services.admission.release(admission_lease)
    execution = services.telemetry.state.backend_execution_records[-1]
    logger.info(
        "backend_execution_completed",
        streaming=False,
        chosen_backend=final_backend_name,
        response_id=chat_response.id,
        total_latency_ms=execution.total_latency_ms,
        ttft_ms=execution.ttft_ms,
        output_tokens=execution.output_tokens,
        tokens_per_second=execution.tokens_per_second,
    )
    logger.info(
        "chat_completion_succeeded",
        chosen_backend=final_backend_name,
        response_id=chat_response.id,
    )
    _schedule_shadow_traffic(
        background_tasks=background_tasks,
        services=services,
        chat_request=chat_request,
        context=route_context,
        decision=decision,
        primary_backend_name=final_backend_name,
        plan=shadow_plan,
    )
    for key, value in _route_metadata_headers(decision).items():
        response.headers[key] = value
    response.headers["x-switchyard-admission-decision"] = admission_lease.decision.model_dump_json(
        exclude_none=True
    )
    await services.trace_capture.capture_chat_completion(
        request_timestamp=request_timestamp,
        request_id=request_id,
        logical_alias=chat_request.model,
        execution_target=execution_target,
        request_context=route_context,
        route_decision=decision,
        chosen_backend=final_backend_name,
        stream=False,
        status_code=200,
        latency_ms=execution.total_latency_ms,
        ttft_ms=execution.ttft_ms,
        output_tokens=execution.output_tokens,
        fallback_used=decision.explanation.fallback_used if decision.explanation else False,
        error=None,
        error_category=None,
        request_payload=chat_request,
        response_payload=chat_response,
    )
    return chat_response


async def _stream_chat_completion(
    *,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
    request: Request,
    services: GatewayServices,
    decision: RouteDecision,
    request_timestamp: datetime,
    execution_target: ExecutionTarget,
    admission_lease: AdmissionLease,
) -> AsyncIterator[str]:
    decision_policy = decision.policy.value
    route_backends = _route_backends_for_execution(decision)
    stream_started = perf_counter()
    remote_permit: RemotePermit | None = None

    try:
        last_backend_instance_id: str | None = None
        for attempt_number, backend_name in enumerate(route_backends, start=1):
            adapter = services.registry.get(backend_name)
            (
                breaker_allowed,
                breaker_state,
                breaker_reason,
            ) = services.circuit_breaker.begin_execution(backend_name)
            if not breaker_allowed:
                decision.protected_backends[backend_name] = (
                    breaker_reason or "backend circuit is open"
                )
                services.telemetry.record_route_attempt(
                    request_id=context.request_id,
                    policy=decision_policy,
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    selected_by_router=attempt_number == 1,
                    outcome="skipped_circuit_open",
                    error=breaker_reason,
                )
                logger.warning(
                    "route_fallback_skipped_protected_backend",
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    breaker_phase=breaker_state.phase.value,
                    error=breaker_reason,
                )
                continue
            decision.circuit_breaker_state = breaker_state
            labels = await _resolve_backend_labels(
                adapter=adapter,
                requested_model=chat_request.model,
            )
            bind_request_context(
                chosen_backend=labels.backend_name,
                backend_type=labels.backend_type,
                model=labels.model,
                model_identifier=labels.model_identifier,
                execution_mode=labels.execution_mode,
                worker_transport=labels.worker_transport,
                route_policy=decision_policy,
            )
            if await _is_backend_unavailable(adapter):
                _append_execution_event(
                    decision,
                    f"attempt={attempt_number} backend={backend_name} outcome=skipped_unhealthy",
                )
                services.telemetry.record_route_attempt(
                    request_id=context.request_id,
                    policy=decision_policy,
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    selected_by_router=attempt_number == 1,
                    outcome="skipped_unhealthy",
                    error="backend health is unavailable",
                )
                if attempt_number == len(route_backends):
                    break
                logger.warning(
                    "route_fallback_skipped_unhealthy_backend",
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    fallback_used=attempt_number > 1,
                )
                continue

            logger.info("backend_execution_started", backend_name=backend_name, streaming=True)
            execution_start = perf_counter()
            ttft_ms: float | None = None
            output_fragments: list[str] = []
            response_id: str | None = None
            emitted_chunk = False
            backend_instance_id: str | None = None
            try:
                remote_permit = await _maybe_acquire_remote_permit(
                    services=services,
                    context=context,
                    serving_target=chat_request.model,
                    backend_name=backend_name,
                    permit=remote_permit,
                )
                backend_instance_id, chunk_stream = await _stream_backend_response(
                    adapter=adapter,
                    chat_request=chat_request,
                    context=context,
                )
                last_backend_instance_id = backend_instance_id
                async for chunk in chunk_stream:
                    emitted_chunk = True
                    response_id = chunk.id
                    if ttft_ms is None and _chunk_has_visible_tokens(chunk):
                        ttft_ms = (perf_counter() - execution_start) * 1000
                    output_fragments.extend(_chunk_content_fragments(chunk))
                    yield _format_sse_event(chunk)
            except Exception as exc:
                event = (
                    f"attempt={attempt_number} backend={backend_name} "
                    f"outcome=failed error={str(exc)}"
                )
                _append_execution_event(decision, event)
                services.telemetry.record_route_attempt(
                    request_id=context.request_id,
                    policy=decision_policy,
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    selected_by_router=attempt_number == 1,
                    outcome="failed",
                    error=str(exc),
                )
                logger.warning(
                    "route_fallback_backend_failed",
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    error=str(exc),
                    fallback_used=attempt_number > 1,
                )
                if emitted_chunk:
                    failed_latency_ms = (perf_counter() - execution_start) * 1000
                    failed_output_tokens = estimate_token_count("".join(output_fragments).strip())
                    _set_execution_outcome(
                        decision,
                        executed_backend=backend_name,
                        fallback_used=attempt_number > 1,
                        final_outcome="failed_after_stream_started",
                    )
                    _record_execution_observation(
                        services,
                        decision,
                        executed_backend=backend_name,
                        backend_instance_id=backend_instance_id,
                        latency_ms=failed_latency_ms,
                        ttft_ms=ttft_ms,
                        output_tokens=failed_output_tokens,
                        status_code=503,
                        error_category="stream_error",
                        final_outcome="failed_after_stream_started",
                    )
                    await services.trace_capture.capture_chat_completion(
                        request_timestamp=request_timestamp,
                        request_id=context.request_id,
                        logical_alias=chat_request.model,
                        execution_target=execution_target,
                        request_context=context,
                        route_decision=decision,
                        chosen_backend=backend_name,
                        stream=True,
                        status_code=503,
                        latency_ms=failed_latency_ms,
                        ttft_ms=ttft_ms,
                        output_tokens=failed_output_tokens,
                        fallback_used=attempt_number > 1,
                        error=str(exc),
                        error_category="stream_error",
                        request_payload=chat_request,
                        response_payload="".join(output_fragments).strip(),
                    )
                    services.circuit_breaker.record_failure(
                        backend_name,
                        reason=_classify_backend_failure(exc),
                    )
                    raise
                services.circuit_breaker.record_failure(
                    backend_name,
                    reason=_classify_backend_failure(exc),
                )
                if remote_permit is not None:
                    services.spillover.record_remote_instability()
                    quarantine_triggered = _record_remote_transport_failure(
                        services=services,
                        backend_name=backend_name,
                        error=str(exc),
                    )
                    services.operator.record_remote_transport_error(
                        request_id=context.request_id,
                        backend_name=backend_name,
                        error=str(exc),
                        cooldown_triggered=services.spillover.inspect_state().cooldown_active,
                        quarantine_triggered=quarantine_triggered,
                    )
                continue

            _append_execution_event(
                decision,
                f"attempt={attempt_number} backend={backend_name} outcome=succeeded",
            )
            services.telemetry.record_route_attempt(
                request_id=context.request_id,
                policy=decision_policy,
                backend_name=backend_name,
                attempt_number=attempt_number,
                selected_by_router=attempt_number == 1,
                outcome="succeeded",
            )
            execution = services.telemetry.record_backend_execution(
                route=request.url.path,
                method=request.method,
                status_code=200,
                streaming=True,
                labels=labels,
                total_latency_ms=(perf_counter() - execution_start) * 1000,
                ttft_ms=ttft_ms,
                output_tokens=estimate_token_count("".join(output_fragments).strip()),
            )
            logger.info(
                "backend_execution_completed",
                streaming=True,
                chosen_backend=backend_name,
                response_id=response_id,
                total_latency_ms=execution.total_latency_ms,
                ttft_ms=execution.ttft_ms,
                output_tokens=execution.output_tokens,
                tokens_per_second=execution.tokens_per_second,
            )
            logger.info(
                "chat_completion_stream_finished",
                chosen_backend=backend_name,
                route_policy=decision_policy,
                response_id=response_id,
                fallback_used=attempt_number > 1,
            )
            _set_execution_outcome(
                decision,
                executed_backend=backend_name,
                fallback_used=attempt_number > 1,
                final_outcome="succeeded",
            )
            shadow_plan = services.shadow.plan(
                request=chat_request,
                context=context,
                decision=decision,
                primary_backend_name=backend_name,
            )
            _record_execution_observation(
                services,
                decision,
                executed_backend=backend_name,
                backend_instance_id=backend_instance_id,
                latency_ms=execution.total_latency_ms,
                ttft_ms=execution.ttft_ms,
                output_tokens=execution.output_tokens,
                status_code=200,
                error_category=None,
                final_outcome="succeeded",
            )
            _update_session_affinity(
                services=services,
                context=context,
                decision=decision,
                backend_name=backend_name,
                fallback_used=attempt_number > 1,
            )
            _record_remote_backend_success(
                services=services,
                backend_name=backend_name,
            )
            _launch_streaming_shadow_traffic(
                services=services,
                chat_request=chat_request,
                context=context,
                decision=decision,
                primary_backend_name=backend_name,
                plan=shadow_plan,
            )
            await services.trace_capture.capture_chat_completion(
                request_timestamp=request_timestamp,
                request_id=context.request_id,
                logical_alias=chat_request.model,
                execution_target=execution_target,
                request_context=context,
                route_decision=decision,
                chosen_backend=backend_name,
                stream=True,
                status_code=200,
                latency_ms=execution.total_latency_ms,
                ttft_ms=execution.ttft_ms,
                output_tokens=execution.output_tokens,
                fallback_used=attempt_number > 1,
                error=None,
                error_category=None,
                request_payload=chat_request,
                response_payload="".join(output_fragments).strip(),
            )
            services.circuit_breaker.record_success(backend_name)
            yield "data: [DONE]\n\n"
            return

        msg = (
            f"all backend candidates failed for model '{chat_request.model}' "
            f"under policy '{decision_policy}'"
        )
        _set_execution_outcome(
            decision,
            executed_backend=None,
            fallback_used=bool(decision.fallback_backends),
            final_outcome="failed_no_safe_fallback",
        )
        _record_execution_observation(
            services,
            decision,
            executed_backend=None,
            backend_instance_id=last_backend_instance_id,
            latency_ms=(perf_counter() - stream_started) * 1000,
            ttft_ms=None,
            output_tokens=None,
            status_code=503,
            error_category="execution_error",
            final_outcome="failed_no_safe_fallback",
        )
        await services.trace_capture.capture_chat_completion(
            request_timestamp=request_timestamp,
            request_id=context.request_id,
            logical_alias=chat_request.model,
            execution_target=execution_target,
            request_context=context,
            route_decision=decision,
            chosen_backend=None,
            stream=True,
            status_code=503,
            latency_ms=(
                0.0
                if decision.execution_observation is None
                or decision.execution_observation.latency_ms is None
                else decision.execution_observation.latency_ms
            ),
            ttft_ms=None,
            output_tokens=None,
            fallback_used=bool(decision.fallback_backends),
            error=msg,
            error_category="execution_error",
            request_payload=chat_request,
            response_payload=None,
        )
        raise BackendExecutionExhaustedError(msg)
    finally:
        services.spillover.release_remote_permit(remote_permit)
        await services.admission.release(admission_lease)


async def _generate_with_fallback(
    *,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
    services: GatewayServices,
    decision: RouteDecision,
    route: str,
    method: str,
) -> tuple[ChatCompletionResponse, str, ShadowLaunchPlan | None]:
    decision_policy = decision.policy.value
    execution_errors: list[str] = []
    remote_permit: RemotePermit | None = None
    last_backend_instance_id: str | None = None

    try:
        for attempt_number, backend_name in enumerate(
            _route_backends_for_execution(decision), start=1
        ):
            adapter = services.registry.get(backend_name)
            (
                breaker_allowed,
                breaker_state,
                breaker_reason,
            ) = services.circuit_breaker.begin_execution(backend_name)
            if not breaker_allowed:
                decision.protected_backends[backend_name] = (
                    breaker_reason or "backend circuit is open"
                )
                services.telemetry.record_route_attempt(
                    request_id=context.request_id,
                    policy=decision_policy,
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    selected_by_router=attempt_number == 1,
                    outcome="skipped_circuit_open",
                    error=breaker_reason,
                )
                logger.warning(
                    "route_fallback_skipped_protected_backend",
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    breaker_phase=breaker_state.phase.value,
                    error=breaker_reason,
                )
                continue
            decision.circuit_breaker_state = breaker_state
            labels = await _resolve_backend_labels(
                adapter=adapter,
                requested_model=chat_request.model,
            )
            bind_request_context(
                chosen_backend=labels.backend_name,
                backend_type=labels.backend_type,
                model=labels.model,
                model_identifier=labels.model_identifier,
                execution_mode=labels.execution_mode,
                worker_transport=labels.worker_transport,
                route_policy=decision_policy,
            )
            if await _is_backend_unavailable(adapter):
                error = "backend health is unavailable"
                execution_errors.append(f"{backend_name}: {error}")
                _append_execution_event(
                    decision,
                    f"attempt={attempt_number} backend={backend_name} outcome=skipped_unhealthy",
                )
                services.telemetry.record_route_attempt(
                    request_id=context.request_id,
                    policy=decision_policy,
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    selected_by_router=attempt_number == 1,
                    outcome="skipped_unhealthy",
                    error=error,
                )
                logger.warning(
                    "route_fallback_skipped_unhealthy_backend",
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    fallback_used=attempt_number > 1,
                )
                continue

            logger.info("backend_execution_started", backend_name=backend_name, streaming=False)
            execution_start = perf_counter()
            backend_instance_id: str | None = None
            try:
                remote_permit = await _maybe_acquire_remote_permit(
                    services=services,
                    context=context,
                    serving_target=chat_request.model,
                    backend_name=backend_name,
                    permit=remote_permit,
                )
                chat_response, backend_instance_id = await _generate_backend_response(
                    adapter=adapter,
                    chat_request=chat_request,
                    context=context,
                )
                last_backend_instance_id = backend_instance_id
            except Exception as exc:
                execution_errors.append(f"{backend_name}: {exc}")
                _append_execution_event(
                    decision,
                    "attempt="
                    f"{attempt_number} backend={backend_name} "
                    f"outcome=failed error={str(exc)}",
                )
                services.telemetry.record_route_attempt(
                    request_id=context.request_id,
                    policy=decision_policy,
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    selected_by_router=attempt_number == 1,
                    outcome="failed",
                    error=str(exc),
                )
                logger.warning(
                    "route_fallback_backend_failed",
                    backend_name=backend_name,
                    attempt_number=attempt_number,
                    error=str(exc),
                    fallback_used=attempt_number > 1,
                )
                services.circuit_breaker.record_failure(
                    backend_name,
                    reason=_classify_backend_failure(exc),
                )
                if remote_permit is not None:
                    services.spillover.record_remote_instability()
                    quarantine_triggered = _record_remote_transport_failure(
                        services=services,
                        backend_name=backend_name,
                        error=str(exc),
                    )
                    services.operator.record_remote_transport_error(
                        request_id=context.request_id,
                        backend_name=backend_name,
                        error=str(exc),
                        cooldown_triggered=services.spillover.inspect_state().cooldown_active,
                        quarantine_triggered=quarantine_triggered,
                    )
                continue

            _append_execution_event(
                decision,
                f"attempt={attempt_number} backend={backend_name} outcome=succeeded",
            )
            services.telemetry.record_route_attempt(
                request_id=context.request_id,
                policy=decision_policy,
                backend_name=backend_name,
                attempt_number=attempt_number,
                selected_by_router=attempt_number == 1,
                outcome="succeeded",
            )
            execution = services.telemetry.record_backend_execution(
                route=route,
                method=method,
                status_code=200,
                streaming=False,
                labels=labels,
                total_latency_ms=(perf_counter() - execution_start) * 1000,
                ttft_ms=None,
                output_tokens=chat_response.usage.completion_tokens,
            )
            _set_execution_outcome(
                decision,
                executed_backend=backend_name,
                fallback_used=attempt_number > 1,
                final_outcome="succeeded",
            )
            shadow_plan = services.shadow.plan(
                request=chat_request,
                context=context,
                decision=decision,
                primary_backend_name=backend_name,
            )
            _record_execution_observation(
                services,
                decision,
                executed_backend=backend_name,
                backend_instance_id=backend_instance_id,
                latency_ms=execution.total_latency_ms,
                ttft_ms=execution.ttft_ms,
                output_tokens=execution.output_tokens,
                status_code=200,
                error_category=None,
                final_outcome="succeeded",
            )
            _update_session_affinity(
                services=services,
                context=context,
                decision=decision,
                backend_name=backend_name,
                fallback_used=attempt_number > 1,
            )
            _record_remote_backend_success(
                services=services,
                backend_name=backend_name,
            )
            services.circuit_breaker.record_success(backend_name)
            return chat_response, backend_name, shadow_plan
    finally:
        services.spillover.release_remote_permit(remote_permit)

    msg = (
        f"all backend candidates failed for model '{chat_request.model}' "
        f"under policy '{decision_policy}': {'; '.join(execution_errors)}"
    )
    _set_execution_outcome(
        decision,
        executed_backend=None,
        fallback_used=bool(decision.fallback_backends),
        final_outcome="failed_no_safe_fallback",
    )
    _record_execution_observation(
        services,
        decision,
        executed_backend=None,
        backend_instance_id=last_backend_instance_id,
        latency_ms=None,
        ttft_ms=None,
        output_tokens=None,
        status_code=503,
        error_category="execution_error",
        final_outcome="failed_no_safe_fallback",
    )
    raise BackendExecutionExhaustedError(msg)


def _format_sse_event(chunk: ChatCompletionChunk) -> str:
    payload = json.dumps(chunk.model_dump(mode="json"), separators=(",", ":"))
    return f"data: {payload}\n\n"


async def _generate_backend_response(
    *,
    adapter: BackendAdapter,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
) -> tuple[ChatCompletionResponse, str | None]:
    generate_with_instance = getattr(adapter, "generate_with_instance", None)
    if callable(generate_with_instance):
        return await cast(
            _ObservedInstanceGeneratingAdapter,
            adapter,
        ).generate_with_instance(chat_request, context)
    return await adapter.generate(chat_request, context), None


async def _stream_backend_response(
    *,
    adapter: BackendAdapter,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
) -> tuple[str | None, AsyncIterator[ChatCompletionChunk]]:
    stream_generate_with_instance = getattr(adapter, "stream_generate_with_instance", None)
    if callable(stream_generate_with_instance):
        return await cast(
            _ObservedInstanceStreamingAdapter,
            adapter,
        ).stream_generate_with_instance(chat_request, context)
    return None, adapter.stream_generate(chat_request, context)


async def _resolve_backend_labels(
    *,
    adapter: BackendAdapter,
    requested_model: str,
) -> BackendLabels:
    status = await adapter.status()
    logical_model = status.capabilities.resolve_logical_model(requested_model)
    model_identifier = (
        (None if logical_model is None else logical_model.model_identifier)
        or status.metadata.get("model_identifier")
        or status.capabilities.default_model
        or requested_model
    )
    return BackendLabels(
        backend_name=adapter.name,
        backend_type=status.capabilities.backend_type.value,
        model=requested_model,
        model_identifier=model_identifier,
        execution_mode=status.metadata.get("execution_mode", "in_process"),
        worker_transport=status.metadata.get("worker_transport", "in_process"),
    )


async def _is_backend_unavailable(adapter: BackendAdapter) -> bool:
    health = await adapter.health()
    return health.state is BackendHealthState.UNAVAILABLE


def _route_backends_for_execution(decision: RouteDecision) -> list[str]:
    return [decision.backend_name, *decision.fallback_backends[:_FALLBACK_RETRY_BUDGET]]


def _classify_backend_failure(exc: Exception) -> str:
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout_like_failure"
    return "invocation_failure"


def _append_execution_event(decision: RouteDecision, event: str) -> None:
    if decision.explanation is None:
        return
    decision.explanation.execution_events.append(event)


def _set_execution_outcome(
    decision: RouteDecision,
    *,
    executed_backend: str | None,
    fallback_used: bool,
    final_outcome: str,
) -> None:
    if decision.explanation is None:
        return
    decision.explanation.executed_backend = executed_backend
    decision.explanation.fallback_used = fallback_used
    decision.explanation.final_outcome = final_outcome
    if (
        fallback_used
        and RouteSelectionReasonCode.FALLBACK_EXECUTION
        not in decision.explanation.selection_reason_codes
    ):
        decision.explanation.selection_reason_codes.append(
            RouteSelectionReasonCode.FALLBACK_EXECUTION
        )


def _record_execution_observation(
    services: GatewayServices,
    decision: RouteDecision,
    *,
    executed_backend: str | None,
    backend_instance_id: str | None = None,
    latency_ms: float | None,
    ttft_ms: float | None,
    output_tokens: int | None,
    status_code: int | None,
    error_category: str | None,
    final_outcome: str,
) -> None:
    queue_delay_ms = (
        None if decision.admission_decision is None else decision.admission_decision.queue_wait_ms
    )
    decision.execution_observation = RouteExecutionObservation(
        executed_backend=executed_backend,
        backend_instance_id=backend_instance_id,
        queue_delay_ms=queue_delay_ms,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        output_tokens=output_tokens,
        status_code=status_code,
        error_category=error_category,
        final_outcome=final_outcome,
    )
    remote_candidate_count = sum(
        1 for candidate in decision.considered_backends if candidate.startswith("remote-worker:")
    )
    services.operator.record_route_example(
        decision=decision,
        executed_backend=executed_backend,
        remote_candidate_count=remote_candidate_count,
        final_outcome=final_outcome,
        fallback_used=executed_backend is not None and executed_backend != decision.backend_name,
    )
    if (
        final_outcome == "succeeded"
        and decision.request_features is not None
        and decision.request_features.prefix_fingerprint is not None
    ):
        services.prefix_locality.observe_execution(
            serving_target=decision.serving_target,
            request_features=decision.request_features,
            backend_name=executed_backend,
            backend_instance_id=decision.execution_observation.backend_instance_id,
        )


def _record_remote_backend_success(
    *,
    services: GatewayServices,
    backend_name: str,
) -> None:
    if backend_name.startswith("remote-worker:"):
        services.cloud_rollout.record_backend_success(backend_name)


def _record_remote_transport_failure(
    *,
    services: GatewayServices,
    backend_name: str,
    error: str,
) -> bool:
    if not backend_name.startswith("remote-worker:"):
        return False
    threshold_reached = services.cloud_rollout.record_backend_failure(backend_name)
    if not threshold_reached:
        return False
    for worker in services.remote_workers.snapshot().workers:
        if worker.backend_name != backend_name or worker.quarantined:
            continue
        services.remote_workers.set_quarantined(
            worker.worker_id,
            enabled=True,
            reason=f"auto-quarantined after repeated transport failures: {error[:160]}",
        )
        return True
    return False


def _update_session_affinity(
    *,
    services: GatewayServices,
    context: RequestContext,
    decision: RouteDecision,
    backend_name: str,
    fallback_used: bool,
) -> None:
    if context.session_id is None or context.internal_backend_pin is not None:
        return
    if not services.session_affinity.enabled:
        return
    annotations = _ensure_route_annotations(decision)
    skip_reason = None
    if decision.canary_policy is not None and decision.canary_policy.enabled:
        skip_reason = "canary routing is active"
    elif annotations.operator_override_applied:
        skip_reason = "operator override selected the current path"
    elif annotations.cloud_routing_reason == "spillover":
        skip_reason = "spillover selected the current path"
    elif annotations.cloud_rollout_disposition in {"selected", "explicit_canary"}:
        skip_reason = "cloud rollout selected the current path"
    if skip_reason is not None:
        _ensure_route_annotations(decision).notes.append(
            f"session affinity update skipped because {skip_reason}"
        )
        return
    affinity_key = decision.session_affinity_key or SessionAffinityKey(
        tenant_id=context.tenant_id,
        session_id=context.session_id,
        serving_target=decision.serving_target,
    )
    previous_backend = (
        decision.sticky_route.backend_name if decision.sticky_route is not None else None
    )
    decision.session_affinity_key = affinity_key
    decision.sticky_route = services.session_affinity.bind(
        affinity_key,
        backend_name=backend_name,
        reason=(
            f"sticky route rebound after failover from '{previous_backend}'"
            if fallback_used and previous_backend is not None and previous_backend != backend_name
            else "sticky route refreshed after successful execution"
        ),
    )
    if previous_backend == backend_name:
        if annotations.affinity_disposition is AffinityDisposition.NOT_REQUESTED:
            annotations.affinity_disposition = AffinityDisposition.CREATED
        return
    if fallback_used and previous_backend is not None:
        annotations.notes.append(
            f"sticky backend failover from '{previous_backend}' to '{backend_name}'"
        )
    annotations.affinity_disposition = AffinityDisposition.CREATED


def _schedule_shadow_traffic(
    *,
    background_tasks: BackgroundTasks,
    services: GatewayServices,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
    decision: RouteDecision,
    primary_backend_name: str,
    plan: ShadowLaunchPlan | None = None,
) -> None:
    if plan is None:
        plan = services.shadow.plan(
            request=chat_request,
            context=context,
            decision=decision,
            primary_backend_name=primary_backend_name,
        )
    if plan is None:
        return
    background_tasks.add_task(
        services.shadow.execute_plan,
        plan=plan,
        request=chat_request,
        context=context,
        registry=services.registry,
        router=services.router,
        telemetry=services.telemetry,
    )


def _launch_streaming_shadow_traffic(
    *,
    services: GatewayServices,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
    decision: RouteDecision,
    primary_backend_name: str,
    plan: ShadowLaunchPlan | None = None,
) -> None:
    if plan is None:
        plan = services.shadow.plan(
            request=chat_request,
            context=context,
            decision=decision,
            primary_backend_name=primary_backend_name,
        )
    if plan is None:
        return
    services.shadow.launch(
        plan=plan,
        request=chat_request,
        context=context,
        registry=services.registry,
        router=services.router,
        telemetry=services.telemetry,
    )


def _ensure_route_annotations(decision: RouteDecision) -> RouteAnnotations:
    if decision.annotations is None:
        decision.annotations = RouteAnnotations()
    return decision.annotations


def _chunk_has_visible_tokens(chunk: ChatCompletionChunk) -> bool:
    return any(
        choice.delta.content is not None and choice.delta.content.strip() != ""
        for choice in chunk.choices
    )


def _chunk_content_fragments(chunk: ChatCompletionChunk) -> list[str]:
    return [
        choice.delta.content
        for choice in chunk.choices
        if choice.delta.content is not None and choice.delta.content != ""
    ]


def _resolve_request_policy(
    *,
    request: Request,
    default_policy: RoutingPolicy,
) -> RoutingPolicy:
    raw_policy = request.headers.get("x-switchyard-routing-policy")
    if raw_policy is None:
        return default_policy
    try:
        return RoutingPolicy(raw_policy)
    except ValueError as exc:
        msg = (
            "invalid x-switchyard-routing-policy header: "
            f"{raw_policy!r}. Expected one of: "
            f"{', '.join(policy.value for policy in RoutingPolicy)}"
        )
        raise InvalidRequestContextError(msg) from exc


def _resolve_workload_shape(request: Request) -> WorkloadShape:
    raw_workload = request.headers.get("x-switchyard-workload-shape")
    if raw_workload is None:
        return WorkloadShape.INTERACTIVE
    try:
        return WorkloadShape(raw_workload)
    except ValueError as exc:
        msg = (
            "invalid x-switchyard-workload-shape header: "
            f"{raw_workload!r}. Expected one of: "
            f"{', '.join(shape.value for shape in WorkloadShape)}"
        )
        raise InvalidRequestContextError(msg) from exc


def _resolve_request_class(request: Request) -> RequestClass:
    raw_request_class = request.headers.get("x-switchyard-request-class")
    if raw_request_class is None:
        return RequestClass.STANDARD
    try:
        return RequestClass(raw_request_class)
    except ValueError as exc:
        msg = (
            "invalid x-switchyard-request-class header: "
            f"{raw_request_class!r}. Expected one of: "
            f"{', '.join(request_class.value for request_class in RequestClass)}"
        )
        raise InvalidRequestContextError(msg) from exc


def _resolve_internal_backend_pin(request: Request) -> str | None:
    raw_backend_pin = request.headers.get("x-switchyard-internal-backend-pin")
    if raw_backend_pin is None:
        return None
    normalized = raw_backend_pin.strip()
    if not normalized:
        msg = "invalid x-switchyard-internal-backend-pin header: value must not be empty"
        raise InvalidRequestContextError(msg)
    return normalized


def _resolve_tenant_id(request: Request) -> str:
    raw_tenant_id = request.headers.get("x-switchyard-tenant-id")
    if raw_tenant_id is None:
        return "default"
    normalized = raw_tenant_id.strip()
    if not normalized:
        msg = "invalid x-switchyard-tenant-id header: value must not be empty"
        raise InvalidRequestContextError(msg)
    return normalized


def _resolve_tenant_tier(request: Request) -> TenantTier:
    raw_tenant_tier = request.headers.get("x-switchyard-tenant-tier")
    if raw_tenant_tier is None:
        return TenantTier.STANDARD
    try:
        return TenantTier(raw_tenant_tier)
    except ValueError as exc:
        msg = (
            "invalid x-switchyard-tenant-tier header: "
            f"{raw_tenant_tier!r}. Expected one of: "
            f"{', '.join(tier.value for tier in TenantTier)}"
        )
        raise InvalidRequestContextError(msg) from exc


def _resolve_session_id(request: Request) -> str | None:
    raw_session_id = request.headers.get("x-switchyard-session-id")
    if raw_session_id is None:
        return None
    normalized = raw_session_id.strip()
    if not normalized:
        msg = "invalid x-switchyard-session-id header: value must not be empty"
        raise InvalidRequestContextError(msg)
    return normalized


def _blocked_backend_reasons(
    *,
    services: GatewayServices,
    context: RequestContext,
    serving_target: str,
    alias_override: AliasRoutingOverrideState | None,
) -> tuple[dict[str, str], dict[str, tuple[str, float | None]]]:
    reasons: dict[str, str] = {}
    rollout_decisions: dict[str, tuple[str, float | None]] = {}
    if alias_override is not None:
        for backend_name in alias_override.disabled_backends:
            reasons[backend_name] = "operator disabled backend for this serving target"
    for worker in services.remote_workers.snapshot().workers:
        if serving_target not in worker.serving_targets:
            continue
        if worker.quarantined:
            reasons[worker.backend_name] = "registered cloud worker is quarantined"
            continue
        if worker.lifecycle_state.value == "draining":
            reasons[worker.backend_name] = "registered cloud worker is draining"
            continue
        if not worker.usable:
            reasons[worker.backend_name] = (
                worker.eligibility_reasons[0]
                if worker.eligibility_reasons
                else "registered cloud worker is not currently usable"
            )
            continue
        if "canary-only" in worker.tags:
            cloud_rollout = services.cloud_rollout.evaluate_canary_only_candidate(
                context=context,
                serving_target=serving_target,
                backend_name=worker.backend_name,
            )
            rollout_decisions[worker.backend_name] = (
                cloud_rollout.disposition,
                cloud_rollout.bucket_percentage,
            )
            if not cloud_rollout.allowed:
                reasons[worker.backend_name] = cloud_rollout.reason or "canary-only backend blocked"
    return reasons, rollout_decisions


def _annotate_registered_cloud_rollout(
    *,
    decision: RouteDecision,
    rollout_decisions: dict[str, tuple[str, float | None]],
) -> None:
    if not rollout_decisions:
        return
    annotations = _ensure_route_annotations(decision)
    if decision.backend_name in rollout_decisions:
        disposition, bucket_percentage = rollout_decisions[decision.backend_name]
        annotations.cloud_rollout_disposition = disposition
        annotations.cloud_rollout_bucket_percentage = (
            None if bucket_percentage is None else round(bucket_percentage, 3)
        )
        if disposition == "selected" and annotations.cloud_routing_reason is None:
            annotations.cloud_routing_reason = "cloud_rollout"
        if (
            disposition == "selected"
            and decision.explanation is not None
            and RouteSelectionReasonCode.CLOUD_ROLLOUT
            not in decision.explanation.selection_reason_codes
        ):
            decision.explanation.selection_reason_codes.append(
                RouteSelectionReasonCode.CLOUD_ROLLOUT
            )
            note = "registered canary-only cloud worker admitted by rollout"
            if note not in decision.explanation.selected_reason:
                decision.explanation.selected_reason.append(note)
            if note not in annotations.notes:
                annotations.notes.append(note)
        return
    first_disposition, first_bucket_percentage = next(iter(rollout_decisions.values()))
    annotations.cloud_rollout_disposition = first_disposition
    annotations.cloud_rollout_bucket_percentage = (
        None if first_bucket_percentage is None else round(first_bucket_percentage, 3)
    )


def _annotate_operator_override(
    *,
    decision: RouteDecision,
    alias_override: AliasRoutingOverrideState | None,
) -> None:
    if alias_override is None or alias_override.pinned_backend is None:
        return
    annotations = _ensure_route_annotations(decision)
    annotations.operator_override_applied = True
    if decision.backend_name.startswith("remote-worker:"):
        annotations.cloud_routing_reason = "operator_override"
    override_note = f"operator override pinned backend={alias_override.pinned_backend}"
    if override_note not in annotations.notes:
        annotations.notes.append(override_note)
    if decision.explanation is not None:
        if (
            RouteSelectionReasonCode.OPERATOR_OVERRIDE
            not in decision.explanation.selection_reason_codes
        ):
            decision.explanation.selection_reason_codes.append(
                RouteSelectionReasonCode.OPERATOR_OVERRIDE
            )
        if override_note not in decision.explanation.selected_reason:
            decision.explanation.selected_reason.append(override_note)


def _route_metadata_headers(decision: RouteDecision) -> dict[str, str]:
    payload = json.dumps(
        decision.model_dump(mode="json", exclude_none=True),
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "x-switchyard-route-decision": payload,
        "x-switchyard-route-selected-backend": decision.backend_name,
    }


def _hybrid_operator_summary(
    *,
    services: GatewayServices,
    runtime_backends: list[BackendRuntimeSummary] | None = None,
    remote_worker_snapshot: RegisteredRemoteWorkerSnapshot | None = None,
) -> HybridOperatorRuntimeSummary:
    resolved_runtime_backends = [] if runtime_backends is None else runtime_backends
    resolved_remote_worker_snapshot = (
        services.remote_workers.snapshot()
        if remote_worker_snapshot is None
        else remote_worker_snapshot
    )
    return services.operator.inspect_state(
        remote_effectively_enabled=services.spillover.enabled,
        cloud_rollout_runtime=services.cloud_rollout.inspect_state(),
        spillover_runtime=services.spillover.inspect_state(),
        remote_worker_snapshot=resolved_remote_worker_snapshot,
        known_serving_targets=_known_serving_targets(
            services=services,
            runtime_backends=resolved_runtime_backends,
            remote_worker_snapshot=resolved_remote_worker_snapshot,
        ),
        alias_overrides=services.alias_overrides.list_overrides(),
    )


def _known_serving_targets(
    *,
    services: GatewayServices,
    runtime_backends: list[BackendRuntimeSummary],
    remote_worker_snapshot: RegisteredRemoteWorkerSnapshot,
) -> list[str]:
    targets: set[str] = set()
    for model in services.settings.local_models:
        targets.add(model.serving_target or model.alias)
    for backend in runtime_backends:
        for logical_target in backend.logical_targets:
            targets.add(logical_target.alias)
    for worker in remote_worker_snapshot.workers:
        targets.update(worker.serving_targets)
    return sorted(targets)


async def _remote_candidate_snapshots(
    *,
    services: GatewayServices,
    serving_target: str,
    context: RequestContext,
) -> list[BackendStatusSnapshot]:
    snapshot = await services.registry.snapshots_for_target(
        serving_target,
        pinned_backend_name=context.internal_backend_pin,
    )
    return [
        item
        for item in snapshot.deployments
        if item.capabilities.device_class is not None
        and (
            item.capabilities.device_class.value == "remote"
            or (
                item.deployment is not None
                and item.deployment.execution_mode is not None
                and item.deployment.execution_mode.value in {"remote_worker", "external_service"}
            )
            or item.capabilities.execution_mode.value in {"remote_worker", "external_service"}
        )
    ]


async def _maybe_acquire_remote_permit(
    *,
    services: GatewayServices,
    context: RequestContext,
    serving_target: str,
    backend_name: str,
    permit: RemotePermit | None,
) -> RemotePermit | None:
    if permit is not None:
        return permit
    remote_candidates = await _remote_candidate_snapshots(
        services=services,
        serving_target=serving_target,
        context=context,
    )
    if not any(candidate.name == backend_name for candidate in remote_candidates):
        return None
    try:
        return services.spillover.acquire_remote_permit(
            context=context,
            backend_name=backend_name,
            remote_candidates=remote_candidates,
        )
    except SpilloverPermitRejectedError as exc:
        raise AdmissionRejectedError(
            AdmissionDecision(
                state=AdmissionDecisionState.REJECTED,
                reason_code=exc.decision.reason_code,
                reason=exc.decision.reason,
                limiter_key=context.tenant_id,
                cooldown_until=services.spillover.inspect_state().cooldown_until,
            ),
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        ) from exc


async def _admit_request(
    *,
    chat_request: ChatCompletionRequest,
    context: RequestContext,
    services: GatewayServices,
    request_timestamp: datetime,
    execution_target: ExecutionTarget,
) -> AdmissionLease:
    try:
        lease = await services.admission.acquire(
            context,
            serving_target=chat_request.model,
        )
    except AdmissionRejectedError as exc:
        remote_candidates = await _remote_candidate_snapshots(
            services=services,
            serving_target=chat_request.model,
            context=context,
        )
        spillover = services.spillover.evaluate_local_admission_failure(
            context=context,
            remote_candidates=remote_candidates,
        )
        if spillover.allowed:
            context.force_remote_candidates_only = True
            decision = spillover_bypass_decision(
                limiter_key=exc.decision.limiter_key or context.tenant_id,
                reason="local admission rejected; request may use explicit remote spillover",
            )
            services.telemetry.record_admission_decision(
                request_id=context.request_id,
                tenant_id=context.tenant_id,
                request_class=context.request_class.value,
                state=decision.state.value,
                reason_code=(None if decision.reason_code is None else decision.reason_code.value),
                queue_depth=0,
                queue_wait_ms=decision.queue_wait_ms,
                status_code=200,
            )
            return AdmissionLease(
                request_id=context.request_id,
                tenant_id=context.tenant_id,
                request_class=context.request_class.value,
                decision=decision,
            )
        if spillover.reason_code is not None:
            exc = AdmissionRejectedError(
                AdmissionDecision(
                    state=AdmissionDecisionState.REJECTED,
                    reason_code=spillover.reason_code,
                    reason=spillover.reason,
                    limiter_key=exc.decision.limiter_key,
                    queue_snapshot=exc.decision.queue_snapshot,
                    queue_wait_ms=exc.decision.queue_wait_ms,
                ),
                status_code=exc.status_code,
            )
        queue_depth = 0
        if exc.decision.queue_snapshot is not None:
            queue_depth = exc.decision.queue_snapshot.current_depth
        services.telemetry.record_admission_decision(
            request_id=context.request_id,
            tenant_id=context.tenant_id,
            request_class=context.request_class.value,
            state=exc.decision.state.value,
            reason_code=(
                None if exc.decision.reason_code is None else exc.decision.reason_code.value
            ),
            queue_depth=queue_depth,
            queue_wait_ms=exc.decision.queue_wait_ms,
            status_code=exc.status_code,
        )
        logger.warning(
            "admission_rejected",
            tenant_id=context.tenant_id,
            request_class=context.request_class.value,
            session_id=context.session_id,
            reason_code=(
                None if exc.decision.reason_code is None else exc.decision.reason_code.value
            ),
            queue_depth=queue_depth,
            status_code=exc.status_code,
        )
        await services.trace_capture.capture_chat_completion(
            request_timestamp=request_timestamp,
            request_id=context.request_id,
            logical_alias=chat_request.model,
            execution_target=execution_target,
            request_context=context,
            route_decision=None,
            chosen_backend=None,
            stream=chat_request.stream,
            status_code=exc.status_code,
            latency_ms=0.0,
            ttft_ms=None,
            output_tokens=None,
            fallback_used=False,
            error=str(exc),
            error_category="admission_rejected",
            request_payload=chat_request,
            response_payload=None,
            admission_decision=exc.decision,
        )
        raise

    queue_depth = 0
    if lease.decision.queue_snapshot is not None:
        queue_depth = lease.decision.queue_snapshot.current_depth
    services.telemetry.record_admission_decision(
        request_id=context.request_id,
        tenant_id=context.tenant_id,
        request_class=context.request_class.value,
        state=lease.decision.state.value,
        reason_code=(
            None if lease.decision.reason_code is None else lease.decision.reason_code.value
        ),
        queue_depth=queue_depth,
        queue_wait_ms=lease.decision.queue_wait_ms,
        status_code=200,
    )
    logger.info(
        "admission_decision",
        tenant_id=context.tenant_id,
        request_class=context.request_class.value,
        session_id=context.session_id,
        state=lease.decision.state.value,
        reason_code=(
            None if lease.decision.reason_code is None else lease.decision.reason_code.value
        ),
        queue_depth=queue_depth,
        queue_wait_ms=lease.decision.queue_wait_ms,
    )
    return lease


def _execution_target_from_context(
    *,
    model_alias: str,
    context: RequestContext,
) -> ExecutionTarget:
    if context.internal_backend_pin is not None:
        return ExecutionTarget(
            target_type=ExecutionTargetType.PINNED_BACKEND,
            model_alias=model_alias,
            pinned_backend=context.internal_backend_pin,
        )
    return ExecutionTarget(
        target_type=ExecutionTargetType.ROUTING_POLICY,
        model_alias=model_alias,
        routing_policy=context.policy,
    )
