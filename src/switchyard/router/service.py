"""Pure Python routing service."""

from __future__ import annotations

from switchyard.adapters.registry import AdapterRegistry
from switchyard.control.affinity import SessionAffinityService
from switchyard.control.canary import CanaryRoutingService
from switchyard.control.circuit import CircuitBreakerService
from switchyard.router.features import extract_request_feature_vector
from switchyard.router.policies import CandidateScore, rejection_reason, score_candidate
from switchyard.schemas.backend import BackendHealthState
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import (
    AffinityDisposition,
    CircuitBreakerPhase,
    PolicyReference,
    RequestContext,
    RolloutDisposition,
    RouteAnnotations,
    RouteCandidateExplanation,
    RouteDecision,
    RouteEligibilityState,
    RouteExplanation,
    RouteSelectionReasonCode,
    RouteTelemetryMetadata,
    SessionAffinityKey,
)


class NoRouteAvailableError(RuntimeError):
    """Raised when no backend can satisfy the request."""


class RouterService:
    """Selects a backend for a request using deterministic Phase 0 policies."""

    def __init__(
        self,
        registry: AdapterRegistry,
        circuit_breaker: CircuitBreakerService | None = None,
        session_affinity: SessionAffinityService | None = None,
        canary_routing: CanaryRoutingService | None = None,
    ) -> None:
        self._registry = registry
        self._circuit_breaker = circuit_breaker
        self._session_affinity = session_affinity
        self._canary_routing = canary_routing

    async def route(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> RouteDecision:
        """Choose the best backend for a request and return a typed decision."""

        rejected_backends: dict[str, str] = {}
        admission_limited_backends: dict[str, str] = {}
        protected_backends: dict[str, str] = {}
        candidates: list[CandidateScore] = []
        explanations: list[RouteCandidateExplanation] = []
        degraded_backends: list[str] = []
        annotations = RouteAnnotations()
        affinity_key = None
        sticky_backend_name: str | None = None
        sticky_route = None
        sticky_lookup_reason: str | None = None
        request_features = context.request_features or extract_request_feature_vector(
            request, context
        )
        context.request_features = request_features
        policy_reference = PolicyReference(policy_id=context.policy.value)
        target_snapshot = await self._registry.snapshots_for_target(
            request.model,
            pinned_backend_name=context.internal_backend_pin,
        )
        if context.session_id is not None:
            affinity_key = context.session_affinity_key or SessionAffinityKey(
                tenant_id=context.tenant_id,
                session_id=context.session_id,
                serving_target=request.model,
            )
        if affinity_key is not None and context.internal_backend_pin is not None:
            annotations.notes.append("session affinity bypassed by internal backend pin")
        elif affinity_key is not None and self._session_affinity is not None:
            lookup = self._session_affinity.lookup(affinity_key)
            annotations.affinity_disposition = lookup.disposition
            if lookup.reason is not None:
                sticky_lookup_reason = lookup.reason
                annotations.notes.append(lookup.reason)
            if lookup.sticky_route is not None:
                sticky_route = lookup.sticky_route
                sticky_backend_name = lookup.sticky_route.backend_name

        for snapshot in target_snapshot.deployments:
            breaker_state = None
            breaker_reason = None
            if self._circuit_breaker is not None:
                _, breaker_state, breaker_reason = self._circuit_breaker.allow_routing(
                    snapshot.name
                )
                if breaker_state.phase is not CircuitBreakerPhase.CLOSED:
                    snapshot.health.circuit_open = breaker_state.phase is CircuitBreakerPhase.OPEN
                    snapshot.health.circuit_reason = breaker_reason or breaker_state.reason
            if snapshot.health.state is BackendHealthState.DEGRADED:
                degraded_backends.append(snapshot.name)

            rejection = rejection_reason(
                snapshot=snapshot,
                request=request,
                context=context,
                policy=context.policy,
            )
            if rejection is not None:
                rejected_backends[snapshot.name] = rejection.reason
                if rejection.category == "admission":
                    admission_limited_backends[snapshot.name] = rejection.reason
                if rejection.category == "protection":
                    protected_backends[snapshot.name] = rejection.reason
                explanations.append(
                    RouteCandidateExplanation(
                        backend_name=snapshot.name,
                        serving_target=request.model,
                        eligibility_state=RouteEligibilityState.REJECTED,
                        reason_codes=[],
                        rationale=[f"policy={context.policy.value}"],
                        rejection_reason=rejection.reason,
                        deployment=snapshot.deployment,
                        engine_type=snapshot.capabilities.engine_type,
                    )
                )
                continue
            candidate = score_candidate(snapshot=snapshot, policy=context.policy)
            candidates.append(candidate)
            explanations.append(
                RouteCandidateExplanation(
                    backend_name=snapshot.name,
                    serving_target=request.model,
                    eligibility_state=RouteEligibilityState.ELIGIBLE,
                    score=round(candidate.score, 3),
                    reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
                    rationale=candidate.rationale,
                    deployment=snapshot.deployment,
                    engine_type=snapshot.capabilities.engine_type,
                )
            )

        if not candidates:
            msg = (
                f"no backend available for model '{request.model}' "
                f"under policy '{context.policy.value}'"
            )
            raise NoRouteAvailableError(msg)

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.snapshot.health.latency_ms or float("inf"),
                candidate.snapshot.name,
            ),
        )
        chosen = ranked[0]
        chosen_rationale = chosen.rationale
        selected_reason_codes = [RouteSelectionReasonCode.POLICY_SCORE]
        canary_policy = None
        if (
            context.internal_backend_pin is not None
            and self._canary_routing is not None
            and self._canary_routing.enabled
        ):
            annotations.notes.append("canary routing bypassed by internal backend pin")
        if sticky_backend_name is not None:
            sticky_candidate = next(
                (
                    candidate
                    for candidate in ranked
                    if candidate.snapshot.name == sticky_backend_name
                ),
                None,
            )
            if sticky_candidate is not None:
                chosen = sticky_candidate
                chosen_rationale = [
                    f"session_affinity=reused backend={sticky_backend_name}",
                    *sticky_candidate.rationale,
                ]
                selected_reason_codes = [RouteSelectionReasonCode.SESSION_AFFINITY]
                fallback_backends = [
                    candidate.snapshot.name
                    for candidate in ranked
                    if candidate.snapshot.name != sticky_backend_name
                ]
                annotations.affinity_disposition = AffinityDisposition.REUSED
            else:
                if annotations.affinity_disposition is AffinityDisposition.REUSED:
                    annotations.affinity_disposition = AffinityDisposition.MISSED
                reason = rejected_backends.get(sticky_backend_name) or sticky_lookup_reason or (
                    f"sticky backend '{sticky_backend_name}' is no longer eligible"
                )
                detail = (
                    f"sticky backend '{sticky_backend_name}' is no longer eligible: {reason}"
                )
                if reason not in annotations.notes:
                    annotations.notes.append(reason)
                if detail not in annotations.notes:
                    annotations.notes.append(detail)
                fallback_backends = [candidate.snapshot.name for candidate in ranked[1:]]
        else:
            canary_service = self._canary_routing
            if canary_service is not None:
                canary_policy = canary_service.match_policy(serving_target=request.model)
            if canary_policy is not None:
                assert canary_service is not None
                annotations.notes.append(f"canary policy '{canary_policy.policy_name}' matched")
                canary_selection = canary_service.select(
                    context=context,
                    policy=canary_policy,
                )
                annotations.rollout_disposition = canary_selection.disposition
                if canary_selection.reason is not None:
                    annotations.notes.append(canary_selection.reason)
                if (
                    canary_selection.disposition is RolloutDisposition.CANARY
                    and canary_selection.selected_backend is not None
                ):
                    canary_candidate = next(
                        (
                            candidate
                            for candidate in ranked
                            if candidate.snapshot.name == canary_selection.selected_backend
                        ),
                        None,
                    )
                    if canary_candidate is not None:
                        chosen = canary_candidate
                        chosen_rationale = [
                            f"canary_rollout=selected backend={canary_candidate.snapshot.name}",
                            *canary_candidate.rationale,
                        ]
                        selected_reason_codes = [RouteSelectionReasonCode.CANARY_SELECTED]
                    else:
                        annotations.notes.append(
                            "canary candidate was ineligible; routing stayed on baseline"
                        )
                elif (
                    canary_selection.disposition is RolloutDisposition.BASELINE
                    and canary_selection.selected_backend is not None
                ):
                    baseline_candidate = next(
                        (
                            candidate
                            for candidate in ranked
                            if candidate.snapshot.name == canary_selection.selected_backend
                        ),
                        None,
                    )
                    if baseline_candidate is not None:
                        chosen = baseline_candidate
                        chosen_rationale = [
                            f"canary_rollout=baseline backend={baseline_candidate.snapshot.name}",
                            *baseline_candidate.rationale,
                        ]
                        selected_reason_codes = [RouteSelectionReasonCode.CANARY_BASELINE]
                    else:
                        annotations.notes.append(
                            "configured canary baseline backend was ineligible; "
                            "using scored baseline"
                        )
            fallback_backends = [
                candidate.snapshot.name
                for candidate in ranked
                if candidate.snapshot.name != chosen.snapshot.name
            ]
        considered_backends = [chosen.snapshot.name, *fallback_backends]

        return RouteDecision(
            backend_name=chosen.snapshot.name,
            serving_target=request.model,
            policy=context.policy,
            request_id=context.request_id,
            workload_shape=context.workload_shape,
            rationale=chosen_rationale,
            score=round(chosen.score, 3),
            considered_backends=considered_backends,
            rejected_backends=rejected_backends,
            admission_limited_backends=admission_limited_backends,
            protected_backends=protected_backends,
            degraded_backends=degraded_backends,
            fallback_backends=fallback_backends,
            circuit_breaker_state=(
                self._circuit_breaker.state_for(chosen.snapshot.name)
                if self._circuit_breaker is not None
                else None
            ),
            session_affinity_key=affinity_key,
            sticky_route=(
                sticky_route
                if sticky_route is not None and sticky_route.backend_name == chosen.snapshot.name
                else None
            ),
            canary_policy=canary_policy,
            annotations=annotations,
            telemetry_metadata=RouteTelemetryMetadata(
                tenant_id=context.tenant_id,
                tenant_tier=context.tenant_tier,
                request_class=context.request_class,
                session_affinity_enabled=(
                    affinity_key is not None
                    and context.internal_backend_pin is None
                    and self._session_affinity is not None
                    and self._session_affinity.enabled
                ),
                circuit_breaker_enabled=self._circuit_breaker is not None,
                canary_enabled=canary_policy is not None,
                labels={"request_id": context.request_id},
            ),
            request_features=request_features,
            policy_reference=policy_reference,
            selected_deployment=chosen.snapshot.deployment,
            explanation=RouteExplanation(
                serving_target=request.model,
                candidates=explanations,
                request_features=request_features,
                policy_reference=policy_reference,
                selected_backend=chosen.snapshot.name,
                selection_reason_codes=selected_reason_codes,
                selected_reason=chosen_rationale,
                tie_breaker="score, latency_ms, backend_name",
            ),
        )
