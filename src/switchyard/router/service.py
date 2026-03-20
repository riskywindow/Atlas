"""Pure Python routing service."""

from __future__ import annotations

from switchyard.adapters.registry import AdapterRegistry
from switchyard.control.affinity import SessionAffinityService
from switchyard.control.canary import CanaryRoutingService
from switchyard.control.circuit import CircuitBreakerService
from switchyard.control.locality import PrefixLocalityService
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.spillover import RemoteSpilloverControlService
from switchyard.router.features import extract_request_feature_vector
from switchyard.router.policies import (
    CandidateRejection,
    PolicyRegistry,
    rejection_reason,
)
from switchyard.router.policies import (
    _is_remote_snapshot as _is_remote_candidate,
)
from switchyard.schemas.backend import BackendHealthState
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import (
    AffinityDisposition,
    CircuitBreakerPhase,
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
        prefix_locality: PrefixLocalityService | None = None,
        canary_routing: CanaryRoutingService | None = None,
        policy_registry: PolicyRegistry | None = None,
        policy_rollout: PolicyRolloutService | None = None,
        spillover: RemoteSpilloverControlService | None = None,
    ) -> None:
        self._registry = registry
        self._circuit_breaker = circuit_breaker
        self._session_affinity = session_affinity
        self._prefix_locality = prefix_locality
        self._canary_routing = canary_routing
        self._policy_registry = policy_registry or PolicyRegistry()
        self._policy_rollout = policy_rollout
        self._spillover = spillover

    async def route(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> RouteDecision:
        """Choose the best backend for a request and return a typed decision."""

        rejected_backends: dict[str, str] = {}
        admission_limited_backends: dict[str, str] = {}
        protected_backends: dict[str, str] = {}
        eligible_snapshots = []
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

        remote_candidates = [
            snapshot for snapshot in target_snapshot.deployments if _is_remote_candidate(snapshot)
        ]
        spillover_guardrail_triggers: list[str] = []

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
            if (
                rejection is None
                and self._spillover is not None
                and _is_remote_candidate(snapshot)
            ):
                spillover_decision = self._spillover.evaluate_remote_candidate(
                    context=context,
                    snapshot=snapshot,
                    remote_candidates=remote_candidates,
                )
                if not spillover_decision.allowed:
                    rejection = CandidateRejection(
                        reason=spillover_decision.reason or "remote spillover rejected",
                        category="admission",
                        reason_codes=(
                            None
                            if spillover_decision.route_reason_code is None
                            else [spillover_decision.route_reason_code]
                        ),
                    )
                    if spillover_decision.guardrail_trigger is not None:
                        spillover_guardrail_triggers.append(
                            spillover_decision.guardrail_trigger
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
                        reason_codes=rejection.reason_codes or [],
                        rationale=[f"policy={context.policy.value}"],
                        rejection_reason=rejection.reason,
                        deployment=snapshot.deployment,
                        engine_type=snapshot.capabilities.engine_type,
                        logical_model=(
                            None
                            if rejection.logical_model is None
                            else rejection.logical_model.model_copy(deep=True)
                        ),
                    )
                )
                continue
            eligible_snapshots.append(snapshot)

        if not eligible_snapshots:
            msg = (
                f"no backend available for model '{request.model}' "
                f"under policy '{context.policy.value}'"
            )
            raise NoRouteAvailableError(msg)

        rollout_resolution = (
            None
            if self._policy_rollout is None
            else self._policy_rollout.resolve(
                registry=self._policy_registry,
                requested_policy=context.policy,
                request=request,
                context=context,
            )
        )
        primary_policy = (
            self._policy_registry.resolve(context.policy)
            if rollout_resolution is None
            else rollout_resolution.primary_policy
        )
        primary_evaluation = primary_policy.evaluate(
            request=request,
            context=context,
            candidates=eligible_snapshots,
        )
        explanations.extend(
            assessment.to_explanation(serving_target=request.model)
            for assessment in primary_evaluation.assessments
        )
        ranked = primary_evaluation.ranked_backends()
        assessment_by_backend = {
            assessment.snapshot.name: assessment for assessment in primary_evaluation.assessments
        }
        prefix_locality_signal = (
            None
            if self._prefix_locality is None
            else self._prefix_locality.inspect(
                serving_target=request.model,
                request_features=request_features,
                candidate_backends=ranked,
                sticky_backend_name=sticky_backend_name,
                session_affinity_enabled=(
                    affinity_key is not None
                    and context.internal_backend_pin is None
                    and self._session_affinity is not None
                    and self._session_affinity.enabled
                ),
            )
        )
        context.prefix_locality_signal = prefix_locality_signal
        if prefix_locality_signal is not None and prefix_locality_signal.affinity_conflict:
            annotations.notes.append(
                "prefix locality preferred backend differs from the current "
                "session affinity binding"
            )
        shadow_policies = (
            self._policy_registry.shadow_policies
            if rollout_resolution is None
            else rollout_resolution.shadow_policies
        )
        shadow_evaluations = [
            shadow_policy.evaluate(
                request=request,
                context=context,
                candidates=eligible_snapshots,
            )
            for shadow_policy in shadow_policies
        ]
        if self._policy_rollout is not None and rollout_resolution is not None:
            self._policy_rollout.observe_decision(
                context=context,
                resolution=rollout_resolution,
                primary_evaluation=primary_evaluation,
                shadow_evaluations=shadow_evaluations,
                extra_guardrail_triggers=spillover_guardrail_triggers,
            )
        shadow_explanations = [
            evaluation.to_shadow_explanation(serving_target=request.model)
            for evaluation in shadow_evaluations
        ]
        chosen = primary_evaluation.selected_assessment()
        chosen_rationale = list(primary_evaluation.selected_reason)
        selected_reason_codes = list(primary_evaluation.selected_reason_codes)
        if context.force_remote_candidates_only:
            selected_reason_codes.insert(0, RouteSelectionReasonCode.LOCAL_ADMISSION_SPILLOVER)
            chosen_rationale = [
                "local admission overflow allowed explicit remote spillover",
                *chosen_rationale,
            ]
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
                    assessment_by_backend[candidate_name]
                    for candidate_name in ranked
                    if candidate_name == sticky_backend_name
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
                    candidate_name
                    for candidate_name in ranked
                    if candidate_name != sticky_backend_name
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
                fallback_backends = ranked[1:]
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
                            assessment_by_backend[candidate_name]
                            for candidate_name in ranked
                            if candidate_name == canary_selection.selected_backend
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
                            assessment_by_backend[candidate_name]
                            for candidate_name in ranked
                            if candidate_name == canary_selection.selected_backend
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
                candidate_name
                for candidate_name in ranked
                if candidate_name != chosen.snapshot.name
            ]
        considered_backends = [chosen.snapshot.name, *fallback_backends]
        policy_reference = primary_evaluation.policy_reference

        return RouteDecision(
            backend_name=chosen.snapshot.name,
            serving_target=request.model,
            policy=context.policy,
            request_id=context.request_id,
            workload_shape=context.workload_shape,
            rationale=chosen_rationale,
            score=None if chosen.score is None else round(chosen.score, 3),
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
            prefix_locality_signal=prefix_locality_signal,
            policy_reference=policy_reference,
            selected_deployment=chosen.snapshot.deployment,
            explanation=RouteExplanation(
                serving_target=request.model,
                candidates=explanations,
                request_features=request_features,
                prefix_locality_signal=prefix_locality_signal,
                policy_reference=policy_reference,
                selected_backend=chosen.snapshot.name,
                selection_reason_codes=selected_reason_codes,
                selected_reason=chosen_rationale,
                tie_breaker=primary_evaluation.tie_breaker,
                shadow_evaluations=shadow_explanations,
            ),
        )
