"""Pure Python routing service."""

from __future__ import annotations

from switchyard.adapters.registry import AdapterRegistry
from switchyard.router.policies import CandidateScore, rejection_reason, score_candidate
from switchyard.schemas.backend import BackendStatusSnapshot
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import RequestContext, RouteDecision


class NoRouteAvailableError(RuntimeError):
    """Raised when no backend can satisfy the request."""


class RouterService:
    """Selects a backend for a request using deterministic Phase 0 policies."""

    def __init__(self, registry: AdapterRegistry) -> None:
        self._registry = registry

    async def route(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> RouteDecision:
        """Choose the best backend for a request and return a typed decision."""

        rejected_backends: dict[str, str] = {}
        candidates: list[CandidateScore] = []

        for adapter in self._registry.list():
            snapshot = BackendStatusSnapshot(
                name=adapter.name,
                capabilities=await adapter.capabilities(),
                health=await adapter.health(),
            )
            reason = rejection_reason(snapshot=snapshot, request=request, policy=context.policy)
            if reason is not None:
                rejected_backends[snapshot.name] = reason
                continue
            candidates.append(score_candidate(snapshot=snapshot, policy=context.policy))

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
        fallback_backends = [candidate.snapshot.name for candidate in ranked[1:]]

        return RouteDecision(
            backend_name=chosen.snapshot.name,
            policy=context.policy,
            request_id=context.request_id,
            workload_shape=context.workload_shape,
            rationale=chosen.rationale,
            score=round(chosen.score, 3),
            considered_backends=[candidate.snapshot.name for candidate in ranked],
            rejected_backends=rejected_backends,
            fallback_backends=fallback_backends,
        )
