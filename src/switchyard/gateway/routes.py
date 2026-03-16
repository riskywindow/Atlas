"""Gateway routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, status

from switchyard.gateway.dependencies import GatewayServices, get_services
from switchyard.logging import get_logger
from switchyard.schemas.backend import BackendHealthState
from switchyard.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from switchyard.schemas.routing import RequestContext

router = APIRouter()
logger = get_logger(__name__)


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


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    request: Request,
    services: GatewayServices = Depends(get_services),
) -> ChatCompletionResponse:
    """Route a chat completion request and invoke the selected backend."""

    request_id = request.state.request_id
    route_context = RequestContext(request_id=request_id)
    decision = await services.router.route(chat_request, route_context)
    services.telemetry.record_route_decision(
        policy=decision.policy.value,
        backend_name=decision.backend_name,
    )
    logger.info(
        "route_decision",
        chosen_backend=decision.backend_name,
        route_policy=decision.policy.value,
        considered_backends=decision.considered_backends,
        fallback_backends=decision.fallback_backends,
    )
    adapter = services.registry.get(decision.backend_name)
    chat_response = await adapter.generate(chat_request, route_context)
    logger.info(
        "chat_completion_succeeded",
        chosen_backend=decision.backend_name,
        route_policy=decision.policy.value,
        response_id=chat_response.id,
    )
    return chat_response
