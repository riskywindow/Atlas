"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from switchyard.adapters.factory import build_registry_from_settings
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import Settings
from switchyard.control.admission import AdmissionControlService, AdmissionRejectedError
from switchyard.control.affinity import SessionAffinityService
from switchyard.control.canary import CanaryRoutingService
from switchyard.control.circuit import CircuitBreakerService
from switchyard.control.locality import PrefixLocalityService
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.remote_workers import (
    RemoteWorkerRegistrationError,
    RemoteWorkerRegistryService,
)
from switchyard.control.shadow import ShadowTrafficService
from switchyard.gateway.dependencies import GatewayServices, gateway_lifespan
from switchyard.gateway.routes import BackendExecutionExhaustedError, InvalidRequestContextError
from switchyard.gateway.routes import router as gateway_router
from switchyard.gateway.trace_capture import build_trace_capture_service
from switchyard.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from switchyard.router.service import NoRouteAvailableError, RouterService
from switchyard.schemas.chat import ErrorResponse
from switchyard.telemetry import Telemetry, configure_telemetry

logger = get_logger(__name__)

RegistryBuilder = Callable[[Settings], AdapterRegistry]


def create_app(
    *,
    registry: AdapterRegistry | None = None,
    router_service: RouterService | None = None,
    policy_rollout: PolicyRolloutService | None = None,
    session_affinity: SessionAffinityService | None = None,
    prefix_locality: PrefixLocalityService | None = None,
    settings: Settings | None = None,
    telemetry: Telemetry | None = None,
    registry_builder: RegistryBuilder | None = None,
) -> FastAPI:
    """Create the Switchyard FastAPI application."""

    resolved_settings = settings or Settings()
    resolved_registry = registry or (registry_builder or build_registry_from_settings)(
        resolved_settings
    )
    resolved_circuit_breaker = CircuitBreakerService(resolved_settings.phase4.circuit_breakers)
    resolved_session_affinity = session_affinity or SessionAffinityService(
        resolved_settings.phase4.session_affinity
    )
    resolved_prefix_locality = prefix_locality or PrefixLocalityService()
    resolved_canary = CanaryRoutingService(resolved_settings.phase4.canary_routing)
    resolved_shadow = ShadowTrafficService(resolved_settings.phase4.shadow_routing)
    resolved_remote_workers = RemoteWorkerRegistryService(
        resolved_settings.phase7.remote_workers
    )
    resolved_policy_rollout = policy_rollout or PolicyRolloutService(
        resolved_settings.phase4.policy_rollout
    )
    resolved_router = router_service or RouterService(
        resolved_registry,
        circuit_breaker=resolved_circuit_breaker,
        session_affinity=resolved_session_affinity,
        prefix_locality=resolved_prefix_locality,
        canary_routing=resolved_canary,
        policy_rollout=resolved_policy_rollout,
    )
    configure_logging(resolved_settings.log_level)
    resolved_telemetry = telemetry or configure_telemetry(
        resolved_settings.service_name,
        enabled=resolved_settings.otel_enabled,
    )

    app = FastAPI(title="Switchyard Gateway", lifespan=gateway_lifespan)
    app.state.services = GatewayServices(
        settings=resolved_settings,
        registry=resolved_registry,
        router=resolved_router,
        admission=AdmissionControlService(resolved_settings.phase4.admission_control),
        circuit_breaker=resolved_circuit_breaker,
        session_affinity=resolved_session_affinity,
        prefix_locality=resolved_prefix_locality,
        canary=resolved_canary,
        shadow=resolved_shadow,
        policy_rollout=resolved_policy_rollout,
        remote_workers=resolved_remote_workers,
        telemetry=resolved_telemetry,
        trace_capture=build_trace_capture_service(
            mode=resolved_settings.trace_capture_mode,
            output_path=resolved_settings.trace_capture_output_path,
        ),
    )
    resolved_telemetry.instrument_fastapi(app)

    @app.middleware("http")
    async def inject_request_id(request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = request_id
        bind_request_context(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            status_code = response.status_code if response is not None else 500
            latency_ms = (perf_counter() - start) * 1000
            resolved_telemetry.record_request(
                route=request.url.path,
                method=request.method,
                status_code=status_code,
                latency_ms=latency_ms,
            )
            logger.info(
                "request_completed",
                status_code=status_code,
                latency_ms=round(latency_ms, 3),
            )
            clear_request_context()
            if response is not None:
                response.headers["x-request-id"] = request_id

    app.include_router(gateway_router)

    if resolved_settings.metrics_enabled:

        @app.get(resolved_settings.metrics_path, include_in_schema=False)
        async def metrics() -> Response:
            return Response(
                content=resolved_telemetry.render_prometheus_text(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="invalid_request",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload.model_dump(mode="json"),
            headers=_request_id_headers(request_id),
        )

    @app.exception_handler(NoRouteAvailableError)
    async def handle_no_route_error(
        request: Request,
        exc: NoRouteAvailableError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="backend_unavailable",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
            headers=_request_id_headers(request_id),
        )

    @app.exception_handler(BackendExecutionExhaustedError)
    async def handle_backend_execution_exhausted(
        request: Request,
        exc: BackendExecutionExhaustedError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="backend_unavailable",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
            headers=_request_id_headers(request_id),
        )

    @app.exception_handler(AdmissionRejectedError)
    async def handle_admission_rejected(
        request: Request,
        exc: AdmissionRejectedError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="rate_limited",
            message=str(exc),
            request_id=request_id,
        )
        headers = _request_id_headers(request_id)
        headers["x-switchyard-admission-decision"] = exc.decision.model_dump_json(
            exclude_none=True
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
            headers=headers,
        )

    @app.exception_handler(KeyError)
    async def handle_missing_adapter(
        request: Request,
        exc: KeyError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="backend_not_found",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
            headers=_request_id_headers(request_id),
        )

    @app.exception_handler(RemoteWorkerRegistrationError)
    async def handle_remote_worker_registration_error(
        request: Request,
        exc: RemoteWorkerRegistrationError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="remote_worker_registration_error",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=payload.model_dump(mode="json"),
            headers=_request_id_headers(request_id),
        )

    @app.exception_handler(InvalidRequestContextError)
    async def handle_invalid_request_context(
        request: Request,
        exc: InvalidRequestContextError,
    ) -> JSONResponse:
        request_id = request.state.request_id
        payload = ErrorResponse(
            code="invalid_request",
            message=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=payload.model_dump(mode="json"),
            headers=_request_id_headers(request_id),
        )

    return app


def _request_id_headers(request_id: str | None) -> dict[str, str]:
    if request_id is None:
        return {}
    return {"x-request-id": request_id}
