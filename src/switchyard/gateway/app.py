"""FastAPI application factory."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import Settings
from switchyard.gateway.dependencies import GatewayServices, gateway_lifespan
from switchyard.gateway.routes import router as gateway_router
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


def create_app(
    *,
    registry: AdapterRegistry | None = None,
    router_service: RouterService | None = None,
    settings: Settings | None = None,
    telemetry: Telemetry | None = None,
) -> FastAPI:
    """Create the Switchyard FastAPI application."""

    resolved_settings = settings or Settings()
    resolved_registry = registry or AdapterRegistry()
    resolved_router = router_service or RouterService(resolved_registry)
    configure_logging(resolved_settings.log_level)
    resolved_telemetry = telemetry or configure_telemetry(
        resolved_settings.service_name,
        enabled=resolved_settings.otel_enabled,
    )

    app = FastAPI(title="Switchyard Gateway", lifespan=gateway_lifespan)
    app.state.services = GatewayServices(
        registry=resolved_registry,
        router=resolved_router,
        telemetry=resolved_telemetry,
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
        response = await call_next(request)
        latency_ms = (perf_counter() - start) * 1000
        resolved_telemetry.record_request(
            route=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        logger.info(
            "request_completed",
            status_code=response.status_code,
            latency_ms=round(latency_ms, 3),
        )
        clear_request_context()
        response.headers["x-request-id"] = request_id
        return response

    app.include_router(gateway_router)

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

    return app


def _request_id_headers(request_id: str | None) -> dict[str, str]:
    if request_id is None:
        return {}
    return {"x-request-id": request_id}
