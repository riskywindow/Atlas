"""FastAPI worker wrapper for a single backend adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from switchyard.adapters.base import BackendAdapter
from switchyard.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from switchyard.runtime.base import UnsupportedRequestError
from switchyard.schemas.backend import (
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    CapacitySnapshot,
)
from switchyard.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from switchyard.schemas.routing import RequestClass, RequestContext, RoutingPolicy, WorkloadShape
from switchyard.schemas.worker import (
    WorkerCapabilitiesResponse,
    WorkerGenerateRequest,
    WorkerGenerateResponse,
    WorkerHealthResponse,
    WorkerReadinessResponse,
    WorkerRequestMetadata,
    WorkerResponseMetadata,
    WorkerStreamChunkResponse,
    WorkerWarmupRequest,
    WorkerWarmupResponse,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class WorkerServiceState:
    """Mutable worker state exposed through the internal HTTP protocol."""

    adapter: BackendAdapter
    worker_name: str
    warmup_on_start: bool = False
    warmup_model_id: str | None = None
    active_requests: int = 0
    last_warmup_error: str | None = None
    warmup_in_progress: bool = False
    draining: bool = False
    drain_reason: str | None = None
    drain_timeout_seconds: float = 15.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def initialize(self) -> None:
        """Optionally warm the worker during startup."""

        if not self.warmup_on_start:
            return
        try:
            await self.warmup(model_id=self.warmup_model_id)
        except Exception:
            logger.exception("worker_startup_warmup_failed", worker_name=self.worker_name)

    async def capabilities_response(self) -> WorkerCapabilitiesResponse:
        """Return the typed capabilities envelope."""

        capabilities = await self.adapter.capabilities()
        deployment = None
        status = getattr(self.adapter, "status", None)
        if callable(status):
            try:
                status_snapshot = await status()
            except Exception:
                status_snapshot = None
            if status_snapshot is not None:
                deployment = status_snapshot.deployment
        return WorkerCapabilitiesResponse(
            worker_name=self.worker_name,
            runtime=capabilities.runtime.model_copy(deep=True)
            if capabilities.runtime is not None
            else None,
            gpu=capabilities.gpu.model_copy(deep=True) if capabilities.gpu is not None else None,
            backend_type=capabilities.backend_type,
            execution_mode=capabilities.execution_mode,
            capabilities=capabilities,
            deployment=deployment,
        )

    async def health_response(self) -> WorkerHealthResponse:
        """Return the typed health envelope."""

        return WorkerHealthResponse(
            worker_name=self.worker_name,
            health=await self.current_health(),
        )

    async def readiness_response(self) -> WorkerReadinessResponse:
        """Return the typed readiness envelope."""

        capabilities = await self.adapter.capabilities()
        health = await self.current_health()
        return WorkerReadinessResponse(
            worker_name=self.worker_name,
            runtime=capabilities.runtime.model_copy(deep=True)
            if capabilities.runtime is not None
            else None,
            gpu=capabilities.gpu.model_copy(deep=True) if capabilities.gpu is not None else None,
            ready=(
                health.state is not BackendHealthState.UNAVAILABLE
                and health.load_state is BackendLoadState.READY
                and self.last_warmup_error is None
            ),
            active_requests=self.active_requests,
            queue_depth=0,
            observed_capacity=CapacitySnapshot(
                concurrency_limit=capabilities.concurrency_limit,
                active_requests=self.active_requests,
                queue_depth=0,
            ),
            health=health,
        )

    async def warmup(self, model_id: str | None = None) -> WorkerWarmupResponse:
        """Warm the wrapped adapter and record the outcome."""

        async with self._lock:
            self.warmup_in_progress = True
            self.last_warmup_error = None
        try:
            await self.adapter.warmup(model_id=model_id)
        except Exception as exc:
            async with self._lock:
                self.last_warmup_error = str(exc)
            raise
        finally:
            async with self._lock:
                self.warmup_in_progress = False

        return WorkerWarmupResponse(
            worker_name=self.worker_name,
            warmed=True,
            health=await self.current_health(),
        )

    async def current_health(self) -> BackendHealth:
        """Return health with local warmup state applied."""

        health = (await self.adapter.health()).model_copy(deep=True)
        if self.draining:
            health.state = BackendHealthState.DEGRADED
            if self.drain_reason:
                health.detail = (
                    self.drain_reason
                    if "drain" in self.drain_reason.lower()
                    else f"worker draining: {self.drain_reason}"
                )
            else:
                health.detail = "worker draining"
        if self.warmup_in_progress:
            health.load_state = BackendLoadState.WARMING
            if health.detail is None:
                health.detail = "worker warmup in progress"
        if self.last_warmup_error is not None:
            health.state = BackendHealthState.UNAVAILABLE
            health.load_state = BackendLoadState.FAILED
            health.detail = "worker warmup failed"
            health.last_error = self.last_warmup_error
        return health

    @asynccontextmanager
    async def track_request(self) -> AsyncIterator[None]:
        """Track in-flight request execution."""

        async with self._lock:
            self.active_requests += 1
        try:
            yield
        finally:
            async with self._lock:
                self.active_requests -= 1

    async def begin_drain(self, *, reason: str) -> None:
        """Stop accepting new work and advertise a draining posture."""

        async with self._lock:
            self.draining = True
            self.drain_reason = reason

    async def wait_for_drain(self) -> None:
        """Wait for in-flight requests to complete during shutdown."""

        deadline = asyncio.get_running_loop().time() + self.drain_timeout_seconds
        while self.active_requests > 0 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)


def create_worker_app(
    adapter: BackendAdapter,
    *,
    worker_name: str | None = None,
    warmup_on_start: bool = False,
    warmup_model_id: str | None = None,
    log_level: str = "INFO",
    drain_timeout_seconds: float = 15.0,
) -> FastAPI:
    """Create a worker-serving FastAPI app around a single adapter."""

    configure_logging(log_level)
    state = WorkerServiceState(
        adapter=adapter,
        worker_name=worker_name or adapter.name,
        warmup_on_start=warmup_on_start,
        warmup_model_id=warmup_model_id,
        drain_timeout_seconds=drain_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await state.initialize()
        try:
            yield
        finally:
            await state.begin_drain(reason="worker shutdown in progress")
            await state.wait_for_drain()

    app = FastAPI(title=f"Switchyard Worker: {state.worker_name}", lifespan=lifespan)
    app.state.worker = state

    @app.middleware("http")
    async def inject_request_id(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = request_id
        request.state.trace_id = request.headers.get("x-trace-id")
        request.state.timeout_ms = _parse_timeout_ms(
            request.headers.get("x-switchyard-timeout-ms")
        )
        request.state.worker_request_id = str(uuid4())
        bind_request_context(
            request_id=request_id,
            worker_name=state.worker_name,
            path=request.url.path,
            method=request.method,
        )
        start = perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            latency_ms = round((perf_counter() - start) * 1000, 3)
            logger.info(
                "worker_request_completed",
                worker_name=state.worker_name,
                path=request.url.path,
                method=request.method,
                status_code=response.status_code if response is not None else 500,
                latency_ms=latency_ms,
            )
            clear_request_context()
            if response is not None:
                response.headers["x-request-id"] = request_id

    @app.exception_handler(Exception)
    async def handle_worker_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        if isinstance(exc, HTTPException):
            raise exc
        if isinstance(exc, UnsupportedRequestError):
            payload = ErrorResponse(
                code="unsupported_request",
                message=str(exc),
                request_id=getattr(request.state, "request_id", None),
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=payload.model_dump(mode="json"),
            )
        logger.exception(
            "worker_request_failed",
            worker_name=state.worker_name,
            path=request.url.path,
        )
        payload = ErrorResponse(
            code="worker_error",
            message=str(exc),
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
        )

    @app.get("/healthz", response_model=WorkerHealthResponse)
    async def healthz(request: Request) -> WorkerHealthResponse:
        response = await state.health_response()
        response.transport_metadata = _response_metadata(request)
        return response

    @app.get("/internal/worker/ready", response_model=WorkerReadinessResponse)
    async def readiness(request: Request) -> WorkerReadinessResponse:
        response = await state.readiness_response()
        if state.draining:
            response.ready = False
        response.transport_metadata = _response_metadata(request)
        return response

    @app.get("/internal/worker/capabilities", response_model=WorkerCapabilitiesResponse)
    async def capabilities(request: Request) -> WorkerCapabilitiesResponse:
        response = await state.capabilities_response()
        response.transport_metadata = _response_metadata(request)
        return response

    @app.post("/internal/worker/warmup", response_model=WorkerWarmupResponse)
    async def warmup(
        request: Request,
        payload: WorkerWarmupRequest | None = None,
    ) -> WorkerWarmupResponse:
        _apply_request_metadata(
            request,
            payload.transport_metadata if payload is not None else None,
        )
        response = await state.warmup(model_id=payload.model_id if payload is not None else None)
        response.transport_metadata = _response_metadata(request)
        return response

    @app.post("/internal/worker/generate", response_model=WorkerGenerateResponse)
    async def generate(request: Request, payload: WorkerGenerateRequest) -> WorkerGenerateResponse:
        _apply_request_metadata(request, payload.transport_metadata)
        if payload.transport_metadata is None:
            _apply_context_metadata(request, payload.context)
        _raise_if_draining(state)
        async with state.track_request():
            response = await adapter.generate(payload.request, payload.context)
        return WorkerGenerateResponse(
            worker_name=state.worker_name,
            response=response,
            transport_metadata=_response_metadata(request),
        )

    @app.post("/internal/worker/generate/stream")
    async def stream_generate(
        request: Request,
        payload: WorkerGenerateRequest,
    ) -> StreamingResponse:
        _apply_request_metadata(request, payload.transport_metadata)
        if payload.transport_metadata is None:
            _apply_context_metadata(request, payload.context)
        _raise_if_draining(state)
        return StreamingResponse(
            _stream_worker_chunks(
                state=state,
                transport_metadata=_response_metadata(request),
                request=payload.request,
                context=payload.context,
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
    ) -> ChatCompletionResponse | StreamingResponse:
        _raise_if_draining(state)
        context = RequestContext(
            request_id=request.state.request_id,
            policy=RoutingPolicy.LOCAL_ONLY,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_class=RequestClass.STANDARD,
        )
        if payload.stream:
            return StreamingResponse(
                _stream_public_chunks(
                    state=state,
                    request=payload,
                    context=context,
                ),
                media_type="text/event-stream",
            )
        async with state.track_request():
            return await adapter.generate(payload, context)

    return app


async def _stream_worker_chunks(
    *,
    state: WorkerServiceState,
    transport_metadata: WorkerResponseMetadata,
    request: ChatCompletionRequest,
    context: RequestContext,
) -> AsyncIterator[str]:
    async with state.track_request():
        async for chunk in state.adapter.stream_generate(request, context):
            payload = WorkerStreamChunkResponse(
                worker_name=state.worker_name,
                chunk=chunk,
                transport_metadata=transport_metadata,
            )
            yield f"data: {payload.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_public_chunks(
    *,
    state: WorkerServiceState,
    request: ChatCompletionRequest,
    context: RequestContext,
) -> AsyncIterator[str]:
    async with state.track_request():
        async for chunk in state.adapter.stream_generate(request, context):
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def _apply_request_metadata(
    request: Request,
    metadata: WorkerRequestMetadata | None,
) -> None:
    if metadata is None:
        return
    request.state.request_id = metadata.request_id
    request.state.trace_id = metadata.trace_id
    request.state.timeout_ms = metadata.timeout_ms


def _apply_context_metadata(request: Request, context: RequestContext) -> None:
    request.state.request_id = context.request_id
    if context.trace_id is not None:
        request.state.trace_id = context.trace_id


def _response_metadata(request: Request) -> WorkerResponseMetadata:
    return WorkerResponseMetadata(
        request_id=request.state.request_id,
        trace_id=getattr(request.state, "trace_id", None),
        worker_request_id=request.state.worker_request_id,
    )


def _parse_timeout_ms(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _raise_if_draining(state: WorkerServiceState) -> None:
    if not state.draining:
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=state.drain_reason or "worker draining",
    )
