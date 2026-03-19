"""HTTP adapter for Switchyard-internal remote workers."""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from switchyard.config import BackendInstanceConfig, LocalModelConfig
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendDeployment,
    BackendHealth,
    BackendHealthState,
    BackendInstance,
    BackendStatusSnapshot,
    DeviceClass,
    EngineType,
    PerformanceHint,
    QualityHint,
    WorkerTransportType,
)
from switchyard.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from switchyard.schemas.routing import RequestContext
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

TWorkerModel = TypeVar("TWorkerModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class RemoteWorkerInstanceStatus:
    """Resolved point-in-time status for one network worker instance."""

    instance: BackendInstance
    ready: bool
    active_requests: int
    queue_depth: int


class RemoteWorkerError(RuntimeError):
    """Base error for remote worker failures."""


class RemoteWorkerOperation(StrEnum):
    """Typed transport operations against a remote worker."""

    HEALTH = "health"
    READINESS = "readiness"
    CAPABILITIES = "capabilities"
    WARMUP = "warmup"
    GENERATE = "generate"
    STREAM = "stream"


class RemoteWorkerErrorKind(StrEnum):
    """Stable classification for remote worker transport failures."""

    CONNECT = "connect"
    TIMEOUT = "timeout"
    HTTP_STATUS = "http_status"
    REQUEST = "request"
    INVALID_JSON = "invalid_json"
    INVALID_PAYLOAD = "invalid_payload"
    PROTOCOL = "protocol"


class RemoteWorkerTransportError(RemoteWorkerError):
    """Raised when a network call to a worker fails."""

    def __init__(
        self,
        message: str,
        *,
        kind: RemoteWorkerErrorKind,
        operation: RemoteWorkerOperation,
        instance_id: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.operation = operation
        self.instance_id = instance_id
        self.status_code = status_code
        self.retryable = retryable


class RemoteWorkerResponseError(RemoteWorkerError):
    """Raised when a worker returns malformed or invalid data."""

    def __init__(
        self,
        message: str,
        *,
        kind: RemoteWorkerErrorKind,
        operation: RemoteWorkerOperation,
        instance_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.operation = operation
        self.instance_id = instance_id


class RemoteWorkerClient:
    """Typed client for the Switchyard internal worker transport."""

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def health(
        self,
        instance_config: BackendInstanceConfig,
        *,
        metadata: WorkerRequestMetadata | None = None,
    ) -> WorkerHealthResponse:
        payload = await self._request_json(
            "GET",
            instance_config,
            instance_config.health_path,
            operation=RemoteWorkerOperation.HEALTH,
            metadata=metadata,
        )
        return self._validate_payload(
            payload,
            WorkerHealthResponse,
            operation=RemoteWorkerOperation.HEALTH,
            instance_id=instance_config.instance_id,
            request_metadata=metadata,
        )

    async def readiness(
        self,
        instance_config: BackendInstanceConfig,
        *,
        metadata: WorkerRequestMetadata | None = None,
    ) -> WorkerReadinessResponse:
        payload = await self._request_json(
            "GET",
            instance_config,
            instance_config.readiness_path,
            operation=RemoteWorkerOperation.READINESS,
            metadata=metadata,
        )
        return self._validate_payload(
            payload,
            WorkerReadinessResponse,
            operation=RemoteWorkerOperation.READINESS,
            instance_id=instance_config.instance_id,
            request_metadata=metadata,
        )

    async def capabilities(
        self,
        instance_config: BackendInstanceConfig,
        *,
        metadata: WorkerRequestMetadata | None = None,
    ) -> WorkerCapabilitiesResponse:
        payload = await self._request_json(
            "GET",
            instance_config,
            instance_config.capabilities_path,
            operation=RemoteWorkerOperation.CAPABILITIES,
            metadata=metadata,
        )
        return self._validate_payload(
            payload,
            WorkerCapabilitiesResponse,
            operation=RemoteWorkerOperation.CAPABILITIES,
            instance_id=instance_config.instance_id,
            request_metadata=metadata,
        )

    async def warmup(
        self,
        instance_config: BackendInstanceConfig,
        *,
        model_id: str | None = None,
        metadata: WorkerRequestMetadata | None = None,
    ) -> WorkerWarmupResponse:
        payload = await self._request_json(
            "POST",
            instance_config,
            instance_config.warmup_path,
            operation=RemoteWorkerOperation.WARMUP,
            json_body=WorkerWarmupRequest(
                model_id=model_id,
                transport_metadata=metadata,
            ).model_dump(mode="json"),
            metadata=metadata,
        )
        return self._validate_payload(
            payload,
            WorkerWarmupResponse,
            operation=RemoteWorkerOperation.WARMUP,
            instance_id=instance_config.instance_id,
            request_metadata=metadata,
        )

    async def generate(
        self,
        instance_config: BackendInstanceConfig,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        metadata: WorkerRequestMetadata | None = None,
    ) -> WorkerGenerateResponse:
        payload = await self._request_json(
            "POST",
            instance_config,
            instance_config.chat_completions_path,
            operation=RemoteWorkerOperation.GENERATE,
            json_body=WorkerGenerateRequest(
                request=request,
                context=context,
                transport_metadata=metadata,
            ).model_dump(mode="json"),
            metadata=metadata,
        )
        return self._validate_payload(
            payload,
            WorkerGenerateResponse,
            operation=RemoteWorkerOperation.GENERATE,
            instance_id=instance_config.instance_id,
            request_metadata=metadata,
        )

    async def stream_generate(
        self,
        instance_config: BackendInstanceConfig,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        metadata: WorkerRequestMetadata | None = None,
    ) -> AsyncIterator[WorkerStreamChunkResponse]:
        client = self._client or self._build_client(instance_config)
        close_client = self._client is None
        try:
            async with client.stream(
                "POST",
                self._url_for(instance_config, instance_config.stream_chat_completions_path),
                json=WorkerGenerateRequest(
                    request=request,
                    context=context,
                    transport_metadata=metadata,
                ).model_dump(mode="json"),
                headers=self._headers_for_metadata(metadata),
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise RemoteWorkerTransportError(
                        f"remote worker streaming request failed with status "
                        f"{exc.response.status_code}",
                        kind=RemoteWorkerErrorKind.HTTP_STATUS,
                        operation=RemoteWorkerOperation.STREAM,
                        instance_id=instance_config.instance_id,
                        status_code=exc.response.status_code,
                        retryable=exc.response.status_code >= 500,
                    ) from exc

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise RemoteWorkerResponseError(
                            "remote worker stream returned malformed JSON",
                            kind=RemoteWorkerErrorKind.INVALID_JSON,
                            operation=RemoteWorkerOperation.STREAM,
                            instance_id=instance_config.instance_id,
                        ) from exc
                    yield self._validate_payload(
                        payload,
                        WorkerStreamChunkResponse,
                        operation=RemoteWorkerOperation.STREAM,
                        instance_id=instance_config.instance_id,
                        request_metadata=metadata,
                    )
        except httpx.TimeoutException as exc:
            raise RemoteWorkerTransportError(
                "remote worker streaming request timed out",
                kind=RemoteWorkerErrorKind.TIMEOUT,
                operation=RemoteWorkerOperation.STREAM,
                instance_id=instance_config.instance_id,
                retryable=True,
            ) from exc
        except httpx.ConnectError as exc:
            raise RemoteWorkerTransportError(
                f"remote worker streaming request failed: {exc}",
                kind=RemoteWorkerErrorKind.CONNECT,
                operation=RemoteWorkerOperation.STREAM,
                instance_id=instance_config.instance_id,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise RemoteWorkerTransportError(
                f"remote worker streaming request failed: {exc}",
                kind=RemoteWorkerErrorKind.REQUEST,
                operation=RemoteWorkerOperation.STREAM,
                instance_id=instance_config.instance_id,
                retryable=True,
            ) from exc
        finally:
            if close_client:
                await client.aclose()

    def _build_client(self, instance_config: BackendInstanceConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=instance_config.request_timeout_seconds,
                connect=instance_config.connect_timeout_seconds,
            )
        )

    async def _request_json(
        self,
        method: str,
        instance_config: BackendInstanceConfig,
        path: str,
        *,
        operation: RemoteWorkerOperation,
        json_body: dict[str, object] | None = None,
        metadata: WorkerRequestMetadata | None = None,
    ) -> dict[str, object]:
        client = self._client or self._build_client(instance_config)
        close_client = self._client is None
        try:
            response = await client.request(
                method,
                self._url_for(instance_config, path),
                json=json_body,
                headers=self._headers_for_metadata(metadata),
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RemoteWorkerTransportError(
                    f"remote worker request failed with status {exc.response.status_code}",
                    kind=RemoteWorkerErrorKind.HTTP_STATUS,
                    operation=operation,
                    instance_id=instance_config.instance_id,
                    status_code=exc.response.status_code,
                    retryable=exc.response.status_code >= 500,
                ) from exc
            payload = response.json()
            if not isinstance(payload, dict):
                raise RemoteWorkerResponseError(
                    "remote worker response must be a JSON object",
                    kind=RemoteWorkerErrorKind.INVALID_JSON,
                    operation=operation,
                    instance_id=instance_config.instance_id,
                )
            return payload
        except httpx.TimeoutException as exc:
            raise RemoteWorkerTransportError(
                "remote worker request timed out",
                kind=RemoteWorkerErrorKind.TIMEOUT,
                operation=operation,
                instance_id=instance_config.instance_id,
                retryable=True,
            ) from exc
        except httpx.ConnectError as exc:
            raise RemoteWorkerTransportError(
                f"remote worker request failed: {exc}",
                kind=RemoteWorkerErrorKind.CONNECT,
                operation=operation,
                instance_id=instance_config.instance_id,
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise RemoteWorkerTransportError(
                f"remote worker request failed: {exc}",
                kind=RemoteWorkerErrorKind.REQUEST,
                operation=operation,
                instance_id=instance_config.instance_id,
                retryable=True,
            ) from exc
        except json.JSONDecodeError as exc:
            raise RemoteWorkerResponseError(
                "remote worker returned malformed JSON",
                kind=RemoteWorkerErrorKind.INVALID_JSON,
                operation=operation,
                instance_id=instance_config.instance_id,
            ) from exc
        finally:
            if close_client:
                await client.aclose()

    def _validate_payload(
        self,
        payload: dict[str, object],
        model_type: type[TWorkerModel],
        *,
        operation: RemoteWorkerOperation,
        instance_id: str,
        request_metadata: WorkerRequestMetadata | None,
    ) -> TWorkerModel:
        try:
            parsed = model_type.model_validate(payload)
        except ValidationError as exc:
            raise RemoteWorkerResponseError(
                f"remote worker returned malformed {operation.value} payload",
                kind=RemoteWorkerErrorKind.INVALID_PAYLOAD,
                operation=operation,
                instance_id=instance_id,
            ) from exc
        self._validate_response_metadata(
            request_metadata=request_metadata,
            response_metadata=getattr(parsed, "transport_metadata", None),
            operation=operation,
            instance_id=instance_id,
        )
        return parsed

    def _validate_response_metadata(
        self,
        *,
        request_metadata: WorkerRequestMetadata | None,
        response_metadata: WorkerResponseMetadata | None,
        operation: RemoteWorkerOperation,
        instance_id: str,
    ) -> None:
        if request_metadata is None or response_metadata is None:
            return
        if response_metadata.request_id != request_metadata.request_id:
            raise RemoteWorkerResponseError(
                "remote worker response request_id did not match the request metadata",
                kind=RemoteWorkerErrorKind.PROTOCOL,
                operation=operation,
                instance_id=instance_id,
            )

    def _headers_for_metadata(
        self,
        metadata: WorkerRequestMetadata | None,
    ) -> dict[str, str]:
        if metadata is None:
            return {}
        headers = {"x-request-id": metadata.request_id}
        if metadata.trace_id is not None:
            headers["x-trace-id"] = metadata.trace_id
        if metadata.timeout_ms is not None:
            headers["x-switchyard-timeout-ms"] = str(metadata.timeout_ms)
        return headers

    def _url_for(self, instance_config: BackendInstanceConfig, path: str) -> str:
        return f"{instance_config.base_url.rstrip('/')}{path}"


class RemoteWorkerAdapter:
    """Backend adapter that invokes a Switchyard worker over HTTP."""

    def __init__(
        self,
        model_config: LocalModelConfig,
        *,
        instance_config: BackendInstanceConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model_config = model_config
        self.backend_type = model_config.backend_type
        self.name = f"remote-worker:{model_config.alias}"
        resolved_instances = self._resolve_instance_configs(model_config, instance_config)
        if not resolved_instances:
            msg = (
                f"remote worker adapter for alias {model_config.alias!r} requires at least one "
                "network-addressable instance"
            )
            raise ValueError(msg)
        self._instance_configs = resolved_instances
        self._client = client
        self._transport = RemoteWorkerClient(client=client)

    async def health(self) -> BackendHealth:
        statuses = await self._describe_instances()
        return self._aggregate_health(statuses)

    async def capabilities(self) -> BackendCapabilities:
        instance_config = await self._select_instance_config()
        parsed = await self._transport.capabilities(
            instance_config,
            metadata=self._request_metadata(
                request_id=f"capabilities:{instance_config.instance_id}",
            ),
        )
        capabilities = parsed.capabilities.model_copy(deep=True)
        capabilities.configured_priority = self.model_config.configured_priority
        capabilities.configured_weight = self.model_config.configured_weight
        capabilities.serving_targets = [self.model_config.serving_target or self.model_config.alias]
        capabilities.model_aliases = {
            self.model_config.serving_target or self.model_config.alias: (
                self.model_config.model_identifier
            )
        }
        capabilities.default_model = self.model_config.serving_target or self.model_config.alias
        if not capabilities.model_ids:
            capabilities.model_ids = [
                self.model_config.alias,
                self.model_config.model_identifier,
            ]
        if capabilities.device_class is DeviceClass.REMOTE:
            capabilities.device_class = instance_config.device_class or DeviceClass.REMOTE
        capabilities.execution_mode = self.model_config.execution_mode
        if capabilities.runtime is None and self.model_config.runtime is not None:
            capabilities.runtime = self.model_config.runtime.model_copy(deep=True)
        if capabilities.gpu is None and self.model_config.gpu is not None:
            capabilities.gpu = self.model_config.gpu.model_copy(deep=True)
        capabilities.placement = self.model_config.placement.model_copy(deep=True)
        capabilities.cost_profile = self.model_config.cost_profile.model_copy(deep=True)
        capabilities.readiness_hints = self.model_config.readiness_hints.model_copy(deep=True)
        capabilities.trust = self.model_config.trust.model_copy(deep=True)
        capabilities.network_characteristics = self.model_config.network_characteristics.model_copy(
            deep=True
        )
        return capabilities

    async def status(self) -> BackendStatusSnapshot:
        try:
            capabilities = await self.capabilities()
            capabilities_error: str | None = None
        except RemoteWorkerError as exc:
            capabilities = self._fallback_capabilities()
            capabilities_error = str(exc)
        statuses = await self._describe_instances(default_device_class=capabilities.device_class)
        health = self._aggregate_health(statuses)
        if capabilities_error is not None:
            health.state = BackendHealthState.UNAVAILABLE
            health.detail = "remote worker capabilities check failed"
            health.last_error = capabilities_error
        active_requests = sum(status.active_requests for status in statuses)
        queue_depth = sum(status.queue_depth for status in statuses)
        preferred_instance = self._select_status(statuses)
        return BackendStatusSnapshot(
            name=self.name,
            deployment=BackendDeployment(
                name=self.name,
                backend_type=self.model_config.backend_type,
                engine_type=capabilities.engine_type,
                model_identifier=self.model_config.model_identifier,
                serving_targets=[self.model_config.serving_target or self.model_config.alias],
                configured_priority=self.model_config.configured_priority,
                configured_weight=self.model_config.configured_weight,
                deployment_profile=self.model_config.deployment_profile,
                execution_mode=self.model_config.execution_mode,
                environment=self.model_config.environment,
                placement=self.model_config.placement.model_copy(deep=True),
                cost_profile=self.model_config.cost_profile.model_copy(deep=True),
                readiness_hints=self.model_config.readiness_hints.model_copy(deep=True),
                runtime=(
                    capabilities.runtime.model_copy(deep=True)
                    if capabilities.runtime is not None
                    else self.model_config.runtime.model_copy(deep=True)
                    if self.model_config.runtime is not None
                    else None
                ),
                gpu=(
                    capabilities.gpu.model_copy(deep=True)
                    if capabilities.gpu is not None
                    else self.model_config.gpu.model_copy(deep=True)
                    if self.model_config.gpu is not None
                    else None
                ),
                request_features=capabilities.request_features.model_copy(deep=True),
                observed_capacity=(
                    preferred_instance.instance.observed_capacity.model_copy(deep=True)
                    if preferred_instance is not None
                    and preferred_instance.instance.observed_capacity is not None
                    else None
                ),
                instances=[status.instance for status in statuses],
            ),
            instance_inventory=[status.instance for status in statuses],
            capabilities=capabilities,
            health=health,
            active_requests=active_requests,
            queue_depth=queue_depth,
            metadata={
                "adapter_kind": "remote_worker",
                "execution_mode": "remote_worker",
                "worker_transport": preferred_instance.instance.endpoint.transport.value
                if preferred_instance is not None
                else self._instance_configs[0].transport.value,
                "model_identifier": self.model_config.model_identifier,
                "preferred_instance_id": (
                    preferred_instance.instance.instance_id
                    if preferred_instance is not None
                    else ""
                ),
            },
        )

    async def warmup(self, model_id: str | None = None) -> None:
        failures: list[str] = []
        for instance_config in self._instance_configs:
            try:
                await self._transport.warmup(
                    instance_config,
                    model_id=model_id,
                    metadata=self._request_metadata(
                        request_id=f"warmup:{instance_config.instance_id}",
                    ),
                )
            except RemoteWorkerError as exc:
                failures.append(f"{instance_config.instance_id}: {exc}")
        if failures:
            raise RemoteWorkerTransportError(
                "remote worker warmup failed for one or more instances: " + "; ".join(failures),
                kind=RemoteWorkerErrorKind.REQUEST,
                operation=RemoteWorkerOperation.WARMUP,
            )

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        response, _ = await self.generate_with_instance(request, context)
        return response

    async def generate_with_instance(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> tuple[ChatCompletionResponse, str]:
        instance_config = await self._select_instance_config()
        response = await self._generate_from_instance(
            instance_config,
            request=request,
            context=context,
        )
        return response, instance_config.instance_id

    async def _generate_from_instance(
        self,
        instance_config: BackendInstanceConfig,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        parsed = await self._transport.generate(
            instance_config,
            request=request,
            context=context,
            metadata=self._request_metadata(
                request_id=context.request_id,
                trace_id=context.trace_id,
                timeout_seconds=instance_config.request_timeout_seconds,
            ),
        )
        response = parsed.response.model_copy(deep=True)
        response.backend_name = self.name
        return response

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        _, chunk_stream = await self.stream_generate_with_instance(request, context)
        async for chunk in chunk_stream:
            yield chunk

    async def stream_generate_with_instance(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> tuple[str, AsyncIterator[ChatCompletionChunk]]:
        instance_config = await self._select_instance_config()
        return instance_config.instance_id, self._stream_generate_from_instance(
            instance_config,
            request=request,
            context=context,
        )

    async def _stream_generate_from_instance(
        self,
        instance_config: BackendInstanceConfig,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        async for parsed in self._transport.stream_generate(
            instance_config,
            request=request,
            context=context,
            metadata=self._request_metadata(
                request_id=context.request_id,
                trace_id=context.trace_id,
                timeout_seconds=instance_config.request_timeout_seconds,
            ),
        ):
            chunk = parsed.chunk.model_copy(deep=True)
            chunk.backend_name = self.name
            yield chunk

    def _resolve_instance_configs(
        self,
        model_config: LocalModelConfig,
        explicit_instance: BackendInstanceConfig | None,
    ) -> tuple[BackendInstanceConfig, ...]:
        if explicit_instance is not None:
            return (explicit_instance,)
        resolved: list[BackendInstanceConfig] = []
        for instance in model_config.instances:
            if instance.transport is not WorkerTransportType.IN_PROCESS:
                resolved.append(instance)
        return tuple(resolved)

    def _fallback_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_type=self.model_config.backend_type,
            engine_type=EngineType.REMOTE_OPENAI,
            device_class=DeviceClass.REMOTE,
            execution_mode=self.model_config.execution_mode,
            model_ids=[self.model_config.alias, self.model_config.model_identifier],
            serving_targets=[self.model_config.serving_target or self.model_config.alias],
            max_context_tokens=8192,
            supports_streaming=False,
            concurrency_limit=1,
            configured_priority=self.model_config.configured_priority,
            configured_weight=self.model_config.configured_weight,
            quality_hint=QualityHint.BALANCED,
            performance_hint=PerformanceHint.BALANCED,
            model_aliases={
                self.model_config.serving_target or self.model_config.alias: (
                    self.model_config.model_identifier
                )
            },
            default_model=self.model_config.serving_target or self.model_config.alias,
            placement=self.model_config.placement.model_copy(deep=True),
            cost_profile=self.model_config.cost_profile.model_copy(deep=True),
            readiness_hints=self.model_config.readiness_hints.model_copy(deep=True),
            trust=self.model_config.trust.model_copy(deep=True),
            network_characteristics=self.model_config.network_characteristics.model_copy(
                deep=True
            ),
            runtime=self.model_config.runtime.model_copy(deep=True)
            if self.model_config.runtime is not None
            else None,
            gpu=self.model_config.gpu.model_copy(deep=True)
            if self.model_config.gpu is not None
            else None,
        )

    async def _select_instance_config(self) -> BackendInstanceConfig:
        statuses = await self._describe_instances()
        selected = self._select_status(statuses)
        if selected is None:
            return self._instance_configs[0]
        for instance_config in self._instance_configs:
            if instance_config.instance_id == selected.instance.instance_id:
                return instance_config
        return self._instance_configs[0]

    async def _describe_instances(
        self,
        *,
        default_device_class: DeviceClass | None = None,
    ) -> list[RemoteWorkerInstanceStatus]:
        resolved_device_class = default_device_class or DeviceClass.REMOTE
        statuses: list[RemoteWorkerInstanceStatus] = []
        for instance_config in self._instance_configs:
            health = await self._health_for_instance(instance_config)
            readiness: WorkerReadinessResponse | None = None
            ready = False
            active_requests = 0
            queue_depth = 0
            try:
                readiness = await self._transport.readiness(
                    instance_config,
                    metadata=self._request_metadata(
                        request_id=f"readiness:{instance_config.instance_id}",
                    ),
                )
                health = readiness.health
                ready = readiness.ready and health.state is not BackendHealthState.UNAVAILABLE
                active_requests = readiness.active_requests
                queue_depth = readiness.queue_depth
            except RemoteWorkerError as exc:
                health.state = BackendHealthState.UNAVAILABLE
                health.detail = "remote worker readiness check failed"
                health.last_error = str(exc)
            instance = instance_config.to_backend_instance(
                backend_type=self.model_config.backend_type,
                default_device_class=instance_config.device_class or resolved_device_class,
                model_identifier=self.model_config.model_identifier,
            )
            instance.health = health.model_copy(deep=True)
            if readiness is not None:
                if readiness.runtime is not None:
                    instance.runtime = readiness.runtime.model_copy(deep=True)
                if readiness.gpu is not None:
                    instance.gpu = readiness.gpu.model_copy(deep=True)
                if readiness.observed_capacity is not None:
                    instance.observed_capacity = readiness.observed_capacity.model_copy(deep=True)
            instance.last_seen_at = (
                health.checked_at if health.state is not BackendHealthState.UNAVAILABLE else None
            )
            instance.registration.last_heartbeat_at = instance.last_seen_at
            instance.tags = sorted(set(instance.tags))
            instance.metadata.update(
                {
                    "deployment_name": self.name,
                    "ready": str(ready).lower(),
                    "active_requests": str(active_requests),
                    "queue_depth": str(queue_depth),
                }
            )
            statuses.append(
                RemoteWorkerInstanceStatus(
                    instance=instance,
                    ready=ready,
                    active_requests=active_requests,
                    queue_depth=queue_depth,
                )
            )
        return statuses

    async def _health_for_instance(self, instance_config: BackendInstanceConfig) -> BackendHealth:
        try:
            response = await self._transport.health(
                instance_config,
                metadata=self._request_metadata(
                    request_id=f"health:{instance_config.instance_id}",
                ),
            )
        except RemoteWorkerError as exc:
            return BackendHealth(
                state=BackendHealthState.UNAVAILABLE,
                detail="remote worker health check failed",
                last_error=str(exc),
            )
        return response.health

    def _aggregate_health(self, statuses: list[RemoteWorkerInstanceStatus]) -> BackendHealth:
        if not statuses:
            return BackendHealth(
                state=BackendHealthState.UNAVAILABLE,
                detail="remote worker has no configured instances",
            )
        selected = self._select_status(statuses)
        if selected is not None:
            return selected.instance.health.model_copy(deep=True)  # type: ignore[union-attr]
        return statuses[0].instance.health.model_copy(deep=True)  # type: ignore[union-attr]

    def _select_status(
        self,
        statuses: list[RemoteWorkerInstanceStatus],
    ) -> RemoteWorkerInstanceStatus | None:
        eligible = [
            status
            for status in statuses
            if status.ready
            and status.instance.health is not None
            and status.instance.health.state is not BackendHealthState.UNAVAILABLE
        ]
        if not eligible:
            eligible = [
                status
                for status in statuses
                if status.instance.health is not None
                and status.instance.health.state is not BackendHealthState.UNAVAILABLE
            ]
        if not eligible:
            return None
        return sorted(
            eligible,
            key=lambda status: (
                status.queue_depth,
                status.active_requests,
                (
                    status.instance.health.latency_ms
                    if status.instance.health is not None
                    and status.instance.health.latency_ms is not None
                    else 0.0
                ),
                status.instance.instance_id,
            ),
        )[0]

    def _request_metadata(
        self,
        *,
        request_id: str,
        trace_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> WorkerRequestMetadata:
        timeout_ms = None
        if timeout_seconds is not None:
            timeout_ms = max(1, math.ceil(timeout_seconds * 1000))
        return WorkerRequestMetadata(
            request_id=request_id,
            trace_id=trace_id,
            timeout_ms=timeout_ms,
        )
