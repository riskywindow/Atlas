"""HTTP adapter for Switchyard-internal remote workers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
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


class RemoteWorkerTransportError(RemoteWorkerError):
    """Raised when a network call to a worker fails."""


class RemoteWorkerResponseError(RemoteWorkerError):
    """Raised when a worker returns malformed or invalid data."""


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

    async def health(self) -> BackendHealth:
        statuses = await self._describe_instances()
        return self._aggregate_health(statuses)

    async def capabilities(self) -> BackendCapabilities:
        instance_config = await self._select_instance_config()
        payload = await self._request_json(
            "GET",
            instance_config.capabilities_path,
            instance_config=instance_config,
        )
        parsed = self._validate_payload(payload, WorkerCapabilitiesResponse, "capabilities")
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
                environment=self.model_config.environment,
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
                payload = await self._request_json(
                    "POST",
                    instance_config.warmup_path,
                    json_body=WorkerWarmupRequest(model_id=model_id).model_dump(mode="json"),
                    instance_config=instance_config,
                )
                self._validate_payload(payload, WorkerWarmupResponse, "warmup")
            except RemoteWorkerError as exc:
                failures.append(f"{instance_config.instance_id}: {exc}")
        if failures:
            raise RemoteWorkerTransportError(
                "remote worker warmup failed for one or more instances: " + "; ".join(failures)
            )

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        instance_config = await self._select_instance_config()
        payload = await self._request_json(
            "POST",
            instance_config.chat_completions_path,
            json_body=WorkerGenerateRequest(request=request, context=context).model_dump(
                mode="json"
            ),
            instance_config=instance_config,
        )
        parsed = self._validate_payload(payload, WorkerGenerateResponse, "generate")
        response = parsed.response.model_copy(deep=True)
        response.backend_name = self.name
        return response

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        instance_config = await self._select_instance_config()
        client = self._client or self._build_client()
        close_client = self._client is None
        try:
            async with client.stream(
                "POST",
                self._url_for(
                    instance_config,
                    instance_config.stream_chat_completions_path,
                ),
                json=WorkerGenerateRequest(request=request, context=context).model_dump(
                    mode="json"
                ),
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise RemoteWorkerTransportError(
                        f"remote worker streaming request failed with status "
                        f"{exc.response.status_code}"
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
                            "remote worker stream returned malformed JSON"
                        ) from exc
                    parsed = self._validate_payload(payload, WorkerStreamChunkResponse, "stream")
                    chunk = parsed.chunk.model_copy(deep=True)
                    chunk.backend_name = self.name
                    yield chunk
        except httpx.TimeoutException as exc:
            raise RemoteWorkerTransportError("remote worker streaming request timed out") from exc
        except httpx.RequestError as exc:
            raise RemoteWorkerTransportError(
                f"remote worker streaming request failed: {exc}"
            ) from exc
        finally:
            if close_client:
                await client.aclose()

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
        )

    def _build_client(self) -> httpx.AsyncClient:
        default_instance = self._instance_configs[0]
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=default_instance.request_timeout_seconds,
                connect=default_instance.connect_timeout_seconds,
            )
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        instance_config: BackendInstanceConfig | None = None,
    ) -> dict[str, object]:
        resolved_instance = instance_config or self._instance_configs[0]
        client = self._client or self._build_client()
        close_client = self._client is None
        try:
            response = await client.request(
                method,
                self._url_for(resolved_instance, path),
                json=json_body,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RemoteWorkerTransportError(
                    f"remote worker request failed with status {exc.response.status_code}"
                ) from exc
            payload = response.json()
            if not isinstance(payload, dict):
                raise RemoteWorkerResponseError("remote worker response must be a JSON object")
            return payload
        except httpx.TimeoutException as exc:
            raise RemoteWorkerTransportError("remote worker request timed out") from exc
        except httpx.RequestError as exc:
            raise RemoteWorkerTransportError(f"remote worker request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RemoteWorkerResponseError("remote worker returned malformed JSON") from exc
        finally:
            if close_client:
                await client.aclose()

    def _url_for(self, instance_config: BackendInstanceConfig, path: str) -> str:
        return f"{instance_config.base_url.rstrip('/')}{path}"

    def _validate_payload(
        self,
        payload: dict[str, object],
        model_type: type[TWorkerModel],
        operation: str,
    ) -> TWorkerModel:
        try:
            return model_type.model_validate(payload)
        except ValidationError as exc:
            raise RemoteWorkerResponseError(
                f"remote worker returned malformed {operation} payload"
            ) from exc

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
            ready = False
            active_requests = 0
            queue_depth = 0
            try:
                payload = await self._request_json(
                    "GET",
                    instance_config.readiness_path,
                    instance_config=instance_config,
                )
                readiness = self._validate_payload(payload, WorkerReadinessResponse, "readiness")
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
            payload = await self._request_json(
                "GET",
                instance_config.health_path,
                instance_config=instance_config,
            )
            parsed = WorkerHealthResponse.model_validate(payload)
        except RemoteWorkerError as exc:
            return BackendHealth(
                state=BackendHealthState.UNAVAILABLE,
                detail="remote worker health check failed",
                last_error=str(exc),
            )
        return parsed.health

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
