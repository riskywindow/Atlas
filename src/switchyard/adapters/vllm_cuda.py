"""vLLM-CUDA adapter implementation for remote Linux/NVIDIA workers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol

from switchyard.adapters.base import BackendAdapter
from switchyard.config import LocalModelConfig
from switchyard.runtime.base import (
    RuntimeGenerationResult,
    RuntimeHealthSnapshot,
    RuntimeStreamChunk,
    UnsupportedRequestError,
)
from switchyard.runtime.vllm_cuda import VLLMCUDAChatRuntime, VLLMCUDARuntimeCapabilities
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendDeployment,
    BackendHealth,
    BackendImageMetadata,
    BackendLoadState,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
    EngineType,
)
from switchyard.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatRole,
    UsageStats,
)
from switchyard.schemas.routing import RequestContext


class VLLMCUDARuntime(Protocol):
    """Runtime shape required by the concrete vLLM-CUDA adapter."""

    backend_type: BackendType

    def health(self) -> RuntimeHealthSnapshot: ...
    def capabilities(self) -> VLLMCUDARuntimeCapabilities: ...
    def warmup(self) -> None: ...
    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult: ...
    def stream_generate(self, request: ChatCompletionRequest) -> Iterator[RuntimeStreamChunk]: ...


class VLLMCUDAAdapter(BackendAdapter):
    """Backend adapter that translates Switchyard requests into vLLM-CUDA runtime calls."""

    def __init__(
        self,
        model_config: LocalModelConfig,
        *,
        runtime: VLLMCUDARuntime | None = None,
    ) -> None:
        if model_config.backend_type is not BackendType.VLLM_CUDA:
            msg = (
                "VLLMCUDAAdapter requires a LocalModelConfig with "
                "backend_type='vllm_cuda'"
            )
            raise ValueError(msg)

        self.model_config = model_config
        self.name = f"vllm-cuda:{model_config.alias}"
        self.backend_type = BackendType.VLLM_CUDA
        self._runtime = runtime or VLLMCUDAChatRuntime(model_config)

    async def health(self) -> BackendHealth:
        runtime_health = self._runtime.health()
        return BackendHealth(
            state=runtime_health.state,
            load_state=runtime_health.load_state,
            detail=runtime_health.detail,
            last_error=runtime_health.last_error,
            warmed_models=self._warmed_models(runtime_health.load_state),
        )

    async def capabilities(self) -> BackendCapabilities:
        runtime_capabilities = self._runtime.capabilities()
        return BackendCapabilities(
            backend_type=self.backend_type,
            engine_type=EngineType.VLLM_CUDA,
            device_class=DeviceClass.NVIDIA_GPU,
            runtime=runtime_capabilities.runtime.model_copy(deep=True),
            gpu=runtime_capabilities.gpu.model_copy(deep=True),
            model_ids=[self.model_config.alias, self.model_config.model_identifier],
            serving_targets=[self.model_config.serving_target or self.model_config.alias],
            max_context_tokens=runtime_capabilities.max_context_tokens,
            supports_streaming=runtime_capabilities.request_features.supports_streaming,
            concurrency_limit=runtime_capabilities.concurrency_limit,
            configured_priority=self.model_config.configured_priority,
            configured_weight=self.model_config.configured_weight,
            quality_tier=4,
            quality_hint=runtime_capabilities.quality_hint,
            performance_hint=runtime_capabilities.performance_hint,
            execution_mode=self.model_config.execution_mode,
            model_aliases={
                self.model_config.serving_target or self.model_config.alias: (
                    self.model_config.model_identifier
                )
            },
            default_model=self.model_config.serving_target or self.model_config.alias,
            warmup_required=self.model_config.warmup.enabled,
            placement=self.model_config.placement.model_copy(deep=True),
            cost_profile=self.model_config.cost_profile.model_copy(deep=True),
            readiness_hints=self.model_config.readiness_hints.model_copy(deep=True),
            trust=self.model_config.trust.model_copy(deep=True),
            network_characteristics=self.model_config.network_characteristics.model_copy(
                deep=True
            ),
            request_features=runtime_capabilities.request_features.model_copy(deep=True),
        )

    async def status(self) -> BackendStatusSnapshot:
        capabilities = await self.capabilities()
        return BackendStatusSnapshot(
            name=self.name,
            deployment=BackendDeployment(
                name=self.name,
                backend_type=self.backend_type,
                engine_type=EngineType.VLLM_CUDA,
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
                build_metadata=BackendImageMetadata(
                    image_tag=self.model_config.image_tag,
                    build_metadata={
                        **dict(self.model_config.build_metadata),
                        "runtime_boundary": "dependency_gated_vllm_cuda_worker",
                    },
                ),
                runtime=capabilities.runtime.model_copy(deep=True)
                if capabilities.runtime is not None
                else None,
                gpu=(
                    capabilities.gpu.model_copy(deep=True)
                    if capabilities.gpu is not None
                    else None
                ),
                request_features=capabilities.request_features.model_copy(deep=True),
            ),
            capabilities=capabilities,
            health=await self.health(),
            metadata={
                "model_alias": self.model_config.serving_target or self.model_config.alias,
                "model_identifier": self.model_config.model_identifier,
                "deployment_profile": self.model_config.deployment_profile.value,
                "environment": self.model_config.environment,
            },
        )

    async def warmup(self, model_id: str | None = None) -> None:
        self._runtime.warmup()

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        capabilities = await self.capabilities()
        self._validate_request(request=request, capabilities=capabilities)
        result = self._runtime.generate(request)
        response_id = self._build_response_id(request=request, context=context)
        usage = self._build_usage(request=request, result=result)
        return ChatCompletionResponse(
            id=response_id,
            created_at=datetime.now(UTC),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=ChatRole.ASSISTANT, content=result.text),
                )
            ],
            usage=usage,
            backend_name=self.name,
        )

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        capabilities = await self.capabilities()
        self._validate_request(request=request, capabilities=capabilities)

        response_id = self._build_response_id(request=request, context=context)
        created_at = datetime.now(UTC)
        yield ChatCompletionChunk(
            id=response_id,
            created_at=created_at,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(role=ChatRole.ASSISTANT),
                )
            ],
            backend_name=self.name,
        )

        for chunk in self._runtime.stream_generate(request):
            yield self._build_stream_chunk(
                request=request,
                response_id=response_id,
                created_at=created_at,
                chunk=chunk,
            )

    def _validate_request(
        self,
        *,
        request: ChatCompletionRequest,
        capabilities: BackendCapabilities,
    ) -> None:
        if request.stream and not capabilities.request_features.supports_streaming:
            raise UnsupportedRequestError(
                message="streaming is not supported by this vLLM-CUDA worker",
                field_name="stream",
            )
        if (
            any(message.role is ChatRole.SYSTEM for message in request.messages)
            and not capabilities.request_features.supports_system_messages
        ):
            raise UnsupportedRequestError(
                message="system messages are not supported by this vLLM-CUDA worker",
                field_name="messages",
            )

    def _build_stream_chunk(
        self,
        *,
        request: ChatCompletionRequest,
        response_id: str,
        created_at: datetime,
        chunk: RuntimeStreamChunk,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=response_id,
            created_at=created_at,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=chunk.text),
                    finish_reason=chunk.finish_reason,
                )
            ],
            backend_name=self.name,
        )

    def _build_response_id(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> str:
        digest = sha256(
            f"{self.name}:{request.model}:{context.request_id}:{len(request.messages)}".encode()
        ).hexdigest()
        return f"vllmcuda_{digest[:16]}"

    def _build_usage(
        self,
        *,
        request: ChatCompletionRequest,
        result: RuntimeGenerationResult,
    ) -> UsageStats:
        prompt_tokens = result.prompt_tokens
        if prompt_tokens is None:
            prompt_tokens = sum(len(message.content.split()) for message in request.messages)

        completion_tokens = result.completion_tokens
        if completion_tokens is None:
            completion_tokens = len(result.text.split())

        return UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _warmed_models(self, load_state: BackendLoadState) -> list[str]:
        if load_state is BackendLoadState.READY:
            return [self.model_config.alias]
        return []
