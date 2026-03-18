"""MLX-LM adapter implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from hashlib import sha256

from switchyard.adapters.base import BackendAdapter
from switchyard.config import LocalModelConfig
from switchyard.runtime.base import (
    ChatModelRuntime,
    RuntimeGenerationResult,
    RuntimeStreamChunk,
)
from switchyard.runtime.mlx_lm import MLXLMChatRuntime
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
    PerformanceHint,
    QualityHint,
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


class MLXLMAdapter(BackendAdapter):
    """Backend adapter that translates Switchyard requests into MLX runtime calls."""

    def __init__(
        self,
        model_config: LocalModelConfig,
        *,
        runtime: ChatModelRuntime | None = None,
    ) -> None:
        if model_config.backend_type is not BackendType.MLX_LM:
            msg = "MLXLMAdapter requires a LocalModelConfig with backend_type='mlx_lm'"
            raise ValueError(msg)

        self.model_config = model_config
        self.name = f"mlx-lm:{model_config.alias}"
        self.backend_type = BackendType.MLX_LM
        self._runtime = runtime or MLXLMChatRuntime(model_config)

    async def health(self) -> BackendHealth:
        """Return the current health for the MLX runtime."""

        runtime_health = self._runtime.health()
        return BackendHealth(
            state=runtime_health.state,
            load_state=runtime_health.load_state,
            detail=runtime_health.detail,
            last_error=runtime_health.last_error,
            warmed_models=self._warmed_models(runtime_health.load_state),
        )

    async def capabilities(self) -> BackendCapabilities:
        """Return the configured MLX capabilities."""

        return BackendCapabilities(
            backend_type=self.backend_type,
            engine_type=EngineType.MLX,
            device_class=DeviceClass.APPLE_GPU,
            model_ids=[self.model_config.alias, self.model_config.model_identifier],
            serving_targets=[self.model_config.serving_target or self.model_config.alias],
            max_context_tokens=32768,
            supports_streaming=True,
            concurrency_limit=1,
            configured_priority=self.model_config.configured_priority,
            configured_weight=self.model_config.configured_weight,
            quality_tier=4,
            quality_hint=QualityHint.PREMIUM,
            performance_hint=PerformanceHint.BALANCED,
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
        )

    async def status(self) -> BackendStatusSnapshot:
        """Return a point-in-time snapshot for routing and readiness."""

        return BackendStatusSnapshot(
            name=self.name,
            deployment=BackendDeployment(
                name=self.name,
                backend_type=self.backend_type,
                engine_type=EngineType.MLX,
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
                    build_metadata=dict(self.model_config.build_metadata),
                ),
            ),
            capabilities=await self.capabilities(),
            health=await self.health(),
            metadata={
                "model_alias": self.model_config.serving_target or self.model_config.alias,
                "model_identifier": self.model_config.model_identifier,
                "deployment_profile": self.model_config.deployment_profile.value,
                "environment": self.model_config.environment,
            },
        )

    async def warmup(self, model_id: str | None = None) -> None:
        """Warm the configured MLX model."""

        self._runtime.warmup()

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        """Generate a non-streaming chat completion response."""

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
                    finish_reason=result.finish_reason,
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
        """Yield streaming chunks from the MLX runtime."""

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
        return f"mlxcmpl_{digest[:16]}"

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
