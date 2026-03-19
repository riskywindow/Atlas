"""vLLM-CUDA runtime boundary for Linux/NVIDIA remote workers."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

from switchyard.config import LocalModelConfig
from switchyard.runtime.base import (
    RuntimeGenerationResult,
    RuntimeHealthSnapshot,
    RuntimeSamplingParams,
    RuntimeStreamChunk,
)
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendType,
    EngineType,
    GPUDeviceMetadata,
    PerformanceHint,
    QualityHint,
    RequestFeatureSupport,
    RuntimeIdentity,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason

ModuleImporter = Callable[[str], ModuleType]


class VLLMCUDARuntimeError(RuntimeError):
    """Base error for vLLM-CUDA runtime failures."""


class VLLMCUDADependencyError(VLLMCUDARuntimeError):
    """Raised when the optional vLLM dependency is unavailable."""


class VLLMCUDAConfigurationError(VLLMCUDARuntimeError):
    """Raised when vLLM is installed but the runtime is misconfigured."""


@dataclass(frozen=True, slots=True)
class LoadedEngine:
    """Loaded vLLM engine state."""

    engine: object


@dataclass(frozen=True, slots=True)
class VLLMCUDARuntimeCapabilities:
    """Runtime-declared capabilities for a CUDA-backed vLLM worker."""

    runtime: RuntimeIdentity
    gpu: GPUDeviceMetadata
    request_features: RequestFeatureSupport
    max_context_tokens: int
    concurrency_limit: int
    quality_hint: QualityHint
    performance_hint: PerformanceHint
    supports_prefix_cache: bool
    supports_kv_cache_reuse: bool


class VLLMCUDAProvider(Protocol):
    """Provider boundary that isolates direct vLLM-CUDA calls."""

    def ensure_available(self) -> None:
        """Verify that vLLM can be imported."""

    def load(self, model_identifier: str) -> LoadedEngine:
        """Load a vLLM engine."""

    def declare_capabilities(self) -> VLLMCUDARuntimeCapabilities:
        """Return provider-declared capabilities without requiring a loaded engine."""

    def generate_text(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> str:
        """Generate a complete text response."""

    def stream_text(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> Iterator[str]:
        """Yield text segments from streamed generation."""


class ImportedVLLMCUDAProvider:
    """Lazy vLLM provider backed by importlib.

    The exact CUDA-serving invocation remains intentionally narrow here so the worker
    contract can be tested without importing CUDA dependencies in local development or CI.
    """

    def __init__(self, module_importer: ModuleImporter = importlib.import_module) -> None:
        self._module_importer = module_importer
        self._vllm_module: ModuleType | None = None

    def ensure_available(self) -> None:
        self._get_vllm_module()

    def load(self, model_identifier: str) -> LoadedEngine:
        llm_class = getattr(self._get_vllm_module(), "LLM", None)
        if not callable(llm_class):
            msg = "vllm.LLM is unavailable in the installed vLLM package"
            raise VLLMCUDAConfigurationError(msg)
        return LoadedEngine(engine=llm_class(model=model_identifier))

    def declare_capabilities(self) -> VLLMCUDARuntimeCapabilities:
        module = self._get_vllm_module()
        return VLLMCUDARuntimeCapabilities(
            runtime=RuntimeIdentity(
                runtime_family="vllm_cuda",
                runtime_label="vllm_cuda",
                runtime_version=getattr(module, "__version__", None),
                engine_type=EngineType.VLLM_CUDA,
                backend_type=BackendType.VLLM_CUDA,
                api_compatibility="openai_chat_completions",
                metadata={"dependency_mode": "optional_import"},
            ),
            gpu=GPUDeviceMetadata(vendor="nvidia", accelerator_type="cuda", count=1),
            request_features=RequestFeatureSupport(
                supports_streaming=True,
                supports_native_streaming=callable(
                    getattr(module, "stream_generate", None)
                ),
                supports_tools=False,
                supports_system_messages=True,
                limitations=[
                    "Tool calling and JSON-schema response formatting remain feature-gated.",
                ],
            ),
            max_context_tokens=32768,
            concurrency_limit=8,
            quality_hint=QualityHint.PREMIUM,
            performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
            supports_prefix_cache=True,
            supports_kv_cache_reuse=True,
        )

    def generate_text(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> str:
        output = self._invoke_chat(
            loaded_engine=loaded_engine,
            messages=messages,
            params=params,
        )
        return self._coerce_text(output)

    def stream_text(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> Iterator[str]:
        stream_generate = getattr(self._get_vllm_module(), "stream_generate", None)
        if callable(stream_generate):
            sampling_params = self._build_sampling_params(params=params)
            rendered_messages = [
                {"role": message.role.value, "content": message.content} for message in messages
            ]
            responses = stream_generate(
                loaded_engine.engine,
                rendered_messages,
                sampling_params=sampling_params,
            )
            for response in responses:
                chunk_text = self._coerce_text(response)
                if chunk_text:
                    yield chunk_text
            return

        yield self.generate_text(
            loaded_engine=loaded_engine,
            messages=messages,
            params=params,
        )

    def _invoke_chat(
        self,
        *,
        loaded_engine: LoadedEngine,
        messages: Sequence[ChatMessage],
        params: RuntimeSamplingParams,
    ) -> object:
        chat_fn = getattr(loaded_engine.engine, "chat", None)
        if not callable(chat_fn):
            msg = "configured vLLM engine does not expose chat"
            raise VLLMCUDAConfigurationError(msg)

        sampling_params = self._build_sampling_params(params=params)
        rendered_messages = [
            {"role": message.role.value, "content": message.content} for message in messages
        ]
        return chat_fn(rendered_messages, sampling_params=sampling_params)

    def _build_sampling_params(self, *, params: RuntimeSamplingParams) -> object:
        sampling_params_class = getattr(self._get_vllm_module(), "SamplingParams", None)
        if not callable(sampling_params_class):
            msg = "vllm.SamplingParams is unavailable in the installed vLLM package"
            raise VLLMCUDAConfigurationError(msg)

        kwargs: dict[str, object] = {}
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        return sampling_params_class(**kwargs)

    def _get_vllm_module(self) -> ModuleType:
        if self._vllm_module is None:
            try:
                self._vllm_module = self._module_importer("vllm")
            except ImportError as exc:
                msg = "vllm is not installed; install it to enable the vLLM-CUDA worker"
                raise VLLMCUDADependencyError(msg) from exc
        return self._vllm_module

    def _coerce_text(self, output: object) -> str:
        if isinstance(output, str):
            return output

        outputs = getattr(output, "outputs", None)
        if isinstance(outputs, list) and outputs:
            text = getattr(outputs[0], "text", None)
            if isinstance(text, str):
                return text

        if isinstance(output, list) and output:
            first = output[0]
            outputs = getattr(first, "outputs", None)
            if isinstance(outputs, list) and outputs:
                text = getattr(outputs[0], "text", None)
                if isinstance(text, str):
                    return text

        text = getattr(output, "text", None)
        if isinstance(text, str):
            return text

        msg = "vLLM chat returned an unsupported response object"
        raise VLLMCUDAConfigurationError(msg)


class VLLMCUDAChatRuntime:
    """Chat-oriented runtime that isolates Linux/NVIDIA vLLM integration details."""

    backend_type = BackendType.VLLM_CUDA

    def __init__(
        self,
        model_config: LocalModelConfig,
        *,
        provider: VLLMCUDAProvider | None = None,
    ) -> None:
        if model_config.backend_type is not BackendType.VLLM_CUDA:
            msg = (
                "VLLMCUDAChatRuntime requires a LocalModelConfig with "
                "backend_type='vllm_cuda'"
            )
            raise VLLMCUDAConfigurationError(msg)

        self.model_config = model_config
        self._provider = provider or ImportedVLLMCUDAProvider()
        self._loaded_engine: LoadedEngine | None = None
        self._last_error: str | None = None

    def load_model(self) -> None:
        if self._loaded_engine is not None:
            return

        try:
            self._loaded_engine = self._provider.load(self.model_config.model_identifier)
            self._last_error = None
        except VLLMCUDARuntimeError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            msg = (
                "unexpected vLLM-CUDA runtime error while loading model "
                f"{self.model_config.model_identifier!r}: {exc}"
            )
            self._last_error = msg
            raise VLLMCUDARuntimeError(msg) from exc

    def warmup(self) -> None:
        self.load_model()
        if not self.model_config.warmup.enabled:
            return

        self.generate(
            ChatCompletionRequest(
                model=self.model_config.alias,
                messages=[ChatMessage(role=ChatRole.USER, content="ping")],
                max_output_tokens=1,
                temperature=0.0,
                top_p=1.0,
            )
        )

    def health(self) -> RuntimeHealthSnapshot:
        try:
            self._provider.ensure_available()
            capabilities = self.capabilities()
        except VLLMCUDARuntimeError as exc:
            self._last_error = str(exc)
            return RuntimeHealthSnapshot(
                state=BackendHealthState.UNAVAILABLE,
                load_state=BackendLoadState.FAILED,
                detail="vLLM-CUDA runtime unavailable",
                last_error=self._last_error,
            )

        return RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=(
                BackendLoadState.READY
                if self._loaded_engine is not None
                else BackendLoadState.COLD
            ),
            detail=(
                "vLLM-CUDA runtime available"
                if capabilities.request_features.supports_native_streaming
                else "vLLM-CUDA runtime available (streaming via buffered fallback)"
            ),
            last_error=self._last_error,
        )

    def capabilities(self) -> VLLMCUDARuntimeCapabilities:
        try:
            declared = self._provider.declare_capabilities()
        except VLLMCUDARuntimeError:
            declared = VLLMCUDARuntimeCapabilities(
                runtime=RuntimeIdentity(
                    runtime_family="vllm_cuda",
                    runtime_label="vllm_cuda",
                    engine_type=EngineType.VLLM_CUDA,
                    backend_type=BackendType.VLLM_CUDA,
                    api_compatibility="openai_chat_completions",
                    metadata={"dependency_mode": "optional_import"},
                ),
                gpu=GPUDeviceMetadata(vendor="nvidia", accelerator_type="cuda", count=1),
                request_features=RequestFeatureSupport(
                    supports_streaming=True,
                    supports_native_streaming=False,
                    supports_tools=False,
                    supports_system_messages=True,
                    limitations=["Runtime dependency unavailable; native streaming unverified."],
                ),
                max_context_tokens=32768,
                concurrency_limit=8,
                quality_hint=QualityHint.PREMIUM,
                performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
                supports_prefix_cache=False,
                supports_kv_cache_reuse=False,
            )

        runtime = declared.runtime.model_copy(deep=True)
        if self.model_config.runtime is not None:
            runtime = self.model_config.runtime.model_copy(deep=True)
            runtime.engine_type = runtime.engine_type or EngineType.VLLM_CUDA
            runtime.backend_type = runtime.backend_type or BackendType.VLLM_CUDA

        gpu = (
            self.model_config.gpu.model_copy(deep=True)
            if self.model_config.gpu is not None
            else declared.gpu
        )

        return VLLMCUDARuntimeCapabilities(
            runtime=runtime,
            gpu=gpu,
            request_features=declared.request_features.model_copy(deep=True),
            max_context_tokens=declared.max_context_tokens,
            concurrency_limit=declared.concurrency_limit,
            quality_hint=declared.quality_hint,
            performance_hint=declared.performance_hint,
            supports_prefix_cache=declared.supports_prefix_cache,
            supports_kv_cache_reuse=declared.supports_kv_cache_reuse,
        )

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        loaded_engine = self._require_loaded_engine()
        text = self._provider.generate_text(
            loaded_engine=loaded_engine,
            messages=request.messages,
            params=self._sampling_params(request=request),
        )
        return RuntimeGenerationResult(text=text, finish_reason=FinishReason.STOP)

    def stream_generate(self, request: ChatCompletionRequest) -> Iterator[RuntimeStreamChunk]:
        loaded_engine = self._require_loaded_engine()
        chunks = self._provider.stream_text(
            loaded_engine=loaded_engine,
            messages=request.messages,
            params=self._sampling_params(request=request),
        )
        emitted = False
        for chunk in chunks:
            emitted = True
            if chunk:
                yield RuntimeStreamChunk(text=chunk)
        if not emitted:
            yield RuntimeStreamChunk(text="")
        yield RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP)

    def _require_loaded_engine(self) -> LoadedEngine:
        self.load_model()
        if self._loaded_engine is None:
            msg = "vLLM-CUDA engine failed to load"
            raise VLLMCUDARuntimeError(msg)
        return self._loaded_engine

    def _sampling_params(self, *, request: ChatCompletionRequest) -> RuntimeSamplingParams:
        defaults = self.model_config.generation_defaults
        return RuntimeSamplingParams(
            max_output_tokens=request.max_output_tokens or defaults.max_output_tokens,
            temperature=request.temperature if request.temperature != 0.7 else defaults.temperature,
            top_p=request.top_p if request.top_p != 1.0 else defaults.top_p,
        )
