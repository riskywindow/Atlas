"""MLX-LM runtime boundary for Switchyard."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, cast

from switchyard.config import LocalModelConfig
from switchyard.runtime.base import (
    RuntimeGenerationResult,
    RuntimeHealthSnapshot,
    RuntimeSamplingParams,
    RuntimeStreamChunk,
)
from switchyard.schemas.backend import BackendHealthState, BackendLoadState, BackendType
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason

ModuleImporter = Callable[[str], ModuleType]


class MLXLMRuntimeError(RuntimeError):
    """Base error for MLX runtime failures."""


class MLXLMDependencyError(MLXLMRuntimeError):
    """Raised when the optional MLX-LM dependency is unavailable."""


class MLXLMConfigurationError(MLXLMRuntimeError):
    """Raised when MLX-LM is installed but the runtime is misconfigured."""


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded MLX model state."""

    model: object
    tokenizer: object


class MLXLMProvider(Protocol):
    """Provider boundary that isolates direct MLX-LM calls."""

    def ensure_available(self) -> None:
        """Verify that MLX-LM can be imported."""

    def load(self, model_identifier: str) -> LoadedModel:
        """Load an MLX-LM model and tokenizer."""

    def render_prompt(
        self,
        *,
        tokenizer: object,
        messages: Sequence[ChatMessage],
    ) -> str:
        """Render chat messages into an MLX-compatible prompt."""

    def generate_text(
        self,
        *,
        loaded_model: LoadedModel,
        prompt: str,
        params: RuntimeSamplingParams,
    ) -> str:
        """Generate a complete text response."""

    def stream_text(
        self,
        *,
        loaded_model: LoadedModel,
        prompt: str,
        params: RuntimeSamplingParams,
    ) -> Iterator[str]:
        """Yield text segments from streamed generation."""


class ImportedMLXLMProvider:
    """Lazy MLX-LM provider backed by importlib."""

    def __init__(self, module_importer: ModuleImporter = importlib.import_module) -> None:
        self._module_importer = module_importer
        self._mlx_module: ModuleType | None = None
        self._sample_utils_module: ModuleType | None = None

    def ensure_available(self) -> None:
        self._get_mlx_module()

    def load(self, model_identifier: str) -> LoadedModel:
        load_fn = self._require_callable(
            module=self._get_mlx_module(),
            attribute="load",
            context="mlx_lm.load",
        )
        loaded = load_fn(model_identifier)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            msg = "mlx_lm.load must return a (model, tokenizer) tuple"
            raise MLXLMConfigurationError(msg)
        model, tokenizer = loaded
        return LoadedModel(model=model, tokenizer=tokenizer)

    def render_prompt(
        self,
        *,
        tokenizer: object,
        messages: Sequence[ChatMessage],
    ) -> str:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            msg = "configured MLX tokenizer does not expose apply_chat_template"
            raise MLXLMConfigurationError(msg)

        rendered_messages = [
            {"role": message.role.value, "content": message.content} for message in messages
        ]
        try:
            prompt = apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            prompt = apply_chat_template(
                rendered_messages,
                add_generation_prompt=True,
            )

        if not isinstance(prompt, str):
            msg = "mlx tokenizer.apply_chat_template must return a text prompt"
            raise MLXLMConfigurationError(msg)
        return prompt

    def generate_text(
        self,
        *,
        loaded_model: LoadedModel,
        prompt: str,
        params: RuntimeSamplingParams,
    ) -> str:
        generate_fn = self._require_callable(
            module=self._get_mlx_module(),
            attribute="generate",
            context="mlx_lm.generate",
        )
        result = generate_fn(
            loaded_model.model,
            loaded_model.tokenizer,
            **self._generation_kwargs(prompt=prompt, params=params),
        )
        return self._coerce_text(result, context="mlx_lm.generate")

    def stream_text(
        self,
        *,
        loaded_model: LoadedModel,
        prompt: str,
        params: RuntimeSamplingParams,
    ) -> Iterator[str]:
        stream_generate = getattr(self._get_mlx_module(), "stream_generate", None)
        if not callable(stream_generate):
            yield self.generate_text(
                loaded_model=loaded_model,
                prompt=prompt,
                params=params,
            )
            return

        responses = stream_generate(
            loaded_model.model,
            loaded_model.tokenizer,
            prompt,
            **self._stream_generation_kwargs(params=params),
        )
        for response in responses:
            chunk_text = self._coerce_text(response, context="mlx_lm.stream_generate")
            if chunk_text:
                yield chunk_text

    def _get_mlx_module(self) -> ModuleType:
        if self._mlx_module is None:
            try:
                self._mlx_module = self._module_importer("mlx_lm")
            except ImportError as exc:
                msg = "mlx-lm is not installed; install it to enable the MLX backend"
                raise MLXLMDependencyError(msg) from exc
        return self._mlx_module

    def _get_sample_utils_module(self) -> ModuleType:
        if self._sample_utils_module is None:
            try:
                self._sample_utils_module = self._module_importer("mlx_lm.sample_utils")
            except ImportError as exc:
                msg = "mlx_lm.sample_utils is unavailable; cannot construct an MLX sampler"
                raise MLXLMDependencyError(msg) from exc
        return self._sample_utils_module

    def _generation_kwargs(
        self,
        *,
        prompt: str,
        params: RuntimeSamplingParams,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "prompt": prompt,
            "verbose": False,
        }
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens

        sampler = self._build_sampler(params=params)
        if sampler is not None:
            kwargs["sampler"] = sampler
        return kwargs

    def _stream_generation_kwargs(
        self,
        *,
        params: RuntimeSamplingParams,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens

        sampler = self._build_sampler(params=params)
        if sampler is not None:
            kwargs["sampler"] = sampler
        return kwargs

    def _build_sampler(self, *, params: RuntimeSamplingParams) -> object | None:
        if params.temperature is None and params.top_p is None:
            return None

        make_sampler = self._require_callable(
            module=self._get_sample_utils_module(),
            attribute="make_sampler",
            context="mlx_lm.sample_utils.make_sampler",
        )
        temperature = 0.0 if params.temperature is None else params.temperature
        top_p = 1.0 if params.top_p is None else params.top_p
        return make_sampler(temperature, top_p)

    def _require_callable(
        self,
        *,
        module: ModuleType,
        attribute: str,
        context: str,
    ) -> Callable[..., object]:
        value = getattr(module, attribute, None)
        if not callable(value):
            msg = f"{context} is not available in the installed mlx-lm package"
            raise MLXLMConfigurationError(msg)
        return cast(Callable[..., object], value)

    def _coerce_text(self, value: object, *, context: str) -> str:
        if isinstance(value, str):
            return value

        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text

        msg = f"{context} returned an unsupported response object"
        raise MLXLMConfigurationError(msg)


class MLXLMChatRuntime:
    """Chat-oriented runtime that isolates MLX-LM integration details."""

    backend_type = BackendType.MLX_LM

    def __init__(
        self,
        model_config: LocalModelConfig,
        *,
        provider: MLXLMProvider | None = None,
    ) -> None:
        if model_config.backend_type is not BackendType.MLX_LM:
            msg = "MLXLMChatRuntime requires a LocalModelConfig with backend_type='mlx_lm'"
            raise MLXLMConfigurationError(msg)

        self.model_config = model_config
        self._provider = provider or ImportedMLXLMProvider()
        self._loaded_model: LoadedModel | None = None
        self._last_error: str | None = None

    def load_model(self) -> None:
        if self._loaded_model is not None:
            return

        try:
            self._loaded_model = self._provider.load(self.model_config.model_identifier)
            self._last_error = None
        except MLXLMRuntimeError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            msg = (
                "failed to load MLX model "
                f"'{self.model_config.model_identifier}': {exc}"
            )
            self._last_error = msg
            raise MLXLMConfigurationError(msg) from exc

    def warmup(self) -> None:
        self.load_model()
        if not self.model_config.warmup.enabled:
            return

        warmup_request = ChatCompletionRequest(
            model=self.model_config.alias,
            messages=[ChatMessage(role=ChatRole.USER, content="ping")],
            max_output_tokens=1,
        )
        self.generate(warmup_request)

    def health(self) -> RuntimeHealthSnapshot:
        if self._loaded_model is not None:
            return RuntimeHealthSnapshot(
                state=BackendHealthState.HEALTHY,
                load_state=BackendLoadState.READY,
                detail=f"loaded MLX model '{self.model_config.model_identifier}'",
            )

        if self._last_error is not None:
            return RuntimeHealthSnapshot(
                state=BackendHealthState.UNAVAILABLE,
                load_state=BackendLoadState.FAILED,
                detail="MLX runtime failed to initialize",
                last_error=self._last_error,
            )

        try:
            self._provider.ensure_available()
        except MLXLMRuntimeError as exc:
            self._last_error = str(exc)
            return RuntimeHealthSnapshot(
                state=BackendHealthState.UNAVAILABLE,
                load_state=BackendLoadState.FAILED,
                detail="MLX runtime dependency check failed",
                last_error=self._last_error,
            )

        return RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.COLD,
            detail=f"MLX runtime is available for '{self.model_config.model_identifier}'",
        )

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        loaded_model = self._require_loaded_model()
        prompt = self._provider.render_prompt(
            tokenizer=loaded_model.tokenizer,
            messages=request.messages,
        )
        text = self._provider.generate_text(
            loaded_model=loaded_model,
            prompt=prompt,
            params=self._resolve_sampling_params(request),
        )
        return RuntimeGenerationResult(text=text)

    def stream_generate(self, request: ChatCompletionRequest) -> Iterator[RuntimeStreamChunk]:
        loaded_model = self._require_loaded_model()
        prompt = self._provider.render_prompt(
            tokenizer=loaded_model.tokenizer,
            messages=request.messages,
        )
        for chunk_text in self._provider.stream_text(
            loaded_model=loaded_model,
            prompt=prompt,
            params=self._resolve_sampling_params(request),
        ):
            yield RuntimeStreamChunk(text=chunk_text)

        yield RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP)

    def _require_loaded_model(self) -> LoadedModel:
        self.load_model()
        if self._loaded_model is None:
            msg = "MLX model was not loaded"
            raise MLXLMConfigurationError(msg)
        return self._loaded_model

    def _resolve_sampling_params(self, request: ChatCompletionRequest) -> RuntimeSamplingParams:
        defaults = self.model_config.generation_defaults
        explicit_fields = request.model_fields_set

        max_output_tokens = request.max_output_tokens
        if max_output_tokens is None and "max_output_tokens" not in explicit_fields:
            max_output_tokens = defaults.max_output_tokens

        temperature = (
            request.temperature
            if "temperature" in explicit_fields
            else defaults.temperature
        )
        if temperature is None:
            temperature = request.temperature

        top_p = request.top_p if "top_p" in explicit_fields else defaults.top_p
        if top_p is None:
            top_p = request.top_p

        return RuntimeSamplingParams(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
