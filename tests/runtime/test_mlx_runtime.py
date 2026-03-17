from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from switchyard.config import GenerationDefaults, LocalModelConfig, WarmupSettings
from switchyard.runtime import (
    ImportedMLXLMProvider,
    MLXLMChatRuntime,
    MLXLMConfigurationError,
    MLXLMDependencyError,
)
from switchyard.runtime.mlx_lm import ModuleImporter
from switchyard.schemas.backend import BackendHealthState, BackendLoadState, BackendType
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason

type CallLog = dict[str, list[object]]


class FakeTokenizer:
    def __init__(self) -> None:
        self.rendered_messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        self.rendered_messages = messages
        suffix = "<assistant>" if add_generation_prompt else ""
        return "|".join(f"{message['role']}:{message['content']}" for message in messages) + suffix


def build_config(
    *,
    warmup_enabled: bool = False,
    generation_defaults: GenerationDefaults | None = None,
) -> LocalModelConfig:
    return LocalModelConfig(
        alias="local-chat",
        model_identifier="mlx-community/test-model",
        backend_type=BackendType.MLX_LM,
        generation_defaults=generation_defaults or GenerationDefaults(),
        warmup=WarmupSettings(enabled=warmup_enabled),
    )


def build_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="local-chat",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello MLX")],
    )


def make_fake_modules(
    *,
    with_streaming: bool = True,
    text_prefix: str = "generated",
) -> tuple[dict[str, ModuleType], CallLog]:
    calls: CallLog = {
        "load": [],
        "generate": [],
        "stream_generate": [],
        "samplers": [],
    }
    tokenizer = FakeTokenizer()

    mlx_module = ModuleType("mlx_lm")

    def load(model_identifier: str) -> tuple[object, FakeTokenizer]:
        calls["load"].append(model_identifier)
        return object(), tokenizer

    def generate(
        model: object,
        tokenizer_obj: FakeTokenizer,
        *,
        prompt: str,
        verbose: bool,
        max_tokens: int | None = None,
        sampler: object | None = None,
    ) -> str:
        calls["generate"].append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "sampler": sampler,
                "verbose": verbose,
                "tokenizer": tokenizer_obj,
            }
        )
        return f"{text_prefix}:{prompt}"

    mlx_module.load = load  # type: ignore[attr-defined]
    mlx_module.generate = generate  # type: ignore[attr-defined]

    if with_streaming:

        def stream_generate(
            model: object,
            tokenizer_obj: FakeTokenizer,
            prompt: str,
            *,
            max_tokens: int | None = None,
            sampler: object | None = None,
        ) -> list[SimpleNamespace]:
            calls["stream_generate"].append(
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "sampler": sampler,
                    "tokenizer": tokenizer_obj,
                }
            )
            return [
                SimpleNamespace(text="chunk-1"),
                SimpleNamespace(text="chunk-2"),
            ]

        mlx_module.stream_generate = stream_generate  # type: ignore[attr-defined]

    sample_utils_module = ModuleType("mlx_lm.sample_utils")

    def make_sampler(temperature: float, top_p: float) -> dict[str, float]:
        calls["samplers"].append({"temperature": temperature, "top_p": top_p})
        return {"temperature": temperature, "top_p": top_p}

    sample_utils_module.make_sampler = make_sampler  # type: ignore[attr-defined]
    return {"mlx_lm": mlx_module, "mlx_lm.sample_utils": sample_utils_module}, calls


def build_importer(modules: dict[str, ModuleType], imported: list[str]) -> ModuleImporter:
    def importer(name: str) -> ModuleType:
        imported.append(name)
        try:
            return modules[name]
        except KeyError as exc:
            raise ImportError(name) from exc

    return importer


def test_imported_provider_is_lazy_until_used() -> None:
    modules, _ = make_fake_modules()
    imported: list[str] = []

    provider = ImportedMLXLMProvider(module_importer=build_importer(modules, imported))

    assert imported == []

    provider.ensure_available()

    assert imported == ["mlx_lm"]


def test_mlx_runtime_generates_with_fake_modules() -> None:
    modules, calls = make_fake_modules(text_prefix="ok")
    imported: list[str] = []
    runtime = MLXLMChatRuntime(
        build_config(
            generation_defaults=GenerationDefaults(
                max_output_tokens=64,
                temperature=0.2,
                top_p=0.8,
            )
        ),
        provider=ImportedMLXLMProvider(
            module_importer=build_importer(modules, imported),
        ),
    )

    cold_health = runtime.health()
    result = runtime.generate(build_request())
    ready_health = runtime.health()

    assert cold_health.state is BackendHealthState.HEALTHY
    assert cold_health.load_state is BackendLoadState.COLD
    assert result.text == "ok:user:Hello MLX<assistant>"
    assert ready_health.load_state is BackendLoadState.READY
    assert calls["load"] == ["mlx-community/test-model"]
    assert calls["samplers"] == [{"temperature": 0.2, "top_p": 0.8}]
    generate_call = calls["generate"][0]
    assert isinstance(generate_call, dict)
    assert generate_call["prompt"] == "user:Hello MLX<assistant>"
    assert generate_call["max_tokens"] == 64
    assert generate_call["sampler"] == {"temperature": 0.2, "top_p": 0.8}
    assert generate_call["verbose"] is False
    assert imported == ["mlx_lm", "mlx_lm.sample_utils"]


def test_mlx_runtime_streaming_falls_back_when_native_streaming_is_missing() -> None:
    modules, calls = make_fake_modules(with_streaming=False, text_prefix="buffered")
    runtime = MLXLMChatRuntime(
        build_config(),
        provider=ImportedMLXLMProvider(module_importer=build_importer(modules, [])),
    )

    chunks = list(runtime.stream_generate(build_request()))

    assert [chunk.text for chunk in chunks] == ["buffered:user:Hello MLX<assistant>", ""]
    assert chunks[-1].finish_reason is FinishReason.STOP
    assert calls["stream_generate"] == []
    assert len(calls["generate"]) == 1


def test_mlx_runtime_warmup_runs_a_small_generation_when_enabled() -> None:
    modules, calls = make_fake_modules()
    runtime = MLXLMChatRuntime(
        build_config(warmup_enabled=True),
        provider=ImportedMLXLMProvider(module_importer=build_importer(modules, [])),
    )

    runtime.warmup()

    assert calls["load"] == ["mlx-community/test-model"]
    generate_call = calls["generate"][0]
    assert isinstance(generate_call, dict)
    assert generate_call["max_tokens"] == 1
    assert generate_call["prompt"] == "user:ping<assistant>"


def test_mlx_runtime_reports_missing_dependency_cleanly() -> None:
    def missing_importer(_: str) -> ModuleType:
        raise ImportError("mlx_lm")

    runtime = MLXLMChatRuntime(
        build_config(),
        provider=ImportedMLXLMProvider(module_importer=missing_importer),
    )

    health = runtime.health()

    assert health.state is BackendHealthState.UNAVAILABLE
    assert health.load_state is BackendLoadState.FAILED
    assert health.last_error is not None
    assert "mlx-lm is not installed" in health.last_error

    with pytest.raises(MLXLMDependencyError, match="mlx-lm is not installed"):
        runtime.load_model()


def test_mlx_runtime_rejects_misconfigured_tokenizers() -> None:
    modules, _ = make_fake_modules()

    class BrokenTokenizer:
        pass

    broken_module = modules["mlx_lm"]

    def load(_: str) -> tuple[object, BrokenTokenizer]:
        return object(), BrokenTokenizer()

    broken_module.load = load  # type: ignore[attr-defined]

    runtime = MLXLMChatRuntime(
        build_config(),
        provider=ImportedMLXLMProvider(module_importer=build_importer(modules, [])),
    )

    with pytest.raises(
        MLXLMConfigurationError,
        match="configured MLX tokenizer does not expose apply_chat_template",
    ):
        runtime.generate(build_request())
