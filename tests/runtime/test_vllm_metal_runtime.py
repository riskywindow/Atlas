from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from switchyard.config import GenerationDefaults, LocalModelConfig, WarmupSettings
from switchyard.runtime import (
    ImportedVLLMMetalProvider,
    VLLMMetalChatRuntime,
    VLLMMetalConfigurationError,
    VLLMMetalDependencyError,
    VLLMMetalRuntimeCapabilities,
)
from switchyard.runtime.vllm_metal import ModuleImporter
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendType,
    DeviceClass,
    EngineType,
    PerformanceHint,
    QualityHint,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole, FinishReason

type CallLog = dict[str, list[object]]


def build_config(
    *,
    warmup_enabled: bool = False,
    generation_defaults: GenerationDefaults | None = None,
) -> LocalModelConfig:
    return LocalModelConfig(
        alias="metal-chat",
        model_identifier="NousResearch/Meta-Llama-3",
        backend_type=BackendType.VLLM_METAL,
        generation_defaults=generation_defaults or GenerationDefaults(),
        warmup=WarmupSettings(enabled=warmup_enabled),
    )


def build_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="metal-chat",
        messages=[ChatMessage(role=ChatRole.USER, content="Hello vLLM")],
    )


def make_fake_module(
    *,
    text_prefix: str = "generated",
    with_native_streaming: bool = False,
) -> tuple[dict[str, ModuleType], CallLog]:
    calls: CallLog = {
        "init": [],
        "chat": [],
        "sampling_params": [],
    }
    vllm_module = ModuleType("vllm")

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            calls["sampling_params"].append(kwargs)
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(self, *, model: str) -> None:
            calls["init"].append(model)

        def chat(
            self,
            messages: list[dict[str, str]],
            *,
            sampling_params: FakeSamplingParams,
        ) -> list[SimpleNamespace]:
            calls["chat"].append(
                {
                    "messages": messages,
                    "sampling_params": sampling_params.kwargs,
                }
            )
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text=f"{text_prefix}:{messages[-1]['content']}")]
                )
            ]

    vllm_module.LLM = FakeEngine  # type: ignore[attr-defined]
    vllm_module.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    if with_native_streaming:

        def stream_generate(
            engine: object,
            messages: list[dict[str, str]],
            *,
            sampling_params: FakeSamplingParams,
        ) -> list[SimpleNamespace]:
            calls["chat"].append(
                {
                    "messages": messages,
                    "sampling_params": sampling_params.kwargs,
                    "streaming": True,
                }
            )
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text="chunk-1")]),
                SimpleNamespace(outputs=[SimpleNamespace(text="chunk-2")]),
            ]

        vllm_module.stream_generate = stream_generate  # type: ignore[attr-defined]
    return {"vllm": vllm_module}, calls


def build_importer(modules: dict[str, ModuleType], imported: list[str]) -> ModuleImporter:
    def importer(name: str) -> ModuleType:
        imported.append(name)
        try:
            return modules[name]
        except KeyError as exc:
            raise ImportError(name) from exc

    return importer


def test_imported_provider_is_lazy_until_used() -> None:
    modules, _ = make_fake_module()
    imported: list[str] = []

    provider = ImportedVLLMMetalProvider(module_importer=build_importer(modules, imported))

    assert imported == []

    provider.ensure_available()

    assert imported == ["vllm"]


def test_vllm_runtime_generates_with_fake_module() -> None:
    modules, calls = make_fake_module(text_prefix="ok")
    runtime = VLLMMetalChatRuntime(
        build_config(
            generation_defaults=GenerationDefaults(
                max_output_tokens=96,
                temperature=0.1,
                top_p=0.9,
            )
        ),
        provider=ImportedVLLMMetalProvider(module_importer=build_importer(modules, [])),
    )

    cold_health = runtime.health()
    result = runtime.generate(build_request())
    ready_health = runtime.health()

    assert cold_health.state is BackendHealthState.HEALTHY
    assert cold_health.load_state is BackendLoadState.COLD
    assert result.text == "ok:Hello vLLM"
    assert result.finish_reason is FinishReason.STOP
    assert ready_health.load_state is BackendLoadState.READY
    assert calls["init"] == ["NousResearch/Meta-Llama-3"]
    assert calls["sampling_params"] == [
        {"max_tokens": 96, "temperature": 0.1, "top_p": 0.9}
    ]
    chat_call = calls["chat"][0]
    assert isinstance(chat_call, dict)
    assert chat_call["messages"] == [{"role": "user", "content": "Hello vLLM"}]


def test_vllm_provider_declares_capabilities() -> None:
    modules, _ = make_fake_module(with_native_streaming=True)
    provider = ImportedVLLMMetalProvider(module_importer=build_importer(modules, []))

    capabilities = provider.declare_capabilities()

    assert capabilities == VLLMMetalRuntimeCapabilities(
        engine_type=EngineType.VLLM,
        device_class=DeviceClass.APPLE_GPU,
        supports_streaming=True,
        supports_native_streaming=True,
        max_context_tokens=32768,
        quality_hint=QualityHint.PREMIUM,
        performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
        supports_prefix_cache=True,
        supports_kv_cache_reuse=True,
    )


def test_vllm_runtime_warmup_runs_a_small_generation_when_enabled() -> None:
    modules, calls = make_fake_module()
    runtime = VLLMMetalChatRuntime(
        build_config(warmup_enabled=True),
        provider=ImportedVLLMMetalProvider(module_importer=build_importer(modules, [])),
    )

    runtime.warmup()

    chat_call = calls["chat"][0]
    assert isinstance(chat_call, dict)
    assert chat_call["messages"] == [{"role": "user", "content": "ping"}]
    assert calls["sampling_params"] == [{"max_tokens": 1, "temperature": 0.0}]


def test_vllm_runtime_uses_native_streaming_when_available() -> None:
    modules, calls = make_fake_module(with_native_streaming=True)
    runtime = VLLMMetalChatRuntime(
        build_config(),
        provider=ImportedVLLMMetalProvider(module_importer=build_importer(modules, [])),
    )

    chunks = list(runtime.stream_generate(build_request()))

    assert [chunk.text for chunk in chunks] == ["chunk-1", "chunk-2", ""]
    assert chunks[-1].finish_reason is FinishReason.STOP
    assert any(
        isinstance(call, dict) and call.get("streaming") is True for call in calls["chat"]
    )


def test_vllm_runtime_reports_missing_dependency_cleanly() -> None:
    def missing_importer(_: str) -> ModuleType:
        raise ImportError("vllm")

    runtime = VLLMMetalChatRuntime(
        build_config(),
        provider=ImportedVLLMMetalProvider(module_importer=missing_importer),
    )

    health = runtime.health()

    assert health.state is BackendHealthState.UNAVAILABLE
    assert health.load_state is BackendLoadState.FAILED
    assert health.last_error is not None
    assert "vllm is not installed" in health.last_error
    assert runtime.capabilities().supports_native_streaming is False

    with pytest.raises(VLLMMetalDependencyError, match="vllm is not installed"):
        runtime.load_model()


def test_vllm_runtime_rejects_misconfigured_engines() -> None:
    modules, _ = make_fake_module()

    class BrokenEngine:
        def __init__(self, *, model: str) -> None:
            self.model = model

    broken_module = modules["vllm"]
    broken_module.LLM = BrokenEngine  # type: ignore[attr-defined]

    runtime = VLLMMetalChatRuntime(
        build_config(),
        provider=ImportedVLLMMetalProvider(module_importer=build_importer(modules, [])),
    )

    with pytest.raises(
        VLLMMetalConfigurationError,
        match="configured vLLM engine does not expose chat",
    ):
        runtime.generate(build_request())
