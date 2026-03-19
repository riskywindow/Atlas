"""Internal runtime boundaries for backend integrations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from switchyard.runtime.base import (
    ChatModelRuntime,
    RuntimeGenerationResult,
    RuntimeHealthSnapshot,
    RuntimeSamplingParams,
    RuntimeStreamChunk,
    UnsupportedRequestError,
)

if TYPE_CHECKING:
    from switchyard.runtime.mlx_lm import (
        ImportedMLXLMProvider,
        MLXLMChatRuntime,
        MLXLMConfigurationError,
        MLXLMDependencyError,
        MLXLMRuntimeError,
    )
    from switchyard.runtime.vllm_cuda import (
        ImportedVLLMCUDAProvider,
        VLLMCUDAChatRuntime,
        VLLMCUDAConfigurationError,
        VLLMCUDADependencyError,
        VLLMCUDARuntimeCapabilities,
        VLLMCUDARuntimeError,
    )
    from switchyard.runtime.vllm_metal import (
        ImportedVLLMMetalProvider,
        VLLMMetalChatRuntime,
        VLLMMetalConfigurationError,
        VLLMMetalDependencyError,
        VLLMMetalRuntimeCapabilities,
        VLLMMetalRuntimeError,
    )

__all__ = [
    "ChatModelRuntime",
    "ImportedMLXLMProvider",
    "MLXLMChatRuntime",
    "MLXLMConfigurationError",
    "MLXLMDependencyError",
    "MLXLMRuntimeError",
    "RuntimeGenerationResult",
    "RuntimeHealthSnapshot",
    "RuntimeSamplingParams",
    "RuntimeStreamChunk",
    "UnsupportedRequestError",
    "ImportedVLLMCUDAProvider",
    "VLLMCUDAChatRuntime",
    "VLLMCUDARuntimeCapabilities",
    "VLLMCUDAConfigurationError",
    "VLLMCUDADependencyError",
    "VLLMCUDARuntimeError",
    "ImportedVLLMMetalProvider",
    "VLLMMetalChatRuntime",
    "VLLMMetalRuntimeCapabilities",
    "VLLMMetalConfigurationError",
    "VLLMMetalDependencyError",
    "VLLMMetalRuntimeError",
]

_OPTIONAL_RUNTIME_EXPORTS = {
    "ImportedVLLMCUDAProvider": ("switchyard.runtime.vllm_cuda", "ImportedVLLMCUDAProvider"),
    "VLLMCUDAChatRuntime": ("switchyard.runtime.vllm_cuda", "VLLMCUDAChatRuntime"),
    "VLLMCUDARuntimeCapabilities": (
        "switchyard.runtime.vllm_cuda",
        "VLLMCUDARuntimeCapabilities",
    ),
    "VLLMCUDAConfigurationError": (
        "switchyard.runtime.vllm_cuda",
        "VLLMCUDAConfigurationError",
    ),
    "VLLMCUDADependencyError": ("switchyard.runtime.vllm_cuda", "VLLMCUDADependencyError"),
    "VLLMCUDARuntimeError": ("switchyard.runtime.vllm_cuda", "VLLMCUDARuntimeError"),
    "ImportedMLXLMProvider": ("switchyard.runtime.mlx_lm", "ImportedMLXLMProvider"),
    "MLXLMChatRuntime": ("switchyard.runtime.mlx_lm", "MLXLMChatRuntime"),
    "MLXLMConfigurationError": ("switchyard.runtime.mlx_lm", "MLXLMConfigurationError"),
    "MLXLMDependencyError": ("switchyard.runtime.mlx_lm", "MLXLMDependencyError"),
    "MLXLMRuntimeError": ("switchyard.runtime.mlx_lm", "MLXLMRuntimeError"),
    "ImportedVLLMMetalProvider": (
        "switchyard.runtime.vllm_metal",
        "ImportedVLLMMetalProvider",
    ),
    "VLLMMetalChatRuntime": ("switchyard.runtime.vllm_metal", "VLLMMetalChatRuntime"),
    "VLLMMetalRuntimeCapabilities": (
        "switchyard.runtime.vllm_metal",
        "VLLMMetalRuntimeCapabilities",
    ),
    "VLLMMetalConfigurationError": (
        "switchyard.runtime.vllm_metal",
        "VLLMMetalConfigurationError",
    ),
    "VLLMMetalDependencyError": ("switchyard.runtime.vllm_metal", "VLLMMetalDependencyError"),
    "VLLMMetalRuntimeError": ("switchyard.runtime.vllm_metal", "VLLMMetalRuntimeError"),
}


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL_RUNTIME_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, attribute = _OPTIONAL_RUNTIME_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
