"""Backend adapter package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from switchyard.adapters.base import BackendAdapter
from switchyard.adapters.factory import build_registry_from_settings
from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.adapters.registry import AdapterRegistry

if TYPE_CHECKING:
    from switchyard.adapters.mlx_lm import MLXLMAdapter
    from switchyard.adapters.remote_worker import RemoteWorkerAdapter
    from switchyard.adapters.vllm_metal import VLLMMetalAdapter

__all__ = [
    "AdapterRegistry",
    "BackendAdapter",
    "MLXLMAdapter",
    "MockBackendAdapter",
    "MockResponseTemplate",
    "RemoteWorkerAdapter",
    "VLLMMetalAdapter",
    "build_registry_from_settings",
]

_OPTIONAL_ADAPTER_EXPORTS = {
    "MLXLMAdapter": ("switchyard.adapters.mlx_lm", "MLXLMAdapter"),
    "RemoteWorkerAdapter": ("switchyard.adapters.remote_worker", "RemoteWorkerAdapter"),
    "VLLMMetalAdapter": ("switchyard.adapters.vllm_metal", "VLLMMetalAdapter"),
}


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL_ADAPTER_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, attribute = _OPTIONAL_ADAPTER_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
