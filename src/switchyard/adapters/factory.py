"""Adapter registration helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from switchyard.adapters.registry import AdapterRegistry
from switchyard.adapters.remote_worker import RemoteWorkerAdapter
from switchyard.config import LocalModelConfig, Settings
from switchyard.runtime.base import ChatModelRuntime
from switchyard.schemas.backend import BackendType, WorkerTransportType

MLXRuntimeFactory = Callable[[LocalModelConfig], ChatModelRuntime]
VLLMRuntimeFactory = Callable[[LocalModelConfig], ChatModelRuntime]


def build_registry_from_settings(
    settings: Settings,
    *,
    mlx_runtime_factory: MLXRuntimeFactory | None = None,
    vllm_runtime_factory: VLLMRuntimeFactory | None = None,
) -> AdapterRegistry:
    """Build an adapter registry from configured local models."""

    registry = AdapterRegistry()

    for model_config in settings.local_models:
        if _requires_remote_worker_adapter(model_config):
            registry.register(RemoteWorkerAdapter(model_config))
            continue
        if model_config.backend_type is BackendType.MLX_LM:
            from switchyard.adapters.mlx_lm import MLXLMAdapter
            from switchyard.runtime.mlx_lm import MLXLMChatRuntime

            resolved_mlx_runtime_factory = mlx_runtime_factory or cast(
                MLXRuntimeFactory,
                MLXLMChatRuntime,
            )
            registry.register(
                MLXLMAdapter(
                    model_config,
                    runtime=resolved_mlx_runtime_factory(model_config),
                )
            )
        elif model_config.backend_type is BackendType.VLLM_METAL:
            from switchyard.adapters.vllm_metal import VLLMMetalAdapter
            from switchyard.runtime.vllm_metal import VLLMMetalChatRuntime

            resolved_vllm_runtime_factory = vllm_runtime_factory or cast(
                VLLMRuntimeFactory,
                VLLMMetalChatRuntime,
            )
            registry.register(
                VLLMMetalAdapter(
                    model_config,
                    runtime=resolved_vllm_runtime_factory(model_config),
                )
            )
        else:
            msg = (
                "unsupported local backend_type "
                f"{model_config.backend_type.value!r} for alias {model_config.alias!r}"
            )
            raise ValueError(msg)

    return registry


def _requires_remote_worker_adapter(model_config: LocalModelConfig) -> bool:
    if model_config.worker_transport is not WorkerTransportType.IN_PROCESS:
        return True
    return any(
        instance.transport is not WorkerTransportType.IN_PROCESS
        for instance in model_config.instances
    )
