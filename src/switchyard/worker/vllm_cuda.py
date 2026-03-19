"""Concrete Linux/NVIDIA remote worker app for the first vLLM-CUDA path."""

from __future__ import annotations

from fastapi import FastAPI

from switchyard.adapters.vllm_cuda import VLLMCUDAAdapter, VLLMCUDARuntime
from switchyard.runtime.vllm_cuda import VLLMCUDAChatRuntime
from switchyard.schemas.backend import BackendType, DeviceClass
from switchyard.worker.app import create_worker_app
from switchyard.worker.config import RemoteWorkerRuntimeSettings


def create_vllm_cuda_worker_app(
    settings: RemoteWorkerRuntimeSettings,
    *,
    runtime: VLLMCUDARuntime | None = None,
) -> FastAPI:
    """Build a concrete vLLM-CUDA worker around the shared worker protocol."""

    if settings.backend_type is not BackendType.VLLM_CUDA:
        msg = "the concrete remote worker path requires backend_type='vllm_cuda'"
        raise ValueError(msg)
    if settings.device_class is not DeviceClass.NVIDIA_GPU:
        msg = "the concrete remote worker path requires device_class='nvidia_gpu'"
        raise ValueError(msg)

    model_config = settings.to_local_model_config()
    adapter = VLLMCUDAAdapter(
        model_config,
        runtime=runtime or VLLMCUDAChatRuntime(model_config),
    )
    return create_worker_app(
        adapter,
        worker_name=settings.worker_name,
        warmup_on_start=settings.warmup_on_start,
        warmup_model_id=settings.model_identifier,
        log_level=settings.log_level,
        drain_timeout_seconds=settings.drain_timeout_seconds,
    )
