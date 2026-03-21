"""Concrete Linux/NVIDIA remote worker app for the first vLLM-CUDA path."""

from __future__ import annotations

from fastapi import FastAPI

from switchyard.adapters.vllm_cuda import VLLMCUDAAdapter, VLLMCUDARuntime
from switchyard.runtime.vllm_cuda import VLLMCUDAChatRuntime
from switchyard.worker.app import create_worker_app
from switchyard.worker.config import RemoteWorkerRuntimeSettings


def create_vllm_cuda_worker_app(
    settings: RemoteWorkerRuntimeSettings,
    *,
    runtime: VLLMCUDARuntime | None = None,
) -> FastAPI:
    """Build a concrete vLLM-CUDA worker around the shared worker protocol."""

    settings.validate_vllm_cuda_contract()

    model_config = settings.to_local_model_config()
    adapter = VLLMCUDAAdapter(
        model_config,
        runtime=runtime
        or VLLMCUDAChatRuntime(
            model_config,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_model_len=settings.max_model_len,
            trust_remote_code=settings.trust_remote_code,
        ),
    )
    return create_worker_app(
        adapter,
        worker_name=settings.worker_name,
        warmup_on_start=settings.warmup_on_start,
        warmup_model_id=settings.model_identifier,
        log_level=settings.log_level,
        drain_timeout_seconds=settings.drain_timeout_seconds,
    )
