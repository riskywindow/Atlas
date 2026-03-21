"""CLI for serving the concrete dependency-gated vLLM-CUDA remote worker."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable

import typer
import uvicorn

from switchyard.runtime.vllm_cuda import ImportedVLLMCUDAProvider, VLLMCUDARuntimeError
from switchyard.worker.config import RemoteWorkerRuntimeSettings
from switchyard.worker.vllm_cuda import create_vllm_cuda_worker_app

app = typer.Typer(help="Serve the concrete Linux/NVIDIA vLLM-CUDA worker process.")

ServerRunner = Callable[..., None]


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind address override."),
    port: int | None = typer.Option(
        None,
        "--port",
        min=1,
        max=65535,
        help="Bind port override.",
    ),
) -> None:
    """Serve the concrete vLLM-CUDA worker using environment-driven settings."""

    settings = RemoteWorkerRuntimeSettings()
    serve_vllm_cuda_worker_process(
        settings=settings,
        host=host,
        port=port,
    )


@app.command("check-config")
def check_config(
    verify_runtime_import: bool = typer.Option(
        True,
        "--verify-runtime-import/--no-verify-runtime-import",
        help="Try importing the optional vLLM dependency during preflight.",
    ),
    require_nvidia_smi: bool = typer.Option(
        False,
        "--require-nvidia-smi/--no-require-nvidia-smi",
        help="Require nvidia-smi to be present during preflight.",
    ),
    fail_on_issues: bool = typer.Option(
        False,
        "--fail-on-issues/--no-fail-on-issues",
        help="Exit non-zero when preflight detects bring-up blockers.",
    ),
) -> None:
    """Validate the rented-GPU worker environment contract and print a JSON report."""

    settings = RemoteWorkerRuntimeSettings()
    payload = build_vllm_cuda_preflight_report(
        settings=settings,
        verify_runtime_import=verify_runtime_import,
        require_nvidia_smi=require_nvidia_smi,
    )
    typer.echo(json.dumps(payload, indent=2))
    if fail_on_issues and payload["issues"]:
        raise typer.Exit(code=1)


def serve_vllm_cuda_worker_process(
    *,
    settings: RemoteWorkerRuntimeSettings,
    host: str | None = None,
    port: int | None = None,
    server_runner: ServerRunner = uvicorn.run,
) -> None:
    """Run the concrete vLLM-CUDA remote worker process."""

    app_instance = create_vllm_cuda_worker_app(settings)
    server_runner(
        app_instance,
        host=host or settings.host,
        port=port or settings.port,
        log_level=settings.log_level.lower(),
    )


def main() -> None:
    """Entrypoint for `python -m switchyard.worker.vllm_cuda_cli`."""

    app()


if __name__ == "__main__":
    main()


def build_vllm_cuda_preflight_report(
    *,
    settings: RemoteWorkerRuntimeSettings,
    verify_runtime_import: bool,
    require_nvidia_smi: bool,
) -> dict[str, object]:
    """Build a small operator-facing preflight report for rented-GPU bring-up."""

    issues: list[str] = []
    notes: list[str] = []
    try:
        settings.validate_vllm_cuda_contract()
    except ValueError as exc:
        issues.append(str(exc))
    runtime_import_ok: bool | None = None
    runtime_import_error: str | None = None
    if verify_runtime_import:
        try:
            ImportedVLLMCUDAProvider().ensure_available()
            runtime_import_ok = True
        except VLLMCUDARuntimeError as exc:
            runtime_import_ok = False
            runtime_import_error = str(exc)
            issues.append(str(exc))
    else:
        notes.append("runtime import verification was skipped")
    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi_present = nvidia_smi_path is not None
    if require_nvidia_smi and not nvidia_smi_present:
        issues.append(
            "nvidia-smi was not found; verify NVIDIA drivers and container GPU runtime "
            "on the rented host"
        )
    if not require_nvidia_smi and not nvidia_smi_present:
        notes.append("nvidia-smi was not found during preflight")
    return {
        "status": "ok" if not issues else "error",
        "worker_name": settings.worker_name,
        "worker_id": settings.worker_id,
        "serving_target": settings.serving_target,
        "model_identifier": settings.model_identifier,
        "backend_type": settings.backend_type.value,
        "device_class": settings.device_class.value,
        "engine_type": settings.engine_type.value,
        "provider": settings.provider,
        "region": settings.region,
        "control_plane_url": settings.control_plane_url,
        "auth_mode": settings.auth_mode.value,
        "tensor_parallel_size": settings.tensor_parallel_size,
        "gpu_count": settings.gpu_count,
        "runtime_import_verified": verify_runtime_import,
        "runtime_import_ok": runtime_import_ok,
        "runtime_import_error": runtime_import_error,
        "nvidia_smi_required": require_nvidia_smi,
        "nvidia_smi_present": nvidia_smi_present,
        "nvidia_smi_path": nvidia_smi_path,
        "issues": issues,
        "notes": notes,
    }
