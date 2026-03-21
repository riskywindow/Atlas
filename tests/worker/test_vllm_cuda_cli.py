from __future__ import annotations

import pytest
from typer.testing import CliRunner

from switchyard.schemas.backend import BackendType, DeviceClass
from switchyard.worker import vllm_cuda_cli
from switchyard.worker.config import RemoteWorkerRuntimeSettings


def test_vllm_cuda_worker_settings_project_runtime_identity() -> None:
    settings = RemoteWorkerRuntimeSettings(
        worker_name="cuda-worker",
        backend_type=BackendType.VLLM_CUDA,
        device_class=DeviceClass.NVIDIA_GPU,
        runtime_version="0.6.3",
        gpu_model="L40S",
        gpu_count=2,
        supports_tools=False,
    )

    runtime = settings.runtime_identity()
    gpu = settings.gpu_metadata()
    request_features = settings.request_feature_support()
    model_config = settings.to_local_model_config()

    assert runtime.runtime_label == "vllm_cuda"
    assert runtime.runtime_version == "0.6.3"
    assert gpu.model == "L40S"
    assert gpu.count == 2
    assert request_features.supports_streaming is True
    assert model_config.backend_type is BackendType.VLLM_CUDA
    assert model_config.execution_mode.value == "remote_worker"
    settings.validate_vllm_cuda_contract()


def test_vllm_cuda_worker_settings_reject_inconsistent_parallelism() -> None:
    settings = RemoteWorkerRuntimeSettings(
        gpu_count=1,
        tensor_parallel_size=2,
    )

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        settings.validate_vllm_cuda_contract()


def test_vllm_cuda_worker_cli_invokes_server(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    settings = RemoteWorkerRuntimeSettings(
        worker_name="cuda-cli",
        host="127.0.0.1",
        port=9300,
        log_level="DEBUG",
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(vllm_cuda_cli, "RemoteWorkerRuntimeSettings", lambda: settings)

    def fake_serve_process(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(vllm_cuda_cli, "serve_vllm_cuda_worker_process", fake_serve_process)

    result = runner.invoke(
        vllm_cuda_cli.app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9400",
        ],
    )

    assert result.exit_code == 0
    assert called["settings"] == settings
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9400


def test_vllm_cuda_worker_check_config_reports_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    settings = RemoteWorkerRuntimeSettings(worker_name="cuda-check", gpu_count=1)

    monkeypatch.setattr(vllm_cuda_cli, "RemoteWorkerRuntimeSettings", lambda: settings)

    def fake_report(**_: object) -> dict[str, object]:
        return {
            "status": "error",
            "worker_name": "cuda-check",
            "issues": ["nvidia-smi was not found"],
            "notes": [],
        }

    monkeypatch.setattr(vllm_cuda_cli, "build_vllm_cuda_preflight_report", fake_report)

    result = runner.invoke(
        vllm_cuda_cli.app,
        ["check-config", "--fail-on-issues"],
    )

    assert result.exit_code == 1
    assert '"status": "error"' in result.stdout


def test_concrete_vllm_cuda_worker_process_runs_server() -> None:
    settings = RemoteWorkerRuntimeSettings(worker_name="cuda-runner", port=9500)
    called: dict[str, object] = {}

    def run_server(app_instance: object, **kwargs: object) -> None:
        called["app"] = app_instance
        called["kwargs"] = kwargs

    vllm_cuda_cli.serve_vllm_cuda_worker_process(
        settings=settings,
        host="127.0.0.1",
        port=9501,
        server_runner=run_server,
    )

    assert called["kwargs"] == {
        "host": "127.0.0.1",
        "port": 9501,
        "log_level": "info",
    }
