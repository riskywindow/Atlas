from __future__ import annotations

import pytest
from typer.testing import CliRunner

from switchyard.schemas.backend import BackendType, DeviceClass
from switchyard.schemas.worker import RemoteWorkerAuthMode
from switchyard.worker import fake_cli
from switchyard.worker.config import RemoteWorkerRuntimeSettings


def test_remote_worker_runtime_settings_load_stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHYARD_REMOTE_WORKER_WORKER_NAME", "stub-cuda")
    monkeypatch.setenv("SWITCHYARD_REMOTE_WORKER_BACKEND_TYPE", "vllm_cuda")
    monkeypatch.setenv("SWITCHYARD_REMOTE_WORKER_DEVICE_CLASS", "nvidia_gpu")
    monkeypatch.setenv("SWITCHYARD_REMOTE_WORKER_AUTH_MODE", "static_token")
    monkeypatch.setenv("SWITCHYARD_REMOTE_WORKER_REGISTRATION_TOKEN", "secret-token")

    settings = RemoteWorkerRuntimeSettings()

    assert settings.worker_name == "stub-cuda"
    assert settings.backend_type is BackendType.VLLM_CUDA
    assert settings.device_class is DeviceClass.NVIDIA_GPU
    assert settings.auth_mode is RemoteWorkerAuthMode.STATIC_TOKEN
    assert settings.registration_token == "secret-token"
    assert settings.to_fake_worker_config().backend_type is BackendType.VLLM_CUDA


def test_remote_worker_runtime_settings_require_matching_auth_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_REMOTE_WORKER_AUTH_MODE", "signed_enrollment")

    with pytest.raises(ValueError, match="enrollment_token"):
        RemoteWorkerRuntimeSettings()


def test_fake_remote_worker_cli_invokes_server(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    settings = RemoteWorkerRuntimeSettings(
        worker_name="cli-stub",
        host="127.0.0.1",
        port=9100,
        log_level="DEBUG",
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(fake_cli, "RemoteWorkerRuntimeSettings", lambda: settings)

    def fake_serve_process(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(fake_cli, "serve_fake_remote_worker_process", fake_serve_process)

    result = runner.invoke(
        fake_cli.app,
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9200",
        ],
    )

    assert result.exit_code == 0
    assert called["settings"] == settings
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 9200
