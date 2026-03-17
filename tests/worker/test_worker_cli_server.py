from __future__ import annotations

import pytest
from typer.testing import CliRunner

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.config import LocalModelConfig, Settings
from switchyard.schemas.backend import BackendType
from switchyard.worker import cli


def test_serve_worker_process_builds_app_and_runs_server() -> None:
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
            ),
        )
    )
    recorded: dict[str, object] = {}

    def build_adapter(model_config: LocalModelConfig) -> MockBackendAdapter:
        recorded["model_alias"] = model_config.alias
        return MockBackendAdapter(name=f"mock:{model_config.alias}")

    def run_server(app_instance: object, **kwargs: object) -> None:
        recorded["app"] = app_instance
        recorded["server_kwargs"] = kwargs

    cli.serve_worker_process(
        settings=settings,
        target="mlx-lm:mlx-chat",
        host="127.0.0.1",
        port=8109,
        warmup_mode="eager",
        warmup_model_id="mlx-community/test-model",
        log_level="DEBUG",
        adapter_builder=build_adapter,
        server_runner=run_server,
    )

    assert recorded["model_alias"] == "mlx-chat"
    assert recorded["server_kwargs"] == {
        "host": "127.0.0.1",
        "port": 8109,
        "log_level": "debug",
    }


def test_worker_cli_invokes_serve_command(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    settings = Settings(
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
            ),
        )
    )
    called: dict[str, object] = {}

    monkeypatch.setattr(cli, "Settings", lambda: settings)

    def fake_serve_worker_process(**kwargs: object) -> None:
        called.update(kwargs)

    monkeypatch.setattr(cli, "serve_worker_process", fake_serve_worker_process)

    result = runner.invoke(
        cli.app,
        [
            "mlx-chat",
            "--host",
            "0.0.0.0",
            "--port",
            "8110",
            "--warmup-mode",
            "lazy",
        ],
    )

    assert result.exit_code == 0
    assert called["target"] == "mlx-chat"
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8110
    assert called["warmup_mode"] == "lazy"
