from __future__ import annotations

import pytest
from typer.testing import CliRunner

import switchyard.control_plane.cli as control_plane_cli
from switchyard.config import Settings


def test_gateway_command_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        called["app"] = app
        called["kwargs"] = kwargs

    monkeypatch.setattr("switchyard.control_plane.cli.uvicorn.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        ["gateway", "--host", "127.0.0.1", "--port", "8010", "--log-level", "DEBUG"],
    )

    assert result.exit_code == 0
    assert called["app"] == "switchyard.gateway:create_app"
    assert called["kwargs"] == {
        "factory": True,
        "host": "127.0.0.1",
        "port": 8010,
        "log_level": "debug",
    }


def test_check_config_command_loads_settings_and_builds_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    called: dict[str, object] = {}

    monkeypatch.setattr(control_plane_cli, "Settings", lambda: settings)

    def fake_create_app(*, settings: Settings) -> object:
        called["settings"] = settings
        return object()

    monkeypatch.setattr(control_plane_cli, "create_app", fake_create_app)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["check-config"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert called["settings"] is settings


def test_doctor_command_runs_local_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_doctor_local() -> dict[str, object]:
        return {
            "diagnostics_source": "config_preflight",
            "worker_deployments": [],
            "supporting_services": [],
        }

    monkeypatch.setattr(control_plane_cli, "_doctor_local", fake_doctor_local)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["doctor"])

    assert result.exit_code == 0
    assert '"diagnostics_source": "config_preflight"' in result.stdout


def test_doctor_command_can_fail_on_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_doctor_local() -> dict[str, object]:
        return {
            "diagnostics_source": "config_preflight",
            "worker_deployments": [
                {"configured_instances": [{"probe": {"status": "unreachable"}}]}
            ],
        }

    monkeypatch.setattr(control_plane_cli, "_doctor_local", fake_doctor_local)

    runner = CliRunner()
    result = runner.invoke(control_plane_cli.app, ["doctor", "--fail-on-issues"])

    assert result.exit_code == 1


def test_doctor_command_queries_remote_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_doctor_remote(*, gateway_base_url: str, admin_path: str) -> dict[str, object]:
        assert gateway_base_url == "http://testserver"
        assert admin_path == "/admin/deployment"
        return {
            "diagnostics_source": "runtime",
            "worker_deployments": [],
            "supporting_services": [],
        }

    monkeypatch.setattr(control_plane_cli, "_doctor_remote", fake_doctor_remote)

    runner = CliRunner()
    result = runner.invoke(
        control_plane_cli.app,
        ["doctor", "--gateway-base-url", "http://testserver"],
    )

    assert result.exit_code == 0
    assert '"diagnostics_source": "runtime"' in result.stdout
