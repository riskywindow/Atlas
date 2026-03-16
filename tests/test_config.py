from pydantic import ValidationError
from pytest import MonkeyPatch

from switchyard.config import AppEnvironment, Settings
from switchyard.schemas.routing import RoutingPolicy


def test_settings_loads_valid_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHYARD_ENV", "test")
    monkeypatch.setenv("SWITCHYARD_GATEWAY_PORT", "9000")
    monkeypatch.setenv("SWITCHYARD_DEFAULT_ROUTING_POLICY", "latency_first")

    settings = Settings()

    assert settings.env is AppEnvironment.TEST
    assert settings.gateway_port == 9000
    assert settings.default_routing_policy is RoutingPolicy.LATENCY_FIRST


def test_settings_rejects_invalid_port(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHYARD_GATEWAY_PORT", "70000")

    try:
        Settings()
    except ValidationError as exc:
        assert "gateway_port" in str(exc)
    else:
        raise AssertionError("Settings should reject ports outside the valid range")
