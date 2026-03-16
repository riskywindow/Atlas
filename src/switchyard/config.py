"""Application configuration for Switchyard."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from switchyard.schemas.routing import RoutingPolicy


class AppEnvironment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    env: AppEnvironment = AppEnvironment.DEVELOPMENT
    log_level: str = "INFO"
    service_name: str = "switchyard-gateway"
    otel_enabled: bool = False
    gateway_host: str = "127.0.0.1"
    gateway_port: int = Field(default=8000, ge=1, le=65535)
    default_routing_policy: RoutingPolicy = RoutingPolicy.BALANCED
    benchmark_output_dir: Path = Path("artifacts/benchmarks")

    model_config = SettingsConfigDict(
        env_prefix="SWITCHYARD_",
        env_file=".env",
        extra="ignore",
    )
