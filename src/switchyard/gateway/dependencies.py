"""Gateway dependency wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Request

from switchyard.adapters.registry import AdapterRegistry
from switchyard.router.service import RouterService
from switchyard.telemetry import Telemetry


@dataclass(slots=True)
class GatewayServices:
    """Shared gateway dependencies injected into route handlers."""

    registry: AdapterRegistry
    router: RouterService
    telemetry: Telemetry


def get_services(request: Request) -> GatewayServices:
    """Return gateway services from application state."""

    services = getattr(request.app.state, "services", None)
    if not isinstance(services, GatewayServices):
        msg = "gateway services are not configured"
        raise RuntimeError(msg)
    return services


@asynccontextmanager
async def gateway_lifespan(_: object) -> AsyncIterator[None]:
    """Phase 0 lifespan hook placeholder."""

    yield
