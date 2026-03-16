"""Telemetry scaffolding for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from starlette.applications import Starlette


@dataclass(slots=True)
class TelemetryState:
    """Local-friendly in-memory telemetry state for tests and development."""

    request_count: int = 0
    request_latency_ms: list[float] = field(default_factory=list)
    route_decision_count: int = 0
    backend_health_snapshots: list[dict[str, Any]] = field(default_factory=list)


class Telemetry:
    """Thin wrapper over local telemetry state and optional OpenTelemetry instruments."""

    def __init__(self, service_name: str, *, enabled: bool = False) -> None:
        self.service_name = service_name
        self.enabled = enabled
        self.state = TelemetryState()
        self._instrumented_apps: set[int] = set()

        resource = Resource.create({"service.name": service_name})
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self._meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(self._meter_provider)
        meter = metrics.get_meter(service_name)
        self._requests_total = meter.create_counter("switchyard_requests_total")
        self._request_latency_ms = meter.create_histogram("switchyard_request_latency_ms")
        self._route_decisions_total = meter.create_counter("switchyard_route_decisions_total")
        self._backend_health = meter.create_up_down_counter("switchyard_backend_health")

    def instrument_fastapi(self, app: Starlette) -> None:
        """Instrument a FastAPI app once when telemetry is enabled."""

        if not self.enabled or id(app) in self._instrumented_apps:
            return
        FastAPIInstrumentor.instrument_app(app)
        self._instrumented_apps.add(id(app))

    def record_request(
        self,
        *,
        route: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record request counters and latency."""

        attributes = {"route": route, "method": method, "status_code": status_code}
        self.state.request_count += 1
        self.state.request_latency_ms.append(latency_ms)
        self._requests_total.add(1, attributes=attributes)
        self._request_latency_ms.record(latency_ms, attributes=attributes)

    def record_route_decision(self, *, policy: str, backend_name: str) -> None:
        """Record a route decision."""

        self.state.route_decision_count += 1
        self._route_decisions_total.add(
            1,
            attributes={"policy": policy, "backend_name": backend_name},
        )

    def record_backend_health_snapshot(
        self,
        *,
        backend_name: str,
        health_state: str,
        latency_ms: float | None,
    ) -> None:
        """Record a backend health snapshot."""

        snapshot = {
            "backend_name": backend_name,
            "health_state": health_state,
            "latency_ms": latency_ms,
        }
        self.state.backend_health_snapshots.append(snapshot)
        self._backend_health.add(
            1,
            attributes={
                "backend_name": backend_name,
                "health_state": health_state,
            },
        )


def configure_telemetry(service_name: str, *, enabled: bool = False) -> Telemetry:
    """Create the Phase 0 telemetry wrapper."""

    return Telemetry(service_name=service_name, enabled=enabled)
