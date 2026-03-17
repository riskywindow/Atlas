"""Gateway dependency wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter

from fastapi import Request

from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import Settings
from switchyard.control.admission import AdmissionControlService
from switchyard.control.affinity import SessionAffinityService
from switchyard.control.canary import CanaryRoutingService
from switchyard.control.circuit import CircuitBreakerService
from switchyard.control.locality import PrefixLocalityService
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.shadow import ShadowTrafficService
from switchyard.gateway.trace_capture import TraceCaptureService
from switchyard.logging import get_logger
from switchyard.router.service import RouterService
from switchyard.telemetry import BackendLabels, Telemetry

logger = get_logger(__name__)


@dataclass(slots=True)
class GatewayServices:
    """Shared gateway dependencies injected into route handlers."""

    settings: Settings
    registry: AdapterRegistry
    router: RouterService
    admission: AdmissionControlService
    circuit_breaker: CircuitBreakerService
    session_affinity: SessionAffinityService
    prefix_locality: PrefixLocalityService
    canary: CanaryRoutingService
    shadow: ShadowTrafficService
    policy_rollout: PolicyRolloutService
    telemetry: Telemetry
    trace_capture: TraceCaptureService


def get_services(request: Request) -> GatewayServices:
    """Return gateway services from application state."""

    services = getattr(request.app.state, "services", None)
    if not isinstance(services, GatewayServices):
        msg = "gateway services are not configured"
        raise RuntimeError(msg)
    return services


@asynccontextmanager
async def gateway_lifespan(app: object) -> AsyncIterator[None]:
    """Warm eager backends during startup when configured."""

    services = getattr(getattr(app, "state", None), "services", None)
    if isinstance(services, GatewayServices):
        for adapter in services.registry.list():
            if not _should_eager_warmup(adapter, services.settings):
                continue

            start = perf_counter()
            success = False
            try:
                await adapter.warmup()
                success = True
            finally:
                latency_ms = (perf_counter() - start) * 1000
                status = await adapter.status()
                labels = BackendLabels(
                    backend_name=adapter.name,
                    backend_type=status.capabilities.backend_type.value,
                    model=status.capabilities.default_model or adapter.name,
                    model_identifier=_resolve_model_identifier(status),
                    execution_mode=status.metadata.get("execution_mode", "in_process"),
                    worker_transport=status.metadata.get("worker_transport", "in_process"),
                )
                record = services.telemetry.record_backend_warmup(
                    labels=labels,
                    readiness_state=status.health.load_state.value,
                    warmup_latency_ms=latency_ms,
                    success=success,
                )
                logger.info(
                    "backend_warmup_completed",
                    backend_name=record.backend_name,
                    backend_type=record.backend_type,
                    model=record.model,
                    model_identifier=record.model_identifier,
                    readiness_state=record.readiness_state,
                    warmup_latency_ms=record.warmup_latency_ms,
                    success=record.success,
                )

    yield


def _should_eager_warmup(adapter: object, settings: Settings) -> bool:
    model_config = getattr(adapter, "model_config", None)
    if model_config is None:
        return False

    for configured_model in settings.local_models:
        if configured_model.alias == getattr(model_config, "alias", None):
            return configured_model.warmup.enabled and configured_model.warmup.eager
    return False


def _resolve_model_identifier(status: object) -> str:
    metadata = getattr(status, "metadata", None)
    if isinstance(metadata, dict):
        model_identifier = metadata.get("model_identifier")
        if isinstance(model_identifier, str) and model_identifier:
            return model_identifier

    capabilities = getattr(status, "capabilities", None)
    default_model = getattr(capabilities, "default_model", None)
    if isinstance(default_model, str) and default_model:
        aliases = getattr(capabilities, "model_aliases", None)
        if isinstance(aliases, dict):
            resolved = aliases.get(default_model)
            if isinstance(resolved, str) and resolved:
                return resolved
        model_ids = getattr(capabilities, "model_ids", None)
        if isinstance(model_ids, list) and model_ids:
            first_model = model_ids[0]
            if isinstance(first_model, str) and first_model:
                return first_model
        return default_model

    return "unknown"
