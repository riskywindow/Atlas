from __future__ import annotations

import pytest

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.router.service import NoRouteAvailableError, RouterService
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import RequestContext, RoutingPolicy, WorkloadShape


def build_request(model: str = "mock-chat") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role=ChatRole.USER, content="route this request")],
    )


def build_context(policy: RoutingPolicy) -> RequestContext:
    return RequestContext(
        request_id=f"req_{policy.value}",
        policy=policy,
        workload_shape=WorkloadShape.INTERACTIVE,
    )


def build_adapter(
    *,
    name: str,
    latency_ms: float,
    quality_tier: int,
    device_class: DeviceClass,
    health_state: BackendHealthState = BackendHealthState.HEALTHY,
    model_ids: list[str] | None = None,
) -> MockBackendAdapter:
    capabilities = BackendCapabilities(
        backend_type=BackendType.MOCK,
        device_class=device_class,
        model_ids=model_ids or ["mock-chat"],
        max_context_tokens=8192,
        supports_streaming=False,
        concurrency_limit=1,
        quality_tier=quality_tier,
    )
    return MockBackendAdapter(
        name=name,
        simulated_latency_ms=latency_ms,
        capability_metadata=capabilities,
        health_state=health_state,
    )


@pytest.mark.asyncio
async def test_router_skips_unavailable_backend() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="fast-unavailable",
            latency_ms=5.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            health_state=BackendHealthState.UNAVAILABLE,
        )
    )
    registry.register(
        build_adapter(
            name="steady-healthy",
            latency_ms=25.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)

    decision = await router.route(build_request(), build_context(RoutingPolicy.LATENCY_FIRST))

    assert decision.backend_name == "steady-healthy"
    assert decision.rejected_backends["fast-unavailable"] == "backend health is unavailable"
    assert decision.considered_backends == ["steady-healthy"]


@pytest.mark.asyncio
async def test_router_policies_choose_different_backends_when_tradeoffs_exist() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="remote-fast",
            latency_ms=5.0,
            quality_tier=1,
            device_class=DeviceClass.REMOTE,
        )
    )
    registry.register(
        build_adapter(
            name="local-balanced",
            latency_ms=20.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="local-premium",
            latency_ms=45.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)
    request = build_request()

    latency_decision = await router.route(request, build_context(RoutingPolicy.LATENCY_FIRST))
    balanced_decision = await router.route(request, build_context(RoutingPolicy.BALANCED))
    quality_decision = await router.route(request, build_context(RoutingPolicy.QUALITY_FIRST))
    local_only_decision = await router.route(request, build_context(RoutingPolicy.LOCAL_ONLY))

    assert latency_decision.backend_name == "remote-fast"
    assert balanced_decision.backend_name == "local-balanced"
    assert quality_decision.backend_name == "local-premium"
    assert local_only_decision.backend_name == "local-premium"
    assert local_only_decision.rejected_backends["remote-fast"] == "policy requires a local backend"


@pytest.mark.asyncio
async def test_router_breaks_ties_by_backend_name() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="alpha",
            latency_ms=15.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="beta",
            latency_ms=15.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "alpha"
    assert decision.fallback_backends == ["beta"]


@pytest.mark.asyncio
async def test_router_raises_when_no_backend_supports_request() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="mock-a",
            latency_ms=10.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
            model_ids=["other-model"],
        )
    )
    router = RouterService(registry)

    with pytest.raises(NoRouteAvailableError, match="no backend available"):
        await router.route(build_request(), build_context(RoutingPolicy.BALANCED))
