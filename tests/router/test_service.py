from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import CanaryRoutingSettings, CircuitBreakerSettings, SessionAffinitySettings
from switchyard.control.affinity import SessionAffinityService
from switchyard.control.canary import CanaryRoutingService
from switchyard.control.circuit import CircuitBreakerService
from switchyard.control.locality import PrefixLocalityService
from switchyard.router.service import NoRouteAvailableError, RouterService
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendType,
    DeviceClass,
    PerformanceHint,
    QualityHint,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import (
    AffinityDisposition,
    CanaryPolicy,
    CircuitBreakerPhase,
    PolicyReference,
    RequestClass,
    RequestContext,
    RequestFeatureVector,
    RolloutDisposition,
    RoutingPolicy,
    SessionAffinityKey,
    TenantTier,
    WeightedBackendAllocation,
    WorkloadShape,
    WorkloadTag,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeAffinityClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


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
    serving_targets: list[str] | None = None,
    supports_streaming: bool = False,
    configured_priority: int = 100,
    configured_weight: float = 1.0,
    quality_hint: QualityHint = QualityHint.BALANCED,
    performance_hint: PerformanceHint = PerformanceHint.BALANCED,
    max_context_tokens: int = 8192,
    concurrency_limit: int = 1,
    active_requests: int = 0,
    queue_depth: int = 0,
    circuit_open: bool = False,
    circuit_reason: str | None = None,
) -> MockBackendAdapter:
    capabilities = BackendCapabilities(
        backend_type=BackendType.MOCK,
        device_class=device_class,
        model_ids=model_ids or ["mock-chat"],
        serving_targets=serving_targets or ["mock-chat"],
        max_context_tokens=max_context_tokens,
        supports_streaming=supports_streaming,
        concurrency_limit=concurrency_limit,
        configured_priority=configured_priority,
        configured_weight=configured_weight,
        quality_tier=quality_tier,
        quality_hint=quality_hint,
        performance_hint=performance_hint,
    )
    return MockBackendAdapter(
        name=name,
        simulated_latency_ms=latency_ms,
        capability_metadata=capabilities,
        health_state=health_state,
        simulated_active_requests=active_requests,
        simulated_queue_depth=queue_depth,
        circuit_open=circuit_open,
        circuit_reason=circuit_reason,
    )


def build_canary_service() -> CanaryRoutingService:
    return CanaryRoutingService(
        CanaryRoutingSettings(
            enabled=True,
            policies=(
                CanaryPolicy(
                    policy_name="chat-rollout",
                    serving_target="chat-shared",
                    enabled=True,
                    baseline_backend="stable-backend",
                    allocations=[
                        WeightedBackendAllocation(
                            backend_name="canary-backend",
                            percentage=20.0,
                        )
                    ],
                ),
            ),
        )
    )


def find_canary_request_id(service: CanaryRoutingService, *, session_id: str | None = None) -> str:
    policy = service.match_policy(serving_target="chat-shared")
    assert policy is not None
    for index in range(10_000):
        request_id = f"req-canary-{index:04d}"
        selection = service.select(
            context=RequestContext(request_id=request_id, session_id=session_id),
            policy=policy,
        )
        if selection.disposition is RolloutDisposition.CANARY:
            return request_id
    raise AssertionError("expected to find a canary-selected request id")


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
    assert decision.serving_target == "mock-chat"
    assert decision.rejected_backends["fast-unavailable"] == "backend health is unavailable"
    assert decision.considered_backends == ["steady-healthy"]
    assert decision.explanation is not None
    assert decision.explanation.selected_backend == "steady-healthy"
    assert decision.request_features is not None
    assert decision.request_features.locality_key
    assert decision.explanation.request_features == decision.request_features
    assert WorkloadTag.SHORT_CHAT in decision.request_features.workload_tags
    assert decision.policy_reference == PolicyReference(
        policy_id="latency_first",
        policy_version="phase6.v1",
    )


@pytest.mark.asyncio
async def test_router_policies_choose_different_backends_when_tradeoffs_exist() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="remote-fast",
            latency_ms=5.0,
            quality_tier=1,
            device_class=DeviceClass.REMOTE,
            performance_hint=PerformanceHint.LATENCY_OPTIMIZED,
        )
    )
    registry.register(
        build_adapter(
            name="local-balanced",
            latency_ms=20.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            configured_priority=10,
            configured_weight=2.0,
            performance_hint=PerformanceHint.BALANCED,
        )
    )
    registry.register(
        build_adapter(
            name="local-premium",
            latency_ms=45.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            quality_hint=QualityHint.PREMIUM,
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
    assert quality_decision.selected_deployment is not None
    assert quality_decision.selected_deployment.name == "local-premium"
    assert quality_decision.request_features is not None
    assert quality_decision.explanation is not None
    assert quality_decision.explanation.policy_reference == quality_decision.policy_reference


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
    assert decision.explanation is not None
    assert len(decision.explanation.candidates) == 2


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
            serving_targets=["other-model"],
        )
    )
    router = RouterService(registry)

    with pytest.raises(NoRouteAvailableError, match="no backend available"):
        await router.route(build_request(), build_context(RoutingPolicy.BALANCED))


@pytest.mark.asyncio
async def test_router_resolves_only_backends_for_logical_target() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="mlx-chat-shared",
            latency_ms=10.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="vllm-chat-shared",
            latency_ms=15.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="other-target",
            latency_ms=1.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            serving_targets=["other-target"],
        )
    )
    router = RouterService(registry)

    decision = await router.route(
        build_request("chat-shared"),
        build_context(RoutingPolicy.BALANCED),
    )

    assert decision.considered_backends == ["vllm-chat-shared", "mlx-chat-shared"]
    assert "other-target" not in decision.considered_backends


@pytest.mark.asyncio
async def test_router_rejects_non_streaming_backend_for_stream_request() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="non-streaming",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            supports_streaming=False,
        )
    )
    registry.register(
        build_adapter(
            name="streaming",
            latency_ms=10.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            supports_streaming=True,
        )
    )
    router = RouterService(registry)
    request = ChatCompletionRequest(
        model="mock-chat",
        stream=True,
        messages=[ChatMessage(role=ChatRole.USER, content="stream this response")],
    )

    decision = await router.route(request, build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "streaming"
    assert decision.rejected_backends["non-streaming"] == "backend does not support streaming"


@pytest.mark.asyncio
async def test_router_rejects_saturated_backend_under_bounded_admission_control() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="busy-standard",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            concurrency_limit=1,
            active_requests=1,
        )
    )
    registry.register(
        build_adapter(
            name="available-standard",
            latency_ms=20.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "available-standard"
    assert decision.admission_limited_backends == {
        "busy-standard": "backend concurrency limit reached"
    }
    assert decision.rejected_backends["busy-standard"] == "backend concurrency limit reached"


@pytest.mark.asyncio
async def test_router_rejects_backend_with_open_circuit() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="protected-backend",
            latency_ms=1.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            circuit_open=True,
            circuit_reason="circuit breaker open after consecutive failures",
        )
    )
    registry.register(
        build_adapter(
            name="fallback-backend",
            latency_ms=25.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "fallback-backend"
    assert decision.protected_backends == {
        "protected-backend": "circuit breaker open after consecutive failures"
    }


@pytest.mark.asyncio
async def test_router_attaches_tenant_and_request_class_metadata() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="tenant-aware",
            latency_ms=5.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)

    decision = await router.route(
        build_request(),
        RequestContext(
            request_id="req-tenant-meta",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            tenant_id="tenant-meta",
            tenant_tier=TenantTier.PRIORITY,
            request_class=RequestClass.LATENCY_SENSITIVE,
            session_id="session-meta",
        ),
    )

    assert decision.telemetry_metadata is not None
    assert decision.telemetry_metadata.tenant_id == "tenant-meta"
    assert decision.telemetry_metadata.request_class is RequestClass.LATENCY_SENSITIVE
    assert decision.session_affinity_key is not None
    assert decision.session_affinity_key.session_id == "session-meta"


@pytest.mark.asyncio
async def test_router_allows_half_open_backend_as_recovery_probe() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="recovering-backend",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
        )
    )
    clock = FakeClock()
    breaker = CircuitBreakerService(
        CircuitBreakerSettings(enabled=True, failure_threshold=1, open_cooldown_seconds=5.0),
        clock=clock,
    )
    breaker.record_failure("recovering-backend", reason="invocation_failure")
    clock.advance(6.0)
    router = RouterService(registry, circuit_breaker=breaker)

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "recovering-backend"
    assert decision.circuit_breaker_state is not None
    assert decision.circuit_breaker_state.phase is CircuitBreakerPhase.HALF_OPEN


@pytest.mark.asyncio
async def test_router_prefers_sticky_backend_when_it_is_still_eligible() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="sticky-backend",
            latency_ms=30.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="faster-backend",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    affinity_clock = FakeAffinityClock()
    affinity = SessionAffinityService(
        SessionAffinitySettings(enabled=True, ttl_seconds=60.0, max_sessions=8),
        clock=affinity_clock,
    )
    affinity.bind(
        SessionAffinityKey(
            tenant_id="tenant-a",
            session_id="session-sticky",
            serving_target="chat-shared",
        ),
        backend_name="sticky-backend",
    )
    router = RouterService(registry, session_affinity=affinity)
    context = RequestContext(
        request_id="req-sticky",
        policy=RoutingPolicy.LATENCY_FIRST,
        workload_shape=WorkloadShape.INTERACTIVE,
        tenant_id="tenant-a",
        session_id="session-sticky",
    )

    decision = await router.route(build_request("chat-shared"), context)

    assert decision.backend_name == "sticky-backend"
    assert decision.annotations is not None
    assert decision.annotations.affinity_disposition is AffinityDisposition.REUSED
    assert decision.sticky_route is not None
    assert decision.sticky_route.backend_name == "sticky-backend"
    assert decision.fallback_backends == ["faster-backend"]


@pytest.mark.asyncio
async def test_router_exposes_prefix_locality_signal_without_overriding_session_affinity() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="sticky-backend",
            latency_ms=30.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="warm-backend",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    affinity = SessionAffinityService(SessionAffinitySettings(enabled=True, ttl_seconds=60.0))
    affinity.bind(
        SessionAffinityKey(
            tenant_id="tenant-a",
            session_id="session-locality",
            serving_target="chat-shared",
        ),
        backend_name="sticky-backend",
    )
    locality = PrefixLocalityService()
    features = RequestFeatureVector(
        message_count=1,
        user_message_count=1,
        prompt_character_count=64,
        prompt_token_estimate=10,
        max_output_tokens=128,
        expected_total_tokens=138,
        repeated_prefix_candidate=True,
        prefix_character_count=32,
        prefix_fingerprint="feedfacecafebeef",
        locality_key="00112233445566778899",
        session_affinity_expected=True,
    )
    locality.observe_request(serving_target="chat-shared", request_features=features)
    locality.observe_execution(
        serving_target="chat-shared",
        request_features=features,
        backend_name="warm-backend",
        backend_instance_id=None,
    )
    router = RouterService(registry, session_affinity=affinity, prefix_locality=locality)
    context = RequestContext(
        request_id="req-prefix-affinity",
        policy=RoutingPolicy.LATENCY_FIRST,
        workload_shape=WorkloadShape.INTERACTIVE,
        tenant_id="tenant-a",
        session_id="session-locality",
        request_features=features,
    )

    decision = await router.route(build_request("chat-shared"), context)

    assert decision.backend_name == "sticky-backend"
    assert decision.prefix_locality_signal is not None
    assert decision.prefix_locality_signal.candidate_local_backend == "warm-backend"
    assert decision.prefix_locality_signal.affinity_conflict is True
    assert decision.annotations is not None
    assert (
        "prefix locality preferred backend differs from the current session affinity binding"
        in decision.annotations.notes
    )


@pytest.mark.asyncio
async def test_router_fails_over_when_sticky_backend_is_ineligible() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="sticky-remote",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.REMOTE,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="local-fallback",
            latency_ms=25.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    affinity = SessionAffinityService(SessionAffinitySettings(enabled=True, ttl_seconds=60.0))
    affinity.bind(
        SessionAffinityKey(
            tenant_id="tenant-a",
            session_id="session-local-only",
            serving_target="chat-shared",
        ),
        backend_name="sticky-remote",
    )
    router = RouterService(registry, session_affinity=affinity)
    context = RequestContext(
        request_id="req-local-only",
        policy=RoutingPolicy.LOCAL_ONLY,
        workload_shape=WorkloadShape.INTERACTIVE,
        tenant_id="tenant-a",
        session_id="session-local-only",
    )

    decision = await router.route(build_request("chat-shared"), context)

    assert decision.backend_name == "local-fallback"
    assert decision.annotations is not None
    assert decision.annotations.affinity_disposition is AffinityDisposition.MISSED
    assert "policy requires a local backend" in decision.annotations.notes
    assert decision.rejected_backends["sticky-remote"] == "policy requires a local backend"


@pytest.mark.asyncio
async def test_router_fails_over_when_sticky_backend_is_open_in_circuit_breaker() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="sticky-protected",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="fallback-backend",
            latency_ms=15.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    affinity = SessionAffinityService(SessionAffinitySettings(enabled=True, ttl_seconds=60.0))
    affinity.bind(
        SessionAffinityKey(
            tenant_id="tenant-a",
            session_id="session-breaker",
            serving_target="chat-shared",
        ),
        backend_name="sticky-protected",
    )
    breaker = CircuitBreakerService(
        CircuitBreakerSettings(enabled=True, failure_threshold=1, open_cooldown_seconds=30.0)
    )
    breaker.record_failure("sticky-protected", reason="invocation_failure")
    router = RouterService(registry, circuit_breaker=breaker, session_affinity=affinity)

    decision = await router.route(
        build_request("chat-shared"),
        RequestContext(
            request_id="req-session-breaker",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            tenant_id="tenant-a",
            session_id="session-breaker",
        ),
    )

    assert decision.backend_name == "fallback-backend"
    assert decision.annotations is not None
    assert decision.annotations.affinity_disposition is AffinityDisposition.MISSED
    assert decision.protected_backends["sticky-protected"] == "invocation_failure"
    assert "invocation_failure" in decision.annotations.notes


@pytest.mark.asyncio
async def test_router_canary_selects_candidate_backend_deterministically() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="stable-backend",
            latency_ms=1.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="canary-backend",
            latency_ms=50.0,
            quality_tier=1,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    canary = build_canary_service()
    router = RouterService(registry, canary_routing=canary)
    request_id = find_canary_request_id(canary)

    decision = await router.route(
        build_request("chat-shared"),
        RequestContext(
            request_id=request_id,
            policy=RoutingPolicy.LATENCY_FIRST,
            workload_shape=WorkloadShape.INTERACTIVE,
        ),
    )

    assert decision.backend_name == "canary-backend"
    assert decision.canary_policy is not None
    assert decision.canary_policy.policy_name == "chat-rollout"
    assert decision.annotations is not None
    assert decision.annotations.rollout_disposition is RolloutDisposition.CANARY
    assert any("selected backend 'canary-backend'" in note for note in decision.annotations.notes)


@pytest.mark.asyncio
async def test_router_canary_bypasses_ineligible_candidate_and_keeps_baseline() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="stable-backend",
            latency_ms=1.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="canary-backend",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
            health_state=BackendHealthState.UNAVAILABLE,
        )
    )
    canary = build_canary_service()
    router = RouterService(registry, canary_routing=canary)
    request_id = find_canary_request_id(canary)

    decision = await router.route(
        build_request("chat-shared"),
        RequestContext(
            request_id=request_id,
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
        ),
    )

    assert decision.backend_name == "stable-backend"
    assert decision.annotations is not None
    assert decision.annotations.rollout_disposition is RolloutDisposition.CANARY
    assert decision.rejected_backends["canary-backend"] == "backend health is unavailable"
    assert (
        "canary candidate was ineligible; routing stayed on baseline"
        in decision.annotations.notes
    )


@pytest.mark.asyncio
async def test_router_session_affinity_overrides_canary_selection() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="stable-backend",
            latency_ms=1.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="canary-backend",
            latency_ms=50.0,
            quality_tier=1,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    canary = build_canary_service()
    request_id = find_canary_request_id(canary, session_id="session-canary")
    affinity = SessionAffinityService(SessionAffinitySettings(enabled=True, ttl_seconds=60.0))
    affinity.bind(
        SessionAffinityKey(
            tenant_id="tenant-a",
            session_id="session-canary",
            serving_target="chat-shared",
        ),
        backend_name="stable-backend",
    )
    router = RouterService(registry, session_affinity=affinity, canary_routing=canary)

    decision = await router.route(
        build_request("chat-shared"),
        RequestContext(
            request_id=request_id,
            policy=RoutingPolicy.LATENCY_FIRST,
            workload_shape=WorkloadShape.INTERACTIVE,
            tenant_id="tenant-a",
            session_id="session-canary",
        ),
    )

    assert decision.backend_name == "stable-backend"
    assert decision.annotations is not None
    assert decision.annotations.affinity_disposition is AffinityDisposition.REUSED
    assert decision.annotations.rollout_disposition is RolloutDisposition.NONE
    assert decision.canary_policy is None


@pytest.mark.asyncio
async def test_router_local_only_bypasses_remote_canary_candidate() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="stable-local",
            latency_ms=10.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="canary-remote",
            latency_ms=1.0,
            quality_tier=4,
            device_class=DeviceClass.REMOTE,
            serving_targets=["chat-shared"],
        )
    )
    canary = CanaryRoutingService(
        CanaryRoutingSettings(
            enabled=True,
            policies=(
                CanaryPolicy(
                    policy_name="remote-rollout",
                    serving_target="chat-shared",
                    enabled=True,
                    baseline_backend="stable-local",
                    allocations=[
                        WeightedBackendAllocation(
                            backend_name="canary-remote",
                            percentage=100.0,
                        )
                    ],
                ),
            ),
        )
    )
    router = RouterService(registry, canary_routing=canary)

    decision = await router.route(
        build_request("chat-shared"),
        RequestContext(
            request_id="req-local-only-canary",
            policy=RoutingPolicy.LOCAL_ONLY,
            workload_shape=WorkloadShape.INTERACTIVE,
        ),
    )

    assert decision.backend_name == "stable-local"
    assert decision.annotations is not None
    assert decision.annotations.rollout_disposition is RolloutDisposition.CANARY
    assert decision.rejected_backends["canary-remote"] == "policy requires a local backend"


@pytest.mark.asyncio
async def test_router_respects_internal_backend_pin() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="primary",
            latency_ms=5.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    registry.register(
        build_adapter(
            name="pinned",
            latency_ms=50.0,
            quality_tier=1,
            device_class=DeviceClass.CPU,
            serving_targets=["chat-shared"],
        )
    )
    router = RouterService(registry, canary_routing=build_canary_service())
    context = RequestContext(
        request_id="req-pinned",
        policy=RoutingPolicy.LATENCY_FIRST,
        workload_shape=WorkloadShape.INTERACTIVE,
        internal_backend_pin="pinned",
        tenant_id="tenant-a",
        session_id="session-pinned",
    )

    decision = await router.route(build_request("chat-shared"), context)

    assert decision.backend_name == "pinned"
    assert decision.considered_backends == ["pinned"]
    assert decision.annotations is not None
    assert "session affinity bypassed by internal backend pin" in decision.annotations.notes
    assert "canary routing bypassed by internal backend pin" in decision.annotations.notes


@pytest.mark.asyncio
async def test_router_rejects_backend_when_request_exceeds_context_window() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="small-context",
            latency_ms=5.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
            max_context_tokens=8,
        )
    )
    registry.register(
        build_adapter(
            name="large-context",
            latency_ms=15.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            max_context_tokens=512,
        )
    )
    router = RouterService(registry)
    request = ChatCompletionRequest(
        model="mock-chat",
        max_output_tokens=32,
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content="one two three four five six seven eight nine ten",
            )
        ],
    )

    decision = await router.route(request, build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "large-context"
    assert decision.rejected_backends["small-context"] == "request exceeds backend max context"
