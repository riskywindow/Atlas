from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import (
    CanaryRoutingSettings,
    CircuitBreakerSettings,
    HybridExecutionSettings,
    PolicyRolloutSettings,
    RemoteTenantSpilloverRule,
    SessionAffinitySettings,
)
from switchyard.control.affinity import SessionAffinityService
from switchyard.control.canary import CanaryRoutingService
from switchyard.control.circuit import CircuitBreakerService
from switchyard.control.locality import PrefixLocalityService
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.spillover import RemoteSpilloverControlService
from switchyard.router.policies import (
    AdaptivePolicyConfig,
    CandidateAssessment,
    PolicyEvaluation,
    PolicyRegistry,
    TransparentAdaptivePolicy,
)
from switchyard.router.service import NoRouteAvailableError, RouterService
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealthState,
    BackendStatusSnapshot,
    BackendType,
    CostBudgetProfile,
    CostProfileClass,
    DeviceClass,
    EngineType,
    LogicalModelTarget,
    ModelEquivalenceKind,
    NetworkCharacteristics,
    NetworkProfile,
    PerformanceHint,
    QualityHint,
    RequestFeatureSupport,
)
from switchyard.schemas.benchmark import (
    CandidateRouteEstimateContext,
    CounterfactualObjective,
    HistoricalRouteEstimate,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import (
    AffinityDisposition,
    CanaryPolicy,
    CircuitBreakerPhase,
    PolicyReference,
    PolicyRolloutMode,
    PrefixHotness,
    PrefixLocalitySignal,
    RequestClass,
    RequestContext,
    RequestFeatureVector,
    RolloutDisposition,
    RouteSelectionReasonCode,
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


class ScoreOverridePolicy:
    def __init__(
        self,
        *,
        policy_id: str,
        compatibility_policy: RoutingPolicy | None = None,
        score_by_backend: dict[str, float],
        reason_code: RouteSelectionReasonCode = RouteSelectionReasonCode.POLICY_SCORE,
    ) -> None:
        self.policy_reference = PolicyReference(
            policy_id=policy_id,
            policy_version="phase6.test",
        )
        self.compatibility_policy = compatibility_policy
        self._score_by_backend = score_by_backend
        self._reason_code = reason_code

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        del request, context
        assessments = sorted(
            [
                CandidateAssessment(
                    snapshot=snapshot,
                    score=self._score_by_backend[snapshot.name],
                    eligible=True,
                    rationale=[
                        f"policy_id={self.policy_reference.policy_id}",
                        f"score_override={self._score_by_backend[snapshot.name]:.3f}",
                    ],
                    reason_codes=[self._reason_code],
                )
                for snapshot in candidates
            ],
            key=lambda assessment: (-(assessment.score or float("-inf")), assessment.snapshot.name),
        )
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=assessments,
            selected_backend=assessments[0].snapshot.name,
            selected_reason_codes=[self._reason_code],
            selected_reason=[f"selected_by={self.policy_reference.policy_id}"],
        )


class FakeHistoricalPredictor:
    def __init__(self, estimates: dict[str, HistoricalRouteEstimate]) -> None:
        self._estimates = estimates

    def estimate(self, context: CandidateRouteEstimateContext) -> HistoricalRouteEstimate:
        estimate = self._estimates[context.backend_name]
        return estimate.model_copy(update={"context": context})


def build_estimate(
    *,
    backend_name: str,
    evidence_count: int,
    sufficient_data: bool,
    expected_latency_ms: float | None = None,
    expected_tokens_per_second: float | None = None,
    expected_error_rate: float | None = None,
    insufficiency_reason: str | None = None,
) -> HistoricalRouteEstimate:
    return HistoricalRouteEstimate(
        context=CandidateRouteEstimateContext(
            model_alias="mock-chat",
            backend_name=backend_name,
        ),
        evidence_count=evidence_count,
        sufficient_data=sufficient_data,
        expected_latency_ms=expected_latency_ms,
        expected_tokens_per_second=expected_tokens_per_second,
        expected_error_rate=expected_error_rate,
        insufficiency_reason=insufficiency_reason,
        rationale=[f"estimate_for={backend_name}"],
    )


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


def build_spillover(
    **overrides: object,
) -> RemoteSpilloverControlService:
    settings = HybridExecutionSettings(
        enabled=True,
        spillover_enabled=True,
    ).model_copy(update=cast(Mapping[str, object], overrides))
    return RemoteSpilloverControlService(settings)


def build_adapter(
    *,
    name: str,
    latency_ms: float,
    quality_tier: int,
    device_class: DeviceClass,
    backend_type: BackendType = BackendType.MOCK,
    engine_type: EngineType = EngineType.MOCK,
    health_state: BackendHealthState = BackendHealthState.HEALTHY,
    model_ids: list[str] | None = None,
    serving_targets: list[str] | None = None,
    logical_models: list[LogicalModelTarget] | None = None,
    supports_streaming: bool = False,
    supports_system_messages: bool = True,
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
    network_profile: NetworkProfile = NetworkProfile.UNKNOWN,
    relative_cost_index: float | None = None,
    cost_profile_name: CostProfileClass = CostProfileClass.UNKNOWN,
    status_metadata: dict[str, str] | None = None,
) -> MockBackendAdapter:
    capabilities = BackendCapabilities(
        backend_type=backend_type,
        engine_type=engine_type,
        device_class=device_class,
        model_ids=model_ids or ["mock-chat"],
        serving_targets=serving_targets or ["mock-chat"],
        logical_models=logical_models or [],
        max_context_tokens=max_context_tokens,
        supports_streaming=supports_streaming,
        concurrency_limit=concurrency_limit,
        configured_priority=configured_priority,
        configured_weight=configured_weight,
        quality_tier=quality_tier,
        quality_hint=quality_hint,
        performance_hint=performance_hint,
        network_characteristics=NetworkCharacteristics(profile=network_profile),
        request_features=RequestFeatureSupport(
            supports_streaming=supports_streaming,
            supports_system_messages=supports_system_messages,
        ),
        cost_profile=CostBudgetProfile(
            profile=cost_profile_name,
            relative_cost_index=relative_cost_index,
        ),
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
        status_metadata=status_metadata,
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
async def test_router_hybrid_modes_score_local_and_remote_candidates_explainably() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-warm",
            latency_ms=20.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            queue_depth=0,
            status_metadata={
                "predicted_latency_ms": "18.0",
                "predicted_queue_delay_ms": "2.0",
                "evidence_sufficient": "true",
                "confidence_score": "0.8",
            },
        )
    )
    registry.register(
        build_adapter(
            name="remote-burst",
            latency_ms=8.0,
            quality_tier=4,
            device_class=DeviceClass.REMOTE,
            queue_depth=0,
            network_profile=NetworkProfile.WAN,
            relative_cost_index=1.2,
            cost_profile_name=CostProfileClass.STANDARD,
            status_metadata={
                "predicted_latency_ms": "12.0",
                "predicted_queue_delay_ms": "1.0",
                "evidence_sufficient": "true",
                "confidence_score": "0.9",
            },
        )
    )
    router = RouterService(registry)

    local_preferred = await router.route(
        build_request(),
        build_context(RoutingPolicy.LOCAL_PREFERRED),
    )
    latency_slo = await router.route(
        build_request(),
        RequestContext(
            request_id="req-latency-slo",
            policy=RoutingPolicy.LATENCY_SLO,
            workload_shape=WorkloadShape.INTERACTIVE,
            max_latency_ms=15,
        ),
    )

    assert local_preferred.backend_name == "local-warm"
    assert local_preferred.explanation is not None
    assert latency_slo.explanation is not None
    assert RouteSelectionReasonCode.HYBRID_LOCAL_PREFERENCE in (
        local_preferred.explanation.selection_reason_codes
    )
    assert latency_slo.backend_name == "remote-burst"
    assert RouteSelectionReasonCode.HYBRID_LATENCY_SLO in (
        latency_slo.explanation.selection_reason_codes
    )
    remote_candidate = next(
        candidate
        for candidate in latency_slo.explanation.candidates
        if candidate.backend_name == "remote-burst"
    )
    assert RouteSelectionReasonCode.NETWORK_PENALTY in remote_candidate.reason_codes
    assert RouteSelectionReasonCode.EVIDENCE_SUFFICIENT in remote_candidate.reason_codes


@pytest.mark.asyncio
async def test_router_burst_to_remote_prefers_remote_when_local_capacity_is_pressed() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-busy",
            latency_ms=18.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            queue_depth=3,
            active_requests=1,
            concurrency_limit=2,
            status_metadata={
                "predicted_latency_ms": "20.0",
                "predicted_queue_delay_ms": "30.0",
                "evidence_sufficient": "true",
            },
        )
    )
    registry.register(
        build_adapter(
            name="remote-spill",
            latency_ms=12.0,
            quality_tier=3,
            device_class=DeviceClass.REMOTE,
            network_profile=NetworkProfile.WAN,
            status_metadata={
                "predicted_latency_ms": "16.0",
                "predicted_queue_delay_ms": "3.0",
                "evidence_sufficient": "true",
            },
        )
    )
    router = RouterService(registry)

    decision = await router.route(build_request(), build_context(RoutingPolicy.BURST_TO_REMOTE))

    assert decision.backend_name == "remote-spill"
    assert decision.explanation is not None
    assert (
        RouteSelectionReasonCode.HYBRID_BURST_REMOTE
        in decision.explanation.selection_reason_codes
    )
    local_candidate = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "local-busy"
    )
    assert RouteSelectionReasonCode.QUEUE_PREDICTION in local_candidate.reason_codes


@pytest.mark.asyncio
async def test_router_force_remote_spillover_rejects_local_candidates_after_local_admission_failure(
) -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-primary",
            latency_ms=40.0,
            quality_tier=5,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="remote-spill",
            latency_ms=25.0,
            quality_tier=5,
            device_class=DeviceClass.REMOTE,
        )
    )
    router = RouterService(registry, spillover=build_spillover())
    context = build_context(RoutingPolicy.BURST_TO_REMOTE).model_copy(
        update={"force_remote_candidates_only": True}
    )

    decision = await router.route(build_request(), context)

    assert decision.backend_name == "remote-spill"
    assert decision.rejected_backends["local-primary"] == (
        "local admission was saturated; request was forced to remote spillover"
    )
    assert decision.explanation is not None
    assert (
        RouteSelectionReasonCode.LOCAL_ADMISSION_SPILLOVER
        in decision.explanation.selection_reason_codes
    )


@pytest.mark.asyncio
async def test_router_remote_budget_exhaustion_keeps_request_local_with_explainable_rejection(
) -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-healthy",
            latency_ms=35.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="remote-budgeted",
            latency_ms=15.0,
            quality_tier=5,
            device_class=DeviceClass.REMOTE,
        )
    )
    spillover = build_spillover(remote_request_budget_per_minute=1)
    spillover.acquire_remote_permit(
        context=build_context(RoutingPolicy.BURST_TO_REMOTE),
        backend_name="remote-budgeted",
        remote_candidates=[await registry.get("remote-budgeted").status()],
    )
    router = RouterService(registry, spillover=spillover)

    decision = await router.route(build_request(), build_context(RoutingPolicy.BURST_TO_REMOTE))

    assert decision.backend_name == "local-healthy"
    assert decision.explanation is not None
    remote_candidate = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "remote-budgeted"
    )
    assert remote_candidate.rejection_reason == (
        "remote spillover budget is exhausted for the current minute"
    )
    assert RouteSelectionReasonCode.REMOTE_BUDGET_GUARDRAIL in remote_candidate.reason_codes


@pytest.mark.asyncio
async def test_router_remote_tenant_restriction_and_cooldown_are_visible() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-safe",
            latency_ms=30.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="remote-guarded",
            latency_ms=12.0,
            quality_tier=5,
            device_class=DeviceClass.REMOTE,
        )
    )
    spillover = build_spillover(
        remote_cooldown_seconds=60.0,
        per_tenant_remote_spillover=(
            RemoteTenantSpilloverRule(tenant_id="tenant-blocked", remote_enabled=False),
        ),
    )
    spillover.record_remote_instability()
    router = RouterService(registry, spillover=spillover)

    denied_context = build_context(RoutingPolicy.BURST_TO_REMOTE).model_copy(
        update={"tenant_id": "tenant-blocked"}
    )
    denied_decision = await router.route(build_request(), denied_context)
    assert denied_decision.explanation is not None
    denied_remote = next(
        candidate
        for candidate in denied_decision.explanation.candidates
        if candidate.backend_name == "remote-guarded"
    )
    assert denied_decision.backend_name == "local-safe"
    assert RouteSelectionReasonCode.REMOTE_TENANT_RESTRICTION in denied_remote.reason_codes

    cooldown_context = build_context(RoutingPolicy.BURST_TO_REMOTE).model_copy(
        update={"tenant_id": "tenant-open"}
    )
    cooldown_decision = await router.route(build_request(), cooldown_context)
    assert cooldown_decision.explanation is not None
    cooldown_remote = next(
        candidate
        for candidate in cooldown_decision.explanation.candidates
        if candidate.backend_name == "remote-guarded"
    )
    assert cooldown_decision.backend_name == "local-safe"
    assert RouteSelectionReasonCode.REMOTE_COOLDOWN in cooldown_remote.reason_codes


@pytest.mark.asyncio
async def test_router_remote_kill_switch_rejects_remote_candidates() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-safe",
            latency_ms=28.0,
            quality_tier=4,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="remote-disabled",
            latency_ms=8.0,
            quality_tier=5,
            device_class=DeviceClass.REMOTE,
        )
    )
    router = RouterService(
        registry,
        spillover=build_spillover(remote_kill_switch_enabled=True),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BURST_TO_REMOTE))

    assert decision.backend_name == "local-safe"
    assert decision.explanation is not None
    remote_candidate = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "remote-disabled"
    )
    assert remote_candidate.rejection_reason == "remote spillover kill switch is active"
    assert RouteSelectionReasonCode.REMOTE_KILL_SWITCH in remote_candidate.reason_codes


@pytest.mark.asyncio
async def test_router_remote_disabled_rejects_remote_candidates_and_preserves_local_compatibility(
) -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="remote-fast",
            latency_ms=6.0,
            quality_tier=4,
            device_class=DeviceClass.REMOTE,
        )
    )
    registry.register(
        build_adapter(
            name="local-safe",
            latency_ms=20.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
        )
    )
    router = RouterService(registry)

    decision = await router.route(build_request(), build_context(RoutingPolicy.REMOTE_DISABLED))

    assert decision.backend_name == "local-safe"
    assert decision.rejected_backends["remote-fast"] == "policy disables remote backends"


@pytest.mark.asyncio
async def test_router_remote_preferred_if_local_unhealthy_and_sparse_evidence_is_explained(
) -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="local-degraded",
            latency_ms=10.0,
            quality_tier=3,
            device_class=DeviceClass.CPU,
            health_state=BackendHealthState.DEGRADED,
            status_metadata={"evidence_sufficient": "false"},
        )
    )
    registry.register(
        build_adapter(
            name="remote-recovery",
            latency_ms=14.0,
            quality_tier=4,
            device_class=DeviceClass.REMOTE,
            network_profile=NetworkProfile.WAN,
            status_metadata={"evidence_sufficient": "false"},
        )
    )
    router = RouterService(registry)

    decision = await router.route(
        build_request(),
        RequestContext(
            request_id="req-remote-unhealthy",
            policy=RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY,
            workload_shape=WorkloadShape.INTERACTIVE,
            prefix_locality_signal=PrefixLocalitySignal(
                serving_target="mock-chat",
                locality_key="0011223344556677",
                prefix_fingerprint="feedfacecafebeef",
                repeated_prefix_detected=True,
                hotness=PrefixHotness.HOT,
                preferred_backend="remote-recovery",
            ),
        ),
    )

    assert decision.backend_name == "remote-recovery"
    assert decision.explanation is not None
    assert RouteSelectionReasonCode.HYBRID_REMOTE_IF_LOCAL_UNHEALTHY in (
        decision.explanation.selection_reason_codes
    )
    remote_candidate = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "remote-recovery"
    )
    assert RouteSelectionReasonCode.PREFIX_LOCALITY in remote_candidate.reason_codes
    assert RouteSelectionReasonCode.EVIDENCE_INSUFFICIENT in remote_candidate.reason_codes


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
async def test_router_resolves_logical_alias_across_local_and_remote_targets() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="mlx-local",
            latency_ms=12.0,
            quality_tier=4,
            device_class=DeviceClass.APPLE_GPU,
            backend_type=BackendType.MLX_LM,
            engine_type=EngineType.MLX,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.EXACT,
                    max_context_tokens=16384,
                    quality_tier=4,
                    quality_hint=QualityHint.PREMIUM,
                )
            ],
            supports_streaming=True,
            configured_priority=10,
        )
    )
    registry.register(
        build_adapter(
            name="metal-local",
            latency_ms=20.0,
            quality_tier=4,
            device_class=DeviceClass.APPLE_GPU,
            backend_type=BackendType.VLLM_METAL,
            engine_type=EngineType.VLLM,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="NousResearch/Meta-Llama-3-8B-Instruct",
                    equivalence=ModelEquivalenceKind.APPROXIMATE,
                    max_context_tokens=32768,
                    quality_tier=4,
                    performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
                    notes=["weights differ from MLX deployment"],
                )
            ],
            supports_streaming=True,
            configured_priority=20,
        )
    )
    registry.register(
        build_adapter(
            name="cuda-remote",
            latency_ms=18.0,
            quality_tier=5,
            device_class=DeviceClass.NVIDIA_GPU,
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM_CUDA,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.EXACT,
                    max_context_tokens=65536,
                    quality_tier=5,
                    quality_hint=QualityHint.PREMIUM,
                    performance_hint=PerformanceHint.THROUGHPUT_OPTIMIZED,
                )
            ],
            supports_streaming=True,
            configured_priority=30,
        )
    )
    router = RouterService(registry)

    decision = await router.route(
        build_request("chat-shared"),
        build_context(RoutingPolicy.BALANCED),
    )

    assert decision.backend_name == "mlx-local"
    assert decision.explanation is not None
    by_backend = {
        candidate.backend_name: candidate for candidate in decision.explanation.candidates
    }
    assert by_backend["mlx-local"].logical_model is not None
    assert by_backend["mlx-local"].logical_model.alias == "chat-shared"
    assert RouteSelectionReasonCode.MODEL_EXACT_EQUIVALENCE in by_backend[
        "mlx-local"
    ].reason_codes
    assert by_backend["metal-local"].logical_model is not None
    assert by_backend["metal-local"].logical_model.equivalence is ModelEquivalenceKind.APPROXIMATE
    assert RouteSelectionReasonCode.MODEL_APPROXIMATE_EQUIVALENCE in by_backend[
        "metal-local"
    ].reason_codes


@pytest.mark.asyncio
async def test_router_rejects_alias_candidates_with_explicit_context_and_feature_mismatches(
) -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="mlx-ineligible",
            latency_ms=10.0,
            quality_tier=4,
            device_class=DeviceClass.APPLE_GPU,
            backend_type=BackendType.MLX_LM,
            engine_type=EngineType.MLX,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.EXACT,
                    max_context_tokens=128,
                    request_features=RequestFeatureSupport(
                        supports_streaming=False,
                        supports_system_messages=True,
                    ),
                )
            ],
            supports_streaming=False,
        )
    )
    registry.register(
        build_adapter(
            name="metal-context-limited",
            latency_ms=14.0,
            quality_tier=4,
            device_class=DeviceClass.APPLE_GPU,
            backend_type=BackendType.VLLM_METAL,
            engine_type=EngineType.VLLM,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.APPROXIMATE,
                    max_context_tokens=200,
                    request_features=RequestFeatureSupport(
                        supports_streaming=True,
                        supports_system_messages=True,
                    ),
                )
            ],
            supports_streaming=True,
        )
    )
    registry.register(
        build_adapter(
            name="cuda-eligible",
            latency_ms=18.0,
            quality_tier=5,
            device_class=DeviceClass.NVIDIA_GPU,
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM_CUDA,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.EXACT,
                    max_context_tokens=8192,
                    request_features=RequestFeatureSupport(
                        supports_streaming=True,
                        supports_system_messages=True,
                    ),
                )
            ],
            supports_streaming=True,
        )
    )
    router = RouterService(registry)
    request = ChatCompletionRequest(
        model="chat-shared",
        stream=True,
        max_output_tokens=256,
        messages=[ChatMessage(role=ChatRole.USER, content="stream this alias request")],
    )

    decision = await router.route(request, build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "cuda-eligible"
    assert decision.explanation is not None
    local_candidate = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "mlx-ineligible"
    )
    assert local_candidate.rejection_reason == "alias does not support streaming on this backend"
    assert RouteSelectionReasonCode.MODEL_FEATURE_UNSUPPORTED in local_candidate.reason_codes
    context_limited = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "metal-context-limited"
    )
    assert context_limited.rejection_reason == "request exceeds alias context window"
    assert RouteSelectionReasonCode.MODEL_CONTEXT_LIMIT in context_limited.reason_codes


@pytest.mark.asyncio
async def test_router_falls_back_to_remote_for_shared_alias_when_local_candidate_is_ineligible(
) -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="mlx-local-unavailable",
            latency_ms=8.0,
            quality_tier=4,
            device_class=DeviceClass.APPLE_GPU,
            backend_type=BackendType.MLX_LM,
            engine_type=EngineType.MLX,
            health_state=BackendHealthState.UNAVAILABLE,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.EXACT,
                    max_context_tokens=32768,
                )
            ],
        )
    )
    registry.register(
        build_adapter(
            name="cuda-remote-fallback",
            latency_ms=16.0,
            quality_tier=5,
            device_class=DeviceClass.NVIDIA_GPU,
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM_CUDA,
            serving_targets=["chat-shared"],
            logical_models=[
                LogicalModelTarget(
                    alias="chat-shared",
                    model_identifier="meta-llama/Llama-3.1-8B-Instruct",
                    equivalence=ModelEquivalenceKind.EXACT,
                    max_context_tokens=65536,
                    quality_tier=5,
                )
            ],
            supports_streaming=True,
        )
    )
    router = RouterService(registry)

    decision = await router.route(
        build_request("chat-shared"),
        build_context(RoutingPolicy.BALANCED),
    )

    assert decision.backend_name == "cuda-remote-fallback"
    assert decision.rejected_backends["mlx-local-unavailable"] == "backend health is unavailable"
    assert decision.explanation is not None
    rejected_local = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "mlx-local-unavailable"
    )
    assert rejected_local.logical_model is not None
    assert rejected_local.logical_model.alias == "chat-shared"


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
    assert (
        decision.rejected_backends["non-streaming"]
        == "alias does not support streaming on this backend"
    )


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
    assert (
        decision.rejected_backends["small-context"]
        == "request exceeds alias context window"
    )


@pytest.mark.asyncio
async def test_router_compatibility_policy_uses_registry_wrapper() -> None:
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
    router = RouterService(registry, policy_registry=PolicyRegistry())

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "local-balanced"
    assert decision.policy_reference == PolicyReference(
        policy_id="balanced",
        policy_version="phase6.v1",
    )


@pytest.mark.asyncio
async def test_router_emits_custom_policy_reason_codes_and_versions() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="alpha",
            latency_ms=20.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="beta",
            latency_ms=10.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    custom_policy = ScoreOverridePolicy(
        policy_id="heuristic-custom-v1",
        compatibility_policy=RoutingPolicy.BALANCED,
        score_by_backend={"alpha": 100.0, "beta": 10.0},
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[custom_policy]),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "alpha"
    assert decision.policy_reference == PolicyReference(
        policy_id="heuristic-custom-v1",
        policy_version="phase6.test",
    )
    assert decision.explanation is not None
    assert decision.explanation.selection_reason_codes == [RouteSelectionReasonCode.POLICY_SCORE]
    assert decision.explanation.selected_reason == ["selected_by=heuristic-custom-v1"]
    candidate_scores = {
        candidate.backend_name: candidate.score for candidate in decision.explanation.candidates
    }
    assert candidate_scores == {"alpha": 100.0, "beta": 10.0}


@pytest.mark.asyncio
async def test_router_records_shadow_policy_without_changing_primary_route() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(
            name="alpha",
            latency_ms=20.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    registry.register(
        build_adapter(
            name="beta",
            latency_ms=10.0,
            quality_tier=2,
            device_class=DeviceClass.CPU,
        )
    )
    primary_policy = ScoreOverridePolicy(
        policy_id="primary-fixed",
        compatibility_policy=RoutingPolicy.BALANCED,
        score_by_backend={"alpha": 100.0, "beta": 10.0},
    )
    shadow_policy = ScoreOverridePolicy(
        policy_id="shadow-predictive",
        score_by_backend={"alpha": 5.0, "beta": 80.0},
        reason_code=RouteSelectionReasonCode.SHADOW_POLICY_SCORE,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(
            primary_policies=[primary_policy],
            shadow_policies=[shadow_policy],
        ),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "alpha"
    assert decision.explanation is not None
    assert len(decision.explanation.shadow_evaluations) == 1
    shadow = decision.explanation.shadow_evaluations[0]
    assert shadow.policy_reference == PolicyReference(
        policy_id="shadow-predictive",
        policy_version="phase6.test",
    )
    assert shadow.selected_backend == "beta"
    assert shadow.selection_reason_codes == [RouteSelectionReasonCode.SHADOW_POLICY_SCORE]
    assert shadow.selected_reason == ["selected_by=shadow-predictive"]


@pytest.mark.asyncio
async def test_adaptive_policy_abstains_on_cold_start_and_falls_back() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=15.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=10.0, quality_tier=3, device_class=DeviceClass.CPU)
    )
    adaptive_policy = TransparentAdaptivePolicy(
        FakeHistoricalPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=0,
                    sufficient_data=False,
                    insufficiency_reason="no matching historical evidence",
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=0,
                    sufficient_data=False,
                    insufficiency_reason="no matching historical evidence",
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-balanced-v1",
            fallback_policy=RoutingPolicy.BALANCED,
            min_confidence_margin=10.0,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "beta"
    assert decision.policy_reference == PolicyReference(
        policy_id="adaptive-balanced-v1",
        policy_version="phase6.v1",
    )
    assert decision.explanation is not None
    assert decision.explanation.selection_reason_codes[:2] == [
        RouteSelectionReasonCode.ADAPTIVE_ABSTAIN,
        RouteSelectionReasonCode.ADAPTIVE_FALLBACK,
    ]
    assert "adaptive policy abstained" in decision.explanation.selected_reason[0]


@pytest.mark.asyncio
async def test_adaptive_policy_abstains_when_top_candidates_are_too_close() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=30.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=10.0, quality_tier=3, device_class=DeviceClass.CPU)
    )
    adaptive_policy = TransparentAdaptivePolicy(
        FakeHistoricalPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=12,
                    sufficient_data=True,
                    expected_latency_ms=50.0,
                    expected_tokens_per_second=70.0,
                    expected_error_rate=0.01,
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=12,
                    sufficient_data=True,
                    expected_latency_ms=55.0,
                    expected_tokens_per_second=70.0,
                    expected_error_rate=0.01,
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-close-margin-v1",
            fallback_policy=RoutingPolicy.BALANCED,
            min_confidence_margin=20.0,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "beta"
    assert decision.explanation is not None
    assert (
        decision.explanation.selection_reason_codes[0]
        is RouteSelectionReasonCode.ADAPTIVE_ABSTAIN
    )


@pytest.mark.asyncio
async def test_adaptive_policy_rejects_unstable_backend() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=30.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=50.0, quality_tier=3, device_class=DeviceClass.CPU)
    )
    adaptive_policy = TransparentAdaptivePolicy(
        FakeHistoricalPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=20,
                    sufficient_data=True,
                    expected_latency_ms=5.0,
                    expected_tokens_per_second=150.0,
                    expected_error_rate=0.40,
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=20,
                    sufficient_data=True,
                    expected_latency_ms=40.0,
                    expected_tokens_per_second=70.0,
                    expected_error_rate=0.01,
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-stability-v1",
            fallback_policy=RoutingPolicy.BALANCED,
            max_expected_error_rate=0.10,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "beta"
    assert decision.explanation is not None
    alpha = next(
        candidate
        for candidate in decision.explanation.candidates
        if candidate.backend_name == "alpha"
    )
    assert alpha.rejection_reason == "predicted error rate exceeds adaptive-policy limit"
    assert alpha.reason_codes == [RouteSelectionReasonCode.ADAPTIVE_ABSTAIN]


@pytest.mark.asyncio
async def test_adaptive_policy_selects_best_supported_candidate_with_exploration_off() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=25.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=30.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    adaptive_policy = TransparentAdaptivePolicy(
        FakeHistoricalPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=20.0,
                    expected_tokens_per_second=90.0,
                    expected_error_rate=0.01,
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=90.0,
                    expected_tokens_per_second=40.0,
                    expected_error_rate=0.05,
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-latency-v1",
            objective=CounterfactualObjective.LATENCY,
            fallback_policy=RoutingPolicy.BALANCED,
            exploration_enabled=False,
            exploration_rate=1.0,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "alpha"
    assert decision.explanation is not None
    assert decision.explanation.selection_reason_codes == [
        RouteSelectionReasonCode.ADAPTIVE_ESTIMATE
    ]


@pytest.mark.asyncio
async def test_adaptive_policy_ignores_exploration_in_deterministic_evaluation_mode() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=25.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=30.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    adaptive_policy = TransparentAdaptivePolicy(
        FakeHistoricalPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=20.0,
                    expected_tokens_per_second=90.0,
                    expected_error_rate=0.01,
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=90.0,
                    expected_tokens_per_second=40.0,
                    expected_error_rate=0.05,
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-deterministic-v1",
            objective=CounterfactualObjective.LATENCY,
            fallback_policy=RoutingPolicy.BALANCED,
            exploration_enabled=True,
            exploration_rate=1.0,
            deterministic_evaluation=True,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "alpha"
    assert decision.explanation is not None
    assert decision.explanation.selection_reason_codes == [
        RouteSelectionReasonCode.ADAPTIVE_ESTIMATE
    ]


@pytest.mark.asyncio
async def test_adaptive_policy_supports_bounded_deterministic_exploration() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=25.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=30.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    adaptive_policy = TransparentAdaptivePolicy(
        FakeHistoricalPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=20.0,
                    expected_tokens_per_second=90.0,
                    expected_error_rate=0.01,
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=90.0,
                    expected_tokens_per_second=40.0,
                    expected_error_rate=0.05,
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-explore-v1",
            objective=CounterfactualObjective.LATENCY,
            fallback_policy=RoutingPolicy.BALANCED,
            exploration_enabled=True,
            exploration_rate=1.0,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )
    context = build_context(RoutingPolicy.BALANCED)
    context.request_id = "req-force-explore"

    decision = await router.route(build_request(), context)

    assert decision.backend_name == "beta"
    assert decision.explanation is not None
    assert decision.explanation.selection_reason_codes == [
        RouteSelectionReasonCode.ADAPTIVE_EXPLORATION
    ]


@pytest.mark.asyncio
async def test_adaptive_policy_scopes_estimates_by_tenant_when_enabled() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=25.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=30.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    captured_contexts: list[CandidateRouteEstimateContext] = []

    class CapturingPredictor(FakeHistoricalPredictor):
        def estimate(self, context: CandidateRouteEstimateContext) -> HistoricalRouteEstimate:
            captured_contexts.append(context)
            return super().estimate(context)

    adaptive_policy = TransparentAdaptivePolicy(
        CapturingPredictor(
            {
                "alpha": build_estimate(
                    backend_name="alpha",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=20.0,
                    expected_tokens_per_second=90.0,
                    expected_error_rate=0.01,
                ),
                "beta": build_estimate(
                    backend_name="beta",
                    evidence_count=15,
                    sufficient_data=True,
                    expected_latency_ms=90.0,
                    expected_tokens_per_second=40.0,
                    expected_error_rate=0.05,
                ),
            }
        ),
        config=AdaptivePolicyConfig(
            policy_id="adaptive-tenant-scope-v1",
            fallback_policy=RoutingPolicy.BALANCED,
            scope_by_tenant=True,
        ),
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(primary_policies=[adaptive_policy]),
    )
    context = RequestContext(
        request_id="req-tenant-scope",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        tenant_id="tenant-gold",
    )

    decision = await router.route(build_request(), context)

    assert decision.backend_name == "alpha"
    assert {estimate_context.tenant_id for estimate_context in captured_contexts} == {"tenant-gold"}


@pytest.mark.asyncio
async def test_policy_rollout_shadow_only_coexists_with_existing_shadow_path() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=20.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=10.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    rollout_candidate = ScoreOverridePolicy(
        policy_id="adaptive-shadow-v1",
        compatibility_policy=RoutingPolicy.BALANCED,
        score_by_backend={"alpha": 100.0, "beta": 5.0},
    )
    existing_shadow = ScoreOverridePolicy(
        policy_id="existing-shadow",
        score_by_backend={"alpha": 1.0, "beta": 10.0},
        reason_code=RouteSelectionReasonCode.SHADOW_POLICY_SCORE,
    )
    rollout = PolicyRolloutService(
        PolicyRolloutSettings(
            mode=PolicyRolloutMode.SHADOW_ONLY,
            candidate_policy_id="adaptive-shadow-v1",
        ),
        candidate_policies=[rollout_candidate],
    )
    router = RouterService(
        registry,
        policy_registry=PolicyRegistry(shadow_policies=[existing_shadow]),
        policy_rollout=rollout,
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "beta"
    assert decision.explanation is not None
    assert [
        shadow.policy_reference.policy_id
        for shadow in decision.explanation.shadow_evaluations
    ] == ["existing-shadow", "adaptive-shadow-v1"]


@pytest.mark.asyncio
async def test_policy_rollout_canary_mode_can_select_candidate_policy() -> None:
    registry = AdapterRegistry()
    registry.register(
        build_adapter(name="alpha", latency_ms=20.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    registry.register(
        build_adapter(name="beta", latency_ms=10.0, quality_tier=2, device_class=DeviceClass.CPU)
    )
    rollout_candidate = ScoreOverridePolicy(
        policy_id="adaptive-canary-v1",
        compatibility_policy=RoutingPolicy.BALANCED,
        score_by_backend={"alpha": 100.0, "beta": 5.0},
    )
    rollout = PolicyRolloutService(
        PolicyRolloutSettings(
            mode=PolicyRolloutMode.CANARY,
            candidate_policy_id="adaptive-canary-v1",
            canary_percentage=100.0,
        ),
        candidate_policies=[rollout_candidate],
    )
    router = RouterService(
        registry,
        policy_rollout=rollout,
    )

    decision = await router.route(build_request(), build_context(RoutingPolicy.BALANCED))

    assert decision.backend_name == "alpha"
    assert decision.policy_reference == PolicyReference(
        policy_id="adaptive-canary-v1",
        policy_version="phase6.test",
    )
