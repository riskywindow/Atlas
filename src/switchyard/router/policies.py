"""Deterministic routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    DeviceClass,
    PerformanceHint,
    QualityHint,
)
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import RequestContext, RoutingPolicy


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Scored backend candidate used by the router service."""

    snapshot: BackendStatusSnapshot
    score: float
    rationale: list[str]


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    """Structured rejection used to preserve overload and protection signals."""

    reason: str
    category: str


def rejection_reason(
    *,
    snapshot: BackendStatusSnapshot,
    request: ChatCompletionRequest,
    context: RequestContext,
    policy: RoutingPolicy,
) -> CandidateRejection | None:
    """Return a rejection reason when a backend should not be considered."""

    if not snapshot.capabilities.supports_model_target(request.model):
        return CandidateRejection(
            reason=f"model '{request.model}' is not supported",
            category="capability",
        )
    if snapshot.health.state is BackendHealthState.UNAVAILABLE:
        return CandidateRejection(
            reason="backend health is unavailable",
            category="health",
        )
    if snapshot.health.load_state is BackendLoadState.FAILED:
        return CandidateRejection(
            reason="backend readiness is failed",
            category="health",
        )
    if snapshot.health.circuit_open:
        detail = snapshot.health.circuit_reason or "backend circuit is open"
        return CandidateRejection(reason=detail, category="protection")
    if snapshot.active_requests >= snapshot.capabilities.concurrency_limit:
        return CandidateRejection(
            reason="backend concurrency limit reached",
            category="admission",
        )
    if request.stream and not snapshot.capabilities.supports_streaming:
        return CandidateRejection(
            reason="backend does not support streaming",
            category="capability",
        )
    if _estimated_request_tokens(request) > snapshot.capabilities.max_context_tokens:
        return CandidateRejection(
            reason="request exceeds backend max context",
            category="capability",
        )
    if (
        policy is RoutingPolicy.LOCAL_ONLY
        and snapshot.capabilities.device_class is DeviceClass.REMOTE
    ):
        return CandidateRejection(
            reason="policy requires a local backend",
            category="policy",
        )
    return None


def score_candidate(*, snapshot: BackendStatusSnapshot, policy: RoutingPolicy) -> CandidateScore:
    """Score an eligible backend snapshot for a routing policy."""

    latency_ms = snapshot.health.latency_ms or 1000.0
    quality = float(snapshot.capabilities.quality_tier)
    local_bonus = 12.0 if snapshot.capabilities.device_class is not DeviceClass.REMOTE else 0.0
    health_bonus = 100.0 if snapshot.health.state is BackendHealthState.HEALTHY else 65.0
    warm_bonus = 8.0 if snapshot.health.load_state is BackendLoadState.READY else 0.0
    priority_bonus = max(0.0, 30.0 - float(snapshot.capabilities.configured_priority) / 4.0)
    weight_bonus = min(snapshot.capabilities.configured_weight * 3.0, 15.0)
    quality_hint_bonus = _quality_hint_bonus(snapshot)
    performance_bonus = _performance_hint_bonus(snapshot, policy=policy)

    if policy is RoutingPolicy.LATENCY_FIRST:
        score = (
            health_bonus
            + warm_bonus
            + priority_bonus
            + weight_bonus
            + performance_bonus
            - latency_ms
        )
    elif policy is RoutingPolicy.QUALITY_FIRST:
        score = (
            health_bonus
            + warm_bonus
            + priority_bonus
            + (quality * 25.0)
            + quality_hint_bonus
            + (weight_bonus / 2.0)
            - (latency_ms / 10.0)
        )
    elif policy is RoutingPolicy.LOCAL_ONLY:
        score = (
            health_bonus
            + warm_bonus
            + local_bonus
            + priority_bonus
            + (quality * 10.0)
            + quality_hint_bonus
            - (latency_ms / 8.0)
        )
    else:
        score = (
            health_bonus
            + warm_bonus
            + local_bonus
            + priority_bonus
            + weight_bonus
            + (quality * 7.0)
            + quality_hint_bonus
            + performance_bonus
            - (latency_ms / 2.0)
        )

    rationale = [
        f"policy={policy.value}",
        f"health={snapshot.health.state.value}",
        f"readiness={snapshot.health.load_state.value}",
        f"latency_ms={latency_ms:.1f}",
        f"quality_tier={snapshot.capabilities.quality_tier}",
        f"quality_hint={snapshot.capabilities.quality_hint.value}",
        f"performance_hint={snapshot.capabilities.performance_hint.value}",
        f"device_class={snapshot.capabilities.device_class.value}",
        f"configured_priority={snapshot.capabilities.configured_priority}",
        f"configured_weight={snapshot.capabilities.configured_weight:.2f}",
    ]
    return CandidateScore(snapshot=snapshot, score=score, rationale=rationale)


def _estimated_request_tokens(request: ChatCompletionRequest) -> int:
    prompt_tokens = sum(len(message.content.split()) for message in request.messages)
    expected_output_tokens = request.max_output_tokens or 256
    return prompt_tokens + expected_output_tokens


def _quality_hint_bonus(snapshot: BackendStatusSnapshot) -> float:
    if snapshot.capabilities.quality_hint is QualityHint.PREMIUM:
        return 12.0
    if snapshot.capabilities.quality_hint is QualityHint.BALANCED:
        return 6.0
    return 0.0


def _performance_hint_bonus(
    snapshot: BackendStatusSnapshot,
    *,
    policy: RoutingPolicy,
) -> float:
    if policy is RoutingPolicy.LATENCY_FIRST and (
        snapshot.capabilities.performance_hint is PerformanceHint.LATENCY_OPTIMIZED
    ):
        return 15.0
    if policy is RoutingPolicy.BALANCED and (
        snapshot.capabilities.performance_hint is PerformanceHint.BALANCED
    ):
        return 8.0
    if policy is RoutingPolicy.QUALITY_FIRST and (
        snapshot.capabilities.performance_hint is PerformanceHint.THROUGHPUT_OPTIMIZED
    ):
        return 4.0
    if policy is RoutingPolicy.LOCAL_ONLY and (
        snapshot.capabilities.device_class is not DeviceClass.REMOTE
    ):
        return 6.0
    return 0.0
