"""Deterministic routing policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from switchyard.schemas.backend import BackendHealthState, BackendStatusSnapshot, DeviceClass
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import RoutingPolicy


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Scored backend candidate used by the router service."""

    snapshot: BackendStatusSnapshot
    score: float
    rationale: list[str]


def rejection_reason(
    *,
    snapshot: BackendStatusSnapshot,
    request: ChatCompletionRequest,
    policy: RoutingPolicy,
) -> str | None:
    """Return a rejection reason when a backend should not be considered."""

    if request.model not in snapshot.capabilities.model_ids:
        return f"model '{request.model}' is not supported"
    if snapshot.health.state is BackendHealthState.UNAVAILABLE:
        return "backend health is unavailable"
    if (
        policy is RoutingPolicy.LOCAL_ONLY
        and snapshot.capabilities.device_class is DeviceClass.REMOTE
    ):
        return "policy requires a local backend"
    return None


def score_candidate(*, snapshot: BackendStatusSnapshot, policy: RoutingPolicy) -> CandidateScore:
    """Score an eligible backend snapshot for a routing policy."""

    latency_ms = snapshot.health.latency_ms or 1000.0
    quality = float(snapshot.capabilities.quality_tier)
    local_bonus = 8.0 if snapshot.capabilities.device_class is not DeviceClass.REMOTE else 0.0
    health_bonus = 100.0 if snapshot.health.state is BackendHealthState.HEALTHY else 65.0

    if policy is RoutingPolicy.LATENCY_FIRST:
        score = health_bonus - latency_ms
    elif policy is RoutingPolicy.QUALITY_FIRST:
        score = health_bonus + (quality * 25.0) - (latency_ms / 10.0)
    elif policy is RoutingPolicy.LOCAL_ONLY:
        score = health_bonus + (quality * 10.0) + local_bonus - (latency_ms / 8.0)
    else:
        score = health_bonus + (quality * 5.0) + local_bonus - (latency_ms / 2.0)

    rationale = [
        f"policy={policy.value}",
        f"health={snapshot.health.state.value}",
        f"latency_ms={latency_ms:.1f}",
        f"quality_tier={snapshot.capabilities.quality_tier}",
        f"device_class={snapshot.capabilities.device_class.value}",
    ]
    return CandidateScore(snapshot=snapshot, score=score, rationale=rationale)
