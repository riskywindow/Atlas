"""Routing policy and scorer helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    DeviceClass,
    PerformanceHint,
    QualityHint,
)
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import (
    PolicyReference,
    RequestContext,
    RouteCandidateExplanation,
    RouteEligibilityState,
    RouteSelectionReasonCode,
    RoutingPolicy,
    ShadowPolicyExplanation,
)

_DEFAULT_POLICY_VERSION = "phase6.v1"
_DEFAULT_TIE_BREAKER = "score, latency_ms, backend_name"


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    """Structured rejection used to preserve overload and protection signals."""

    reason: str
    category: str


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """One policy-specific candidate evaluation."""

    snapshot: BackendStatusSnapshot
    score: float | None
    eligible: bool
    rationale: list[str]
    reason_codes: list[RouteSelectionReasonCode]
    rejection_reason: str | None = None

    def to_explanation(self, *, serving_target: str) -> RouteCandidateExplanation:
        """Convert the assessment into the stable route-explanation schema."""

        return RouteCandidateExplanation(
            backend_name=self.snapshot.name,
            serving_target=serving_target,
            eligibility_state=(
                RouteEligibilityState.ELIGIBLE
                if self.eligible
                else RouteEligibilityState.REJECTED
            ),
            score=None if self.score is None else round(self.score, 3),
            reason_codes=self.reason_codes,
            rationale=self.rationale,
            rejection_reason=self.rejection_reason,
            deployment=self.snapshot.deployment,
            engine_type=self.snapshot.capabilities.engine_type,
        )


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Primary or shadow policy evaluation across candidate backends."""

    policy_reference: PolicyReference
    assessments: list[CandidateAssessment]
    selected_backend: str
    selected_reason_codes: list[RouteSelectionReasonCode]
    selected_reason: list[str]
    tie_breaker: str = _DEFAULT_TIE_BREAKER

    def ranked_backends(self) -> list[str]:
        """Return the evaluated backend ranking for routing and diagnostics."""

        return [assessment.snapshot.name for assessment in self.assessments if assessment.eligible]

    def selected_assessment(self) -> CandidateAssessment:
        """Return the chosen candidate assessment."""

        for assessment in self.assessments:
            if assessment.snapshot.name == self.selected_backend:
                return assessment
        msg = f"selected backend '{self.selected_backend}' was not present in assessments"
        raise ValueError(msg)

    def to_shadow_explanation(self, *, serving_target: str) -> ShadowPolicyExplanation:
        """Render a shadow policy evaluation into the route explanation schema."""

        return ShadowPolicyExplanation(
            policy_reference=self.policy_reference,
            selected_backend=self.selected_backend,
            candidates=[
                assessment.to_explanation(serving_target=serving_target)
                for assessment in self.assessments
            ],
            selection_reason_codes=self.selected_reason_codes,
            selected_reason=self.selected_reason,
            tie_breaker=self.tie_breaker,
        )


class RoutingPolicyScorer(Protocol):
    """Small scorer interface for heuristic, predictive, and adaptive policies."""

    policy_reference: PolicyReference
    compatibility_policy: RoutingPolicy | None

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation: ...


class PolicyRegistry:
    """Registry for primary and shadow routing scorers."""

    def __init__(
        self,
        *,
        primary_policies: Iterable[RoutingPolicyScorer] | None = None,
        shadow_policies: Iterable[RoutingPolicyScorer] | None = None,
    ) -> None:
        resolved_primary = list(primary_policies or compatibility_policies())
        self._primary_by_mode = {
            scorer.compatibility_policy: scorer
            for scorer in resolved_primary
            if scorer.compatibility_policy is not None
        }
        self._shadow_policies = list(shadow_policies or [])

    def resolve(self, policy: RoutingPolicy) -> RoutingPolicyScorer:
        """Resolve the primary scorer for a compatibility routing mode."""

        scorer = self._primary_by_mode.get(policy)
        if scorer is None:
            msg = f"no registered scorer for compatibility routing policy '{policy.value}'"
            raise KeyError(msg)
        return scorer

    @property
    def shadow_policies(self) -> list[RoutingPolicyScorer]:
        """Return configured non-binding shadow scorers."""

        return list(self._shadow_policies)


class CompatibilityRoutingPolicy:
    """Compatibility wrapper that preserves existing fixed routing behavior."""

    compatibility_policy: RoutingPolicy | None

    def __init__(
        self,
        policy: RoutingPolicy,
        *,
        policy_id: str | None = None,
        policy_version: str = _DEFAULT_POLICY_VERSION,
    ) -> None:
        self.compatibility_policy = policy
        self.policy_reference = PolicyReference(
            policy_id=policy.value if policy_id is None else policy_id,
            policy_version=policy_version,
        )

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        """Score candidates using the legacy deterministic routing formulas."""

        assert self.compatibility_policy is not None
        assessments = [
            CandidateAssessment(
                snapshot=snapshot,
                score=_score_snapshot(snapshot=snapshot, policy=self.compatibility_policy),
                eligible=True,
                rationale=_score_rationale(snapshot=snapshot, policy=self.compatibility_policy),
                reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            )
            for snapshot in candidates
        ]
        ranked = sorted(
            assessments,
            key=lambda assessment: (
                -(assessment.score or float("-inf")),
                assessment.snapshot.health.latency_ms or float("inf"),
                assessment.snapshot.name,
            ),
        )
        if not ranked:
            msg = "policy evaluation requires at least one eligible candidate"
            raise ValueError(msg)
        selected = ranked[0]
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=ranked,
            selected_backend=selected.snapshot.name,
            selected_reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            selected_reason=selected.rationale,
        )


def compatibility_policies() -> list[RoutingPolicyScorer]:
    """Return the built-in fixed policies through the new scorer abstraction."""

    return list(CompatibilityRoutingPolicy(policy) for policy in RoutingPolicy)


def rejection_reason(
    *,
    snapshot: BackendStatusSnapshot,
    request: ChatCompletionRequest,
    context: RequestContext,
    policy: RoutingPolicy,
) -> CandidateRejection | None:
    """Return a rejection reason when a backend should not be considered."""

    del context
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


def _estimated_request_tokens(request: ChatCompletionRequest) -> int:
    prompt_tokens = sum(len(message.content.split()) for message in request.messages)
    expected_output_tokens = request.max_output_tokens or 256
    return prompt_tokens + expected_output_tokens


def _score_snapshot(
    *,
    snapshot: BackendStatusSnapshot,
    policy: RoutingPolicy,
) -> float:
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
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + weight_bonus
            + performance_bonus
            - latency_ms
        )
    if policy is RoutingPolicy.QUALITY_FIRST:
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + (quality * 25.0)
            + quality_hint_bonus
            + (weight_bonus / 2.0)
            - (latency_ms / 10.0)
        )
    if policy is RoutingPolicy.LOCAL_ONLY:
        return (
            health_bonus
            + warm_bonus
            + local_bonus
            + priority_bonus
            + (quality * 10.0)
            + quality_hint_bonus
            - (latency_ms / 8.0)
        )
    return (
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


def _score_rationale(
    *,
    snapshot: BackendStatusSnapshot,
    policy: RoutingPolicy,
) -> list[str]:
    latency_ms = snapshot.health.latency_ms or 1000.0
    return [
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
