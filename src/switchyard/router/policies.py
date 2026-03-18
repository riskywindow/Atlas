"""Routing policy and scorer helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from switchyard.schemas.backend import (
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    DeviceClass,
    ExecutionModeLabel,
    NetworkProfile,
    PerformanceHint,
    QualityHint,
    WorkerLocalityClass,
)
from switchyard.schemas.benchmark import (
    CandidateRouteEstimateContext,
    CounterfactualObjective,
    HistoricalRouteEstimate,
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
    reason_codes: list[RouteSelectionReasonCode] | None = None


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


class HistoricalRoutePredictor(Protocol):
    """Minimal predictor contract used by transparent adaptive policies."""

    def estimate(self, context: CandidateRouteEstimateContext) -> HistoricalRouteEstimate: ...


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
            _compatibility_assessment(
                snapshot=snapshot,
                candidates=candidates,
                policy=self.compatibility_policy,
                request=request,
                context=context,
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
            selected_reason_codes=list(selected.reason_codes),
            selected_reason=selected.rationale,
        )


@dataclass(frozen=True, slots=True)
class AdaptivePolicyConfig:
    """Configuration for the transparent historical adaptive policy."""

    policy_id: str
    policy_version: str = _DEFAULT_POLICY_VERSION
    fallback_policy: RoutingPolicy = RoutingPolicy.BALANCED
    objective: CounterfactualObjective = CounterfactualObjective.BALANCED
    min_samples: int = 5
    min_confidence_margin: float = 25.0
    max_expected_error_rate: float = 0.25
    exploration_rate: float = 0.0
    exploration_enabled: bool = False
    deterministic_evaluation: bool = False
    scope_by_tenant: bool = False
    avoid_degraded_backends: bool = True
    evidence_policy_id: str | None = None


class TransparentAdaptivePolicy:
    """Explainable historical adaptive scorer with bounded abstention and exploration."""

    compatibility_policy: RoutingPolicy | None

    def __init__(
        self,
        predictor: HistoricalRoutePredictor,
        *,
        config: AdaptivePolicyConfig,
        compatibility_policy: RoutingPolicy | None = None,
    ) -> None:
        self._predictor = predictor
        self._config = config
        self.compatibility_policy = compatibility_policy
        self.policy_reference = PolicyReference(
            policy_id=config.policy_id,
            policy_version=config.policy_version,
        )
        self._fallback = CompatibilityRoutingPolicy(
            config.fallback_policy,
            policy_id=config.fallback_policy.value,
            policy_version=config.policy_version,
        )

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        """Evaluate candidates with a transparent adaptive heuristic over historical evidence."""

        adaptive_assessments = [
            self._adaptive_assessment(
                request=request,
                context=context,
                snapshot=snapshot,
            )
            for snapshot in candidates
        ]
        eligible = [assessment for assessment in adaptive_assessments if assessment.eligible]
        if not eligible:
            return self._abstain_with_fallback(
                request=request,
                context=context,
                candidates=candidates,
                reason="adaptive policy abstained because no candidate met the evidence threshold",
                adaptive_assessments=adaptive_assessments,
            )
        ranked = _rank_assessments(eligible)
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        if not _confident_enough(
            best=best,
            runner_up=runner_up,
            min_confidence_margin=self._config.min_confidence_margin,
        ):
            return self._abstain_with_fallback(
                request=request,
                context=context,
                candidates=candidates,
                reason=(
                    "adaptive policy abstained because the evidence margin between the top "
                    "candidates was too small"
                ),
                adaptive_assessments=adaptive_assessments,
            )
        selected = best
        selected_reason_codes = [RouteSelectionReasonCode.ADAPTIVE_ESTIMATE]
        selected_reason = [
            f"adaptive policy selected backend={best.snapshot.name}",
            *best.rationale,
        ]
        if self._should_explore(context=context, ranked=ranked):
            selected = ranked[1]
            selected_reason_codes = [RouteSelectionReasonCode.ADAPTIVE_EXPLORATION]
            selected_reason = [
                (
                    "adaptive policy selected a bounded exploration candidate because the "
                    "deterministic exploration gate fired"
                ),
                *selected.rationale,
            ]
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=_rank_assessments(adaptive_assessments),
            selected_backend=selected.snapshot.name,
            selected_reason_codes=selected_reason_codes,
            selected_reason=selected_reason,
        )

    def _adaptive_assessment(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        snapshot: BackendStatusSnapshot,
    ) -> CandidateAssessment:
        estimate = self._predictor.estimate(
            _estimate_context(
                request=request,
                context=context,
                snapshot=snapshot,
                evidence_policy_id=(
                    self._config.evidence_policy_id or self._config.fallback_policy.value
                ),
                scope_by_tenant=self._config.scope_by_tenant,
            )
        )
        rationale = [
            f"policy={self.policy_reference.policy_id}",
            f"fallback_policy={self._config.fallback_policy.value}",
            f"evidence_count={estimate.evidence_count}",
        ]
        if estimate.sufficient_data:
            rationale.append("historical evidence met the configured minimum sample threshold")
        elif estimate.insufficiency_reason is not None:
            rationale.append(estimate.insufficiency_reason)
        if self._config.avoid_degraded_backends and (
            snapshot.health.state is BackendHealthState.DEGRADED
        ):
            return CandidateAssessment(
                snapshot=snapshot,
                score=None,
                eligible=False,
                rationale=[*rationale, "backend health is degraded so adaptive routing abstains"],
                reason_codes=[RouteSelectionReasonCode.ADAPTIVE_ABSTAIN],
                rejection_reason="adaptive policy avoids degraded backends",
            )
        if not estimate.sufficient_data:
            return CandidateAssessment(
                snapshot=snapshot,
                score=None,
                eligible=False,
                rationale=rationale,
                reason_codes=[RouteSelectionReasonCode.ADAPTIVE_ABSTAIN],
                rejection_reason="insufficient historical evidence for adaptive routing",
            )
        if (
            estimate.expected_error_rate is not None
            and estimate.expected_error_rate > self._config.max_expected_error_rate
        ):
            return CandidateAssessment(
                snapshot=snapshot,
                score=None,
                eligible=False,
                rationale=[
                    *rationale,
                    (
                        "adaptive policy rejected the candidate because the predicted error "
                        f"rate {estimate.expected_error_rate:.3f} exceeds the configured limit"
                    ),
                ],
                reason_codes=[RouteSelectionReasonCode.ADAPTIVE_ABSTAIN],
                rejection_reason="predicted error rate exceeds adaptive-policy limit",
            )
        score = _score_estimate(estimate=estimate, objective=self._config.objective)
        return CandidateAssessment(
            snapshot=snapshot,
            score=score,
            eligible=True,
            rationale=[
                *rationale,
                f"objective={self._config.objective.value}",
                f"expected_latency_ms={_fmt_optional(estimate.expected_latency_ms)}",
                f"expected_error_rate={_fmt_optional(estimate.expected_error_rate)}",
                f"expected_tokens_per_second={_fmt_optional(estimate.expected_tokens_per_second)}",
                f"score={score:.6f}",
            ],
            reason_codes=[RouteSelectionReasonCode.ADAPTIVE_ESTIMATE],
        )

    def _abstain_with_fallback(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
        reason: str,
        adaptive_assessments: Sequence[CandidateAssessment],
    ) -> PolicyEvaluation:
        fallback = self._fallback.evaluate(
            request=request,
            context=context,
            candidates=candidates,
        )
        by_backend = {assessment.snapshot.name: assessment for assessment in adaptive_assessments}
        merged = []
        for fallback_assessment in fallback.assessments:
            adaptive_assessment = by_backend.get(fallback_assessment.snapshot.name)
            if adaptive_assessment is None:
                merged.append(fallback_assessment)
                continue
            rationale = [
                reason,
                *adaptive_assessment.rationale,
                *fallback_assessment.rationale,
            ]
            merged.append(
                CandidateAssessment(
                    snapshot=fallback_assessment.snapshot,
                    score=fallback_assessment.score,
                    eligible=True,
                    rationale=rationale,
                    reason_codes=[
                        RouteSelectionReasonCode.ADAPTIVE_FALLBACK,
                        *fallback_assessment.reason_codes,
                    ],
                    rejection_reason=adaptive_assessment.rejection_reason,
                )
            )
        selected = next(
            assessment
            for assessment in merged
            if assessment.snapshot.name == fallback.selected_backend
        )
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=merged,
            selected_backend=fallback.selected_backend,
            selected_reason_codes=[
                RouteSelectionReasonCode.ADAPTIVE_ABSTAIN,
                RouteSelectionReasonCode.ADAPTIVE_FALLBACK,
                *fallback.selected_reason_codes,
            ],
            selected_reason=[reason, *selected.rationale],
        )

    def _should_explore(
        self,
        *,
        context: RequestContext,
        ranked: Sequence[CandidateAssessment],
    ) -> bool:
        if len(ranked) < 2:
            return False
        if self._config.deterministic_evaluation:
            return False
        if not self._config.exploration_enabled:
            return False
        if self._config.exploration_rate <= 0.0:
            return False
        gate = _deterministic_float(
            self.policy_reference.policy_id,
            context.request_id,
            context.tenant_id,
        )
        return gate < self._config.exploration_rate


def _estimate_context(
    *,
    request: ChatCompletionRequest,
    context: RequestContext,
    snapshot: BackendStatusSnapshot,
    evidence_policy_id: str,
    scope_by_tenant: bool,
) -> CandidateRouteEstimateContext:
    request_features = context.request_features
    prefix_signal = context.prefix_locality_signal
    backend_instance_id = (
        snapshot.instance_inventory[0].instance_id if snapshot.instance_inventory else None
    )
    return CandidateRouteEstimateContext(
        model_alias=request.model,
        backend_name=snapshot.name,
        backend_type=snapshot.capabilities.backend_type.value,
        backend_instance_id=backend_instance_id,
        policy_id=evidence_policy_id,
        request_class=context.request_class,
        tenant_id=context.tenant_id if scope_by_tenant else None,
        input_length_bucket=(
            None if request_features is None else request_features.input_length_bucket
        ),
        history_depth_bucket=(
            None if request_features is None else request_features.history_depth_bucket
        ),
        workload_tags=[] if request_features is None else list(request_features.workload_tags),
        prefix_hotness=None if prefix_signal is None else prefix_signal.hotness,
        cache_opportunity=None if prefix_signal is None else prefix_signal.cache_opportunity,
        locality_benefit=(
            None if prefix_signal is None else prefix_signal.likely_benefits_from_locality
        ),
    )


def _score_estimate(
    *,
    estimate: HistoricalRouteEstimate,
    objective: CounterfactualObjective,
) -> float:
    resolved_latency = estimate.expected_latency_ms or 1000.0
    resolved_ttft = estimate.expected_ttft_ms or 0.0
    resolved_error_rate = estimate.expected_error_rate or 0.0
    resolved_throughput = estimate.expected_tokens_per_second or 0.0
    resolved_queue_delay = estimate.expected_queue_delay_ms or 0.0
    if objective is CounterfactualObjective.LATENCY:
        return -(
            resolved_latency
            + resolved_queue_delay
            + (resolved_ttft / 2.0)
            + (resolved_error_rate * 500.0)
        )
    if objective is CounterfactualObjective.THROUGHPUT:
        return (
            resolved_throughput
            - (resolved_error_rate * 200.0)
            - (resolved_latency / 20.0)
        )
    if objective is CounterfactualObjective.RELIABILITY:
        return -(resolved_error_rate * 1000.0) - (resolved_latency / 50.0)
    return (
        resolved_throughput
        - resolved_latency
        - resolved_queue_delay
        - (resolved_ttft / 2.0)
        - (resolved_error_rate * 400.0)
    )


def _rank_assessments(assessments: Sequence[CandidateAssessment]) -> list[CandidateAssessment]:
    return sorted(
        assessments,
        key=lambda assessment: (
            not assessment.eligible,
            -(assessment.score or float("-inf")),
            assessment.snapshot.health.latency_ms or float("inf"),
            assessment.snapshot.name,
        ),
    )


def _confident_enough(
    *,
    best: CandidateAssessment,
    runner_up: CandidateAssessment | None,
    min_confidence_margin: float,
) -> bool:
    if best.score is None:
        return False
    if runner_up is None or runner_up.score is None:
        return True
    return (best.score - runner_up.score) >= min_confidence_margin


def _deterministic_float(*parts: str | None) -> float:
    seed = "::".join(part or "" for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return numerator / float(2**64)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


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
    if policy in {RoutingPolicy.LOCAL_ONLY, RoutingPolicy.REMOTE_DISABLED} and _is_remote_snapshot(
        snapshot
    ):
        return CandidateRejection(
            reason=(
                "policy requires a local backend"
                if policy is RoutingPolicy.LOCAL_ONLY
                else "policy disables remote backends"
            ),
            category="policy",
        )
    if context.force_remote_candidates_only and not _is_remote_snapshot(snapshot):
        return CandidateRejection(
            reason="local admission was saturated; request was forced to remote spillover",
            category="admission",
            reason_codes=[RouteSelectionReasonCode.LOCAL_ADMISSION_SPILLOVER],
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
    request: ChatCompletionRequest | None = None,
    context: RequestContext | None = None,
    candidates: Sequence[BackendStatusSnapshot] | None = None,
) -> float:
    del request
    latency_ms = snapshot.health.latency_ms or 1000.0
    predicted_latency_ms = _predicted_latency_ms(snapshot)
    queue_delay_ms = _predicted_queue_delay_ms(snapshot)
    quality = float(snapshot.capabilities.quality_tier)
    is_remote = _is_remote_snapshot(snapshot)
    local_bonus = 12.0 if not is_remote else 0.0
    health_bonus = 100.0 if snapshot.health.state is BackendHealthState.HEALTHY else 65.0
    warm_bonus = 8.0 if snapshot.health.load_state is BackendLoadState.READY else 0.0
    priority_bonus = max(0.0, 30.0 - float(snapshot.capabilities.configured_priority) / 4.0)
    weight_bonus = min(snapshot.capabilities.configured_weight * 3.0, 15.0)
    quality_hint_bonus = _quality_hint_bonus(snapshot)
    performance_bonus = _performance_hint_bonus(snapshot, policy=policy)
    network_penalty = _network_penalty(snapshot)
    cost_penalty = _cost_penalty(snapshot)
    locality_bonus = _prefix_locality_bonus(snapshot=snapshot, context=context)
    evidence_bonus = _evidence_bonus(snapshot)
    tenant_bonus = _tenant_bonus(snapshot=snapshot, context=context)

    if policy is RoutingPolicy.LATENCY_FIRST:
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + weight_bonus
            + performance_bonus
            - latency_ms
        )
    if policy is RoutingPolicy.LOCAL_PREFERRED:
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + weight_bonus
            + (local_bonus * 2.5)
            + locality_bonus
            + evidence_bonus
            + tenant_bonus
            - (predicted_latency_ms / 3.0)
            - (queue_delay_ms / 2.0)
            - network_penalty
            - cost_penalty
        )
    if policy is RoutingPolicy.BURST_TO_REMOTE:
        burst_remote_bonus = (
            40.0 if is_remote and _local_capacity_pressure(candidates or []) else 0.0
        )
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + weight_bonus
            + locality_bonus
            + evidence_bonus
            + tenant_bonus
            + burst_remote_bonus
            + (0.0 if is_remote else 10.0)
            - (predicted_latency_ms / 2.0)
            - queue_delay_ms
            - (network_penalty / 2.0)
            - cost_penalty
        )
    if policy is RoutingPolicy.LATENCY_SLO:
        target = 0 if context is None or context.max_latency_ms is None else context.max_latency_ms
        projected_latency = predicted_latency_ms + queue_delay_ms
        meets_slo_bonus = 35.0 if target > 0 and projected_latency <= target else 0.0
        misses_slo_penalty = 25.0 if target > 0 and projected_latency > target else 0.0
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + performance_bonus
            + evidence_bonus
            + tenant_bonus
            + meets_slo_bonus
            - projected_latency
            - network_penalty
            - misses_slo_penalty
            - (cost_penalty / 2.0)
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
    if policy is RoutingPolicy.QUALITY_ON_DEMAND:
        demand_bonus = 0.0
        if context is not None and (
            context.tenant_tier.value == "priority"
            or context.workload_shape.value == "evaluation"
        ):
            demand_bonus = 25.0
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + (quality * 25.0)
            + quality_hint_bonus
            + demand_bonus
            + locality_bonus
            + evidence_bonus
            - (predicted_latency_ms / 8.0)
            - (queue_delay_ms / 4.0)
            - (network_penalty / 3.0)
            - (cost_penalty / 2.0)
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
    if policy is RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY:
        remote_bonus = (
            45.0
            if is_remote and not _has_healthy_local_candidate(candidates or [])
            else 0.0
        )
        return (
            health_bonus
            + warm_bonus
            + priority_bonus
            + weight_bonus
            + evidence_bonus
            + locality_bonus
            + remote_bonus
            + (0.0 if is_remote else 8.0)
            - (predicted_latency_ms / 2.5)
            - (queue_delay_ms / 2.0)
            - (network_penalty / 2.0)
            - cost_penalty
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
    context: RequestContext | None = None,
) -> list[str]:
    latency_ms = snapshot.health.latency_ms or 1000.0
    predicted_latency_ms = _predicted_latency_ms(snapshot)
    predicted_queue_delay_ms = _predicted_queue_delay_ms(snapshot)
    return [
        f"policy={policy.value}",
        f"health={snapshot.health.state.value}",
        f"readiness={snapshot.health.load_state.value}",
        f"latency_ms={latency_ms:.1f}",
        f"predicted_latency_ms={predicted_latency_ms:.1f}",
        f"predicted_queue_delay_ms={predicted_queue_delay_ms:.1f}",
        f"quality_tier={snapshot.capabilities.quality_tier}",
        f"quality_hint={snapshot.capabilities.quality_hint.value}",
        f"performance_hint={snapshot.capabilities.performance_hint.value}",
        f"device_class={snapshot.capabilities.device_class.value}",
        f"configured_priority={snapshot.capabilities.configured_priority}",
        f"configured_weight={snapshot.capabilities.configured_weight:.2f}",
        (
            "prefix_locality=preferred"
            if context is not None
            and context.prefix_locality_signal is not None
            and context.prefix_locality_signal.preferred_backend == snapshot.name
            else "prefix_locality=none"
        ),
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


def _compatibility_assessment(
    *,
    snapshot: BackendStatusSnapshot,
    candidates: Sequence[BackendStatusSnapshot],
    policy: RoutingPolicy,
    request: ChatCompletionRequest,
    context: RequestContext,
) -> CandidateAssessment:
    score = _score_snapshot(
        snapshot=snapshot,
        policy=policy,
        request=request,
        context=context,
        candidates=candidates,
    )
    return CandidateAssessment(
        snapshot=snapshot,
        score=score,
        eligible=True,
        rationale=_score_rationale(snapshot=snapshot, policy=policy, context=context),
        reason_codes=_reason_codes_for(snapshot=snapshot, policy=policy, context=context),
    )


def _reason_codes_for(
    *,
    snapshot: BackendStatusSnapshot,
    policy: RoutingPolicy,
    context: RequestContext,
) -> list[RouteSelectionReasonCode]:
    if policy in {
        RoutingPolicy.LATENCY_FIRST,
        RoutingPolicy.BALANCED,
        RoutingPolicy.QUALITY_FIRST,
        RoutingPolicy.LOCAL_ONLY,
    }:
        return [RouteSelectionReasonCode.POLICY_SCORE]
    codes = [RouteSelectionReasonCode.POLICY_SCORE]
    if policy is RoutingPolicy.LOCAL_PREFERRED and not _is_remote_snapshot(snapshot):
        codes.append(RouteSelectionReasonCode.HYBRID_LOCAL_PREFERENCE)
    if policy is RoutingPolicy.BURST_TO_REMOTE and _is_remote_snapshot(snapshot):
        codes.append(RouteSelectionReasonCode.HYBRID_BURST_REMOTE)
    if policy is RoutingPolicy.LATENCY_SLO:
        codes.append(RouteSelectionReasonCode.HYBRID_LATENCY_SLO)
    if policy is RoutingPolicy.QUALITY_ON_DEMAND:
        codes.append(RouteSelectionReasonCode.HYBRID_QUALITY_ON_DEMAND)
    if policy is RoutingPolicy.REMOTE_DISABLED:
        codes.append(RouteSelectionReasonCode.HYBRID_REMOTE_DISABLED)
    if (
        policy is RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY
        and _is_remote_snapshot(snapshot)
    ):
        codes.append(RouteSelectionReasonCode.HYBRID_REMOTE_IF_LOCAL_UNHEALTHY)
    if _is_remote_snapshot(snapshot):
        codes.append(RouteSelectionReasonCode.NETWORK_PENALTY)
    if _predicted_queue_delay_ms(snapshot) > 0:
        codes.append(RouteSelectionReasonCode.QUEUE_PREDICTION)
    if _cost_penalty(snapshot) > 0:
        codes.append(RouteSelectionReasonCode.COST_PRESSURE)
    if (
        context.prefix_locality_signal is not None
        and context.prefix_locality_signal.preferred_backend == snapshot.name
    ):
        codes.append(RouteSelectionReasonCode.PREFIX_LOCALITY)
    codes.append(
        RouteSelectionReasonCode.EVIDENCE_SUFFICIENT
        if _has_sufficient_evidence(snapshot)
        else RouteSelectionReasonCode.EVIDENCE_INSUFFICIENT
    )
    if context.tenant_tier.value == "priority" or context.request_class.value != "standard":
        codes.append(RouteSelectionReasonCode.TENANT_POLICY)
    return codes


def _is_remote_snapshot(snapshot: BackendStatusSnapshot) -> bool:
    if snapshot.capabilities.execution_mode in {
        ExecutionModeLabel.REMOTE_WORKER,
        ExecutionModeLabel.EXTERNAL_SERVICE,
    }:
        return True
    if snapshot.deployment is not None and snapshot.deployment.execution_mode in {
        ExecutionModeLabel.REMOTE_WORKER,
        ExecutionModeLabel.EXTERNAL_SERVICE,
    }:
        return True
    if snapshot.capabilities.device_class is DeviceClass.REMOTE:
        return True
    return any(
        instance.locality_class
        in {
            WorkerLocalityClass.REMOTE_PRIVATE,
            WorkerLocalityClass.REMOTE_CLOUD,
            WorkerLocalityClass.EXTERNAL_SERVICE,
        }
        for instance in snapshot.instance_inventory
    )


def _predicted_latency_ms(snapshot: BackendStatusSnapshot) -> float:
    predicted = snapshot.metadata.get("predicted_latency_ms")
    if predicted is not None:
        try:
            return float(predicted)
        except ValueError:
            pass
    return snapshot.health.latency_ms or 1000.0


def _predicted_queue_delay_ms(snapshot: BackendStatusSnapshot) -> float:
    predicted = snapshot.metadata.get("predicted_queue_delay_ms")
    if predicted is not None:
        try:
            return float(predicted)
        except ValueError:
            pass
    if snapshot.capabilities.concurrency_limit <= 0:
        return 0.0
    base_latency = snapshot.health.latency_ms or 0.0
    return (snapshot.queue_depth / snapshot.capabilities.concurrency_limit) * base_latency


def _network_penalty(snapshot: BackendStatusSnapshot) -> float:
    if not _is_remote_snapshot(snapshot):
        return 0.0
    profile = snapshot.capabilities.network_characteristics.profile
    if profile is NetworkProfile.INTERNET:
        return 30.0
    if profile is NetworkProfile.WAN:
        return 18.0
    if profile is NetworkProfile.LAN:
        return 6.0
    return 12.0


def _cost_penalty(snapshot: BackendStatusSnapshot) -> float:
    relative_cost = snapshot.capabilities.cost_profile.relative_cost_index
    penalty = 0.0 if relative_cost is None else relative_cost * 10.0
    if snapshot.capabilities.cost_profile.profile.value == "premium":
        penalty += 15.0
    elif snapshot.capabilities.cost_profile.profile.value == "standard":
        penalty += 6.0
    return penalty


def _prefix_locality_bonus(
    *,
    snapshot: BackendStatusSnapshot,
    context: RequestContext | None,
) -> float:
    if context is None or context.prefix_locality_signal is None:
        return 0.0
    signal = context.prefix_locality_signal
    if signal.preferred_backend == snapshot.name:
        return 20.0
    if signal.candidate_local_backend == snapshot.name:
        return 8.0
    return 0.0


def _tenant_bonus(
    *,
    snapshot: BackendStatusSnapshot,
    context: RequestContext | None,
) -> float:
    if context is None:
        return 0.0
    if context.tenant_tier.value == "priority":
        if snapshot.capabilities.quality_hint is QualityHint.PREMIUM:
            return 12.0
        if snapshot.capabilities.performance_hint is PerformanceHint.LATENCY_OPTIMIZED:
            return 10.0
    if context.request_class.value == "bulk" and _is_remote_snapshot(snapshot):
        return 6.0
    return 0.0


def _has_sufficient_evidence(snapshot: BackendStatusSnapshot) -> bool:
    return snapshot.metadata.get("evidence_sufficient", "").lower() == "true"


def _evidence_bonus(snapshot: BackendStatusSnapshot) -> float:
    if _has_sufficient_evidence(snapshot):
        confidence = snapshot.metadata.get("confidence_score")
        try:
            return 10.0 + (0.0 if confidence is None else float(confidence) * 10.0)
        except ValueError:
            return 10.0
    return -5.0


def _has_healthy_local_candidate(candidates: Sequence[BackendStatusSnapshot]) -> bool:
    return any(
        not _is_remote_snapshot(candidate)
        and candidate.health.state is BackendHealthState.HEALTHY
        for candidate in candidates
    )


def _local_capacity_pressure(candidates: Sequence[BackendStatusSnapshot]) -> bool:
    local_candidates = [
        candidate for candidate in candidates if not _is_remote_snapshot(candidate)
    ]
    if not local_candidates:
        return False
    return any(
        candidate.active_requests >= candidate.capabilities.concurrency_limit
        or candidate.queue_depth > 0
        or candidate.health.load_state in {BackendLoadState.COLD, BackendLoadState.WARMING}
        for candidate in local_candidates
    )
