"""Local-first rollout controls for intelligent routing policies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from switchyard.config import PolicyRolloutSettings
from switchyard.router.policies import PolicyEvaluation, PolicyRegistry, RoutingPolicyScorer
from switchyard.schemas.admin import (
    PolicyDecisionRuntimeEntry,
    PolicyRolloutRuntimeSummary,
    PolicyRolloutStateSnapshot,
    PolicyRolloutUpdateRequest,
)
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import (
    PolicyReference,
    PolicyRolloutMode,
    RequestContext,
    RouteSelectionReasonCode,
    RoutingPolicy,
)


@dataclass(frozen=True, slots=True)
class PolicyRolloutResolution:
    """Resolved primary and shadow scorers for one request under rollout controls."""

    mode: PolicyRolloutMode
    primary_policy: RoutingPolicyScorer
    shadow_policies: list[RoutingPolicyScorer]
    notes: list[str]
    canary_selected: bool = False
    kill_switch_applied: bool = False


class PolicyRolloutService:
    """Small mutable controller for safe local policy rollout and diagnostics."""

    def __init__(
        self,
        settings: PolicyRolloutSettings,
        *,
        candidate_policies: Iterable[RoutingPolicyScorer] | None = None,
    ) -> None:
        self._candidate_by_id = {
            scorer.policy_reference.policy_id: scorer for scorer in (candidate_policies or [])
        }
        self._initial_mode = settings.mode
        self._initial_canary_percentage = settings.canary_percentage
        self._initial_kill_switch_enabled = settings.kill_switch_enabled
        self._initial_learning_frozen = settings.learning_frozen
        self._max_recent_decisions = settings.max_recent_decisions
        self._mode = settings.mode
        self._candidate_policy_id = settings.candidate_policy_id
        self._shadow_policy_id = settings.shadow_policy_id
        self._canary_percentage = settings.canary_percentage
        self._kill_switch_enabled = settings.kill_switch_enabled
        self._learning_frozen = settings.learning_frozen
        self._recent_decisions: deque[PolicyDecisionRuntimeEntry] = deque(
            maxlen=settings.max_recent_decisions
        )
        self._last_policy_update_at: datetime | None = None
        self._last_learning_event_at: datetime | None = None
        self._last_learning_event: str | None = None
        self._last_guardrail_trigger: str | None = None
        self._notes: list[str] = []

    def resolve(
        self,
        *,
        registry: PolicyRegistry,
        requested_policy: RoutingPolicy,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> PolicyRolloutResolution:
        """Resolve the scorer set for the current request under rollout controls."""

        del request
        baseline = registry.resolve(requested_policy)
        candidate = self._candidate_for_policy(requested_policy)
        configured_shadow = self._configured_shadow_policy(requested_policy)
        base_shadows = registry.shadow_policies

        notes: list[str] = []
        if self._kill_switch_enabled:
            notes.append("policy rollout kill switch forced baseline routing")
            return PolicyRolloutResolution(
                mode=self._mode,
                primary_policy=baseline,
                shadow_policies=base_shadows,
                notes=notes,
                kill_switch_applied=True,
            )
        if self._mode is PolicyRolloutMode.DISABLED or candidate is None:
            if self._mode is not PolicyRolloutMode.DISABLED and candidate is None:
                notes.append("candidate policy unavailable; baseline routing remained active")
            return PolicyRolloutResolution(
                mode=self._mode,
                primary_policy=baseline,
                shadow_policies=base_shadows,
                notes=notes,
            )
        if self._mode is PolicyRolloutMode.SHADOW_ONLY:
            shadow_policy = configured_shadow or candidate
            return PolicyRolloutResolution(
                mode=self._mode,
                primary_policy=baseline,
                shadow_policies=self._merge_shadow_policies(base_shadows, shadow_policy),
                notes=notes,
            )
        if self._mode is PolicyRolloutMode.REPORT_ONLY:
            shadow_policy = configured_shadow or candidate
            return PolicyRolloutResolution(
                mode=self._mode,
                primary_policy=baseline,
                shadow_policies=self._merge_shadow_policies(base_shadows, shadow_policy),
                notes=["report-only mode kept baseline routing active"],
            )
        if self._mode is PolicyRolloutMode.CANARY:
            canary_selected = self._canary_selected(context)
            primary = candidate if canary_selected else baseline
            shadows = base_shadows
            if not canary_selected:
                shadows = self._merge_shadow_policies(shadows, configured_shadow or candidate)
            return PolicyRolloutResolution(
                mode=self._mode,
                primary_policy=primary,
                shadow_policies=shadows,
                notes=[
                    (
                        "policy rollout canary selected candidate policy"
                        if canary_selected
                        else "policy rollout canary kept baseline policy"
                    )
                ],
                canary_selected=canary_selected,
            )
        return PolicyRolloutResolution(
            mode=self._mode,
            primary_policy=candidate,
            shadow_policies=self._merge_shadow_policies(base_shadows, configured_shadow),
            notes=["guarded adaptive policy active"],
        )

    def observe_decision(
        self,
        *,
        context: RequestContext,
        resolution: PolicyRolloutResolution,
        primary_evaluation: PolicyEvaluation,
        shadow_evaluations: Sequence[PolicyEvaluation],
    ) -> None:
        """Record a bounded recent policy-decision history for inspection."""

        guardrail_triggers = self._guardrail_triggers(
            primary_evaluation=primary_evaluation,
            resolution=resolution,
        )
        if guardrail_triggers:
            self._last_guardrail_trigger = guardrail_triggers[-1]
        shadow_policy = shadow_evaluations[0].policy_reference if shadow_evaluations else None
        entry = PolicyDecisionRuntimeEntry(
            request_id=context.request_id,
            requested_policy=context.policy.value,
            rollout_mode=resolution.mode,
            selected_policy=primary_evaluation.policy_reference,
            selected_backend=primary_evaluation.selected_backend,
            shadow_policy=shadow_policy,
            abstained=RouteSelectionReasonCode.ADAPTIVE_ABSTAIN
            in primary_evaluation.selected_reason_codes,
            exploration_used=RouteSelectionReasonCode.ADAPTIVE_EXPLORATION
            in primary_evaluation.selected_reason_codes,
            canary_selected=resolution.canary_selected,
            guardrail_triggers=guardrail_triggers,
            notes=[*resolution.notes, *primary_evaluation.selected_reason],
        )
        self._recent_decisions.append(entry)

    def update(self, request: PolicyRolloutUpdateRequest) -> PolicyRolloutRuntimeSummary:
        """Mutate rollout controls using a small admin request."""

        if request.mode is not None:
            self._mode = request.mode
        if request.canary_percentage is not None:
            self._canary_percentage = request.canary_percentage
        if request.kill_switch_enabled is not None:
            self._kill_switch_enabled = request.kill_switch_enabled
        if request.learning_frozen is not None:
            self._learning_frozen = request.learning_frozen
        self._last_policy_update_at = datetime.now(UTC)
        return self.inspect_state()

    def record_learning_event(self, description: str) -> bool:
        """Record a future adaptive-learning event unless learning is frozen."""

        if self._learning_frozen:
            self._last_guardrail_trigger = "learning_frozen"
            return False
        self._last_learning_event = description
        self._last_learning_event_at = datetime.now(UTC)
        return True

    def reset_state(self) -> PolicyRolloutRuntimeSummary:
        """Reset runtime rollout state back to its initial safe defaults."""

        self._mode = self._initial_mode
        self._canary_percentage = self._initial_canary_percentage
        self._kill_switch_enabled = self._initial_kill_switch_enabled
        self._learning_frozen = self._initial_learning_frozen
        self._recent_decisions.clear()
        self._last_guardrail_trigger = None
        self._last_learning_event = None
        self._last_learning_event_at = None
        self._last_policy_update_at = datetime.now(UTC)
        self._notes = []
        return self.inspect_state()

    def export_state(self) -> PolicyRolloutStateSnapshot:
        """Export the current mutable rollout state for local persistence."""

        return PolicyRolloutStateSnapshot(
            mode=self._mode,
            candidate_policy_id=self._candidate_policy_id,
            shadow_policy_id=self._shadow_policy_id,
            canary_percentage=self._canary_percentage,
            kill_switch_enabled=self._kill_switch_enabled,
            learning_frozen=self._learning_frozen,
            last_policy_update_at=self._last_policy_update_at,
            last_learning_event_at=self._last_learning_event_at,
            last_learning_event=self._last_learning_event,
            recent_decisions=list(self._recent_decisions),
            notes=list(self._notes),
        )

    def import_state(self, snapshot: PolicyRolloutStateSnapshot) -> PolicyRolloutRuntimeSummary:
        """Import a previously exported rollout state snapshot."""

        self._mode = snapshot.mode
        self._candidate_policy_id = snapshot.candidate_policy_id
        self._shadow_policy_id = snapshot.shadow_policy_id
        self._canary_percentage = snapshot.canary_percentage
        self._kill_switch_enabled = snapshot.kill_switch_enabled
        self._learning_frozen = snapshot.learning_frozen
        self._last_policy_update_at = snapshot.last_policy_update_at or datetime.now(UTC)
        self._last_learning_event_at = snapshot.last_learning_event_at
        self._last_learning_event = snapshot.last_learning_event
        self._recent_decisions = deque(snapshot.recent_decisions, maxlen=self._max_recent_decisions)
        self._notes = list(snapshot.notes)
        return self.inspect_state()

    def inspect_state(self) -> PolicyRolloutRuntimeSummary:
        """Return current rollout controls and bounded recent decision history."""

        candidate_policy = self._selected_candidate_policy()
        shadow_policy = self._selected_shadow_policy()
        active_policy = self._active_policy_reference(candidate_policy)
        recent_decisions = list(self._recent_decisions)
        return PolicyRolloutRuntimeSummary(
            mode=self._mode,
            candidate_policy=(
                None if candidate_policy is None else candidate_policy.policy_reference
            ),
            active_policy=active_policy,
            shadow_policy=None if shadow_policy is None else shadow_policy.policy_reference,
            compatibility_policy=(
                None
                if candidate_policy is None or candidate_policy.compatibility_policy is None
                else candidate_policy.compatibility_policy.value
            ),
            canary_percentage=self._canary_percentage,
            kill_switch_enabled=self._kill_switch_enabled,
            learning_frozen=self._learning_frozen,
            exploration_enabled=self._candidate_exploration_enabled(candidate_policy),
            recent_decisions=recent_decisions,
            recent_abstentions=sum(1 for decision in recent_decisions if decision.abstained),
            recent_guardrail_triggers=self._recent_guardrail_triggers(recent_decisions),
            last_policy_update_at=self._last_policy_update_at,
            last_learning_event_at=self._last_learning_event_at,
            last_learning_event=self._last_learning_event,
            last_guardrail_trigger=self._last_guardrail_trigger,
            notes=list(self._notes),
        )

    def _candidate_for_policy(self, requested_policy: RoutingPolicy) -> RoutingPolicyScorer | None:
        candidate = self._selected_candidate_policy()
        if candidate is None:
            return None
        if (
            candidate.compatibility_policy is None
            or candidate.compatibility_policy is requested_policy
        ):
            return candidate
        return None

    def _configured_shadow_policy(
        self,
        requested_policy: RoutingPolicy,
    ) -> RoutingPolicyScorer | None:
        shadow = self._selected_shadow_policy()
        if shadow is None:
            return None
        if shadow.compatibility_policy is None or shadow.compatibility_policy is requested_policy:
            return shadow
        return None

    def _selected_candidate_policy(self) -> RoutingPolicyScorer | None:
        if self._candidate_policy_id is None:
            return None
        return self._candidate_by_id.get(self._candidate_policy_id)

    def _selected_shadow_policy(self) -> RoutingPolicyScorer | None:
        if self._shadow_policy_id is None:
            return None
        return self._candidate_by_id.get(self._shadow_policy_id)

    def _merge_shadow_policies(
        self,
        existing: Sequence[RoutingPolicyScorer],
        additional: RoutingPolicyScorer | None,
    ) -> list[RoutingPolicyScorer]:
        merged = list(existing)
        if additional is None:
            return merged
        if all(
            scorer.policy_reference.policy_id != additional.policy_reference.policy_id
            for scorer in merged
        ):
            merged.append(additional)
        return merged

    def _active_policy_reference(
        self,
        candidate_policy: RoutingPolicyScorer | None,
    ) -> PolicyReference | None:
        if self._kill_switch_enabled:
            return None
        if candidate_policy is None:
            return None
        if self._mode is PolicyRolloutMode.ACTIVE_GUARDED:
            return candidate_policy.policy_reference
        if self._mode is PolicyRolloutMode.CANARY:
            return candidate_policy.policy_reference
        return None

    def _candidate_exploration_enabled(
        self,
        candidate_policy: RoutingPolicyScorer | None,
    ) -> bool:
        if candidate_policy is None:
            return False
        config = getattr(candidate_policy, "_config", None)
        return bool(getattr(config, "exploration_enabled", False))

    def _canary_selected(self, context: RequestContext) -> bool:
        bucket_key = context.session_id or context.request_id
        bucket = _bucket_percentage(
            policy_name=self._candidate_policy_id or "candidate-policy",
            bucket_key=bucket_key,
        )
        return bucket < self._canary_percentage

    def _guardrail_triggers(
        self,
        *,
        primary_evaluation: PolicyEvaluation,
        resolution: PolicyRolloutResolution,
    ) -> list[str]:
        triggers = list(resolution.notes)
        if resolution.kill_switch_applied:
            triggers.append("kill_switch_enabled")
        selected_codes = set(primary_evaluation.selected_reason_codes)
        if RouteSelectionReasonCode.ADAPTIVE_ABSTAIN in selected_codes:
            triggers.append("adaptive_abstain")
        if RouteSelectionReasonCode.ADAPTIVE_FALLBACK in selected_codes:
            triggers.append("adaptive_fallback")
        if RouteSelectionReasonCode.ADAPTIVE_EXPLORATION in selected_codes:
            triggers.append("adaptive_exploration")
        for assessment in primary_evaluation.assessments:
            if assessment.rejection_reason is not None and (
                RouteSelectionReasonCode.ADAPTIVE_ABSTAIN in assessment.reason_codes
            ):
                triggers.append(assessment.rejection_reason)
        return list(dict.fromkeys(triggers))

    def _recent_guardrail_triggers(
        self,
        recent_decisions: Sequence[PolicyDecisionRuntimeEntry],
    ) -> list[str]:
        seen: list[str] = []
        for decision in recent_decisions:
            for trigger in decision.guardrail_triggers:
                if trigger not in seen:
                    seen.append(trigger)
        return seen[-10:]


def _bucket_percentage(*, policy_name: str, bucket_key: str) -> float:
    digest = sha256(f"{policy_name}:{bucket_key}".encode()).digest()
    scaled = int.from_bytes(digest[:8], "big") / float(2**64)
    return scaled * 100.0
