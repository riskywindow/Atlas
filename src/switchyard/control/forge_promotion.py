"""Forge Stage A rollout controls for bounded optimization-profile promotion."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from switchyard.config import Settings
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.control.spillover import (
    HybridMutableRuntimeSettings,
    RemoteSpilloverControlService,
)
from switchyard.optimization import build_trial_optimization_config_profile
from switchyard.router.policies import CompatibilityRoutingPolicy, PolicyEvaluation
from switchyard.schemas.admin import PolicyRolloutStateSnapshot
from switchyard.schemas.backend import BackendStatusSnapshot
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.forge import (
    ForgePromotionAppliedKnobChange,
    ForgePromotionApplyRequest,
    ForgePromotionCompareRequest,
    ForgePromotionComparisonSummary,
    ForgePromotionDecisionRequest,
    ForgePromotionLifecycleEvent,
    ForgePromotionLifecycleState,
    ForgePromotionProposeRequest,
    ForgePromotionRuntimeSummary,
)
from switchyard.schemas.optimization import (
    OptimizationCampaignComparisonArtifact,
    OptimizationConfigProfile,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
)
from switchyard.schemas.routing import (
    PolicyRolloutMode,
    RequestContext,
    RoutingPolicy,
)

_SUPPORTED_RUNTIME_KNOB_IDS = {
    "default_routing_policy",
    "policy_rollout_mode",
    "policy_rollout_canary_percentage",
    "hybrid_spillover_enabled",
    "hybrid_max_remote_share_percent",
    "hybrid_remote_request_budget_per_minute",
    "hybrid_remote_concurrency_cap",
    "hybrid_remote_cooldown_seconds",
}
_HYBRID_RUNTIME_KNOB_IDS = {
    "hybrid_spillover_enabled",
    "hybrid_max_remote_share_percent",
    "hybrid_remote_request_budget_per_minute",
    "hybrid_remote_concurrency_cap",
    "hybrid_remote_cooldown_seconds",
}
_ACTIVE_LIFECYCLE_STATES = {
    ForgePromotionLifecycleState.CANARY_ACTIVE,
    ForgePromotionLifecycleState.COMPARED,
    ForgePromotionLifecycleState.PROMOTED_DEFAULT,
}
_FINAL_LIFECYCLE_STATES = {
    ForgePromotionLifecycleState.ROLLED_BACK,
    ForgePromotionLifecycleState.REJECTED,
}


class _PromotedConfigProfilePolicy:
    """Candidate scorer that evaluates with a promoted routing-policy config profile."""

    compatibility_policy: RoutingPolicy | None = None

    def __init__(self, *, config_profile_id: str, routing_policy: RoutingPolicy) -> None:
        self._delegate = CompatibilityRoutingPolicy(
            routing_policy,
            policy_id=config_profile_id,
            policy_version="phase9.forge-stage-a",
        )
        self.policy_reference = self._delegate.policy_reference

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        return self._delegate.evaluate(
            request=request,
            context=context,
            candidates=candidates,
        )


class ForgePromotionService:
    """Apply reviewed optimization profiles through an explicit bounded lifecycle."""

    def __init__(
        self,
        *,
        settings: Settings,
        baseline_config_profile_id: str,
        max_canary_percentage: float,
        requires_operator_review: bool,
        policy_rollout: PolicyRolloutService,
        spillover: RemoteSpilloverControlService,
    ) -> None:
        self._settings = settings
        self._baseline_config_profile_id = baseline_config_profile_id
        self._max_canary_percentage = max_canary_percentage
        self._requires_operator_review = requires_operator_review
        self._policy_rollout = policy_rollout
        self._spillover = spillover
        self._rollback_state: _PromotionRollbackState | None = None
        self._registered_candidate_policy_id: str | None = None
        self._summary = ForgePromotionRuntimeSummary(
            baseline_config_profile_id=baseline_config_profile_id,
            active_config_profile_id=baseline_config_profile_id,
            requires_operator_review=requires_operator_review,
            notes=["no promoted optimization profile is currently active"],
        )

    def inspect_state(self) -> ForgePromotionRuntimeSummary:
        """Return the current bounded Forge promotion posture."""

        return self._summary.model_copy(deep=True)

    def propose(self, request: ForgePromotionProposeRequest) -> ForgePromotionRuntimeSummary:
        """Create one explicit rollout proposal from a reviewed trial artifact."""

        if (
            self._summary.rollout_artifact_id is not None
            and self._summary.lifecycle_state not in _FINAL_LIFECYCLE_STATES
        ):
            msg = "an existing Forge rollout must be finalized before proposing another"
            raise ValueError(msg)

        trial = request.trial_artifact
        self._validate_trial(trial)
        config_profile = build_trial_optimization_config_profile(
            settings=self._settings,
            trial_artifact=trial,
            campaign_artifact_id=request.campaign_artifact_id,
        )
        if not config_profile.validation.compatible:
            issues = ", ".join(
                issue.issue_kind.value for issue in config_profile.validation.issues
            )
            msg = f"trial cannot be promoted safely as a config profile: {issues}"
            raise ValueError(msg)

        blocked_knobs = self._blocked_knob_changes(config_profile)
        if blocked_knobs:
            msg = "trial includes config changes that cannot be applied safely at runtime"
            raise ValueError(msg)

        rollout_artifact_id = _rollout_artifact_id(
            candidate_configuration_id=trial.candidate_configuration.candidate_configuration_id,
            timestamp=datetime.now(UTC),
        )
        proposed_canary = self._bounded_canary_percentage(
            proposed=(
                0.0
                if trial.promotion_decision is None
                else trial.promotion_decision.canary_percentage
            )
        )
        self._rollback_state = None
        self._summary = ForgePromotionRuntimeSummary(
            rollout_artifact_id=rollout_artifact_id,
            baseline_config_profile_id=self._baseline_config_profile_id,
            active_config_profile_id=self._baseline_config_profile_id,
            candidate_config_profile_id=config_profile.config_profile_id,
            lifecycle_state=ForgePromotionLifecycleState.PROPOSED,
            applied=False,
            campaign_id=trial.campaign_id,
            campaign_artifact_id=request.campaign_artifact_id,
            trial_artifact_id=trial.trial_artifact_id,
            baseline_candidate_configuration_id=trial.baseline_candidate_configuration_id,
            candidate_configuration_id=(
                trial.candidate_configuration.candidate_configuration_id
            ),
            candidate_kind=trial.candidate_configuration.candidate.candidate_kind,
            routing_policy=(
                None
                if config_profile.routing_policy is None
                else config_profile.routing_policy.value
            ),
            rollout_mode=PolicyRolloutMode.DISABLED,
            canary_percentage=proposed_canary,
            recommendation_disposition=(
                None
                if trial.recommendation_summary is None
                else trial.recommendation_summary.disposition
            ),
            promotion_disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
            evidence_kinds=sorted(
                {
                    evidence_kind
                    for evidence_kind in (
                        []
                        if trial.recommendation_summary is None
                        else trial.recommendation_summary.evidence_kinds
                    )
                },
                key=lambda item: item.value,
            ),
            config_profile=config_profile,
            blocked_knob_changes=[],
            rollback_available=False,
            requires_operator_review=self._requires_operator_review,
            lifecycle_events=[
                _lifecycle_event(
                    rollout_artifact_id=rollout_artifact_id,
                    lifecycle_state=ForgePromotionLifecycleState.PROPOSED,
                    promotion_disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
                    notes=[
                        "proposal recorded from a reviewed trial artifact",
                        *list(request.notes),
                    ],
                )
            ],
            notes=[
                "proposal is explicit and reviewable; runtime state is unchanged",
                *list(request.notes),
            ],
        )
        return self.inspect_state()

    def approve(self, request: ForgePromotionDecisionRequest) -> ForgePromotionRuntimeSummary:
        """Approve the current proposal without activating runtime rollout yet."""

        self._require_transition(
            request.rollout_artifact_id,
            expected={ForgePromotionLifecycleState.PROPOSED},
            action="approve",
        )
        return self._replace_summary(
            lifecycle_state=ForgePromotionLifecycleState.APPROVED,
            promotion_disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
            notes=[
                *self._summary.notes,
                *list(_decision_notes(prefix="approved", request=request)),
            ],
        )

    def apply(self, request: ForgePromotionApplyRequest) -> ForgePromotionRuntimeSummary:
        """Activate one approved profile as a bounded canary."""

        self._require_transition(
            request.rollout_artifact_id,
            expected={ForgePromotionLifecycleState.APPROVED},
            action="activate canary for",
        )
        config_profile = self._required_config_profile()
        applied_at = datetime.now(UTC)
        canary_percentage = self._bounded_canary_percentage(
            proposed=self._summary.canary_percentage,
            requested=request.canary_percentage,
        )
        self._rollback_state = _PromotionRollbackState(
            policy_rollout_snapshot=self._policy_rollout.export_state(),
            hybrid_snapshot=self._spillover.export_mutable_settings(),
            candidate_policy_id=self._registered_candidate_policy_id,
        )
        applied_knobs = self._apply_runtime_profile(
            config_profile=config_profile,
            canary_percentage=canary_percentage,
        )
        self._summary = self._summary.model_copy(
            update={
                "active_config_profile_id": config_profile.config_profile_id,
                "lifecycle_state": ForgePromotionLifecycleState.CANARY_ACTIVE,
                "applied": True,
                "rollout_mode": PolicyRolloutMode.CANARY,
                "canary_percentage": canary_percentage,
                "promotion_disposition": OptimizationPromotionDisposition.APPROVED_CANARY,
                "applied_knob_changes": applied_knobs,
                "rollback_available": True,
                "applied_at": applied_at,
                "lifecycle_events": [
                    *self._summary.lifecycle_events,
                    _lifecycle_event(
                        rollout_artifact_id=request.rollout_artifact_id,
                        lifecycle_state=ForgePromotionLifecycleState.CANARY_ACTIVE,
                        promotion_disposition=OptimizationPromotionDisposition.APPROVED_CANARY,
                        notes=[
                            "approved profile was activated as a bounded canary",
                            *list(request.notes),
                        ],
                    ),
                ],
                "notes": [
                    *self._summary.notes,
                    "canary remains bounded and rollback stays explicit",
                    *list(request.notes),
                ],
            },
            deep=True,
        )
        return self.inspect_state()

    def compare(self, request: ForgePromotionCompareRequest) -> ForgePromotionRuntimeSummary:
        """Attach artifact-backed canary-versus-baseline comparison evidence."""

        self._require_transition(
            request.rollout_artifact_id,
            expected={ForgePromotionLifecycleState.CANARY_ACTIVE},
            action="compare",
        )
        comparison = _extract_comparison_summary(
            summary=self._summary,
            comparison_artifact=request.comparison_artifact,
        )
        self._summary = self._summary.model_copy(
            update={
                "lifecycle_state": ForgePromotionLifecycleState.COMPARED,
                "comparison": comparison,
                "lifecycle_events": [
                    *self._summary.lifecycle_events,
                    _lifecycle_event(
                        rollout_artifact_id=request.rollout_artifact_id,
                        lifecycle_state=ForgePromotionLifecycleState.COMPARED,
                        promotion_disposition=OptimizationPromotionDisposition.APPROVED_CANARY,
                        notes=[
                            (
                                "artifact-backed canary comparison was attached to the "
                                "active rollout"
                            ),
                            *list(request.notes),
                        ],
                    ),
                ],
                "notes": [
                    *self._summary.notes,
                    "baseline-versus-canary comparison remains explicit and artifact-backed",
                    *list(request.notes),
                ],
            },
            deep=True,
        )
        return self.inspect_state()

    def promote_default(
        self,
        request: ForgePromotionDecisionRequest,
    ) -> ForgePromotionRuntimeSummary:
        """Promote the compared profile to the default routed policy at runtime."""

        self._require_transition(
            request.rollout_artifact_id,
            expected={ForgePromotionLifecycleState.COMPARED},
            action="promote",
        )
        config_profile = self._required_config_profile()
        if self._registered_candidate_policy_id is None or config_profile.routing_policy is None:
            msg = "promote-default requires an active candidate routing policy canary"
            raise ValueError(msg)
        snapshot = self._policy_rollout.export_state().model_copy(
            update={
                "mode": PolicyRolloutMode.ACTIVE_GUARDED,
                "canary_percentage": 0.0,
                "notes": [
                    *self._policy_rollout.export_state().notes,
                    (
                        "Forge Stage A compared profile promoted to default through "
                        "guarded policy rollout"
                    ),
                ],
            },
            deep=True,
        )
        self._policy_rollout.import_state(snapshot)
        self._summary = self._summary.model_copy(
            update={
                "lifecycle_state": ForgePromotionLifecycleState.PROMOTED_DEFAULT,
                "rollout_mode": PolicyRolloutMode.ACTIVE_GUARDED,
                "canary_percentage": 0.0,
                "promotion_disposition": OptimizationPromotionDisposition.PROMOTED_DEFAULT,
                "lifecycle_events": [
                    *self._summary.lifecycle_events,
                    _lifecycle_event(
                        rollout_artifact_id=request.rollout_artifact_id,
                        lifecycle_state=ForgePromotionLifecycleState.PROMOTED_DEFAULT,
                        promotion_disposition=OptimizationPromotionDisposition.PROMOTED_DEFAULT,
                        notes=[
                            "compared canary was explicitly promoted to the default policy",
                            *list(_decision_notes(prefix="promoted", request=request)),
                        ],
                    ),
                ],
                "notes": [
                    *self._summary.notes,
                    "default promotion remains bounded to reversible runtime state",
                    *list(_decision_notes(prefix="promoted", request=request)),
                ],
            },
            deep=True,
        )
        return self.inspect_state()

    def reject(self, request: ForgePromotionDecisionRequest) -> ForgePromotionRuntimeSummary:
        """Reject the current proposal and restore baseline runtime posture when needed."""

        self._assert_rollout_artifact_id(request.rollout_artifact_id)
        lifecycle_state = self._required_lifecycle_state()
        if lifecycle_state in _FINAL_LIFECYCLE_STATES:
            msg = f"cannot reject a rollout that is already {lifecycle_state.value}"
            raise ValueError(msg)
        if lifecycle_state in _ACTIVE_LIFECYCLE_STATES:
            self._restore_runtime_state()
        self._summary = self._summary.model_copy(
            update={
                "active_config_profile_id": self._baseline_config_profile_id,
                "lifecycle_state": ForgePromotionLifecycleState.REJECTED,
                "applied": False,
                "rollout_mode": PolicyRolloutMode.DISABLED,
                "canary_percentage": 0.0,
                "promotion_disposition": OptimizationPromotionDisposition.REJECTED,
                "rollback_available": False,
                "last_reset_at": datetime.now(UTC),
                "lifecycle_events": [
                    *self._summary.lifecycle_events,
                    _lifecycle_event(
                        rollout_artifact_id=request.rollout_artifact_id,
                        lifecycle_state=ForgePromotionLifecycleState.REJECTED,
                        promotion_disposition=OptimizationPromotionDisposition.REJECTED,
                        notes=list(_decision_notes(prefix="rejected", request=request)),
                    ),
                ],
                "notes": [
                    *self._summary.notes,
                    *list(_decision_notes(prefix="rejected", request=request)),
                ],
            },
            deep=True,
        )
        return self.inspect_state()

    def reset(self, request: ForgePromotionDecisionRequest) -> ForgePromotionRuntimeSummary:
        """Roll back any canary or promoted-default runtime state to the baseline."""

        self._require_transition(
            request.rollout_artifact_id,
            expected=_ACTIVE_LIFECYCLE_STATES,
            action="roll back",
        )
        self._restore_runtime_state()
        self._summary = self._summary.model_copy(
            update={
                "active_config_profile_id": self._baseline_config_profile_id,
                "lifecycle_state": ForgePromotionLifecycleState.ROLLED_BACK,
                "applied": False,
                "rollout_mode": PolicyRolloutMode.DISABLED,
                "canary_percentage": 0.0,
                "promotion_disposition": OptimizationPromotionDisposition.ROLLED_BACK,
                "rollback_available": False,
                "last_reset_at": datetime.now(UTC),
                "lifecycle_events": [
                    *self._summary.lifecycle_events,
                    _lifecycle_event(
                        rollout_artifact_id=request.rollout_artifact_id,
                        lifecycle_state=ForgePromotionLifecycleState.ROLLED_BACK,
                        promotion_disposition=OptimizationPromotionDisposition.ROLLED_BACK,
                        notes=list(_decision_notes(prefix="rolled back", request=request)),
                    ),
                ],
                "notes": [
                    *self._summary.notes,
                    *list(_decision_notes(prefix="rolled back", request=request)),
                ],
            },
            deep=True,
        )
        return self.inspect_state()

    def _validate_trial(self, trial: object) -> None:
        recommendation = getattr(trial, "recommendation_summary", None)
        decision = getattr(trial, "promotion_decision", None)
        candidate_configuration = getattr(trial, "candidate_configuration", None)
        if recommendation is None:
            msg = "trial is missing recommendation_summary"
            raise ValueError(msg)
        if (
            recommendation.disposition
            is not OptimizationRecommendationDisposition.PROMOTE_CANDIDATE
        ):
            msg = "trial recommendation does not promote this candidate"
            raise ValueError(msg)
        if decision is None:
            msg = "trial is missing promotion_decision"
            raise ValueError(msg)
        if decision.disposition not in {
            OptimizationPromotionDisposition.RECOMMEND_CANARY,
            OptimizationPromotionDisposition.APPROVED_CANARY,
        }:
            msg = "trial promotion_decision does not approve a canary rollout"
            raise ValueError(msg)
        if (
            candidate_configuration is None
            or candidate_configuration.candidate_configuration_id
            != decision.candidate_configuration_id
        ):
            msg = "promotion_decision does not match candidate_configuration"
            raise ValueError(msg)

    def _blocked_knob_changes(
        self,
        config_profile: OptimizationConfigProfile,
    ) -> list[ForgePromotionAppliedKnobChange]:
        blocked: list[ForgePromotionAppliedKnobChange] = []
        changed_knob_ids = {change.knob_id for change in config_profile.changes}
        for change in config_profile.changes:
            if (
                change.knob_id in {"policy_rollout_mode", "policy_rollout_canary_percentage"}
                and "default_routing_policy" not in changed_knob_ids
            ):
                blocked.append(
                    ForgePromotionAppliedKnobChange(
                        knob_id=change.knob_id,
                        config_path=change.config_path,
                        runtime_mutable=True,
                        applied=False,
                        baseline_value=change.baseline_value,
                        candidate_value=change.candidate_value,
                        notes=[
                            "rollout-only changes require an executable candidate routing policy"
                        ],
                    )
                )
                continue
            if change.knob_id in _SUPPORTED_RUNTIME_KNOB_IDS:
                continue
            blocked.append(
                ForgePromotionAppliedKnobChange(
                    knob_id=change.knob_id,
                    config_path=change.config_path,
                    runtime_mutable=change.mutable_at_runtime,
                    applied=False,
                    baseline_value=change.baseline_value,
                    candidate_value=change.candidate_value,
                    notes=[
                        "runtime rollout only supports bounded policy and hybrid guardrail knobs"
                    ],
                )
            )
        return blocked

    def _apply_runtime_profile(
        self,
        *,
        config_profile: OptimizationConfigProfile,
        canary_percentage: float,
    ) -> list[ForgePromotionAppliedKnobChange]:
        applied_knobs: list[ForgePromotionAppliedKnobChange] = []
        change_by_id = {change.knob_id: change for change in config_profile.changes}
        routing_policy_change = change_by_id.get("default_routing_policy")
        if routing_policy_change is not None:
            candidate_routing_policy = config_profile.routing_policy
            if candidate_routing_policy is None:
                msg = "default_routing_policy changes require candidate routing_policy"
                raise ValueError(msg)
            candidate_policy = _PromotedConfigProfilePolicy(
                config_profile_id=config_profile.config_profile_id,
                routing_policy=candidate_routing_policy,
            )
            self._policy_rollout.register_candidate_policy(candidate_policy)
            self._registered_candidate_policy_id = candidate_policy.policy_reference.policy_id
            exported_state = self._policy_rollout.export_state()
            snapshot = exported_state.model_copy(
                update={
                    "candidate_policy_id": self._registered_candidate_policy_id,
                    "mode": PolicyRolloutMode.CANARY,
                    "canary_percentage": canary_percentage,
                    "notes": [
                        *exported_state.notes,
                        "Forge Stage A config profile applied as a reversible canary",
                    ],
                },
                deep=True,
            )
            self._policy_rollout.import_state(snapshot)
            applied_knobs.extend(
                [
                    ForgePromotionAppliedKnobChange(
                        knob_id="default_routing_policy",
                        config_path=routing_policy_change.config_path,
                        runtime_mutable=True,
                        applied=True,
                        baseline_value=routing_policy_change.baseline_value,
                        candidate_value=(
                            None
                            if config_profile.routing_policy is None
                            else config_profile.routing_policy.value
                        ),
                        notes=[
                            (
                                "candidate routing policy is active only through "
                                "the policy rollout canary"
                            )
                        ],
                    ),
                    ForgePromotionAppliedKnobChange(
                        knob_id="policy_rollout_mode",
                        config_path="phase4.policy_rollout.mode",
                        runtime_mutable=True,
                        applied=True,
                        baseline_value=exported_state.mode.value,
                        candidate_value=PolicyRolloutMode.CANARY.value,
                        notes=["promotion forces a bounded canary rather than direct cutover"],
                    ),
                    ForgePromotionAppliedKnobChange(
                        knob_id="policy_rollout_canary_percentage",
                        config_path="phase4.policy_rollout.canary_percentage",
                        runtime_mutable=True,
                        applied=True,
                        baseline_value=exported_state.canary_percentage,
                        candidate_value=canary_percentage,
                        notes=["canary percentage is capped by the exported optimization profile"],
                    ),
                ]
            )

        hybrid_updates: dict[str, bool | int | float | str | list[str] | None] = {}
        for change in config_profile.changes:
            if change.knob_id not in _HYBRID_RUNTIME_KNOB_IDS:
                continue
            hybrid_updates[change.knob_id] = change.candidate_value
            applied_knobs.append(
                ForgePromotionAppliedKnobChange(
                    knob_id=change.knob_id,
                    config_path=change.config_path,
                    runtime_mutable=True,
                    applied=True,
                    baseline_value=change.baseline_value,
                    candidate_value=change.candidate_value,
                    notes=["hybrid guardrail was updated through the spillover controller"],
                )
            )
        if hybrid_updates:
            self._spillover.apply_mutable_settings(
                spillover_enabled=_as_bool(hybrid_updates.get("hybrid_spillover_enabled")),
                max_remote_share_percent=_as_float(
                    hybrid_updates.get("hybrid_max_remote_share_percent")
                ),
                remote_request_budget_per_minute=_as_optional_int(
                    hybrid_updates.get("hybrid_remote_request_budget_per_minute")
                ),
                remote_concurrency_cap=_as_optional_int(
                    hybrid_updates.get("hybrid_remote_concurrency_cap")
                ),
                remote_cooldown_seconds=_as_float(
                    hybrid_updates.get("hybrid_remote_cooldown_seconds")
                ),
            )
        return applied_knobs

    def _restore_runtime_state(self) -> None:
        rollback_state = self._rollback_state
        if rollback_state is not None:
            self._policy_rollout.import_state(rollback_state.policy_rollout_snapshot)
            self._spillover.restore_mutable_settings(rollback_state.hybrid_snapshot)
        if self._registered_candidate_policy_id is not None:
            self._policy_rollout.unregister_candidate_policy(self._registered_candidate_policy_id)
            self._registered_candidate_policy_id = None
        self._rollback_state = None

    def _require_transition(
        self,
        rollout_artifact_id: str,
        *,
        expected: set[ForgePromotionLifecycleState],
        action: str,
    ) -> None:
        self._assert_rollout_artifact_id(rollout_artifact_id)
        lifecycle_state = self._required_lifecycle_state()
        if lifecycle_state not in expected:
            allowed = ", ".join(
                state.value for state in sorted(expected, key=lambda item: item.value)
            )
            msg = f"cannot {action} a rollout in state {lifecycle_state.value}; expected {allowed}"
            raise ValueError(msg)

    def _assert_rollout_artifact_id(self, rollout_artifact_id: str) -> None:
        if self._summary.rollout_artifact_id != rollout_artifact_id:
            msg = "rollout_artifact_id does not match the current Forge rollout"
            raise ValueError(msg)

    def _required_lifecycle_state(self) -> ForgePromotionLifecycleState:
        if self._summary.lifecycle_state is None:
            msg = "no Forge rollout proposal exists"
            raise ValueError(msg)
        return self._summary.lifecycle_state

    def _required_config_profile(self) -> OptimizationConfigProfile:
        if self._summary.config_profile is None:
            msg = "Forge rollout is missing its materialized config profile"
            raise ValueError(msg)
        return self._summary.config_profile

    def _replace_summary(
        self,
        *,
        lifecycle_state: ForgePromotionLifecycleState,
        promotion_disposition: OptimizationPromotionDisposition,
        notes: list[str],
    ) -> ForgePromotionRuntimeSummary:
        rollout_artifact_id = self._summary.rollout_artifact_id
        if rollout_artifact_id is None:
            msg = "no Forge rollout proposal exists"
            raise ValueError(msg)
        self._summary = self._summary.model_copy(
            update={
                "lifecycle_state": lifecycle_state,
                "promotion_disposition": promotion_disposition,
                "lifecycle_events": [
                    *self._summary.lifecycle_events,
                    _lifecycle_event(
                        rollout_artifact_id=rollout_artifact_id,
                        lifecycle_state=lifecycle_state,
                        promotion_disposition=promotion_disposition,
                        notes=notes,
                    ),
                ],
                "notes": notes,
            },
            deep=True,
        )
        return self.inspect_state()

    def _bounded_canary_percentage(
        self,
        *,
        proposed: float,
        requested: float | None = None,
    ) -> float:
        resolved = proposed if requested is None else requested
        return min(self._max_canary_percentage, max(0.0, resolved))


class _PromotionRollbackState:
    def __init__(
        self,
        *,
        policy_rollout_snapshot: PolicyRolloutStateSnapshot,
        hybrid_snapshot: HybridMutableRuntimeSettings,
        candidate_policy_id: str | None,
    ) -> None:
        self.policy_rollout_snapshot = policy_rollout_snapshot
        self.hybrid_snapshot = hybrid_snapshot
        self.candidate_policy_id = candidate_policy_id


def _extract_comparison_summary(
    *,
    summary: ForgePromotionRuntimeSummary,
    comparison_artifact: OptimizationCampaignComparisonArtifact,
) -> ForgePromotionComparisonSummary:
    campaign_id = summary.campaign_id
    candidate_configuration_id = summary.candidate_configuration_id
    baseline_candidate_configuration_id = summary.baseline_candidate_configuration_id
    candidate_config_profile_id = summary.candidate_config_profile_id
    if campaign_id is None or candidate_configuration_id is None:
        msg = "Forge rollout is missing campaign or candidate identity"
        raise ValueError(msg)
    if comparison_artifact.campaign_id != campaign_id:
        msg = "comparison artifact campaign_id does not match the active Forge rollout"
        raise ValueError(msg)
    if (
        baseline_candidate_configuration_id is not None
        and comparison_artifact.baseline_candidate_configuration_id
        != baseline_candidate_configuration_id
    ):
        msg = "comparison artifact baseline candidate does not match the active Forge rollout"
        raise ValueError(msg)

    candidate_comparison = next(
        (
            comparison
            for comparison in comparison_artifact.candidate_comparisons
            if comparison.candidate_configuration_id == candidate_configuration_id
        ),
        None,
    )
    if candidate_comparison is None:
        msg = "comparison artifact does not contain the active candidate configuration"
        raise ValueError(msg)
    if (
        candidate_config_profile_id is not None
        and candidate_comparison.config_profile_id != candidate_config_profile_id
    ):
        msg = "comparison artifact config profile does not match the active Forge rollout"
        raise ValueError(msg)

    recommendation = candidate_comparison.recommendation_summary
    return ForgePromotionComparisonSummary(
        comparison_artifact_id=comparison_artifact.comparison_artifact_id,
        campaign_id=comparison_artifact.campaign_id,
        candidate_configuration_id=candidate_comparison.candidate_configuration_id,
        baseline_candidate_configuration_id=(
            comparison_artifact.baseline_candidate_configuration_id
        ),
        config_profile_id=candidate_comparison.config_profile_id,
        rank=candidate_comparison.rank,
        pareto_optimal=candidate_comparison.pareto_optimal,
        dominated=candidate_comparison.dominated,
        recommendation_disposition=recommendation.disposition,
        recommendation_label=recommendation.recommendation_label,
        evidence_kinds=list(recommendation.evidence_kinds),
        improved_objective_ids=list(recommendation.improved_objective_ids),
        regressed_objective_ids=list(recommendation.regressed_objective_ids),
        satisfied_constraint_ids=list(recommendation.satisfied_constraint_ids),
        violated_constraint_ids=list(recommendation.violated_constraint_ids),
        benefited_workload_families=list(recommendation.benefited_workload_families),
        regressed_workload_families=list(recommendation.regressed_workload_families),
        rationale=list(recommendation.rationale),
        notes=list(candidate_comparison.notes),
    )


def _lifecycle_event(
    *,
    rollout_artifact_id: str,
    lifecycle_state: ForgePromotionLifecycleState,
    promotion_disposition: OptimizationPromotionDisposition,
    notes: list[str],
) -> ForgePromotionLifecycleEvent:
    timestamp = datetime.now(UTC)
    return ForgePromotionLifecycleEvent(
        event_id=(
            f"{rollout_artifact_id}:{lifecycle_state.value}:{int(timestamp.timestamp() * 1000)}"
        ),
        lifecycle_state=lifecycle_state,
        promotion_disposition=promotion_disposition,
        recorded_at=timestamp,
        notes=notes,
    )


def _rollout_artifact_id(
    *,
    candidate_configuration_id: str,
    timestamp: datetime,
) -> str:
    return (
        f"forge-rollout-{candidate_configuration_id}-{int(timestamp.timestamp() * 1000)}"
    )


def _decision_notes(
    *,
    prefix: str,
    request: ForgePromotionDecisionRequest,
) -> list[str]:
    notes = list(request.notes)
    if request.reason is not None:
        notes.append(f"{prefix}: {request.reason}")
    return notes


def _as_bool(value: bool | int | float | str | list[str] | None) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_float(value: bool | int | float | str | list[str] | None) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_optional_int(value: bool | int | float | str | list[str] | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
