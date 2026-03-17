from __future__ import annotations

from collections.abc import Sequence

from switchyard.config import PolicyRolloutSettings
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.router.policies import CandidateAssessment, PolicyEvaluation, PolicyRegistry
from switchyard.schemas.admin import PolicyRolloutStateSnapshot, PolicyRolloutUpdateRequest
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import (
    PolicyReference,
    PolicyRolloutMode,
    RequestContext,
    RouteSelectionReasonCode,
    RoutingPolicy,
    WorkloadShape,
)


class FakePolicy:
    def __init__(
        self,
        *,
        policy_id: str,
        compatibility_policy: RoutingPolicy | None = None,
    ) -> None:
        self.policy_reference = PolicyReference(
            policy_id=policy_id,
            policy_version="phase6.test",
        )
        self.compatibility_policy = compatibility_policy

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        del request, context
        assessments = [
            CandidateAssessment(
                snapshot=candidate,
                score=100.0 if candidate.name == "alpha" else 1.0,
                eligible=True,
                rationale=[f"selected_by={self.policy_reference.policy_id}"],
                reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            )
            for candidate in candidates
        ]
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=assessments,
            selected_backend="alpha",
            selected_reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            selected_reason=[f"selected_by={self.policy_reference.policy_id}"],
        )


def build_context() -> RequestContext:
    return RequestContext(
        request_id="req-rollout",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
    )


def build_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="mock-chat",
        messages=[ChatMessage(role=ChatRole.USER, content="hello")],
    )


def build_snapshot(name: str) -> BackendStatusSnapshot:
    return BackendStatusSnapshot(
        name=name,
        capabilities=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            concurrency_limit=1,
        ),
        health=BackendHealth(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            latency_ms=10.0,
        ),
    )


def test_policy_rollout_mode_transitions_and_kill_switch() -> None:
    candidate = FakePolicy(
        policy_id="adaptive-rollout-v1",
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    service = PolicyRolloutService(
        PolicyRolloutSettings(
            mode=PolicyRolloutMode.SHADOW_ONLY,
            candidate_policy_id="adaptive-rollout-v1",
        ),
        candidate_policies=[candidate],
    )
    registry = PolicyRegistry()

    shadow_resolution = service.resolve(
        registry=registry,
        requested_policy=RoutingPolicy.BALANCED,
        request=build_request(),
        context=build_context(),
    )
    assert shadow_resolution.primary_policy.policy_reference.policy_id == "balanced"
    assert [policy.policy_reference.policy_id for policy in shadow_resolution.shadow_policies] == [
        "adaptive-rollout-v1"
    ]

    service.update(
        PolicyRolloutUpdateRequest(mode=PolicyRolloutMode.ACTIVE_GUARDED)
    )
    active_resolution = service.resolve(
        registry=registry,
        requested_policy=RoutingPolicy.BALANCED,
        request=build_request(),
        context=build_context(),
    )
    assert active_resolution.primary_policy.policy_reference.policy_id == "adaptive-rollout-v1"

    service.update(PolicyRolloutUpdateRequest(kill_switch_enabled=True))
    killed_resolution = service.resolve(
        registry=registry,
        requested_policy=RoutingPolicy.BALANCED,
        request=build_request(),
        context=build_context(),
    )
    assert killed_resolution.primary_policy.policy_reference.policy_id == "balanced"
    assert service.inspect_state().kill_switch_enabled is True


def test_policy_rollout_freeze_learning_blocks_events() -> None:
    service = PolicyRolloutService(PolicyRolloutSettings())

    assert service.record_learning_event("warm history refresh") is True
    assert service.inspect_state().last_learning_event == "warm history refresh"

    service.update(PolicyRolloutUpdateRequest(learning_frozen=True))
    assert service.record_learning_event("should be blocked") is False
    assert service.inspect_state().last_guardrail_trigger == "learning_frozen"


def test_policy_rollout_reset_and_export_import_state() -> None:
    candidate = FakePolicy(
        policy_id="adaptive-rollout-v1",
        compatibility_policy=RoutingPolicy.BALANCED,
    )
    service = PolicyRolloutService(
        PolicyRolloutSettings(
            mode=PolicyRolloutMode.CANARY,
            candidate_policy_id="adaptive-rollout-v1",
            canary_percentage=50.0,
        ),
        candidate_policies=[candidate],
    )
    evaluation = candidate.evaluate(
        request=build_request(),
        context=build_context(),
        candidates=[build_snapshot("alpha"), build_snapshot("beta")],
    )
    service.observe_decision(
        context=build_context(),
        resolution=service.resolve(
            registry=PolicyRegistry(),
            requested_policy=RoutingPolicy.BALANCED,
            request=build_request(),
            context=build_context(),
        ),
        primary_evaluation=evaluation,
        shadow_evaluations=[],
    )

    snapshot = service.export_state()
    assert snapshot.mode is PolicyRolloutMode.CANARY
    assert len(snapshot.recent_decisions) == 1

    reset = service.reset_state()
    assert reset.mode is PolicyRolloutMode.CANARY
    assert reset.recent_decisions == []

    imported = service.import_state(
        PolicyRolloutStateSnapshot(
            mode=PolicyRolloutMode.ACTIVE_GUARDED,
            canary_percentage=0.0,
            kill_switch_enabled=False,
            learning_frozen=True,
            recent_decisions=snapshot.recent_decisions,
            notes=["imported"],
        )
    )
    assert imported.mode is PolicyRolloutMode.ACTIVE_GUARDED
    assert imported.learning_frozen is True
    assert len(imported.recent_decisions) == 1
