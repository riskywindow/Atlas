from __future__ import annotations

from switchyard.config import CloudTrafficRolloutSettings
from switchyard.control.cloud_rollout import CloudTrafficRolloutService
from switchyard.schemas.admin import CloudRolloutUpdateRequest
from switchyard.schemas.routing import RequestContext, RoutingPolicy, WorkloadShape


def build_context(*, request_id: str = "req-cloud-rollout") -> RequestContext:
    return RequestContext(
        request_id=request_id,
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        session_id="session-cloud-rollout",
    )


def test_cloud_rollout_blocks_canary_only_backend_by_default() -> None:
    service = CloudTrafficRolloutService(CloudTrafficRolloutSettings())

    decision = service.evaluate_canary_only_candidate(
        context=build_context(),
        serving_target="chat-shared",
        backend_name="remote-worker:gpu-canary",
    )

    assert decision.allowed is False
    assert decision.disposition == "disabled"
    assert service.inspect_state().recent_blocked_count == 1


def test_cloud_rollout_allows_explicit_pin_even_when_disabled() -> None:
    service = CloudTrafficRolloutService(CloudTrafficRolloutSettings())
    context = build_context().model_copy(
        update={"internal_backend_pin": "remote-worker:gpu-canary"}
    )

    decision = service.evaluate_canary_only_candidate(
        context=context,
        serving_target="chat-shared",
        backend_name="remote-worker:gpu-canary",
    )

    assert decision.allowed is True
    assert decision.disposition == "explicit_pin"


def test_cloud_rollout_does_not_let_canary_selection_bypass_disabled_or_kill_switch() -> None:
    service = CloudTrafficRolloutService(CloudTrafficRolloutSettings())
    context = build_context(request_id="req-cloud-rollout-canary-gate")

    disabled = service.evaluate_canary_only_candidate(
        context=context,
        serving_target="chat-shared",
        backend_name="remote-worker:gpu-canary",
        explicitly_selected_backend="remote-worker:gpu-canary",
    )
    assert disabled.allowed is False
    assert disabled.disposition == "disabled"

    service.update(CloudRolloutUpdateRequest(enabled=True, canary_percentage=100.0))
    service.update(CloudRolloutUpdateRequest(kill_switch_enabled=True))
    killed = service.evaluate_canary_only_candidate(
        context=context.model_copy(update={"request_id": "req-cloud-rollout-canary-killed"}),
        serving_target="chat-shared",
        backend_name="remote-worker:gpu-canary",
        explicitly_selected_backend="remote-worker:gpu-canary",
    )
    assert killed.allowed is False
    assert killed.disposition == "kill_switch"


def test_cloud_rollout_runtime_update_and_kill_switch_are_reversible() -> None:
    service = CloudTrafficRolloutService(CloudTrafficRolloutSettings())
    context = build_context(request_id="req-cloud-rollout-2")

    enabled = service.update(
        CloudRolloutUpdateRequest(enabled=True, canary_percentage=100.0)
    )
    assert enabled.enabled is True
    assert enabled.canary_percentage == 100.0

    selected = service.evaluate_canary_only_candidate(
        context=context,
        serving_target="chat-shared",
        backend_name="remote-worker:gpu-canary",
    )
    assert selected.allowed is True
    assert selected.disposition == "selected"

    killed = service.update(CloudRolloutUpdateRequest(kill_switch_enabled=True))
    assert killed.kill_switch_enabled is True

    blocked = service.evaluate_canary_only_candidate(
        context=context.model_copy(update={"request_id": "req-cloud-rollout-3"}),
        serving_target="chat-shared",
        backend_name="remote-worker:gpu-canary",
    )
    assert blocked.allowed is False
    assert blocked.disposition == "kill_switch"


def test_cloud_rollout_tracks_repeated_failures_for_auto_quarantine() -> None:
    service = CloudTrafficRolloutService(
        CloudTrafficRolloutSettings(auto_quarantine_failure_threshold=2)
    )

    assert service.record_backend_failure("remote-worker:gpu-canary") is False
    assert service.record_backend_failure("remote-worker:gpu-canary") is True
    assert service.inspect_state().consecutive_failures_by_backend == {
        "remote-worker:gpu-canary": 2
    }

    service.record_backend_success("remote-worker:gpu-canary")
    assert service.inspect_state().consecutive_failures_by_backend == {}
