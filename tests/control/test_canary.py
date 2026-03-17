from switchyard.config import CanaryRoutingSettings
from switchyard.control.canary import CanaryRoutingService
from switchyard.schemas.routing import (
    CanaryPolicy,
    RequestContext,
    RolloutDisposition,
    WeightedBackendAllocation,
)


def build_policy() -> CanaryPolicy:
    return CanaryPolicy(
        policy_name="chat-rollout",
        serving_target="chat-shared",
        enabled=True,
        baseline_backend="stable-backend",
        allocations=[
            WeightedBackendAllocation(backend_name="canary-backend", percentage=20.0)
        ],
    )


def test_canary_selection_is_deterministic_for_same_request_key() -> None:
    service = CanaryRoutingService(
        CanaryRoutingSettings(enabled=True, policies=(build_policy(),))
    )
    context = RequestContext(request_id="req-123", session_id="sess-123")

    first = service.select(context=context, policy=build_policy())
    second = service.select(context=context, policy=build_policy())

    assert first == second


def test_canary_distribution_is_close_to_configured_percentage() -> None:
    service = CanaryRoutingService(
        CanaryRoutingSettings(enabled=True, policies=(build_policy(),))
    )
    canary_count = 0
    total = 1000

    for index in range(total):
        selection = service.select(
            context=RequestContext(request_id=f"req-{index:04d}"),
            policy=build_policy(),
        )
        if selection.disposition is RolloutDisposition.CANARY:
            canary_count += 1

    assert 150 <= canary_count <= 250


def test_canary_selection_uses_session_id_for_stable_multi_turn_routing() -> None:
    service = CanaryRoutingService(
        CanaryRoutingSettings(enabled=True, policies=(build_policy(),))
    )
    first = service.select(
        context=RequestContext(request_id="req-a", session_id="sess-stable"),
        policy=build_policy(),
    )
    second = service.select(
        context=RequestContext(request_id="req-b", session_id="sess-stable"),
        policy=build_policy(),
    )

    assert first == second
