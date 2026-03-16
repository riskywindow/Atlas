from pydantic import ValidationError

from switchyard.schemas.routing import (
    RequestContext,
    RouteDecision,
    RoutingPolicy,
    WorkloadShape,
)


def test_route_decision_valid_case() -> None:
    context = RequestContext(
        request_id="req_123",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        max_latency_ms=250,
    )
    decision = RouteDecision(
        backend_name="mock-a",
        policy=context.policy,
        request_id=context.request_id,
        workload_shape=context.workload_shape,
        rationale=["lowest observed latency"],
        considered_backends=["mock-a", "mock-b"],
        fallback_backends=["mock-b"],
    )

    assert decision.policy is RoutingPolicy.BALANCED
    assert decision.model_dump(mode="json")["workload_shape"] == "interactive"


def test_route_decision_rejects_self_fallback() -> None:
    try:
        RouteDecision(
            backend_name="mock-a",
            policy=RoutingPolicy.LOCAL_ONLY,
            request_id="req_456",
            workload_shape=WorkloadShape.BATCH,
            rationale=["must stay local"],
            considered_backends=["mock-a"],
            fallback_backends=["mock-a"],
        )
    except ValidationError as exc:
        assert "fallback_backends" in str(exc)
    else:
        raise AssertionError("RouteDecision should reject self-referential fallbacks")
