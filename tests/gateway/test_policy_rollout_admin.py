from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import PolicyRolloutSettings
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.gateway.app import create_app
from switchyard.router.policies import CandidateAssessment, PolicyEvaluation
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.routing import (
    PolicyReference,
    RequestContext,
    RouteSelectionReasonCode,
    RoutingPolicy,
)


class FixedPolicy:
    compatibility_policy: RoutingPolicy | None

    def __init__(self, policy_id: str) -> None:
        self.policy_reference = PolicyReference(
            policy_id=policy_id,
            policy_version="phase6.test",
        )
        self.compatibility_policy = RoutingPolicy.BALANCED

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        del request, context
        assessments = sorted(
            [
                CandidateAssessment(
                    snapshot=snapshot,
                    score=100.0 if snapshot.name == "alpha" else 1.0,
                    eligible=True,
                    rationale=[f"selected_by={self.policy_reference.policy_id}"],
                    reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
                )
                for snapshot in candidates
            ],
            key=lambda assessment: (-(assessment.score or float("-inf")), assessment.snapshot.name),
        )
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=assessments,
            selected_backend=assessments[0].snapshot.name,
            selected_reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            selected_reason=[f"selected_by={self.policy_reference.policy_id}"],
        )


def build_adapter(*, name: str, latency_ms: float) -> MockBackendAdapter:
    return MockBackendAdapter(
        name=name,
        simulated_latency_ms=latency_ms,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            concurrency_limit=1,
        ),
    )


def test_admin_policy_rollout_endpoints_control_runtime_state() -> None:
    registry = AdapterRegistry()
    registry.register(build_adapter(name="alpha", latency_ms=20.0))
    registry.register(build_adapter(name="beta", latency_ms=5.0))
    rollout = PolicyRolloutService(
        PolicyRolloutSettings(
            candidate_policy_id="adaptive-admin-v1",
            max_recent_decisions=5,
        ),
        candidate_policies=[FixedPolicy("adaptive-admin-v1")],
    )
    app = create_app(
        registry=registry,
        policy_rollout=rollout,
    )
    client = TestClient(app)

    update = client.post(
        "/admin/policy-rollout",
        json={
            "mode": "report_only",
            "kill_switch_enabled": False,
            "learning_frozen": True,
        },
    )
    assert update.status_code == 200
    assert update.json()["mode"] == "report_only"
    assert update.json()["learning_frozen"] is True

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-policy-admin"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "inspect rollout"}],
        },
    )
    assert response.status_code == 200

    runtime = client.get("/admin/runtime")
    assert runtime.status_code == 200
    payload = runtime.json()
    assert payload["policy_rollout"]["mode"] == "report_only"
    assert payload["policy_rollout"]["candidate_policy"]["policy_id"] == "adaptive-admin-v1"
    assert payload["policy_rollout"]["recent_decisions"][0]["request_id"] == "req-policy-admin"

    exported = client.get("/admin/policy-rollout/export")
    assert exported.status_code == 200
    assert exported.json()["recent_decisions"][0]["request_id"] == "req-policy-admin"

    reset = client.post("/admin/policy-rollout/reset")
    assert reset.status_code == 200
    assert reset.json()["recent_decisions"] == []

    imported = client.post(
        "/admin/policy-rollout/import",
        json=exported.json(),
    )
    assert imported.status_code == 200
    assert imported.json()["recent_decisions"][0]["request_id"] == "req-policy-admin"
    assert imported.json()["learning_frozen"] is True
