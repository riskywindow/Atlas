from datetime import UTC, datetime

from pydantic import ValidationError

from switchyard.schemas.routing import (
    AdmissionDecision,
    AdmissionDecisionState,
    AffinityDisposition,
    CanaryPolicy,
    LimiterMode,
    LimiterState,
    PolicyReference,
    QueueSnapshot,
    RequestClass,
    RequestContext,
    RequestFeatureVector,
    RouteAnnotations,
    RouteCandidateExplanation,
    RouteDecision,
    RouteEligibilityState,
    RouteExecutionObservation,
    RouteExplanation,
    RouteSelectionReasonCode,
    RouteTelemetryMetadata,
    RoutingPolicy,
    SessionAffinityKey,
    ShadowDisposition,
    ShadowPolicy,
    ShadowRouteEvidence,
    StickyRouteRecord,
    TenantIdentity,
    TenantTier,
    TopologySnapshotReference,
    WeightedBackendAllocation,
    WorkloadShape,
)


def test_route_decision_valid_case() -> None:
    context = RequestContext(
        request_id="req_123",
        policy=RoutingPolicy.BALANCED,
        workload_shape=WorkloadShape.INTERACTIVE,
        max_latency_ms=250,
        tenant_id="tenant-a",
        tenant_tier=TenantTier.PRIORITY,
        session_id="sess-123",
        request_class=RequestClass.LATENCY_SENSITIVE,
    )
    decision = RouteDecision(
        backend_name="mock-a",
        serving_target="chat-default",
        policy=context.policy,
        request_id=context.request_id,
        workload_shape=context.workload_shape,
        rationale=["lowest observed latency"],
        considered_backends=["mock-a", "mock-b"],
        rejected_backends={"mock-c": "backend concurrency limit reached"},
        admission_limited_backends={"mock-c": "backend concurrency limit reached"},
        degraded_backends=["mock-b"],
        fallback_backends=["mock-b"],
        admission_decision=AdmissionDecision(
            state=AdmissionDecisionState.ADMITTED,
            limiter_key="tenant-a:latency_sensitive",
        ),
        queue_snapshot=QueueSnapshot(
            queue_name="chat-default",
            current_depth=0,
            max_depth=4,
        ),
        limiter_state=LimiterState(
            limiter_key="tenant-a:latency_sensitive",
            mode=LimiterMode.ENFORCING,
            in_flight_requests=1,
            concurrency_limit=2,
        ),
        session_affinity_key=SessionAffinityKey(
            tenant_id="tenant-a",
            session_id="sess-123",
            serving_target="chat-default",
        ),
        sticky_route=StickyRouteRecord(
            affinity_key=SessionAffinityKey(
                tenant_id="tenant-a",
                session_id="sess-123",
                serving_target="chat-default",
            ),
            backend_name="mock-a",
            expires_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
            bound_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        ),
        canary_policy=CanaryPolicy(
            policy_name="chat-default-rollout",
            serving_target="chat-default",
            enabled=True,
            allocations=[WeightedBackendAllocation(backend_name="mock-a", percentage=5.0)],
        ),
        shadow_policy=ShadowPolicy(
            policy_name="chat-default-shadow",
            enabled=True,
            serving_target="chat-default",
            tenant_id="tenant-a",
            target_backend="mock-b",
            sampling_rate=0.1,
        ),
        shadow_decision=ShadowRouteEvidence(
            policy_name="chat-default-shadow",
            disposition=ShadowDisposition.SHADOWED,
            target_backend="mock-b",
            sampling_rate=0.1,
            decision_reason="launched",
        ),
        annotations=RouteAnnotations(
            overload_state=AdmissionDecisionState.ADMITTED,
            affinity_disposition=AffinityDisposition.REUSED,
            shadow_disposition=ShadowDisposition.SHADOWED,
        ),
        telemetry_metadata=RouteTelemetryMetadata(
            tenant_id="tenant-a",
            tenant_tier=TenantTier.PRIORITY,
            request_class=RequestClass.LATENCY_SENSITIVE,
            session_affinity_enabled=True,
            admission_control_enabled=True,
            canary_enabled=True,
            shadow_enabled=True,
        ),
        request_features=RequestFeatureVector(
            message_count=1,
            user_message_count=1,
            prompt_character_count=42,
            prompt_token_estimate=8,
            max_output_tokens=128,
            expected_total_tokens=136,
            repeated_prefix_candidate=True,
            prefix_character_count=28,
            prefix_fingerprint="0123456789abcdef",
            locality_key="abcdef0123456789abcd",
        ),
        policy_reference=PolicyReference(policy_id="balanced", policy_version="phase6.v1"),
        topology_reference=TopologySnapshotReference(
            topology_snapshot_id="topology-001",
            capture_source="gateway_admin_runtime",
            artifact_run_id="run-001",
        ),
        execution_observation=RouteExecutionObservation(
            executed_backend="mock-a",
            queue_delay_ms=3.0,
            ttft_ms=12.0,
            latency_ms=42.0,
            output_tokens=18,
            status_code=200,
            final_outcome="succeeded",
        ),
        explanation=RouteExplanation(
            serving_target="chat-default",
            request_features=RequestFeatureVector(
                message_count=1,
                user_message_count=1,
                prompt_character_count=42,
                prompt_token_estimate=8,
                max_output_tokens=128,
                expected_total_tokens=136,
                repeated_prefix_candidate=True,
                prefix_character_count=28,
                prefix_fingerprint="0123456789abcdef",
                locality_key="abcdef0123456789abcd",
            ),
            policy_reference=PolicyReference(policy_id="balanced", policy_version="phase6.v1"),
            candidates=[
                RouteCandidateExplanation(
                    backend_name="mock-a",
                    serving_target="chat-default",
                    eligibility_state=RouteEligibilityState.ELIGIBLE,
                    score=42.0,
                    reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
                    rationale=["lowest observed latency"],
                ),
                RouteCandidateExplanation(
                    backend_name="mock-b",
                    serving_target="chat-default",
                    eligibility_state=RouteEligibilityState.ELIGIBLE,
                    score=21.0,
                    rationale=["slower"],
                ),
            ],
            selected_backend="mock-a",
            selection_reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            selected_reason=["lowest observed latency"],
            tie_breaker="score, latency_ms, backend_name",
        ),
    )

    assert decision.policy is RoutingPolicy.BALANCED
    assert decision.serving_target == "chat-default"
    assert decision.model_dump(mode="json")["workload_shape"] == "interactive"
    assert decision.model_dump(mode="json")["admission_limited_backends"] == {
        "mock-c": "backend concurrency limit reached"
    }
    assert decision.model_dump(mode="json")["telemetry_metadata"]["request_class"] == (
        "latency_sensitive"
    )
    assert decision.explanation is not None
    assert decision.explanation.selected_backend == "mock-a"
    assert decision.explanation.request_features is not None
    assert decision.explanation.request_features.repeated_prefix_candidate is True
    assert decision.policy_reference == PolicyReference(
        policy_id="balanced",
        policy_version="phase6.v1",
    )
    assert decision.execution_observation is not None
    assert decision.execution_observation.final_outcome == "succeeded"


def test_shadow_policy_rejects_multiple_target_kinds() -> None:
    try:
        ShadowPolicy(
            policy_name="invalid-shadow",
            enabled=True,
            target_backend="mock-a",
            target_alias="chat-shadow",
        )
    except ValidationError as exc:
        assert "target_backend or target_alias" in str(exc) or "either" in str(exc)
    else:
        raise AssertionError("ShadowPolicy should reject backend and alias targets together")


def test_route_decision_rejects_self_fallback() -> None:
    try:
        RouteDecision(
            backend_name="mock-a",
            serving_target="chat-default",
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


def test_route_candidate_explanation_requires_rejection_reason_for_rejected_candidates() -> None:
    try:
        RouteCandidateExplanation(
            backend_name="mock-a",
            serving_target="chat-default",
            eligibility_state=RouteEligibilityState.REJECTED,
            rationale=["health unavailable"],
        )
    except ValidationError as exc:
        assert "rejection_reason" in str(exc)
    else:
        raise AssertionError("Rejected route candidates should require a rejection_reason")


def test_route_decision_rejects_non_subset_protection_maps() -> None:
    try:
        RouteDecision(
            backend_name="mock-a",
            serving_target="chat-default",
            policy=RoutingPolicy.BALANCED,
            request_id="req-protection",
            workload_shape=WorkloadShape.INTERACTIVE,
            rationale=["selected"],
            considered_backends=["mock-a"],
            rejected_backends={},
            protected_backends={"mock-b": "circuit open"},
        )
    except ValidationError as exc:
        assert "protected_backends" in str(exc)
    else:
        raise AssertionError(
            "RouteDecision should reject protection maps outside rejected_backends"
        )


def test_route_explanation_compact_reason_is_stable() -> None:
    explanation = RouteExplanation(
        serving_target="chat-default",
        candidates=[
            RouteCandidateExplanation(
                backend_name="mock-a",
                serving_target="chat-default",
                eligibility_state=RouteEligibilityState.ELIGIBLE,
                score=42.0,
                rationale=["lowest observed latency"],
            )
        ],
        selected_backend="mock-a",
        selected_reason=["lowest observed latency", "local backend"],
        tie_breaker="score, latency_ms, backend_name",
        final_outcome="succeeded",
    )

    assert explanation.compact_reason() == (
        "target=chat-default | selected=mock-a | "
        "reason=lowest observed latency; local backend | "
        "tie_breaker=score, latency_ms, backend_name | outcome=succeeded"
    )


def test_request_context_populates_tenant_identity() -> None:
    context = RequestContext(
        request_id="req-context",
        tenant_id="tenant-b",
        tenant_tier=TenantTier.STANDARD,
        request_class=RequestClass.BULK,
        session_id="sess-b",
    )

    assert context.tenant == TenantIdentity(
        tenant_id="tenant-b",
        tenant_tier=TenantTier.STANDARD,
        request_class=RequestClass.BULK,
    )
    assert context.session_affinity_key is None


def test_admission_decision_requires_reason_for_rejections() -> None:
    try:
        AdmissionDecision(state=AdmissionDecisionState.REJECTED)
    except ValidationError as exc:
        assert "reason" in str(exc)
    else:
        raise AssertionError("Rejected admission decisions should require a reason")


def test_queue_snapshot_rejects_depth_above_bound() -> None:
    try:
        QueueSnapshot(queue_name="chat", current_depth=3, max_depth=2)
    except ValidationError as exc:
        assert "current_depth" in str(exc)
    else:
        raise AssertionError("QueueSnapshot should reject depth above max_depth")


def test_canary_policy_rejects_percentages_over_100() -> None:
    try:
        CanaryPolicy(
            policy_name="overflow",
            serving_target="chat",
            enabled=True,
            allocations=[
                WeightedBackendAllocation(backend_name="a", percentage=60.0),
                WeightedBackendAllocation(backend_name="b", percentage=50.0),
            ],
        )
    except ValidationError as exc:
        assert "percentages" in str(exc)
    else:
        raise AssertionError("CanaryPolicy should reject allocations over 100 percent")
