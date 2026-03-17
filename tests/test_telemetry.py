from datetime import UTC, datetime

from switchyard.telemetry import (
    BackendLabels,
    compute_tokens_per_second,
    configure_telemetry,
    estimate_token_count,
)


def test_estimate_token_count_uses_whitespace_tokenization() -> None:
    assert estimate_token_count("hello  local metrics test") == 4
    assert estimate_token_count("") == 0


def test_compute_tokens_per_second_handles_empty_values() -> None:
    assert compute_tokens_per_second(output_tokens=0, total_latency_ms=10.0) is None
    assert compute_tokens_per_second(output_tokens=4, total_latency_ms=0.0) is None
    assert compute_tokens_per_second(output_tokens=4, total_latency_ms=2000.0) == 2.0


def test_telemetry_records_request_route_and_health_state() -> None:
    telemetry = configure_telemetry("switchyard-test")

    telemetry.record_request(
        route="/v1/chat/completions",
        method="POST",
        status_code=200,
        latency_ms=12.5,
    )
    admission_record = telemetry.record_admission_decision(
        request_id="req-1",
        tenant_id="tenant-a",
        request_class="standard",
        state="admitted",
        reason_code=None,
        queue_depth=0,
        queue_wait_ms=None,
        status_code=200,
    )
    route_decision = telemetry.record_route_decision(
        request_id="req-1",
        tenant_id="tenant-a",
        session_id="session-1",
        requested_model="mock-chat",
        serving_target="mock-chat",
        policy="balanced",
        backend_name="mock-a",
        considered_backends=["mock-a", "mock-b"],
        fallback_backends=["mock-b"],
        rejected_backends={"mock-c": "backend health is unavailable"},
        admission_limited_backends={},
        protected_backends={},
        degraded_backends=[],
        route_reason="target=mock-chat | selected=mock-a | reason=latency",
        route_latency_ms=4.5,
    )
    route_attempt = telemetry.record_route_attempt(
        request_id="req-1",
        policy="balanced",
        backend_name="mock-a",
        attempt_number=1,
        selected_by_router=True,
        outcome="succeeded",
    )
    telemetry.record_backend_health_snapshot(
        backend_name="mock-a",
        health_state="healthy",
        latency_ms=8.0,
    )
    execution = telemetry.record_backend_execution(
        route="/v1/chat/completions",
        method="POST",
        status_code=200,
        streaming=True,
        labels=BackendLabels(
            backend_name="mock-a",
            backend_type="mock",
            model="mock-chat",
            model_identifier="mock-chat",
        ),
        total_latency_ms=500.0,
        ttft_ms=125.0,
        output_tokens=10,
    )
    warmup = telemetry.record_backend_warmup(
        labels=BackendLabels(
            backend_name="mock-a",
            backend_type="mock",
            model="mock-chat",
            model_identifier="mock-chat",
        ),
        readiness_state="ready",
        warmup_latency_ms=25.0,
        success=True,
    )
    shadow = telemetry.record_shadow_execution(
        primary_request_id="req-1",
        shadow_request_id="req-1:shadow:policy",
        policy_name="shadow-policy",
        target_kind="backend",
        configured_target="mock-shadow",
        resolved_backend_name="mock-shadow",
        requested_model="mock-chat",
        launched_at=datetime(2026, 1, 1, tzinfo=UTC),
        success=True,
        latency_ms=3.5,
        error=None,
    )

    assert telemetry.state.request_count == 1
    assert telemetry.state.request_latency_ms == [12.5]
    assert telemetry.state.admission_records == [admission_record]
    assert telemetry.state.route_decision_count == 1
    assert telemetry.state.route_decision_records == [route_decision]
    assert route_decision.candidate_backend_count == 2
    assert route_decision.fallback_occurred is True
    assert telemetry.state.route_attempt_records == [route_attempt]
    assert execution.tokens_per_second == 20.0
    assert warmup.readiness_state == "ready"
    assert telemetry.state.backend_health_snapshots == [
        {
            "backend_name": "mock-a",
            "health_state": "healthy",
            "latency_ms": 8.0,
        }
    ]
    assert telemetry.state.backend_execution_records == [execution]
    assert telemetry.state.backend_warmup_records == [warmup]
    assert telemetry.state.shadow_execution_records == [shadow]


def test_telemetry_renders_prometheus_text() -> None:
    telemetry = configure_telemetry("switchyard-test")
    telemetry.record_request(route="/readyz", method="GET", status_code=200, latency_ms=5.0)
    telemetry.record_admission_decision(
        request_id="req-metrics",
        tenant_id="default",
        request_class="standard",
        state="admitted",
        reason_code=None,
        queue_depth=0,
        queue_wait_ms=None,
        status_code=200,
    )
    telemetry.record_route_decision(
        request_id="req-metrics",
        tenant_id="default",
        session_id=None,
        requested_model="mock-chat",
        serving_target="mock-chat",
        policy="balanced",
        backend_name="mock-a",
        considered_backends=["mock-a"],
        fallback_backends=[],
        rejected_backends={},
        admission_limited_backends={},
        protected_backends={},
        degraded_backends=[],
        route_reason='target=mock-chat | selected=mock-a | reason=lowest "latency"',
        route_latency_ms=1.0,
    )
    telemetry.record_route_attempt(
        request_id="req-metrics",
        policy="balanced",
        backend_name="mock-a",
        attempt_number=1,
        selected_by_router=True,
        outcome="succeeded",
    )
    telemetry.record_backend_execution(
        route="/v1/chat/completions",
        method="POST",
        status_code=200,
        streaming=False,
        labels=BackendLabels(
            backend_name="mock-a",
            backend_type="mock",
            model="mock-chat",
            model_identifier="mock-chat",
        ),
        total_latency_ms=50.0,
        ttft_ms=None,
        output_tokens=5,
    )
    telemetry.record_shadow_execution(
        primary_request_id="req-metrics",
        shadow_request_id="req-metrics:shadow:policy",
        policy_name="shadow-policy",
        target_kind="backend",
        configured_target="mock-b",
        resolved_backend_name="mock-b",
        requested_model="mock-chat",
        launched_at=datetime(2026, 1, 1, tzinfo=UTC),
        success=True,
        latency_ms=7.0,
        error=None,
    )

    rendered = telemetry.render_prometheus_text()

    assert "switchyard_requests_total 1" in rendered
    assert "switchyard_admission_decisions_total 1" in rendered
    assert "switchyard_route_attempts_total 1" in rendered
    assert 'switchyard_route_decision_latency_ms{candidate_backend_count="1"' in rendered
    assert (
        'route_reason="target=mock-chat | selected=mock-a | '
        'reason=lowest \\"latency\\""' in rendered
    )
    assert 'switchyard_backend_request_latency_ms{backend_name="mock-a"' in rendered
    assert 'switchyard_backend_output_tokens{backend_name="mock-a"' in rendered
    assert 'switchyard_shadow_latency_ms{configured_target="mock-b"' in rendered
