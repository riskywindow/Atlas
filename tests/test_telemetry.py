from switchyard.telemetry import configure_telemetry


def test_telemetry_records_request_route_and_health_state() -> None:
    telemetry = configure_telemetry("switchyard-test")

    telemetry.record_request(
        route="/v1/chat/completions",
        method="POST",
        status_code=200,
        latency_ms=12.5,
    )
    telemetry.record_route_decision(policy="balanced", backend_name="mock-a")
    telemetry.record_backend_health_snapshot(
        backend_name="mock-a",
        health_state="healthy",
        latency_ms=8.0,
    )

    assert telemetry.state.request_count == 1
    assert telemetry.state.request_latency_ms == [12.5]
    assert telemetry.state.route_decision_count == 1
    assert telemetry.state.backend_health_snapshots == [
        {
            "backend_name": "mock-a",
            "health_state": "healthy",
            "latency_ms": 8.0,
        }
    ]
