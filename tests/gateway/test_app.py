from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import CaptureFixture

from switchyard.adapters.factory import build_registry_from_settings
from switchyard.adapters.mock import MockBackendAdapter, MockResponseTemplate
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import (
    AdmissionControlSettings,
    AppEnvironment,
    CanaryRoutingSettings,
    CircuitBreakerSettings,
    GenerationDefaults,
    HybridExecutionSettings,
    LocalModelConfig,
    Phase4ControlPlaneSettings,
    Phase7ControlPlaneSettings,
    SessionAffinitySettings,
    Settings,
    ShadowRoutingSettings,
    TenantLimitConfig,
    WarmupSettings,
)
from switchyard.gateway import create_app
from switchyard.runtime import (
    RuntimeGenerationResult,
    RuntimeHealthSnapshot,
    RuntimeStreamChunk,
)
from switchyard.runtime.base import ChatModelRuntime
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FinishReason,
)
from switchyard.schemas.routing import (
    CanaryPolicy,
    RequestClass,
    RequestContext,
    RoutingPolicy,
    ShadowDisposition,
    ShadowPolicy,
    WeightedBackendAllocation,
)
from switchyard.telemetry import Telemetry, configure_telemetry


def parse_sse_events(body: str) -> list[str]:
    return [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def build_client(*, unavailable: bool = False) -> tuple[TestClient, Telemetry]:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-gateway",
            health_state=(
                BackendHealthState.UNAVAILABLE if unavailable else BackendHealthState.HEALTHY
            ),
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
                quality_tier=3,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} said={user_message}"
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    return TestClient(app), telemetry


def test_health_and_readiness_endpoints() -> None:
    client, telemetry = build_client()

    health = client.get("/healthz")
    ready = client.get("/readyz", headers={"x-request-id": "req-ready"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert "x-request-id" in health.headers
    assert ready.status_code == 200
    assert ready.headers["x-request-id"] == "req-ready"
    assert ready.json()["status"] == "ready"
    assert ready.json()["adapters"] == ["mock-gateway"]
    assert telemetry.state.request_count == 2
    assert telemetry.state.backend_health_snapshots[0]["backend_name"] == "mock-gateway"


def test_chat_completions_returns_deterministic_mock_response(capsys: CaptureFixture[str]) -> None:
    client, telemetry = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-123"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert payload["backend_name"] == "mock-gateway"
    assert payload["id"] == "mockcmpl_c4c89bb9a8139e6f"
    assert payload["choices"][0]["message"]["content"] == (
        "backend=mock-gateway request=req-123 said=Hello gateway"
    )
    assert telemetry.state.route_decision_count == 1
    route_record = telemetry.state.route_decision_records[0]
    assert route_record.tenant_id == "default"
    assert route_record.session_id is None
    execution = telemetry.state.backend_execution_records[0]
    assert execution.backend_name == "mock-gateway"
    assert execution.model == "mock-chat"
    assert execution.model_identifier == "mock-chat"
    assert execution.output_tokens == payload["usage"]["completion_tokens"]
    assert execution.ttft_ms is None
    assert execution.tokens_per_second is not None

    log_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"event":' in line
    ]
    route_log = next(line for line in log_lines if line["event"] == "route_decision")
    backend_log = next(
        line for line in log_lines if line["event"] == "backend_execution_completed"
    )
    completion_log = next(
        line for line in log_lines if line["event"] == "chat_completion_succeeded"
    )

    assert route_log["request_id"] == "req-123"
    assert route_log["chosen_backend"] == "mock-gateway"
    assert backend_log["request_id"] == "req-123"
    assert backend_log["chosen_backend"] == "mock-gateway"
    assert backend_log["output_tokens"] == payload["usage"]["completion_tokens"]
    assert completion_log["request_id"] == "req-123"
    assert completion_log["chosen_backend"] == "mock-gateway"


def test_chat_completions_capture_tenant_and_session_context() -> None:
    client, telemetry = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-tenant-session",
            "x-switchyard-tenant-id": "tenant-a",
            "x-switchyard-tenant-tier": "priority",
            "x-switchyard-session-id": "session-42",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello tenant-aware gateway"}],
        },
    )

    assert response.status_code == 200
    route_record = telemetry.state.route_decision_records[0]
    assert route_record.tenant_id == "tenant-a"
    assert route_record.session_id == "session-42"


def test_chat_completions_capture_request_class_in_route_header() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-request-class",
            "x-switchyard-request-class": "latency_sensitive",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello request class"}],
        },
    )

    assert response.status_code == 200
    route_header = json.loads(response.headers["x-switchyard-route-decision"])
    assert route_header["telemetry_metadata"]["request_class"] == "latency_sensitive"


def test_chat_completions_reuse_sticky_backend_for_same_session() -> None:
    registry = AdapterRegistry()
    sticky_backend = MockBackendAdapter(
        name="mock-sticky",
        simulated_latency_ms=5.0,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            quality_tier=3,
        ),
        response_template=MockResponseTemplate(content="sticky={backend_name}"),
    )
    faster_backend = MockBackendAdapter(
        name="mock-faster",
        simulated_latency_ms=25.0,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            quality_tier=4,
        ),
        response_template=MockResponseTemplate(content="sticky={backend_name}"),
    )
    registry.register(sticky_backend)
    registry.register(faster_backend)
    client = TestClient(
        create_app(
            registry=registry,
            settings=Settings(
                env=AppEnvironment.TEST,
                default_routing_policy=RoutingPolicy.LATENCY_FIRST,
                phase4=Phase4ControlPlaneSettings(
                    session_affinity=SessionAffinitySettings(
                        enabled=True,
                        ttl_seconds=120.0,
                        max_sessions=16,
                    )
                ),
            ),
        )
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"x-switchyard-session-id": "session-sticky"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "turn one"}],
        },
    )
    sticky_backend._simulated_latency_ms = 50.0
    faster_backend._simulated_latency_ms = 1.0
    second = client.post(
        "/v1/chat/completions",
        headers={"x-switchyard-session-id": "session-sticky"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "turn two"}],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["backend_name"] == "mock-sticky"
    assert second.json()["backend_name"] == "mock-sticky"
    first_route = json.loads(first.headers["x-switchyard-route-decision"])
    second_route = json.loads(second.headers["x-switchyard-route-decision"])
    assert first_route["annotations"]["affinity_disposition"] == "created"
    assert second_route["annotations"]["affinity_disposition"] == "reused"
    assert second_route["sticky_route"]["backend_name"] == "mock-sticky"


def test_chat_completions_fail_over_from_ineligible_sticky_backend() -> None:
    registry = AdapterRegistry()
    sticky_backend = MockBackendAdapter(
        name="mock-sticky",
        simulated_latency_ms=5.0,
        simulated_active_requests=0,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            concurrency_limit=1,
            quality_tier=3,
        ),
        response_template=MockResponseTemplate(content="sticky={backend_name}"),
    )
    fallback_backend = MockBackendAdapter(
        name="mock-fallback",
        simulated_latency_ms=10.0,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            concurrency_limit=1,
            quality_tier=2,
        ),
        response_template=MockResponseTemplate(content="sticky={backend_name}"),
    )
    registry.register(sticky_backend)
    registry.register(fallback_backend)
    client = TestClient(
        create_app(
            registry=registry,
            settings=Settings(
                env=AppEnvironment.TEST,
                default_routing_policy=RoutingPolicy.LATENCY_FIRST,
                phase4=Phase4ControlPlaneSettings(
                    session_affinity=SessionAffinitySettings(
                        enabled=True,
                        ttl_seconds=120.0,
                        max_sessions=16,
                    )
                ),
            ),
        )
    )

    first = client.post(
        "/v1/chat/completions",
        headers={"x-switchyard-session-id": "session-failover"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "turn one"}],
        },
    )
    sticky_backend._simulated_active_requests = 1
    second = client.post(
        "/v1/chat/completions",
        headers={"x-switchyard-session-id": "session-failover"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "turn two"}],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["backend_name"] == "mock-sticky"
    assert second.json()["backend_name"] == "mock-fallback"
    second_route = json.loads(second.headers["x-switchyard-route-decision"])
    assert second_route["annotations"]["affinity_disposition"] == "created"
    assert "backend concurrency limit reached" in second_route["annotations"]["notes"]
    assert (
        "sticky backend 'mock-sticky' is no longer eligible: backend concurrency limit reached"
        in second_route["annotations"]["notes"]
    )
    assert second_route["sticky_route"]["backend_name"] == "mock-fallback"


def test_chat_completions_do_not_launch_shadow_traffic_by_default() -> None:
    client, telemetry = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-no-shadow"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "no shadow"}],
        },
    )

    assert response.status_code == 200
    route_header = json.loads(response.headers["x-switchyard-route-decision"])
    assert route_header["annotations"]["shadow_disposition"] == "disabled"
    assert telemetry.state.shadow_execution_records == []


def test_chat_completions_launch_shadow_traffic_without_affecting_primary_response() -> None:
    registry = AdapterRegistry()
    primary = MockBackendAdapter(
        name="mock-primary",
        simulated_latency_ms=1.0,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            quality_tier=4,
        ),
        response_template=MockResponseTemplate(content="primary={backend_name}"),
    )
    shadow = MockBackendAdapter(
        name="mock-shadow",
        simulated_latency_ms=1.0,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-shadow-alias"],
            max_context_tokens=8192,
            quality_tier=2,
        ),
        response_template=MockResponseTemplate(content="shadow={backend_name}"),
    )
    registry.register(primary)
    registry.register(shadow)
    telemetry = configure_telemetry("switchyard-shadow-test")
    client = TestClient(
        create_app(
            registry=registry,
            telemetry=telemetry,
            settings=Settings(
                env=AppEnvironment.TEST,
                phase4=Phase4ControlPlaneSettings(
                    shadow_routing=ShadowRoutingSettings(
                        enabled=True,
                        policies=(
                            ShadowPolicy(
                                policy_name="tenant-shadow",
                                enabled=True,
                                serving_target="mock-chat",
                                tenant_id="tenant-shadow",
                                target_backend="mock-shadow",
                                sampling_rate=1.0,
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-shadow-primary",
            "x-switchyard-tenant-id": "tenant-shadow",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "shadow me"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["backend_name"] == "mock-primary"
    route_header = json.loads(response.headers["x-switchyard-route-decision"])
    assert route_header["annotations"]["shadow_disposition"] == ShadowDisposition.SHADOWED.value
    assert route_header["shadow_policy"]["policy_name"] == "tenant-shadow"
    assert len(telemetry.state.shadow_execution_records) == 1
    shadow_record = telemetry.state.shadow_execution_records[0]
    assert shadow_record.primary_request_id == "req-shadow-primary"
    assert shadow_record.resolved_backend_name == "mock-shadow"
    assert shadow_record.success is True


def test_chat_completions_surface_canary_route_metadata() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="stable-backend",
            simulated_latency_ms=1.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                serving_targets=["chat-shared"],
                max_context_tokens=8192,
                quality_tier=5,
            ),
            response_template=MockResponseTemplate(content="stable={backend_name}"),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="canary-backend",
            simulated_latency_ms=25.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                serving_targets=["chat-shared"],
                max_context_tokens=8192,
                quality_tier=1,
            ),
            response_template=MockResponseTemplate(content="canary={backend_name}"),
        )
    )
    client = TestClient(
        create_app(
            registry=registry,
            settings=Settings(
                env=AppEnvironment.TEST,
                default_routing_policy=RoutingPolicy.LATENCY_FIRST,
                phase4=Phase4ControlPlaneSettings(
                    canary_routing=CanaryRoutingSettings(
                        enabled=True,
                        policies=(
                            CanaryPolicy(
                                policy_name="chat-rollout",
                                serving_target="chat-shared",
                                enabled=True,
                                baseline_backend="stable-backend",
                                allocations=[
                                    WeightedBackendAllocation(
                                        backend_name="canary-backend",
                                        percentage=100.0,
                                    )
                                ],
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-canary-header"},
        json={
            "model": "chat-shared",
            "messages": [{"role": "user", "content": "canary header"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["backend_name"] == "canary-backend"
    route_header = json.loads(response.headers["x-switchyard-route-decision"])
    assert route_header["annotations"]["rollout_disposition"] == "canary"
    assert route_header["canary_policy"]["policy_name"] == "chat-rollout"


def test_chat_completions_shadow_failures_do_not_break_primary_response() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-primary",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=4,
            ),
        )
    )
    registry.register(
        FailingMockBackendAdapter(
            name="mock-shadow-failing",
            fail_on_generate=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-shadow-alias"],
                max_context_tokens=8192,
                quality_tier=1,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-shadow-failure-test")
    client = TestClient(
        create_app(
            registry=registry,
            telemetry=telemetry,
            settings=Settings(
                env=AppEnvironment.TEST,
                phase4=Phase4ControlPlaneSettings(
                    shadow_routing=ShadowRoutingSettings(
                        enabled=True,
                        policies=(
                            ShadowPolicy(
                                policy_name="shadow-failure",
                                enabled=True,
                                serving_target="mock-chat",
                                target_backend="mock-shadow-failing",
                                sampling_rate=1.0,
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-shadow-failure"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "shadow failure"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["backend_name"] == "mock-primary"
    assert len(telemetry.state.shadow_execution_records) == 1
    shadow_record = telemetry.state.shadow_execution_records[0]
    assert shadow_record.primary_request_id == "req-shadow-failure"
    assert shadow_record.success is False
    assert shadow_record.error == "simulated generate failure"


def test_chat_completions_prevent_shadow_alias_loops() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-gateway",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=3,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-shadow-loop-test")
    client = TestClient(
        create_app(
            registry=registry,
            telemetry=telemetry,
            settings=Settings(
                env=AppEnvironment.TEST,
                phase4=Phase4ControlPlaneSettings(
                    shadow_routing=ShadowRoutingSettings(
                        enabled=True,
                        policies=(
                            ShadowPolicy(
                                policy_name="loop-shadow",
                                enabled=True,
                                serving_target="mock-chat",
                                target_alias="mock-chat",
                                sampling_rate=1.0,
                            ),
                        ),
                    )
                ),
            ),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-shadow-loop"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "loop prevention"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["backend_name"] == "mock-gateway"
    assert len(telemetry.state.shadow_execution_records) == 1
    shadow_record = telemetry.state.shadow_execution_records[0]
    assert shadow_record.primary_request_id == "req-shadow-loop"
    assert shadow_record.success is False
    assert shadow_record.error == "shadow target resolved to primary backend"


def test_chat_completions_reject_circuit_open_backend_during_routing() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-protected",
            circuit_open=True,
            circuit_reason="circuit breaker open after failures",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=5,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-fallback",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=1,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-circuit"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    assert response.status_code == 200
    route_record = telemetry.state.route_decision_records[0]
    assert route_record.protected_backends == {
        "mock-protected": "circuit breaker open after failures"
    }


def test_chat_completions_open_circuit_after_repeated_backend_failures() -> None:
    registry = AdapterRegistry()
    registry.register(
        FailingMockBackendAdapter(
            name="mock-flaky-primary",
            fail_on_generate=True,
            simulated_latency_ms=1.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=5,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-steady-fallback",
            simulated_latency_ms=5.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=1,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-breaker-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            phase4=Phase4ControlPlaneSettings(
                circuit_breakers=CircuitBreakerSettings(
                    enabled=True,
                    failure_threshold=2,
                    recovery_success_threshold=1,
                    open_cooldown_seconds=30.0,
                ),
                admission_control=AdmissionControlSettings(enabled=False),
            ),
        ),
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-breaker-1"},
        json={"model": "mock-chat", "messages": [{"role": "user", "content": "first"}]},
    )
    second = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-breaker-2"},
        json={"model": "mock-chat", "messages": [{"role": "user", "content": "second"}]},
    )
    third = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-breaker-3"},
        json={"model": "mock-chat", "messages": [{"role": "user", "content": "third"}]},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    third_route = json.loads(third.headers["x-switchyard-route-decision"])
    assert third_route["backend_name"] == "mock-steady-fallback"
    assert third_route["protected_backends"] == {
        "mock-flaky-primary": "invocation_failure"
    }


def test_chat_completions_streams_sse_chunks_for_mock_backend() -> None:
    client, telemetry = build_client()

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-request-id": "req-stream"},
        json={
            "model": "mock-chat",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    ) as response:
        body = response.read().decode()

    events = parse_sse_events(body)
    chunk_payloads = [json.loads(event) for event in events[:-1]]

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-stream"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events[-1] == "[DONE]"
    assert chunk_payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunk_payloads[0]["backend_name"] == "mock-gateway"
    assert chunk_payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert telemetry.state.route_decision_count == 1
    execution = telemetry.state.backend_execution_records[0]
    assert execution.streaming is True
    assert execution.backend_name == "mock-gateway"
    assert execution.output_tokens > 0
    assert execution.ttft_ms is not None
    assert execution.tokens_per_second is not None


def test_chat_completions_returns_typed_validation_error() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-invalid"},
        json={"model": "", "messages": []},
    )

    payload = response.json()

    assert response.status_code == 422
    assert response.headers["x-request-id"] == "req-invalid"
    assert payload["code"] == "invalid_request"
    assert payload["request_id"] == "req-invalid"


def test_chat_completions_returns_backend_unavailable_error() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-miss"},
        json={
            "model": "unsupported-model",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 503
    assert response.headers["x-request-id"] == "req-miss"
    assert payload["code"] == "backend_unavailable"
    assert payload["request_id"] == "req-miss"


def test_chat_completions_respects_default_routing_policy_from_settings(
    capsys: CaptureFixture[str],
) -> None:
    client, _ = build_client()
    app = cast(FastAPI, client.app)
    app.state.services.settings.default_routing_policy = RoutingPolicy.LATENCY_FIRST

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-policy"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    assert response.status_code == 200
    log_lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"event":' in line
    ]
    route_log = next(line for line in log_lines if line["event"] == "route_decision")
    assert route_log["route_policy"] == "latency_first"


def test_chat_completions_rejects_invalid_request_context_headers() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-bad-context",
            "x-switchyard-routing-policy": "not-a-policy",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 422
    assert response.headers["x-request-id"] == "req-bad-context"
    assert payload["code"] == "invalid_request"
    assert "x-switchyard-routing-policy" in payload["message"]


def test_chat_completions_rejects_invalid_tenant_tier_header() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-bad-tier",
            "x-switchyard-tenant-tier": "gold",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 422
    assert payload["code"] == "invalid_request"
    assert "x-switchyard-tenant-tier" in payload["message"]


def test_chat_completions_rejects_invalid_request_class_header() -> None:
    client, _ = build_client()

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-bad-request-class",
            "x-switchyard-request-class": "urgent",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    payload = response.json()

    assert response.status_code == 422
    assert payload["code"] == "invalid_request"
    assert "x-switchyard-request-class" in payload["message"]


@pytest.mark.asyncio
async def test_chat_completions_return_429_when_admission_control_rejects() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-admission",
            simulated_latency_ms=25.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=False,
                quality_tier=3,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-admission-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            phase4=Phase4ControlPlaneSettings(
                admission_control=AdmissionControlSettings(
                    enabled=True,
                    global_concurrency_cap=1,
                    global_queue_size=0,
                    default_concurrency_cap=1,
                )
            ),
        ),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"x-request-id": "req-admit-1", "x-switchyard-tenant-id": "tenant-a"},
                json={
                    "model": "mock-chat",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
        )
        await asyncio.sleep(0)
        second_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={"x-request-id": "req-admit-2", "x-switchyard-tenant-id": "tenant-b"},
                json={
                    "model": "mock-chat",
                    "messages": [{"role": "user", "content": "second"}],
                },
            )
        )
        first_response, second_response = await asyncio.gather(first_task, second_task)

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.json()["code"] == "rate_limited"
    admission_header = json.loads(second_response.headers["x-switchyard-admission-decision"])
    assert admission_header["state"] == "rejected"
    assert telemetry.state.admission_records[-1].status_code == 429


@pytest.mark.asyncio
async def test_chat_completions_can_spill_to_remote_when_local_admission_is_full() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-local",
            simulated_latency_ms=25.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=False,
                quality_tier=4,
            ),
            response_template=MockResponseTemplate(content="local:{backend_name}"),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-remote",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.REMOTE,
                model_ids=["mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=False,
                quality_tier=4,
            ),
            response_template=MockResponseTemplate(content="remote:{backend_name}"),
            status_metadata={"execution_mode": "remote_worker"},
        )
    )
    telemetry = configure_telemetry("switchyard-admission-spillover-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            default_routing_policy=RoutingPolicy.LOCAL_PREFERRED,
            phase4=Phase4ControlPlaneSettings(
                admission_control=AdmissionControlSettings(
                    enabled=True,
                    global_concurrency_cap=1,
                    global_queue_size=0,
                    default_concurrency_cap=1,
                )
            ),
            phase7=Phase7ControlPlaneSettings(
                hybrid_execution=HybridExecutionSettings(
                    enabled=True,
                    spillover_enabled=True,
                )
            ),
        ),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first_task = asyncio.create_task(
            client.post(
                "/v1/chat/completions",
                headers={
                    "x-request-id": "req-spill-1",
                    "x-switchyard-routing-policy": "local_preferred",
                },
                json={
                    "model": "mock-chat",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
        )
        await asyncio.sleep(0)
        second_response = await client.post(
            "/v1/chat/completions",
            headers={
                "x-request-id": "req-spill-2",
                "x-switchyard-routing-policy": "burst_to_remote",
            },
            json={
                "model": "mock-chat",
                "messages": [{"role": "user", "content": "second"}],
            },
        )
        first_response = await first_task

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    response_contents = {
        first_response.json()["choices"][0]["message"]["content"],
        second_response.json()["choices"][0]["message"]["content"],
    }
    assert response_contents == {"local:mock-local", "remote:mock-remote"}
    spillover_response = next(
        response
        for response in (first_response, second_response)
        if response.json()["choices"][0]["message"]["content"] == "remote:mock-remote"
    )
    admission_header = json.loads(spillover_response.headers["x-switchyard-admission-decision"])
    assert admission_header["state"] == "bypassed"
    assert admission_header["reason_code"] == "local_admission_spillover_eligible"


def test_chat_completions_accepts_internal_backend_pin_header() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-primary",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-primary", "mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=5,
            ),
            response_template=MockResponseTemplate(content="primary"),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-pinned",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-pinned", "mock-chat"],
                serving_targets=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=1,
            ),
            response_template=MockResponseTemplate(content="pinned"),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-pinned-header",
            "x-switchyard-internal-backend-pin": "mock-pinned",
        },
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Hello gateway"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["backend_name"] == "mock-pinned"


def test_readiness_fails_when_all_backends_are_unavailable() -> None:
    client, _ = build_client(unavailable=True)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


class FakeMLXRuntime:
    backend_type = BackendType.MLX_LM

    def load_model(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def health(self) -> RuntimeHealthSnapshot:
        return RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            detail="fake mlx runtime ready",
        )

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        return RuntimeGenerationResult(
            text=f"mlx handled {request.messages[-1].content}",
            finish_reason=FinishReason.STOP,
            prompt_tokens=3,
            completion_tokens=4,
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[RuntimeStreamChunk]:
        return iter(
            [
                RuntimeStreamChunk(text="mlx "),
                RuntimeStreamChunk(text="streamed"),
                RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP),
            ]
        )


class FakeVLLMRuntime:
    backend_type = BackendType.VLLM_METAL

    def load_model(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def health(self) -> RuntimeHealthSnapshot:
        return RuntimeHealthSnapshot(
            state=BackendHealthState.HEALTHY,
            load_state=BackendLoadState.READY,
            detail="fake vllm runtime ready",
        )

    def generate(self, request: ChatCompletionRequest) -> RuntimeGenerationResult:
        return RuntimeGenerationResult(
            text=f"vllm handled {request.messages[-1].content}",
            finish_reason=FinishReason.STOP,
            prompt_tokens=4,
            completion_tokens=4,
        )

    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Iterator[RuntimeStreamChunk]:
        return iter(
            [
                RuntimeStreamChunk(text="vllm "),
                RuntimeStreamChunk(text="streamed"),
                RuntimeStreamChunk(text="", finish_reason=FinishReason.STOP),
            ]
        )


class FlakyMockBackendAdapter(MockBackendAdapter):
    def __init__(
        self,
        *,
        fail_health_after_route: bool = False,
        name: str = "mock-backend",
        simulated_latency_ms: float = 0.0,
        health_state: BackendHealthState = BackendHealthState.HEALTHY,
        capability_metadata: BackendCapabilities | None = None,
        response_template: MockResponseTemplate | None = None,
        health_detail: str | None = None,
        error_rate: float | None = None,
        stream_chunk_size: int = 3,
    ) -> None:
        super().__init__(
            name=name,
            simulated_latency_ms=simulated_latency_ms,
            health_state=health_state,
            capability_metadata=capability_metadata,
            response_template=response_template,
            health_detail=health_detail,
            error_rate=error_rate,
            stream_chunk_size=stream_chunk_size,
        )
        self._fail_health_after_route = fail_health_after_route
        self._health_calls = 0

    async def health(self) -> BackendHealth:
        self._health_calls += 1
        if self._fail_health_after_route and self._health_calls >= 2:
            return BackendHealth(
                state=BackendHealthState.UNAVAILABLE,
                latency_ms=1.0,
                load_state=BackendLoadState.FAILED,
                detail="simulated post-route failure",
            )
        return await super().health()


class FailingMockBackendAdapter(MockBackendAdapter):
    def __init__(
        self,
        *,
        fail_on_generate: bool = False,
        fail_before_first_stream_chunk: bool = False,
        fail_after_first_stream_chunk: bool = False,
        name: str = "mock-backend",
        simulated_latency_ms: float = 0.0,
        health_state: BackendHealthState = BackendHealthState.HEALTHY,
        capability_metadata: BackendCapabilities | None = None,
        response_template: MockResponseTemplate | None = None,
        health_detail: str | None = None,
        error_rate: float | None = None,
        stream_chunk_size: int = 3,
    ) -> None:
        super().__init__(
            name=name,
            simulated_latency_ms=simulated_latency_ms,
            health_state=health_state,
            capability_metadata=capability_metadata,
            response_template=response_template,
            health_detail=health_detail,
            error_rate=error_rate,
            stream_chunk_size=stream_chunk_size,
        )
        self._fail_on_generate = fail_on_generate
        self._fail_before_first_stream_chunk = fail_before_first_stream_chunk
        self._fail_after_first_stream_chunk = fail_after_first_stream_chunk

    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse:
        if self._fail_on_generate:
            raise RuntimeError("simulated generate failure")
        return await super().generate(request, context)

    async def stream_generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if self._fail_before_first_stream_chunk:
            raise RuntimeError("simulated stream startup failure")

        first_chunk = True
        async for chunk in super().stream_generate(request, context):
            yield chunk
            if self._fail_after_first_stream_chunk and first_chunk:
                first_chunk = False
                raise RuntimeError("simulated mid-stream failure")
            first_chunk = False


def test_create_app_registers_configured_mlx_backend_and_serves_request() -> None:
    settings = Settings(
        env=AppEnvironment.TEST,
        log_level="INFO",
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
        ),
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        settings=settings,
        telemetry=telemetry,
        registry_builder=lambda resolved_settings: build_registry_from_settings(
            resolved_settings,
            mlx_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeMLXRuntime()),
        ),
    )
    client = TestClient(app)

    ready = client.get("/readyz")
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-mlx-gateway"},
        json={
            "model": "mlx-chat",
            "messages": [{"role": "user", "content": "Hello real backend"}],
        },
    )

    assert ready.status_code == 200
    assert ready.json()["adapters"] == ["mlx-lm:mlx-chat"]
    assert response.status_code == 200
    assert response.json()["backend_name"] == "mlx-lm:mlx-chat"
    assert response.json()["choices"][0]["message"]["content"] == (
        "mlx handled Hello real backend"
    )
    assert telemetry.state.route_decision_count == 1


def test_create_app_records_eager_warmup_metrics_for_real_backend() -> None:
    settings = Settings(
        env=AppEnvironment.TEST,
        log_level="INFO",
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True, eager=True),
            ),
        ),
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        settings=settings,
        telemetry=telemetry,
        registry_builder=lambda resolved_settings: build_registry_from_settings(
            resolved_settings,
            mlx_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeMLXRuntime()),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    warmup = telemetry.state.backend_warmup_records[0]
    assert warmup.backend_name == "mlx-lm:mlx-chat"
    assert warmup.model_identifier == "mlx-community/test-model"
    assert warmup.readiness_state == "ready"
    assert warmup.success is True


def test_create_app_streams_configured_mlx_backend() -> None:
    settings = Settings(
        env=AppEnvironment.TEST,
        log_level="INFO",
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
        ),
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        settings=settings,
        telemetry=telemetry,
        registry_builder=lambda resolved_settings: build_registry_from_settings(
            resolved_settings,
            mlx_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeMLXRuntime()),
        ),
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-request-id": "req-mlx-stream"},
        json={
            "model": "mlx-chat",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello real backend"}],
        },
    ) as response:
        body = response.read().decode()

    events = parse_sse_events(body)
    chunk_payloads = [json.loads(event) for event in events[:-1]]

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-mlx-stream"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events[-1] == "[DONE]"
    assert chunk_payloads[0]["backend_name"] == "mlx-lm:mlx-chat"
    assert chunk_payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert [payload["choices"][0]["delta"]["content"] for payload in chunk_payloads[1:]] == [
        "mlx ",
        "streamed",
        "",
    ]
    assert chunk_payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert telemetry.state.route_decision_count == 1
    execution = telemetry.state.backend_execution_records[0]
    assert execution.backend_name == "mlx-lm:mlx-chat"
    assert execution.model_identifier == "mlx-community/test-model"
    assert execution.ttft_ms is not None


def test_create_app_streams_multi_backend_alias_via_router_to_mlx() -> None:
    settings = Settings(
        env=AppEnvironment.TEST,
        log_level="INFO",
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                serving_target="chat-shared",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                configured_priority=10,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
            LocalModelConfig(
                alias="metal-chat",
                serving_target="chat-shared",
                model_identifier="NousResearch/Meta-Llama-3",
                backend_type=BackendType.VLLM_METAL,
                configured_priority=50,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
        ),
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        settings=settings,
        telemetry=telemetry,
        registry_builder=lambda resolved_settings: build_registry_from_settings(
            resolved_settings,
            mlx_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeMLXRuntime()),
            vllm_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeVLLMRuntime()),
        ),
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-shared-mlx-stream",
            "x-switchyard-routing-policy": "balanced",
        },
        json={
            "model": "chat-shared",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello shared backend"}],
        },
    ) as response:
        body = response.read().decode()

    events = parse_sse_events(body)
    chunk_payloads = [json.loads(event) for event in events[:-1]]

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-shared-mlx-stream"
    assert events[-1] == "[DONE]"
    assert chunk_payloads[0]["backend_name"] == "mlx-lm:mlx-chat"
    assert chunk_payloads[0]["model"] == "chat-shared"
    assert [payload["choices"][0]["delta"]["content"] for payload in chunk_payloads[1:]] == [
        "mlx ",
        "streamed",
        "",
    ]
    assert telemetry.state.route_decision_records[0].requested_model == "chat-shared"
    assert telemetry.state.route_decision_records[0].chosen_backend == "mlx-lm:mlx-chat"
    assert telemetry.state.backend_execution_records[0].streaming is True
    assert telemetry.state.backend_execution_records[0].backend_name == "mlx-lm:mlx-chat"
    assert telemetry.state.backend_execution_records[0].model == "chat-shared"
    assert (
        telemetry.state.backend_execution_records[0].model_identifier
        == "mlx-community/test-model"
    )


def test_create_app_streams_multi_backend_alias_via_router_to_vllm() -> None:
    settings = Settings(
        env=AppEnvironment.TEST,
        log_level="INFO",
        local_models=(
            LocalModelConfig(
                alias="mlx-chat",
                serving_target="chat-shared",
                model_identifier="mlx-community/test-model",
                backend_type=BackendType.MLX_LM,
                configured_priority=60,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
            LocalModelConfig(
                alias="metal-chat",
                serving_target="chat-shared",
                model_identifier="NousResearch/Meta-Llama-3",
                backend_type=BackendType.VLLM_METAL,
                configured_priority=10,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
        ),
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        settings=settings,
        telemetry=telemetry,
        registry_builder=lambda resolved_settings: build_registry_from_settings(
            resolved_settings,
            mlx_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeMLXRuntime()),
            vllm_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeVLLMRuntime()),
        ),
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-shared-vllm-stream",
            "x-switchyard-routing-policy": "latency_first",
        },
        json={
            "model": "chat-shared",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello shared backend"}],
        },
    ) as response:
        body = response.read().decode()

    events = parse_sse_events(body)
    chunk_payloads = [json.loads(event) for event in events[:-1]]

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-shared-vllm-stream"
    assert events[-1] == "[DONE]"
    assert chunk_payloads[0]["backend_name"] == "vllm-metal:metal-chat"
    assert chunk_payloads[0]["model"] == "chat-shared"
    assert [payload["choices"][0]["delta"]["content"] for payload in chunk_payloads[1:]] == [
        "vllm ",
        "streamed",
        "",
    ]
    assert telemetry.state.route_decision_records[0].requested_model == "chat-shared"
    assert telemetry.state.route_decision_records[0].chosen_backend == "vllm-metal:metal-chat"
    assert telemetry.state.backend_execution_records[0].streaming is True
    assert telemetry.state.backend_execution_records[0].backend_name == "vllm-metal:metal-chat"
    assert telemetry.state.backend_execution_records[0].model == "chat-shared"
    assert (
        telemetry.state.backend_execution_records[0].model_identifier
        == "NousResearch/Meta-Llama-3"
    )


def test_create_app_registers_configured_vllm_backend_and_serves_request() -> None:
    settings = Settings(
        env=AppEnvironment.TEST,
        log_level="INFO",
        local_models=(
            LocalModelConfig(
                alias="metal-chat",
                model_identifier="NousResearch/Meta-Llama-3",
                backend_type=BackendType.VLLM_METAL,
                generation_defaults=GenerationDefaults(),
                warmup=WarmupSettings(enabled=True),
            ),
        ),
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        settings=settings,
        telemetry=telemetry,
        registry_builder=lambda resolved_settings: build_registry_from_settings(
            resolved_settings,
            vllm_runtime_factory=lambda _config: cast(ChatModelRuntime, FakeVLLMRuntime()),
        ),
    )
    client = TestClient(app)

    ready = client.get("/readyz")
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-vllm-gateway"},
        json={
            "model": "metal-chat",
            "messages": [{"role": "user", "content": "Hello vllm backend"}],
        },
    )

    assert ready.status_code == 200
    assert ready.json()["adapters"] == ["vllm-metal:metal-chat"]
    assert response.status_code == 200
    assert response.json()["backend_name"] == "vllm-metal:metal-chat"
    assert response.json()["choices"][0]["message"]["content"] == (
        "vllm handled Hello vllm backend"
    )
    assert telemetry.state.route_decision_count == 1
    execution = telemetry.state.backend_execution_records[0]
    assert execution.backend_name == "vllm-metal:metal-chat"
    assert execution.model_identifier == "NousResearch/Meta-Llama-3"


def test_metrics_endpoint_is_optional_and_local_friendly() -> None:
    registry = AdapterRegistry()
    registry.register(MockBackendAdapter(name="mock-gateway"))
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            metrics_enabled=True,
        ),
    )
    client = TestClient(app)

    client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "hello metrics"}],
        },
    )
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "switchyard_requests_total" in response.text
    assert "switchyard_backend_request_latency_ms" in response.text


def test_chat_completions_falls_back_when_primary_turns_unhealthy() -> None:
    registry = AdapterRegistry()
    registry.register(
        FlakyMockBackendAdapter(
            name="mock-primary",
            simulated_latency_ms=1.0,
            fail_health_after_route=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=4,
            ),
            response_template=MockResponseTemplate(content="primary should not respond"),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-fallback",
            simulated_latency_ms=5.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=3,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} said={user_message}"
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-fallback"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Fallback please"}],
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["backend_name"] == "mock-fallback"
    assert payload["choices"][0]["message"]["content"] == (
        "backend=mock-fallback request=req-fallback said=Fallback please"
    )
    assert telemetry.state.route_decision_records[0].chosen_backend == "mock-primary"
    assert telemetry.state.route_attempt_records[0].backend_name == "mock-primary"
    assert telemetry.state.route_attempt_records[0].outcome == "skipped_unhealthy"
    assert telemetry.state.route_attempt_records[1].backend_name == "mock-fallback"
    assert telemetry.state.route_attempt_records[1].selected_by_router is False
    assert telemetry.state.backend_execution_records[0].backend_name == "mock-fallback"


def test_chat_completions_falls_back_when_primary_generate_fails_before_response() -> None:
    registry = AdapterRegistry()
    registry.register(
        FailingMockBackendAdapter(
            name="mock-primary",
            fail_on_generate=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=5,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-fallback",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=3,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} said={user_message}"
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-generate-fallback"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Fallback please"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["backend_name"] == "mock-fallback"
    assert telemetry.state.route_attempt_records[0].outcome == "failed"
    assert telemetry.state.route_attempt_records[1].backend_name == "mock-fallback"
    assert telemetry.state.route_decision_records[0].fallback_backends == ["mock-fallback"]


def test_chat_completions_returns_503_when_no_safe_fallback_exists() -> None:
    registry = AdapterRegistry()
    registry.register(
        FailingMockBackendAdapter(
            name="mock-primary",
            fail_on_generate=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=5,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-no-fallback"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Fallback please"}],
        },
    )

    assert response.status_code == 503
    assert response.json()["code"] == "backend_unavailable"
    assert len(telemetry.state.route_attempt_records) == 1
    assert telemetry.state.route_attempt_records[0].backend_name == "mock-primary"


def test_streaming_chat_completion_falls_back_before_first_chunk_only() -> None:
    registry = AdapterRegistry()
    registry.register(
        FailingMockBackendAdapter(
            name="mock-primary",
            fail_before_first_stream_chunk=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
                quality_tier=5,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-fallback",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
                quality_tier=3,
            ),
            response_template=MockResponseTemplate(
                content="backend={backend_name} request={request_id} said={user_message}"
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-request-id": "req-stream-fallback"},
        json={
            "model": "mock-chat",
            "stream": True,
            "messages": [{"role": "user", "content": "Fallback please"}],
        },
    ) as response:
        body = response.read().decode()

    events = parse_sse_events(body)
    chunk_payloads = [json.loads(event) for event in events[:-1]]

    assert response.status_code == 200
    assert events[-1] == "[DONE]"
    assert chunk_payloads[0]["backend_name"] == "mock-fallback"
    assert telemetry.state.route_attempt_records[0].backend_name == "mock-primary"
    assert telemetry.state.route_attempt_records[0].outcome == "failed"
    assert telemetry.state.route_attempt_records[1].backend_name == "mock-fallback"


def test_streaming_chat_completion_does_not_failover_after_first_chunk() -> None:
    registry = AdapterRegistry()
    registry.register(
        FailingMockBackendAdapter(
            name="mock-primary",
            fail_after_first_stream_chunk=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
                quality_tier=5,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-fallback",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
                quality_tier=3,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    with pytest.raises(RuntimeError, match="simulated mid-stream failure"):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"x-request-id": "req-mid-stream"},
            json={
                "model": "mock-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "Fallback please"}],
            },
        ) as response:
            response.read()

    assert telemetry.state.route_attempt_records[0].backend_name == "mock-primary"
    assert all(
        record.backend_name != "mock-fallback"
        for record in telemetry.state.route_attempt_records
    )


def test_chat_completions_uses_only_one_fallback_attempt() -> None:
    registry = AdapterRegistry()
    registry.register(
        FailingMockBackendAdapter(
            name="mock-primary",
            fail_on_generate=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=5,
                configured_priority=10,
            ),
        )
    )
    registry.register(
        FailingMockBackendAdapter(
            name="mock-fallback-one",
            fail_on_generate=True,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=4,
                configured_priority=20,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-fallback-two",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mock-chat"],
                max_context_tokens=8192,
                quality_tier=3,
                configured_priority=30,
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-fallback-budget"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "Fallback please"}],
        },
    )

    assert response.status_code == 503
    assert [record.backend_name for record in telemetry.state.route_attempt_records] == [
        "mock-primary",
        "mock-fallback-one",
    ]


def test_admin_runtime_endpoint_reports_phase4_runtime_state() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-admin",
            simulated_latency_ms=5.0,
            simulated_queue_depth=2,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                serving_targets=["chat-shared"],
                max_context_tokens=8192,
                quality_tier=3,
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-admin-canary",
            simulated_latency_ms=10.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                serving_targets=["chat-shared"],
                max_context_tokens=8192,
                quality_tier=2,
            ),
        )
    )
    app = create_app(
        registry=registry,
        settings=Settings(
            env=AppEnvironment.TEST,
            default_routing_policy=RoutingPolicy.BALANCED,
            phase4=Phase4ControlPlaneSettings(
                admission_control=AdmissionControlSettings(
                    enabled=True,
                    global_concurrency_cap=8,
                    global_queue_size=4,
                    per_tenant_limits=(
                        TenantLimitConfig(
                            tenant_id="tenant-a",
                            request_class=RequestClass.BULK,
                            concurrency_cap=1,
                            queue_size=0,
                        ),
                    ),
                ),
                circuit_breakers=CircuitBreakerSettings(
                    enabled=True,
                    failure_threshold=1,
                    open_cooldown_seconds=30.0,
                ),
                session_affinity=SessionAffinitySettings(
                    enabled=True,
                    ttl_seconds=120.0,
                    max_sessions=32,
                ),
                canary_routing=CanaryRoutingSettings(
                    enabled=True,
                    policies=(
                            CanaryPolicy(
                                policy_name="admin-rollout",
                                serving_target="chat-shared",
                                enabled=True,
                                baseline_backend="mock-admin",
                                allocations=[
                                    WeightedBackendAllocation(
                                        backend_name="mock-admin-canary",
                                        percentage=5.0,
                                    )
                                ],
                            ),
                    ),
                ),
                shadow_routing=ShadowRoutingSettings(
                    enabled=True,
                    policies=(
                        ShadowPolicy(
                            policy_name="admin-shadow",
                            enabled=True,
                            serving_target="chat-shared",
                            target_backend="mock-admin",
                            sampling_rate=0.1,
                        ),
                    ),
                ),
            ),
        ),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-request-id": "req-admin-runtime",
            "x-switchyard-session-id": "session-admin",
        },
        json={
            "model": "chat-shared",
            "messages": [{"role": "user", "content": "prime runtime state"}],
        },
    )
    assert response.status_code == 200

    services = app.state.services
    services.circuit_breaker.record_failure("mock-admin", reason="invocation_failure")

    runtime = client.get("/admin/runtime")

    assert runtime.status_code == 200
    payload = runtime.json()
    assert payload["backends"][0]["backend_name"] == "mock-admin"
    assert payload["backends"][0]["logical_targets"][0]["alias"] == "chat-shared"
    assert payload["backends"][0]["logical_targets"][0]["equivalence"] == "exact"
    assert payload["backends"][0]["queue_depth"] == 2
    assert payload["admission"]["enabled"] is True
    assert payload["admission"]["global_concurrency_cap"] == 8
    assert payload["admission"]["tenant_limiters"] == [
        {
            "tenant_id": "tenant-a",
            "request_class": "bulk",
            "in_flight_requests": 0,
            "concurrency_cap": 1,
        }
    ]
    assert payload["circuit_breakers"]["enabled"] is True
    assert payload["circuit_breakers"]["backends"][0]["phase"] == "open"
    assert payload["canary_routing"]["policies"][0]["policy_name"] == "admin-rollout"
    assert payload["shadow_routing"]["policies"][0]["policy_name"] == "admin-shadow"
    assert payload["policy_rollout"]["mode"] == "disabled"
    assert payload["session_affinity"]["enabled"] is True
    assert payload["session_affinity"]["active_bindings"] == 0
    assert payload["session_affinity"]["bindings_by_target"] == {}
    assert payload["hybrid_execution"]["enabled"] is False
    assert payload["hybrid_execution"]["local_capable_backends"] == 2
    assert "local_preferred" in payload["hybrid_execution"]["remote_policy_eligible"]
    assert "remote_disabled" in payload["hybrid_execution"]["remote_policy_ineligible"]
    assert payload["hybrid_operator"]["remote_effectively_enabled"] is False
    assert payload["hybrid_operator"]["recent_route_example_count"] == 1
    assert payload["hybrid_operator"]["recent_placement_distribution"]["local_count"] == 1
    assert (
        payload["hybrid_operator"]["recent_route_examples"][0]["request_id"]
        == "req-admin-runtime"
    )
    assert payload["hybrid_operator"]["recent_route_examples"][0]["execution_path"] == "local"
    assert payload["remote_workers"]["heartbeat_timeout_seconds"] == 30.0
    assert payload["routing_features"]["feature_version"] == "phase6.v2"
    assert "repeated_prefix" in payload["routing_features"]["workload_tags"]
    assert payload["prefix_locality"]["enabled"] is True
    assert payload["prefix_locality"]["prefix_plaintext_retained"] is False
    assert payload["prefix_locality"]["collision_scope"] == (
        "serving_target+locality_key+prefix_fingerprint"
    )
