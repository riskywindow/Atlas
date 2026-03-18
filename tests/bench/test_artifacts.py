from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.bench.artifacts import (
    _build_request_for_scenario,
    build_gateway_scenario,
    build_replay_plan,
    build_run_id,
    build_synthetic_scenario,
    compare_benchmark_runs,
    compare_synthetic_policies,
    compare_trace_execution_targets,
    compare_workload_execution_targets,
    load_benchmark_artifact_model,
    load_captured_traces,
    parse_prometheus_text,
    render_artifact_bundle_markdown,
    render_comparison_report_markdown,
    render_loaded_artifact_markdown,
    render_run_report_markdown,
    render_simulation_comparison_report_markdown,
    render_target_comparison_report_markdown,
    run_gateway_benchmark,
    run_synthetic_benchmark,
    run_trace_replay_benchmark,
    run_workload_manifest_benchmark,
    summarize_records,
    validate_replayable_traces,
    write_artifact,
)
from switchyard.bench.recommendations import build_policy_recommendation_report
from switchyard.bench.simulation import compare_candidate_policies_offline
from switchyard.bench.workloads import build_workload_manifest
from switchyard.config import AppEnvironment, Settings
from switchyard.gateway import create_app
from switchyard.schemas.backend import BackendCapabilities, BackendType, DeviceClass
from switchyard.schemas.benchmark import (
    BenchmarkArtifactSchemaVersion,
    BenchmarkComparisonDeltaSummary,
    BenchmarkComparisonSideSummary,
    BenchmarkDeploymentTarget,
    BenchmarkEnvironmentMetadata,
    BenchmarkRequestRecord,
    BenchmarkRunArtifact,
    BenchmarkScenario,
    BenchmarkSummary,
    BenchmarkTargetComparisonArtifact,
    BenchmarkWarmupConfig,
    CacheObservation,
    CapturedTraceRecord,
    ComparisonSourceKind,
    CounterfactualObjective,
    CounterfactualSimulationComparisonArtifact,
    ExecutionTarget,
    ExecutionTargetType,
    ExplainablePolicySpec,
    HybridBenchmarkSummary,
    HybridComparisonOutcome,
    HybridComparisonSummary,
    HybridConditionProfile,
    HybridConditionSource,
    HybridExecutionContext,
    HybridExecutionPath,
    PolicyRecommendationReportArtifact,
    RemoteBudgetOutcome,
    RemoteTemperature,
    ReplayMode,
    ScenarioDelta,
    SimulationEvidenceKind,
    TraceCaptureMode,
    WorkloadGenerationConfig,
    WorkloadPattern,
    WorkloadScenarioFamily,
)
from switchyard.schemas.chat import UsageStats
from switchyard.schemas.routing import (
    AdmissionDecision,
    AdmissionDecisionState,
    AffinityDisposition,
    CanaryPolicy,
    CircuitBreakerPhase,
    CircuitBreakerState,
    RequestClass,
    RolloutDisposition,
    RouteAnnotations,
    RouteDecision,
    RoutingPolicy,
    ShadowDisposition,
    ShadowPolicy,
    WorkloadShape,
)
from switchyard.schemas.worker import RemoteWorkerAuthMode
from switchyard.telemetry import configure_telemetry


@pytest.mark.asyncio
async def test_run_synthetic_benchmark_builds_artifact() -> None:
    scenario = build_synthetic_scenario(
        request_count=2,
        policy=RoutingPolicy.BALANCED,
        workload_generation=WorkloadGenerationConfig(
            pattern=WorkloadPattern.REPEATED_PREFIX,
            seed=11,
            shared_prefix="shared-context",
        ),
    )
    artifact = await run_synthetic_benchmark(
        scenario=scenario,
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert artifact.run_id == "20260315T120000Z_balanced"
    assert artifact.request_count == 2
    assert artifact.summary.success_count == 2
    assert artifact.backends_involved == ["mock-local-fast"]
    assert artifact.records[0].route_explanation is not None
    assert artifact.records[0].route_candidate_count == 2
    assert artifact.records[0].routing_policy is RoutingPolicy.BALANCED
    assert artifact.summary.total_output_tokens > 0
    assert artifact.summary.chosen_backend_counts == {"mock-local-fast": 2}
    assert artifact.environment.benchmark_mode == "synthetic"
    assert artifact.scenario.workload_generation.pattern is WorkloadPattern.REPEATED_PREFIX


@pytest.mark.asyncio
async def test_run_gateway_benchmark_builds_phase1_artifact() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mlx-lm:mlx-chat",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["mlx-chat"],
                max_context_tokens=8192,
                supports_streaming=True,
                concurrency_limit=1,
                model_aliases={"mlx-chat": "mlx-community/test-model"},
                default_model="mlx-chat",
            ),
        )
    )
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
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        artifact = await run_gateway_benchmark(
            scenario=build_gateway_scenario(model="mlx-chat", request_count=2),
            gateway_base_url="http://testserver",
            client=client,
            timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        )

    assert artifact.run_id == "20260315T120000Z_balanced"
    assert artifact.request_count == 2
    assert artifact.summary.success_count == 2
    assert artifact.summary.avg_ttft_ms is not None
    assert artifact.summary.total_output_tokens > 0
    assert artifact.backends_involved == ["mlx-lm:mlx-chat"]
    assert artifact.backend_types_involved == ["mock"]
    assert artifact.model_aliases_involved == ["mlx-chat"]
    assert artifact.environment.benchmark_mode == "gateway"
    assert artifact.environment.metadata["metrics_enabled"] == "True"


@pytest.mark.asyncio
async def test_run_gateway_benchmark_captures_deployed_topology_metadata() -> None:
    app = FastAPI()

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse("", media_type="text/plain")

    @app.get("/admin/runtime")
    async def admin_runtime() -> dict[str, object]:
        return cast(
            dict[str, object],
            {
                "captured_at": "2026-03-16T12:00:00Z",
                "backends": [
                {
                    "backend_name": "mlx-lm:chat-mlx",
                    "backend_type": "mlx_lm",
                    "deployment_profile": "host_native",
                    "environment": "local",
                    "health_state": "healthy",
                    "load_state": "ready",
                    "latency_ms": 1.2,
                    "active_requests": 0,
                    "queue_depth": 0,
                    "circuit_open": False,
                    "circuit_reason": None,
                    "instances": [
                        {
                            "instance_id": "mlx-worker-01",
                            "source_of_truth": "static_config",
                            "endpoint": "http://host.docker.internal:8101",
                            "transport": "http",
                            "device_class": "apple_gpu",
                            "locality": "local",
                            "registration_state": "static",
                            "health_state": "healthy",
                            "load_state": "ready",
                            "last_seen_at": "2026-03-16T11:59:55Z",
                            "tags": ["local", "compose"],
                        }
                    ],
                }
            ],
            "admission": {
                "enabled": False,
                "global_concurrency_cap": 1,
                "global_queue_size": 0,
                "in_flight_total": 0,
                "queued_requests": 0,
                "oldest_queue_age_ms": None,
                "tenant_limiters": [],
            },
            "circuit_breakers": {"enabled": False, "backends": []},
            "canary_routing": {"enabled": False, "default_percentage": 0.0, "policies": []},
            "shadow_routing": {
                "enabled": False,
                "default_sampling_rate": 0.0,
                "active_tasks": 0,
                "policies": [],
            },
            "policy_rollout": {
                "mode": "disabled",
                "candidate_policy": None,
                "active_policy": None,
                "shadow_policy": None,
                "compatibility_policy": None,
                "canary_percentage": 0.0,
                "kill_switch_enabled": False,
                "learning_frozen": False,
                "exploration_enabled": False,
                "recent_decisions": [],
                "recent_abstentions": 0,
                "recent_guardrail_triggers": [],
                "last_policy_update_at": None,
                "last_learning_event_at": None,
                "last_learning_event": None,
                "last_guardrail_trigger": None,
                "notes": [],
            },
            "session_affinity": {
                "enabled": False,
                "ttl_seconds": 60.0,
                "max_sessions": 1,
                "active_bindings": 0,
                "bindings_by_target": {},
            },
            "hybrid_execution": {
                "enabled": False,
                "prefer_local": True,
                "spillover_enabled": False,
                "require_healthy_local_backends": True,
                "max_remote_share_percent": 0.0,
                "remote_request_budget_per_minute": None,
                "allowed_remote_environments": [],
                "local_capable_backends": 1,
                "remote_capable_backends": 0,
                "healthy_local_backends": 1,
                "healthy_remote_backends": 0,
                "degraded_remote_backends": 0,
                "unavailable_remote_backends": 0,
                "remote_instance_count": 0,
                "notes": [],
            },
            "remote_workers": {
                "secure_registration_required": False,
                "auth_mode": "none",
                "dynamic_registration_enabled": False,
                "heartbeat_timeout_seconds": 30.0,
                "stale_eviction_seconds": 300.0,
                "registration_token_name": None,
                "allow_static_instances": True,
                "static_instance_count": 1,
                "registered_instance_count": 0,
                "discovered_instance_count": 0,
                "stale_instance_count": 0,
                "ready_instance_count": 0,
                "draining_instance_count": 0,
                "unhealthy_instance_count": 0,
                "lost_instance_count": 0,
                "retired_instance_count": 0,
                "notes": [],
            },
            "remote_worker_registry": {
                "secure_registration_required": False,
                "auth_mode": "none",
                "dynamic_registration_enabled": False,
                "heartbeat_timeout_seconds": 30.0,
                "stale_eviction_seconds": 300.0,
                "registration_token_name": None,
                "worker_count": 1,
                "stale_worker_count": 0,
                "ready_worker_count": 1,
                "live_worker_count": 1,
                "draining_worker_count": 0,
                "unhealthy_worker_count": 0,
                "lost_worker_count": 0,
                "retired_worker_count": 0,
                "workers": [
                    {
                        "worker_id": "remote-mlx-01",
                        "worker_name": "mlx-worker",
                        "environment": "local",
                        "serving_targets": ["chat-shared"],
                        "backend_name": "remote-worker:mlx-worker",
                        "backend_type": "mlx_lm",
                        "lifecycle_state": "ready",
                        "registration_state": "registered",
                        "registered_at": "2026-03-16T12:00:00Z",
                        "last_heartbeat_at": "2026-03-16T12:00:00Z",
                        "expires_at": "2026-03-16T12:00:30Z",
                        "deregistered_at": None,
                        "stale": False,
                        "live": True,
                        "ready": True,
                        "active_requests": 0,
                        "queue_depth": 0,
                        "heartbeat_count": 3,
                        "capabilities": {
                            "backend_type": "mlx_lm",
                            "engine_type": "mlx",
                            "device_class": "apple_gpu",
                            "execution_mode": "host_native",
                            "model_ids": ["chat-mlx"],
                            "serving_targets": ["chat-shared"],
                            "max_context_tokens": 8192,
                            "supports_streaming": True,
                            "concurrency_limit": 2,
                            "configured_priority": 100,
                            "configured_weight": 1.0,
                            "quality_tier": 1,
                            "quality_hint": "balanced",
                            "performance_hint": "balanced",
                            "model_aliases": {},
                            "default_model": None,
                            "supports_tools": False,
                            "warmup_required": False,
                            "cache_capabilities": {
                                "supports_prefix_cache": False,
                                "supports_prompt_cache_read": False,
                                "supports_prompt_cache_write": False,
                                "supports_kv_cache_reuse": False,
                            },
                            "placement": {
                                "provider": None,
                                "region": None,
                                "zone": None,
                                "provider_tags": {},
                            },
                            "cost_profile": {
                                "profile": "unknown",
                                "budget_bucket": None,
                                "relative_cost_index": None,
                                "currency": None,
                                "metadata": {},
                            },
                            "readiness_hints": {
                                "cold_start_likely": False,
                                "warm_pool_enabled": False,
                                "estimated_cold_start_ms": None,
                                "estimated_warm_ttl_seconds": None,
                            },
                            "trust": {
                                "auth_state": "unknown",
                                "trust_state": "unknown",
                                "identity_provider": None,
                                "principal": None,
                                "detail": None,
                            },
                            "network_characteristics": {
                                "profile": "unknown",
                                "expected_rtt_ms": None,
                                "expected_jitter_ms": None,
                                "expected_bandwidth_mbps": None,
                                "notes": [],
                            },
                        },
                        "token_verified": False,
                        "instance": {
                            "instance_id": "remote-mlx-01",
                            "endpoint": {
                                "base_url": "http://host.docker.internal:8101",
                                "transport": "http",
                                "health_path": "/healthz",
                                "readiness_path": "/internal/worker/ready",
                                "capabilities_path": "/internal/worker/capabilities",
                                "warmup_path": "/internal/worker/warmup",
                                "chat_completions_path": "/internal/worker/generate",
                                "stream_chat_completions_path": "/internal/worker/generate/stream",
                                "reachability": {
                                    "reachable_from_control_plane": True,
                                    "requires_tunnel": False,
                                    "auth_state": "unknown",
                                    "trust_state": "unknown",
                                    "last_verified_at": None,
                                    "detail": None,
                                },
                            },
                            "source_of_truth": "registered",
                            "backend_type": "mlx_lm",
                            "device_class": "apple_gpu",
                            "model_identifier": "chat-mlx",
                            "locality": "local",
                            "locality_class": "local_network",
                            "execution_mode": "host_native",
                            "placement": {
                                "provider": None,
                                "region": None,
                                "zone": None,
                                "provider_tags": {},
                            },
                            "cost_profile": {
                                "profile": "unknown",
                                "budget_bucket": None,
                                "relative_cost_index": None,
                                "currency": None,
                                "metadata": {},
                            },
                            "readiness_hints": {
                                "cold_start_likely": False,
                                "warm_pool_enabled": False,
                                "estimated_cold_start_ms": None,
                                "estimated_warm_ttl_seconds": None,
                            },
                            "trust": {
                                "auth_state": "unknown",
                                "trust_state": "unknown",
                                "identity_provider": None,
                                "principal": None,
                                "detail": None,
                            },
                            "network_characteristics": {
                                "profile": "lan",
                                "expected_rtt_ms": None,
                                "expected_jitter_ms": None,
                                "expected_bandwidth_mbps": None,
                                "notes": [],
                            },
                            "tags": ["registered", "remote"],
                            "registration": {
                                "state": "registered",
                                "lifecycle_state": "ready",
                                "registered_at": "2026-03-16T12:00:00Z",
                                "last_heartbeat_at": "2026-03-16T12:00:00Z",
                                "expires_at": "2026-03-16T12:00:30Z",
                                "deregistered_at": None,
                                "heartbeat_count": 3,
                                "source": "dynamic_registration",
                                "detail": None,
                            },
                            "health": {
                                "state": "healthy",
                                "checked_at": "2026-03-16T12:00:00Z",
                                "latency_ms": 3.5,
                                "error_rate": None,
                                "detail": None,
                                "load_state": "ready",
                                "warmed_models": [],
                                "last_error": None,
                                "circuit_open": False,
                                "circuit_reason": None,
                            },
                            "last_seen_at": "2026-03-16T12:00:00Z",
                            "image_metadata": None,
                            "metadata": {"worker_name": "mlx-worker", "ready": "true"},
                        },
                        "metadata": {},
                    }
                ],
                "recent_events": [
                    {
                        "event_type": "registered",
                        "recorded_at": "2026-03-16T12:00:00Z",
                        "worker_id": "remote-mlx-01",
                        "lifecycle_state": "ready",
                        "detail": "worker registered",
                        "metadata": {},
                    }
                ],
            },
            "routing_features": {
                "feature_version": "phase6.v2",
                "input_length_buckets": ["tiny", "short", "medium", "long", "very_long"],
                "history_depth_buckets": ["single_turn", "short_history", "deep_history"],
                "workload_tags": [
                    "short_chat",
                    "long_context",
                    "repeated_prefix",
                    "burst_candidate",
                    "session_continuation",
                    "streaming",
                    "latency_sensitive",
                    "bulk",
                    "priority_tenant",
                ],
                "prefix_fingerprint_algorithm": "sha256_truncated_16_hex",
                "prefix_plaintext_retained": False,
            },
                "prefix_locality": {
                    "enabled": True,
                    "ttl_seconds": 300.0,
                    "max_prefixes": 256,
                    "active_prefixes": 1,
                    "hot_prefixes": 1,
                    "tracked_serving_targets": {"mock-chat": 1},
                    "hottest_prefixes": [
                        {
                            "serving_target": "mock-chat",
                            "locality_key": "00112233445566778899",
                            "prefix_fingerprint": "feedfacecafebeef",
                            "recent_request_count": 3,
                            "hotness": "hot",
                            "preferred_backend": "mock-a",
                            "preferred_instance_id": None,
                            "last_seen_at": "2026-03-16T00:00:00Z",
                        }
                    ],
                    "prefix_fingerprint_algorithm": "sha256_truncated_16_hex",
                    "prefix_plaintext_retained": False,
                    "collision_scope": "serving_target+locality_key+prefix_fingerprint",
                },
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions() -> dict[str, object]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_763_310_400,
            "model": "chat-shared",
            "backend_name": "mlx-lm:chat-mlx",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        artifact = await run_gateway_benchmark(
            scenario=build_gateway_scenario(model="chat-shared", request_count=1),
            gateway_base_url="http://testserver",
            client=client,
            deployment_target=BenchmarkDeploymentTarget.COMPOSE,
            timestamp=datetime(2026, 3, 16, 12, 0, tzinfo=UTC),
        )

    assert artifact.environment.deployment_target is BenchmarkDeploymentTarget.COMPOSE
    assert artifact.environment.topology_capture_source == "gateway_admin_runtime"
    assert artifact.environment.worker_instance_inventory[0].instance_id == "mlx-worker-01"
    assert (
        artifact.environment.worker_instance_inventory[0].endpoint.base_url
        == "http://host.docker.internal:8101"
    )
    assert artifact.environment.deployed_topology[0].address == "http://testserver"
    assert artifact.environment.deployed_topology[1].metadata["backend_type"] == "mlx_lm"
    assert artifact.environment.metadata["topology_captured_at"] == "2026-03-16T12:00:00+00:00"
    assert artifact.environment.hybrid_execution is not None
    assert artifact.environment.hybrid_execution.remote_capable_backends == 0
    assert artifact.environment.remote_workers is not None
    assert artifact.environment.remote_workers.auth_mode is RemoteWorkerAuthMode.NONE
    assert artifact.environment.remote_worker_snapshot is not None
    assert artifact.environment.remote_worker_snapshot.workers[0].lifecycle_state.value == "ready"


def test_render_run_report_markdown_includes_hybrid_runtime_sections() -> None:
    artifact = BenchmarkRunArtifact.model_validate(
        {
            "run_id": "phase7-remote-report",
            "timestamp": "2026-03-17T12:00:00Z",
            "scenario": {
                "name": "remote-aware",
                "model": "chat-shared",
                "model_alias": "chat-shared",
                "policy": "burst_to_remote",
                "workload_shape": "interactive",
                "request_count": 1,
            },
            "policy": "burst_to_remote",
            "execution_target": {
                "target_type": "routing_policy",
                "model_alias": "chat-shared",
                "routing_policy": "burst_to_remote",
            },
            "backends_involved": ["mlx-lm:chat-mlx", "remote-worker:vllm-cuda"],
            "backend_types_involved": ["mlx_lm", "vllm_cuda"],
            "model_aliases_involved": ["chat-shared"],
            "request_count": 1,
            "summary": {
                "request_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "avg_latency_ms": 25.0,
                "p50_latency_ms": 25.0,
                "p95_latency_ms": 25.0,
                "avg_ttft_ms": 10.0,
                "p50_ttft_ms": 10.0,
                "p95_ttft_ms": 10.0,
                "total_output_tokens": 8,
                "avg_output_tokens": 8.0,
                "avg_tokens_per_second": 16.0,
                "p95_tokens_per_second": 16.0,
                "fallback_count": 0,
                "chosen_backend_counts": {"remote-worker:vllm-cuda": 1},
            },
            "environment": {
                "benchmark_mode": "workload_manifest",
                "platform": "macOS-15.0",
                "machine": "arm64",
                "python_version": "3.12.2",
                "worker_instance_inventory": [
                    {
                        "instance_id": "local-mlx-01",
                        "endpoint": {
                            "base_url": "http://127.0.0.1:8101",
                            "transport": "http",
                        },
                        "source_of_truth": "static_config",
                        "backend_type": "mlx_lm",
                        "device_class": "apple_gpu",
                        "locality": "local",
                        "locality_class": "local_host",
                        "execution_mode": "host_native",
                        "registration": {"state": "static"},
                    },
                    {
                        "instance_id": "remote-cuda-01",
                        "endpoint": {
                            "base_url": "https://remote-cuda.internal",
                            "transport": "https",
                        },
                        "source_of_truth": "registered",
                        "backend_type": "vllm_cuda",
                        "device_class": "nvidia_gpu",
                        "locality": "remote",
                        "locality_class": "remote_cloud",
                        "execution_mode": "remote_worker",
                        "registration": {"state": "registered"},
                    },
                ],
                "hybrid_execution": {
                    "enabled": True,
                    "prefer_local": True,
                    "spillover_enabled": True,
                    "require_healthy_local_backends": True,
                    "max_remote_share_percent": 25.0,
                    "remote_request_budget_per_minute": 60,
                    "remote_concurrency_cap": 4,
                    "remote_kill_switch_enabled": False,
                    "remote_cooldown_seconds": 30.0,
                    "allow_high_priority_remote_escalation": True,
                    "allowed_remote_environments": ["staging"],
                    "tenant_remote_policy_count": 1,
                    "local_capable_backends": 1,
                    "remote_capable_backends": 1,
                    "healthy_local_backends": 1,
                    "healthy_remote_backends": 1,
                    "degraded_remote_backends": 0,
                    "unavailable_remote_backends": 0,
                    "remote_instance_count": 1,
                    "remote_budget_requests_used": 3,
                    "remote_budget_requests_remaining": 57,
                    "remote_in_flight_requests": 1,
                    "cooldown_active": False,
                    "notes": ["remote spillover within configured budget"],
                },
                "remote_workers": {
                    "secure_registration_required": True,
                    "auth_mode": "static_token",
                    "dynamic_registration_enabled": True,
                    "heartbeat_timeout_seconds": 30.0,
                    "stale_eviction_seconds": 300.0,
                    "allow_static_instances": True,
                    "static_instance_count": 1,
                    "registered_instance_count": 1,
                    "discovered_instance_count": 0,
                    "stale_instance_count": 0,
                    "ready_instance_count": 1,
                    "draining_instance_count": 0,
                    "unhealthy_instance_count": 0,
                    "lost_instance_count": 0,
                    "retired_instance_count": 0,
                    "notes": ["remote registration healthy"],
                },
            },
            "records": [
                {
                    "request_id": "req-1",
                    "backend_name": "remote-worker:vllm-cuda",
                    "backend_type": "vllm_cuda",
                    "model_alias": "chat-shared",
                    "started_at": "2026-03-17T12:00:00Z",
                    "completed_at": "2026-03-17T12:00:00Z",
                    "latency_ms": 25.0,
                    "output_tokens": 8,
                    "tokens_per_second": 16.0,
                    "success": True,
                    "status_code": 200,
                }
            ],
        }
    )

    markdown = render_run_report_markdown(artifact)

    assert "## Hybrid Execution" in markdown
    assert "## Remote Worker Lifecycle" in markdown
    assert "Captured locality mix: `1 local / 1 remote / 0 external`" in markdown
    assert "Remote budget usage: `3` used / `57` remaining" in markdown
    assert "Auth mode: `static_token`" in markdown


def test_summarize_records_captures_hybrid_evidence_honestly() -> None:
    started_at = datetime(2026, 3, 17, 12, 0, tzinfo=UTC)
    records = [
        BenchmarkRequestRecord(
            request_id="req-local",
            backend_name="mock-local",
            started_at=started_at,
            completed_at=started_at,
            latency_ms=8.0,
            success=True,
            status_code=200,
            hybrid_context=HybridExecutionContext(
                observed_execution_path=HybridExecutionPath.LOCAL_ONLY,
                observed_budget_outcome=RemoteBudgetOutcome.DISABLED,
            ),
        ),
        BenchmarkRequestRecord(
            request_id="req-hybrid",
            backend_name="remote-worker:cuda-a",
            started_at=started_at,
            completed_at=started_at,
            latency_ms=20.0,
            success=True,
            status_code=200,
            hybrid_context=HybridExecutionContext(
                observed_execution_path=HybridExecutionPath.HYBRID_SPILLOVER,
                observed_remote_temperature=RemoteTemperature.WARM,
                injected_condition=HybridConditionProfile(
                    source=HybridConditionSource.INJECTED_MOCK,
                    execution_path=HybridExecutionPath.HYBRID_SPILLOVER,
                    remote_temperature=RemoteTemperature.WARM,
                    budget_outcome=RemoteBudgetOutcome.WITHIN_BUDGET,
                    network_penalty_ms=18.0,
                    modeled_cost=0.05,
                ),
            ),
        ),
        BenchmarkRequestRecord(
            request_id="req-unsupported",
            backend_name="mock-local",
            started_at=started_at,
            completed_at=started_at,
            latency_ms=9.0,
            success=True,
            status_code=200,
        ),
    ]

    summary = summarize_records(records)

    assert summary.hybrid_summary is not None
    assert summary.hybrid_summary.local_only_count == 1
    assert summary.hybrid_summary.hybrid_spillover_count == 1
    assert summary.hybrid_summary.injected_condition_count == 1
    assert summary.hybrid_summary.unsupported_count == 1
    assert summary.hybrid_summary.total_modeled_cost == 0.05


def test_compare_benchmark_runs_marks_hybrid_benefit_and_uncertainty() -> None:
    started_at = datetime(2026, 3, 17, 12, 0, tzinfo=UTC)
    left = BenchmarkRunArtifact(
        run_id="left-local",
        timestamp=started_at,
        scenario=BenchmarkScenario(
            name="hybrid-compare",
            model="chat-shared",
            policy=RoutingPolicy.LOCAL_ONLY,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=2,
        ),
        policy=RoutingPolicy.LOCAL_ONLY,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias="chat-shared",
            routing_policy=RoutingPolicy.LOCAL_ONLY,
        ),
        backends_involved=["mock-local"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=2,
        summary=BenchmarkSummary(
            request_count=2,
            success_count=2,
            failure_count=0,
            avg_latency_ms=20.0,
            p50_latency_ms=20.0,
            p95_latency_ms=22.0,
            avg_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=10,
            avg_output_tokens=5.0,
            avg_tokens_per_second=10.0,
            p95_tokens_per_second=10.0,
            fallback_count=0,
            chosen_backend_counts={"mock-local": 2},
            hybrid_summary=HybridBenchmarkSummary(local_only_count=2, observed_runtime_count=2),
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="workload_manifest"),
        records=[
            BenchmarkRequestRecord(
                request_id="req-a-left",
                workload_item_id="item-a",
                backend_name="mock-local",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=30.0,
                success=True,
                status_code=200,
                hybrid_context=HybridExecutionContext(
                    observed_execution_path=HybridExecutionPath.LOCAL_ONLY,
                    observed_budget_outcome=RemoteBudgetOutcome.DISABLED,
                ),
            ),
            BenchmarkRequestRecord(
                request_id="req-b-left",
                workload_item_id="item-b",
                backend_name="mock-local",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=10.0,
                success=True,
                status_code=200,
            ),
        ],
    )
    right = BenchmarkRunArtifact(
        run_id="right-hybrid",
        timestamp=started_at,
        scenario=BenchmarkScenario(
            name="hybrid-compare",
            model="chat-shared",
            policy=RoutingPolicy.BURST_TO_REMOTE,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=2,
        ),
        policy=RoutingPolicy.BURST_TO_REMOTE,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias="chat-shared",
            routing_policy=RoutingPolicy.BURST_TO_REMOTE,
        ),
        backends_involved=["remote-worker:cuda-a", "mock-local"],
        backend_types_involved=["mock", "vllm_cuda"],
        model_aliases_involved=["chat-shared"],
        request_count=2,
        summary=BenchmarkSummary(
            request_count=2,
            success_count=2,
            failure_count=0,
            avg_latency_ms=14.0,
            p50_latency_ms=14.0,
            p95_latency_ms=18.0,
            avg_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=10,
            avg_output_tokens=5.0,
            avg_tokens_per_second=12.0,
            p95_tokens_per_second=12.0,
            fallback_count=0,
            chosen_backend_counts={"mock-local": 1, "remote-worker:cuda-a": 1},
            hybrid_summary=HybridBenchmarkSummary(
                local_only_count=1,
                hybrid_spillover_count=1,
                observed_runtime_count=1,
                injected_condition_count=1,
                total_modeled_cost=0.05,
                avg_modeled_cost=0.05,
            ),
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="workload_manifest"),
        records=[
            BenchmarkRequestRecord(
                request_id="req-a-right",
                workload_item_id="item-a",
                backend_name="remote-worker:cuda-a",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=12.0,
                success=True,
                status_code=200,
                hybrid_context=HybridExecutionContext(
                    observed_execution_path=HybridExecutionPath.HYBRID_SPILLOVER,
                    observed_remote_temperature=RemoteTemperature.WARM,
                    injected_condition=HybridConditionProfile(
                        source=HybridConditionSource.INJECTED_MOCK,
                        execution_path=HybridExecutionPath.HYBRID_SPILLOVER,
                        remote_temperature=RemoteTemperature.WARM,
                        budget_outcome=RemoteBudgetOutcome.WITHIN_BUDGET,
                        network_penalty_ms=18.0,
                        modeled_cost=0.05,
                    ),
                ),
            ),
            BenchmarkRequestRecord(
                request_id="req-b-right",
                workload_item_id="item-b",
                backend_name="mock-local",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=16.0,
                success=True,
                status_code=200,
                hybrid_context=HybridExecutionContext(
                    predictor_condition=HybridConditionProfile(
                        source=HybridConditionSource.PREDICTOR_ESTIMATE,
                        execution_path=HybridExecutionPath.HYBRID_SPILLOVER,
                        budget_outcome=RemoteBudgetOutcome.EXHAUSTED,
                        network_penalty_ms=22.0,
                        modeled_cost=0.07,
                    )
                ),
            ),
        ],
    )

    comparison = compare_benchmark_runs(
        left_artifact=left,
        right_artifact=right,
        source_kind=ComparisonSourceKind.WORKLOAD_MANIFEST,
        source_name="hybrid-compare",
    )

    assert comparison.delta.hybrid_summary is not None
    assert comparison.delta.hybrid_summary.beneficial_count == 1
    assert comparison.delta.hybrid_summary.low_confidence_count == 0
    assert comparison.delta.hybrid_summary.predictor_estimate_count == 1
    assert (
        comparison.delta.notable_scenario_deltas[0].hybrid_outcome
        is HybridComparisonOutcome.BENEFICIAL
    )
    assert any(
        delta.evidence_kind is SimulationEvidenceKind.PREDICTOR_ESTIMATE
        for delta in comparison.delta.notable_scenario_deltas
    )


def test_render_target_comparison_report_marks_unsupported_hybrid_evidence() -> None:
    artifact = BenchmarkTargetComparisonArtifact(
        comparison_id="hybrid-report",
        source_kind=ComparisonSourceKind.WORKLOAD_MANIFEST,
        source_name="hybrid-report-source",
        request_count=1,
        left=BenchmarkComparisonSideSummary(
            run_id="left",
            execution_target=ExecutionTarget(
                target_type=ExecutionTargetType.ROUTING_POLICY,
                model_alias="chat-shared",
                routing_policy=RoutingPolicy.LOCAL_ONLY,
            ),
            request_count=1,
            success_rate=1.0,
            error_rate=0.0,
            fallback_rate=0.0,
            p50_latency_ms=10.0,
            p95_latency_ms=10.0,
            route_distribution={"mock-local": 1},
            backend_distribution={"mock-local": 1},
        ),
        right=BenchmarkComparisonSideSummary(
            run_id="right",
            execution_target=ExecutionTarget(
                target_type=ExecutionTargetType.ROUTING_POLICY,
                model_alias="chat-shared",
                routing_policy=RoutingPolicy.BURST_TO_REMOTE,
            ),
            request_count=1,
            success_rate=1.0,
            error_rate=0.0,
            fallback_rate=0.0,
            p50_latency_ms=12.0,
            p95_latency_ms=12.0,
            route_distribution={"remote-worker:cuda-a": 1},
            backend_distribution={"remote-worker:cuda-a": 1},
        ),
        delta=BenchmarkComparisonDeltaSummary(
            p50_latency_delta_ms=2.0,
            p95_latency_delta_ms=2.0,
            hybrid_summary=HybridComparisonSummary(
                unsupported_count=1,
                low_confidence_count=1,
                notes=["comparison relied on injected/mock remote conditions"],
            ),
            notable_scenario_deltas=[
                ScenarioDelta(
                    key="workload:item-a",
                    left_request_id="left-1",
                    right_request_id="right-1",
                    latency_delta_ms=2.0,
                    evidence_kind=SimulationEvidenceKind.LOW_CONFIDENCE_ESTIMATE,
                    hybrid_outcome=HybridComparisonOutcome.UNSUPPORTED,
                    condition_sources=[HybridConditionSource.INJECTED_MOCK],
                    notes=["comparison relied on injected/mock remote conditions"],
                )
            ],
        ),
    )

    markdown = render_target_comparison_report_markdown(artifact)

    assert "## Hybrid Evaluation" in markdown
    assert "unsupported" in markdown
    assert "comparison relied on injected/mock remote conditions" in markdown


@pytest.mark.asyncio
async def test_run_workload_manifest_benchmark_builds_rich_artifact() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-workload",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                concurrency_limit=1,
                default_model="chat-shared",
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-workload-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            metrics_enabled=True,
        ),
    )
    transport = httpx.ASGITransport(app=app)
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=2,
        seed=13,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        artifact = await run_workload_manifest_benchmark(
            scenario=scenario,
            gateway_base_url="http://testserver",
            execution_target=ExecutionTarget(
                target_type=ExecutionTargetType.ROUTING_POLICY,
                model_alias="chat-shared",
                routing_policy=RoutingPolicy.BALANCED,
            ),
            warmup=BenchmarkWarmupConfig(enabled=True, request_count=1),
            metrics_path="/metrics",
            client=client,
            timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        )

    assert artifact.environment.benchmark_mode == "workload_manifest"
    assert artifact.run_config.warmup.request_count == 1
    assert artifact.request_count == 2
    assert artifact.records[0].workload_item_id == scenario.items[0].item_id
    assert artifact.records[0].scenario_family is WorkloadScenarioFamily.SHORT_CHAT
    assert artifact.records[0].route_decision is not None
    assert artifact.records[0].route_decision.backend_name == "mock-workload"
    assert artifact.records[0].route_candidate_count == 1
    assert artifact.records[0].cache_observation == CacheObservation()
    assert artifact.summary.success_count == 2
    assert artifact.summary.family_summaries[WorkloadScenarioFamily.SHORT_CHAT].request_count == 2


@pytest.mark.asyncio
async def test_workload_manifest_propagates_tenant_request_context() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-workload-context",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-context"],
                max_context_tokens=8192,
                supports_streaming=False,
                concurrency_limit=1,
                default_model="chat-context",
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-workload-context-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(env=AppEnvironment.TEST, log_level="INFO"),
    )
    transport = httpx.ASGITransport(app=app)
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-context",
        request_count=1,
        seed=13,
    )
    scenario.items[0].metadata.update(
        {
            "tenant_id": "tenant-workload",
            "request_class": "bulk",
            "session_id": "session-workload",
        }
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        artifact = await run_workload_manifest_benchmark(
            scenario=scenario,
            gateway_base_url="http://testserver",
            client=client,
            timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        )

    assert artifact.records[0].tenant_id == "tenant-workload"
    assert artifact.records[0].request_class is RequestClass.BULK
    assert artifact.records[0].session_id == "session-workload"


def test_build_replay_plan_preserves_timestamp_order() -> None:
    target = ExecutionTarget(
        target_type=ExecutionTargetType.ROUTING_POLICY,
        model_alias="chat-shared",
        routing_policy=RoutingPolicy.BALANCED,
    )
    traces = [
        CapturedTraceRecord(
            record_id="trace-late",
            request_id="req-late",
            request_timestamp=datetime(2026, 3, 16, 12, 0, 2, tzinfo=UTC),
            execution_target=target,
            tenant_id="tenant-late",
            request_class=RequestClass.BULK,
            session_id="session-late",
            stream=False,
            fallback_used=False,
            status_code=200,
            latency_ms=10.0,
            normalized_request_payload={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "later"}],
                "stream": False,
            },
        ),
        CapturedTraceRecord(
            record_id="trace-early",
            request_id="req-early",
            request_timestamp=datetime(2026, 3, 16, 12, 0, 1, tzinfo=UTC),
            execution_target=target,
            tenant_id="tenant-early",
            request_class=RequestClass.LATENCY_SENSITIVE,
            session_id="session-early",
            stream=False,
            fallback_used=False,
            status_code=200,
            latency_ms=10.0,
            normalized_request_payload={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "earlier"}],
                "stream": False,
            },
        ),
    ]

    plan = build_replay_plan(
        traces=traces,
        execution_target=target,
        replay_mode=ReplayMode.PRESERVE_ORDER_WITHOUT_ORIGINAL_TIMING,
        concurrency=8,
        source_run_id="capture",
    )

    assert [request.source_request_id for request in plan.requests] == ["req-early", "req-late"]
    assert plan.requests[1].original_interarrival_ms == 1000.0
    assert plan.concurrency == 1
    assert plan.requests[0].tenant_id == "tenant-early"
    assert plan.requests[0].request_class is RequestClass.LATENCY_SENSITIVE
    assert plan.requests[0].session_id == "session-early"


def test_load_captured_traces_reads_jsonl(tmp_path: Path) -> None:
    target = ExecutionTarget(
        target_type=ExecutionTargetType.LOGICAL_ALIAS,
        model_alias="chat-shared",
    )
    trace = CapturedTraceRecord(
        record_id="trace-1",
        request_id="req-1",
        execution_target=target,
        fallback_used=False,
        status_code=200,
        latency_ms=10.0,
        normalized_request_payload={
            "model": "chat-shared",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")

    traces = load_captured_traces(trace_path)

    assert [loaded.record_id for loaded in traces] == ["trace-1"]


def test_validate_replayable_traces_rejects_metadata_only_trace() -> None:
    target = ExecutionTarget(
        target_type=ExecutionTargetType.LOGICAL_ALIAS,
        model_alias="chat-shared",
    )
    trace = CapturedTraceRecord(
        record_id="trace-1",
        request_id="req-1",
        execution_target=target,
        capture_mode=TraceCaptureMode.METADATA_ONLY,
        fallback_used=False,
        status_code=200,
        latency_ms=10.0,
        normalized_request_payload={
            "model": "chat-shared",
            "message_count": 1,
            "roles": ["user"],
            "stream": False,
        },
    )

    with pytest.raises(ValueError, match="metadata_only is not replayable"):
        validate_replayable_traces([trace], model_alias="chat-shared")


def test_compare_benchmark_runs_builds_delta_summary() -> None:
    started_at = datetime(2026, 3, 16, 12, 0, tzinfo=UTC)
    left = BenchmarkRunArtifact(
        run_id="run-left",
        timestamp=started_at,
        scenario=BenchmarkScenario(
            name="compare",
            model="chat-shared",
            policy=RoutingPolicy.BALANCED,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=1,
        ),
        policy=RoutingPolicy.BALANCED,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.ROUTING_POLICY,
            model_alias="chat-shared",
            routing_policy=RoutingPolicy.BALANCED,
        ),
        backends_involved=["mock-a"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=10.0,
            p50_latency_ms=10.0,
            p95_latency_ms=10.0,
            avg_ttft_ms=3.0,
            p50_ttft_ms=3.0,
            p95_ttft_ms=3.0,
            total_output_tokens=4,
            avg_output_tokens=4.0,
            avg_tokens_per_second=20.0,
            p95_tokens_per_second=20.0,
            fallback_count=0,
            chosen_backend_counts={"mock-a": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="workload_manifest"),
        records=[
            BenchmarkRequestRecord(
                request_id="left-1",
                workload_item_id="item-1",
                backend_name="mock-a",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=10.0,
                ttft_ms=3.0,
                tokens_per_second=20.0,
                success=True,
                status_code=200,
            )
        ],
    )
    right = BenchmarkRunArtifact(
        run_id="run-right",
        timestamp=started_at,
        scenario=BenchmarkScenario(
            name="compare",
            model="chat-shared",
            policy=RoutingPolicy.QUALITY_FIRST,
            workload_shape=WorkloadShape.INTERACTIVE,
            request_count=1,
        ),
        policy=RoutingPolicy.QUALITY_FIRST,
        execution_target=ExecutionTarget(
            target_type=ExecutionTargetType.PINNED_BACKEND,
            model_alias="chat-shared",
            pinned_backend="mock-b",
        ),
        backends_involved=["mock-b"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=0,
            failure_count=1,
            avg_latency_ms=16.0,
            p50_latency_ms=16.0,
            p95_latency_ms=16.0,
            avg_ttft_ms=None,
            p50_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=0,
            avg_output_tokens=0.0,
            avg_tokens_per_second=None,
            p95_tokens_per_second=None,
            fallback_count=1,
            chosen_backend_counts={"mock-b": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(benchmark_mode="workload_manifest"),
        records=[
            BenchmarkRequestRecord(
                request_id="right-1",
                workload_item_id="item-1",
                backend_name="mock-b",
                started_at=started_at,
                completed_at=started_at,
                latency_ms=16.0,
                success=False,
                status_code=500,
                error="boom",
            )
        ],
    )

    artifact = compare_benchmark_runs(
        left_artifact=left,
        right_artifact=right,
        source_kind=ComparisonSourceKind.WORKLOAD_MANIFEST,
        source_name="compare",
    )

    assert artifact.delta.success_rate_delta == -1.0
    assert artifact.delta.fallback_rate_delta == 1.0
    assert artifact.delta.p50_latency_delta_ms == 6.0
    assert artifact.delta.backend_distribution_delta == {"mock-a": -1, "mock-b": 1}
    assert artifact.delta.notable_scenario_deltas[0].backend_changed is True
    assert "Switchyard Target Comparison" in render_target_comparison_report_markdown(artifact)


@pytest.mark.asyncio
async def test_run_trace_replay_benchmark_builds_artifact() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-replay",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                concurrency_limit=4,
                default_model="chat-shared",
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-replay-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            metrics_enabled=True,
        ),
    )
    transport = httpx.ASGITransport(app=app)
    target = ExecutionTarget(
        target_type=ExecutionTargetType.PINNED_BACKEND,
        model_alias="chat-shared",
        pinned_backend="mock-replay",
    )
    traces = [
        CapturedTraceRecord(
            record_id="trace-1",
            request_id="req-1",
            request_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC),
            execution_target=target,
            stream=False,
            fallback_used=False,
            status_code=200,
            latency_ms=10.0,
            normalized_request_payload={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "replay prompt 1"}],
                "stream": False,
            },
        ),
        CapturedTraceRecord(
            record_id="trace-2",
            request_id="req-2",
            request_timestamp=datetime(2026, 3, 16, 12, 0, 1, tzinfo=UTC),
            execution_target=target,
            stream=False,
            fallback_used=False,
            status_code=200,
            latency_ms=12.0,
            normalized_request_payload={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "replay prompt 2"}],
                "stream": False,
            },
        ),
    ]

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        artifact = await run_trace_replay_benchmark(
            traces=traces,
            gateway_base_url="http://testserver",
            execution_target=target,
            replay_mode=ReplayMode.FIXED_CONCURRENCY,
            concurrency=2,
            warmup=BenchmarkWarmupConfig(enabled=True, request_count=1),
            source_run_id="gateway-traces",
            client=client,
            timestamp=datetime(2026, 3, 16, 12, 30, tzinfo=UTC),
        )

    assert artifact.environment.benchmark_mode == "trace_replay"
    assert artifact.run_config.replay_mode is ReplayMode.FIXED_CONCURRENCY
    assert artifact.run_config.warmup.request_count == 1
    assert artifact.execution_target is not None
    assert artifact.execution_target.pinned_backend == "mock-replay"
    assert artifact.records[0].source_request_id == "req-1"
    assert artifact.records[0].source_trace_record_id == "trace-1"
    assert artifact.records[0].replay_correlation_id == "replay_req-1"
    assert artifact.records[0].cache_observation == CacheObservation()
    assert artifact.summary.success_count == 2
    assert artifact.backends_involved == ["mock-replay"]


@pytest.mark.asyncio
async def test_repeated_prefix_workload_auto_preconditions_warmup() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-prefix",
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                concurrency_limit=1,
                default_model="chat-shared",
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-prefix-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            metrics_enabled=True,
        ),
    )
    transport = httpx.ASGITransport(app=app)
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.REPEATED_PREFIX,
        model_alias="chat-shared",
        request_count=2,
        seed=21,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        artifact = await run_workload_manifest_benchmark(
            scenario=scenario,
            gateway_base_url="http://testserver",
            execution_target=ExecutionTarget(
                target_type=ExecutionTargetType.ROUTING_POLICY,
                model_alias="chat-shared",
                routing_policy=RoutingPolicy.BALANCED,
            ),
            warmup=BenchmarkWarmupConfig(),
            client=client,
            timestamp=datetime(2026, 3, 16, 14, 0, tzinfo=UTC),
        )

    assert artifact.run_config.warmup.enabled is True
    assert artifact.run_config.warmup.request_count == 1
    assert (
        artifact.summary.family_summaries[WorkloadScenarioFamily.REPEATED_PREFIX].request_count
        == 2
    )


def test_summarize_records_groups_by_family() -> None:
    started_at = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 3, 15, 12, 0, 0, 1000, tzinfo=UTC)
    summary = summarize_records(
        [
            BenchmarkRequestRecord(
                request_id="req_1",
                scenario_family=WorkloadScenarioFamily.SHORT_CHAT,
                backend_name="mock-a",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=10.0,
                success=True,
                status_code=200,
            ),
            BenchmarkRequestRecord(
                request_id="req_2",
                scenario_family=WorkloadScenarioFamily.REPEATED_PREFIX,
                backend_name="mock-b",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=20.0,
                success=False,
                status_code=500,
                error="boom",
            ),
        ]
    )

    assert summary.family_summaries[WorkloadScenarioFamily.SHORT_CHAT].request_count == 1
    assert summary.family_summaries[WorkloadScenarioFamily.REPEATED_PREFIX].failure_count == 1


@pytest.mark.asyncio
async def test_compare_workload_execution_targets_builds_two_target_diff() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-a",
            simulated_latency_ms=5.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                default_model="chat-shared",
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-b",
            simulated_latency_ms=8.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                default_model="chat-shared",
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-compare-workload-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            metrics_enabled=True,
        ),
    )
    transport = httpx.ASGITransport(app=app)
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=2,
        seed=17,
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        comparison = await compare_workload_execution_targets(
            scenario=scenario,
            gateway_base_url="http://testserver",
            left_target=ExecutionTarget(
                target_type=ExecutionTargetType.PINNED_BACKEND,
                model_alias="chat-shared",
                pinned_backend="mock-a",
            ),
            right_target=ExecutionTarget(
                target_type=ExecutionTargetType.PINNED_BACKEND,
                model_alias="chat-shared",
                pinned_backend="mock-b",
            ),
            metrics_path="/metrics",
            client=client,
            timestamp=datetime(2026, 3, 16, 13, 0, tzinfo=UTC),
        )

    assert comparison.source_kind is ComparisonSourceKind.WORKLOAD_MANIFEST
    assert comparison.request_count == 2
    assert comparison.left.execution_target.pinned_backend == "mock-a"
    assert comparison.right.execution_target.pinned_backend == "mock-b"
    assert comparison.left.backend_distribution == {"mock-a": 2}
    assert comparison.right.backend_distribution == {"mock-b": 2}
    assert comparison.delta.backend_distribution_delta == {"mock-a": -2, "mock-b": 2}


@pytest.mark.asyncio
async def test_compare_trace_execution_targets_builds_two_target_diff() -> None:
    registry = AdapterRegistry()
    registry.register(
        MockBackendAdapter(
            name="mock-replay-a",
            simulated_latency_ms=5.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                default_model="chat-shared",
            ),
        )
    )
    registry.register(
        MockBackendAdapter(
            name="mock-replay-b",
            simulated_latency_ms=6.0,
            capability_metadata=BackendCapabilities(
                backend_type=BackendType.MOCK,
                device_class=DeviceClass.CPU,
                model_ids=["chat-shared"],
                max_context_tokens=8192,
                supports_streaming=True,
                default_model="chat-shared",
            ),
        )
    )
    telemetry = configure_telemetry("switchyard-compare-trace-test")
    app = create_app(
        registry=registry,
        telemetry=telemetry,
        settings=Settings(
            env=AppEnvironment.TEST,
            log_level="INFO",
            metrics_enabled=True,
        ),
    )
    transport = httpx.ASGITransport(app=app)
    target = ExecutionTarget(
        target_type=ExecutionTargetType.LOGICAL_ALIAS,
        model_alias="chat-shared",
    )
    traces = [
        CapturedTraceRecord(
            record_id="trace-1",
            request_id="req-1",
            request_timestamp=datetime(2026, 3, 16, 12, 0, 0, tzinfo=UTC),
            execution_target=target,
            stream=False,
            fallback_used=False,
            status_code=200,
            latency_ms=10.0,
            normalized_request_payload={
                "model": "chat-shared",
                "messages": [{"role": "user", "content": "trace compare"}],
                "stream": False,
            },
        )
    ]
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        comparison = await compare_trace_execution_targets(
            traces=traces,
            gateway_base_url="http://testserver",
            left_target=ExecutionTarget(
                target_type=ExecutionTargetType.PINNED_BACKEND,
                model_alias="chat-shared",
                pinned_backend="mock-replay-a",
            ),
            right_target=ExecutionTarget(
                target_type=ExecutionTargetType.PINNED_BACKEND,
                model_alias="chat-shared",
                pinned_backend="mock-replay-b",
            ),
            source_run_id="trace-capture",
            client=client,
            timestamp=datetime(2026, 3, 16, 13, 30, tzinfo=UTC),
        )

    assert comparison.source_kind is ComparisonSourceKind.TRACE_SET
    assert comparison.left.backend_distribution == {"mock-replay-a": 1}
    assert comparison.right.backend_distribution == {"mock-replay-b": 1}


@pytest.mark.asyncio
async def test_compare_synthetic_policies_builds_comparison_artifact() -> None:
    comparison = await compare_synthetic_policies(
        request_count=2,
        pinned_backends=["mock-local-fast"],
        workload_generation=WorkloadGenerationConfig(
            pattern=WorkloadPattern.BURSTY,
            seed=3,
            burst_size=2,
        ),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert comparison.run_id == "20260315T120000Z_comparison"
    assert comparison.request_count == 2
    assert len(comparison.results) == len(RoutingPolicy) * 2
    assert comparison.best_policy_by_latency in RoutingPolicy
    assert comparison.best_result_by_latency in {
        result.comparison_label for result in comparison.results
    }
    assert any(result.internal_backend_pin == "mock-local-fast" for result in comparison.results)


@pytest.mark.asyncio
async def test_synthetic_benchmark_applies_repeated_prefix_prompts() -> None:
    scenario = build_synthetic_scenario(
        request_count=1,
        workload_generation=WorkloadGenerationConfig(
            pattern=WorkloadPattern.REPEATED_PREFIX,
            seed=5,
            shared_prefix="phase3-prefix",
        ),
    )
    request = _build_request_for_scenario(scenario=scenario, index=0)
    artifact = await run_synthetic_benchmark(
        scenario=scenario,
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    assert request.messages[0].content.startswith("phase3-prefix\n")
    markdown = render_run_report_markdown(artifact)
    assert "repeated_prefix" in markdown
    assert "phase3-prefix" in markdown
    assert "## Run Metadata" in markdown
    assert "## Environment" in markdown
    assert "## Benchmark Configuration" in markdown
    assert "## Per-Scenario Table" in markdown
    assert "## Route and Backend Distributions" in markdown
    assert "## Fallback and Error Summary" in markdown
    assert "- Error categories: `none`" in markdown


@pytest.mark.asyncio
async def test_compare_synthetic_report_mentions_best_result() -> None:
    comparison = await compare_synthetic_policies(
        request_count=1,
        policies=[RoutingPolicy.BALANCED],
        workload_generation=WorkloadGenerationConfig(
            pattern=WorkloadPattern.BURSTY,
            seed=19,
            burst_size=1,
        ),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )

    markdown = render_comparison_report_markdown(comparison)
    assert comparison.best_result_by_latency in markdown
    assert "## Top Takeaways" in markdown
    assert "## Results" in markdown


def test_render_run_report_markdown_summarizes_phase4_control_plane_signals() -> None:
    started_at = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 3, 15, 12, 0, 0, 5000, tzinfo=UTC)
    scenario = build_synthetic_scenario(request_count=2)
    route_decision = RouteDecision(
        backend_name="mock-canary",
        serving_target="chat-shared",
        policy=RoutingPolicy.BALANCED,
        request_id="req-phase4",
        workload_shape=WorkloadShape.INTERACTIVE,
        rationale=["canary bucket matched"],
        considered_backends=["mock-canary", "mock-stable"],
        admission_decision=AdmissionDecision(
            state=AdmissionDecisionState.QUEUED,
            limiter_key="tenant-rollout",
            queue_wait_ms=12.5,
        ),
        circuit_breaker_state=CircuitBreakerState(
            backend_name="mock-canary",
            phase=CircuitBreakerPhase.HALF_OPEN,
            failure_count=3,
            reason="timeout_failure",
        ),
        canary_policy=CanaryPolicy(
            policy_name="chat-shared-rollout",
            serving_target="chat-shared",
            enabled=True,
            baseline_backend="mock-stable",
        ),
        shadow_policy=ShadowPolicy(
            policy_name="chat-shared-shadow",
            enabled=True,
            serving_target="chat-shared",
            target_backend="mock-shadow",
            sampling_rate=0.25,
        ),
        annotations=RouteAnnotations(
            overload_state=AdmissionDecisionState.QUEUED,
            breaker_phase=CircuitBreakerPhase.HALF_OPEN,
            affinity_disposition=AffinityDisposition.MISSED,
            rollout_disposition=RolloutDisposition.CANARY,
            shadow_disposition=ShadowDisposition.SHADOWED,
            notes=[
                "sticky backend unavailable after cooldown",
                "canary policy 'chat-shared-rollout' matched",
            ],
        ),
    )
    artifact = BenchmarkRunArtifact(
        run_id="phase4-markdown",
        timestamp=started_at,
        scenario=scenario,
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-canary", "mock-stable"],
        backend_types_involved=["mock"],
        model_aliases_involved=["chat-shared"],
        request_count=2,
        summary=BenchmarkSummary(
            request_count=2,
            success_count=2,
            failure_count=0,
            avg_latency_ms=5.0,
            p50_latency_ms=5.0,
            p95_latency_ms=5.0,
            avg_ttft_ms=None,
            p50_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=4,
            avg_output_tokens=2.0,
            avg_tokens_per_second=None,
            p95_tokens_per_second=None,
            fallback_count=0,
            chosen_backend_counts={"mock-canary": 1, "mock-stable": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="gateway",
            canary_percentage=10.0,
            shadow_sampling_rate=0.25,
        ),
        records=[
            BenchmarkRequestRecord(
                request_id="req-phase4",
                scenario_family=WorkloadScenarioFamily.CANARY_ROLLOUT,
                tenant_id="tenant-rollout",
                request_class=RequestClass.LATENCY_SENSITIVE,
                session_id="canary-session-1",
                backend_name="mock-canary",
                backend_type="mock",
                model_alias="chat-shared",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=5.0,
                output_tokens=2,
                success=True,
                status_code=200,
                route_decision=route_decision,
                usage=UsageStats(prompt_tokens=2, completion_tokens=2, total_tokens=4),
            ),
            BenchmarkRequestRecord(
                request_id="req-stable",
                scenario_family=WorkloadScenarioFamily.SHADOW_TRAFFIC,
                tenant_id="tenant-shadow",
                request_class=RequestClass.STANDARD,
                session_id="shadow-session-1",
                backend_name="mock-stable",
                backend_type="mock",
                model_alias="chat-shared",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=5.0,
                output_tokens=2,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=2, completion_tokens=2, total_tokens=4),
            ),
        ],
    )

    markdown = render_run_report_markdown(artifact)

    assert "## Phase 4 Control Plane" in markdown
    assert "- Admission outcomes: `queued`: `1`" in markdown
    assert "- Queue waits: `count=1, avg_ms=12.500, p95_ms=12.500`" in markdown
    assert "- Breaker phases: `half_open`: `1`" in markdown
    assert "- Breaker reasons: `timeout_failure`: `1`" in markdown
    assert "- Session affinity: `missed`: `1`" in markdown
    assert "- Canary rollout: `canary`: `1`" in markdown
    assert "- Canary policies: `chat-shared-rollout`: `1`" in markdown
    assert "- Shadow traffic: `shadowed`: `1`" in markdown
    assert "- Shadow targets: `mock-shadow`: `1`" in markdown
    assert "sticky backend unavailable after cooldown" in markdown


@pytest.mark.asyncio
async def test_render_artifact_bundle_markdown_joins_multiple_artifacts() -> None:
    run_artifact = await run_synthetic_benchmark(
        scenario=build_synthetic_scenario(request_count=1),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )
    comparison_artifact = await compare_synthetic_policies(
        request_count=1,
        policies=[RoutingPolicy.BALANCED],
        timestamp=datetime(2026, 3, 15, 12, 1, tzinfo=UTC),
    )

    markdown = render_artifact_bundle_markdown([run_artifact, comparison_artifact])

    assert "# Switchyard Benchmark Report:" in markdown
    assert "# Switchyard Comparison Report:" in markdown
    assert "\n\n---\n\n" in markdown


@pytest.mark.asyncio
async def test_load_benchmark_artifact_model_round_trips_run_artifact(tmp_path: Path) -> None:
    artifact = await run_synthetic_benchmark(
        scenario=build_synthetic_scenario(request_count=1),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )
    artifact_path = write_artifact(artifact, tmp_path / "run-artifact.json")

    loaded = load_benchmark_artifact_model(artifact_path)

    assert isinstance(loaded, BenchmarkRunArtifact)
    assert loaded.run_id == artifact.run_id


@pytest.mark.asyncio
async def test_load_benchmark_artifact_model_accepts_v2_run_artifact_payload(
    tmp_path: Path,
) -> None:
    artifact = await run_synthetic_benchmark(
        scenario=build_synthetic_scenario(request_count=1),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
    )
    payload = artifact.model_dump(mode="json")
    payload["schema_version"] = BenchmarkArtifactSchemaVersion.V2.value
    payload["records"][0].pop("request_features", None)
    payload["records"][0].pop("policy_reference", None)
    payload["records"][0].pop("topology_reference", None)
    payload["records"][0].pop("execution_observation", None)
    payload["environment"].pop("topology_reference", None)

    artifact_path = tmp_path / "run-artifact-v2.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_benchmark_artifact_model(artifact_path)

    assert isinstance(loaded, BenchmarkRunArtifact)
    assert loaded.schema_version is BenchmarkArtifactSchemaVersion.V2


def test_summarize_records_includes_phase1_metrics() -> None:
    started_at = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 3, 15, 12, 0, 0, 1000, tzinfo=UTC)
    summary = summarize_records(
        [
            BenchmarkRequestRecord(
                request_id="req_1",
                backend_name="mlx-lm:mlx-chat",
                backend_type="mlx_lm",
                model_alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=10.0,
                ttft_ms=4.0,
                output_tokens=6,
                tokens_per_second=600.0,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=4, completion_tokens=6, total_tokens=10),
            ),
            BenchmarkRequestRecord(
                request_id="req_2",
                backend_name="mlx-lm:mlx-chat",
                backend_type="mlx_lm",
                model_alias="mlx-chat",
                model_identifier="mlx-community/test-model",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=20.0,
                ttft_ms=6.0,
                output_tokens=4,
                tokens_per_second=200.0,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=4, completion_tokens=4, total_tokens=8),
            ),
        ]
    )

    assert summary.avg_latency_ms == 15.0
    assert summary.p50_latency_ms == 10.0
    assert summary.p95_latency_ms == 20.0
    assert summary.avg_ttft_ms == 5.0
    assert summary.total_output_tokens == 10
    assert summary.avg_tokens_per_second == 400.0
    assert summary.fallback_count == 0
    assert summary.chosen_backend_counts == {"mlx-lm:mlx-chat": 2}


def test_parse_prometheus_text_extracts_labels() -> None:
    samples = parse_prometheus_text(
        "# HELP switchyard_backend_request_latency_ms Backend latency\n"
        'switchyard_backend_request_latency_ms{backend_name="mlx-lm:mlx-chat",'
        'index="1",model="mlx-chat"} 12.5\n'
        'switchyard_route_decision_latency_ms{backend_name="mlx-lm:mlx-chat",'
        'index="1",route_reason="target=mlx-chat | selected=mlx-lm:mlx-chat | '
        'reason=lowest latency | tie_breaker=score, latency_ms, backend_name"} 3.0\n'
    )

    assert samples[0].name == "switchyard_backend_request_latency_ms"
    assert samples[0].labels["backend_name"] == "mlx-lm:mlx-chat"
    assert samples[1].value == 3.0
    assert samples[1].labels["route_reason"].endswith(
        "tie_breaker=score, latency_ms, backend_name"
    )


def test_write_artifact_outputs_stable_json(tmp_path: Path) -> None:
    scenario = build_synthetic_scenario(request_count=1)

    started_at = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
    completed_at = datetime(2026, 3, 15, 12, 0, 0, 1000, tzinfo=UTC)
    benchmark_artifact = BenchmarkRunArtifact(
        run_id=build_run_id(
            run_timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
            policy=RoutingPolicy.BALANCED,
        ),
        timestamp=datetime(2026, 3, 15, 12, 0, tzinfo=UTC),
        git_sha="abcdef123456",
        scenario=scenario,
        policy=RoutingPolicy.BALANCED,
        backends_involved=["mock-local-fast"],
        backend_types_involved=["mock"],
        model_aliases_involved=["mock-chat"],
        request_count=1,
        summary=BenchmarkSummary(
            request_count=1,
            success_count=1,
            failure_count=0,
            avg_latency_ms=1.0,
            p50_latency_ms=1.0,
            p95_latency_ms=1.0,
            avg_ttft_ms=None,
            p95_ttft_ms=None,
            total_output_tokens=4,
            avg_output_tokens=4.0,
            avg_tokens_per_second=4.0,
            p95_tokens_per_second=4.0,
            fallback_count=0,
            chosen_backend_counts={"mock-local-fast": 1},
        ),
        environment=BenchmarkEnvironmentMetadata(
            benchmark_mode="synthetic",
            stream=False,
            timeout_seconds=30.0,
        ),
        records=[
            BenchmarkRequestRecord(
                request_id="synthetic_phase0_0000",
                backend_name="mock-local-fast",
                backend_type="mock",
                model_alias="mock-chat",
                model_identifier="mock-chat",
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=1.0,
                output_tokens=4,
                tokens_per_second=4.0,
                success=True,
                status_code=200,
                usage=UsageStats(prompt_tokens=4, completion_tokens=4, total_tokens=8),
            )
        ],
    )
    output_path = tmp_path / "artifact.json"

    write_artifact(benchmark_artifact, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "20260315T120000Z_balanced"
    assert payload["summary"]["avg_latency_ms"] == 1.0
    assert payload["environment"]["benchmark_mode"] == "synthetic"
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_load_benchmark_artifact_model_accepts_simulation_comparison_payload(
    tmp_path: Path,
) -> None:
    artifact = compare_candidate_policies_offline(
        policies=[
            ExplainablePolicySpec(
                policy_id="balanced-offline",
                objective=CounterfactualObjective.BALANCED,
            )
        ],
        evaluation_artifacts=[],
        evaluation_trace_records=[
            CapturedTraceRecord(
                record_id="trace-1",
                request_id="trace-req-1",
                execution_target=ExecutionTarget(
                    target_type=ExecutionTargetType.LOGICAL_ALIAS,
                    model_alias="chat-shared",
                ),
                logical_alias="chat-shared",
                chosen_backend="mock-a",
                latency_ms=12.0,
                status_code=200,
            )
        ],
        history_artifacts=[],
        history_trace_records=[],
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )
    artifact_path = tmp_path / "simulation-comparison.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_benchmark_artifact_model(artifact_path)

    assert isinstance(loaded, CounterfactualSimulationComparisonArtifact)
    assert loaded.simulation_comparison_id == artifact.simulation_comparison_id


def test_render_simulation_comparison_report_markdown_mentions_limitations() -> None:
    artifact = compare_candidate_policies_offline(
        policies=[
            ExplainablePolicySpec(
                policy_id="unsupported",
                objective=CounterfactualObjective.BALANCED,
            )
        ],
        evaluation_artifacts=[],
        evaluation_trace_records=[
            CapturedTraceRecord(
                record_id="trace-unsupported",
                request_id="trace-unsupported",
                execution_target=ExecutionTarget(
                    target_type=ExecutionTargetType.LOGICAL_ALIAS,
                    model_alias="chat-shared",
                ),
                logical_alias="chat-shared",
                chosen_backend="mock-a",
            )
        ],
        history_artifacts=[],
        history_trace_records=[],
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )

    markdown = render_simulation_comparison_report_markdown(artifact)

    assert "# Switchyard Simulation Comparison Report:" in markdown
    assert "## Limitations" in markdown


def test_load_benchmark_artifact_model_accepts_policy_recommendation_payload(
    tmp_path: Path,
) -> None:
    comparison = compare_candidate_policies_offline(
        policies=[
            ExplainablePolicySpec(
                policy_id="balanced-offline",
                objective=CounterfactualObjective.BALANCED,
            )
        ],
        evaluation_artifacts=[],
        evaluation_trace_records=[
            CapturedTraceRecord(
                record_id="trace-guidance",
                request_id="trace-guidance",
                execution_target=ExecutionTarget(
                    target_type=ExecutionTargetType.LOGICAL_ALIAS,
                    model_alias="chat-shared",
                ),
                logical_alias="chat-shared",
                chosen_backend="mock-a",
                latency_ms=12.0,
                status_code=200,
            )
        ],
        history_artifacts=[],
        history_trace_records=[],
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )
    report = build_policy_recommendation_report(
        [comparison],
        report_id="policy-guidance",
        timestamp=datetime(2026, 3, 17, tzinfo=UTC),
    )
    artifact_path = tmp_path / "policy-guidance.json"
    artifact_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_benchmark_artifact_model(artifact_path)

    assert isinstance(loaded, PolicyRecommendationReportArtifact)
    assert loaded.recommendation_report_id == "policy-guidance"
    markdown = render_loaded_artifact_markdown(loaded)
    assert "# Switchyard Policy Recommendation Report:" in markdown
