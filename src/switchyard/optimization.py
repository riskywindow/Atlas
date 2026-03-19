"""Helpers for exporting optimization-ready control-plane surfaces."""

from __future__ import annotations

import hashlib
import json

from switchyard.config import Settings
from switchyard.schemas.benchmark import (
    BenchmarkConfigFingerprint,
    BenchmarkConfigKnob,
    BenchmarkConfigKnobCategory,
    BenchmarkConfigSnapshot,
    BenchmarkRunConfig,
)
from switchyard.schemas.optimization import (
    OptimizationEvidenceProfile,
    OptimizationKnobSurface,
    OptimizationKnobType,
    OptimizationProfile,
)


def build_optimization_profile(settings: Settings) -> OptimizationProfile:
    """Build a typed snapshot of tunable routing knobs for later Stage A workflows."""

    optimization = settings.optimization
    rollout = settings.phase4.policy_rollout
    hybrid = settings.phase7.hybrid_execution

    notes = [
        "profile is informational and does not mutate live routing behavior",
        "later Stage A should pair this profile with benchmark, replay, and simulation artifacts",
    ]
    if optimization.promotion_requires_operator_review:
        notes.append("optimized policies still require explicit operator review before promotion")
    if hybrid.enabled and hybrid.prefer_local:
        notes.append(
            "local-first posture remains the default even when remote execution is enabled"
        )

    return OptimizationProfile(
        profile_id=optimization.profile_id,
        active_routing_policy=settings.default_routing_policy,
        active_rollout_mode=rollout.mode,
        allowlisted_routing_policies=list(optimization.allowlisted_routing_policies),
        allowlisted_rollout_modes=list(optimization.allowlisted_rollout_modes),
        candidate_policy_id=rollout.candidate_policy_id,
        shadow_policy_id=rollout.shadow_policy_id,
        hybrid_remote_enabled=hybrid.enabled,
        worker_launch_presets=list(optimization.worker_launch_presets),
        evidence=OptimizationEvidenceProfile(
            objective=optimization.objective,
            min_evidence_count=optimization.min_evidence_count,
            max_predicted_error_rate=optimization.max_predicted_error_rate,
            max_predicted_latency_regression_ms=optimization.max_predicted_latency_regression_ms,
            require_observed_backend_evidence=optimization.require_observed_backend_evidence,
            promotion_requires_operator_review=optimization.promotion_requires_operator_review,
            notes=[
                (
                    "evidence thresholds are intended for offline policy comparison and "
                    "recommendation flows"
                )
            ],
        ),
        knobs=_build_knob_surfaces(settings),
        notes=notes,
    )


def attach_benchmark_config_snapshot(
    *,
    settings: Settings,
    run_config: BenchmarkRunConfig,
) -> BenchmarkRunConfig:
    """Attach a canonical immutable config snapshot and fingerprint to one run config."""

    snapshot = build_benchmark_config_snapshot(settings=settings, run_config=run_config)
    return run_config.model_copy(
        update={
            "immutable_config": snapshot,
            "config_fingerprint": snapshot.fingerprint,
        },
        deep=True,
    )


def build_benchmark_config_snapshot(
    *,
    settings: Settings,
    run_config: BenchmarkRunConfig,
) -> BenchmarkConfigSnapshot:
    """Build the bounded benchmark-facing configuration truth for one run."""

    profile = build_optimization_profile(settings)
    knobs = _benchmark_runner_knobs(run_config) + _settings_knobs(settings)
    notes = [
        "captures bounded benchmark-relevant control-plane and worker-launch knobs",
        "batching knobs are absent because the current architecture does not expose them",
    ]
    return BenchmarkConfigSnapshot(
        profile_id=profile.profile_id,
        fingerprint=_fingerprint_benchmark_knobs(
            profile_id=profile.profile_id,
            knobs=knobs,
            notes=notes,
        ),
        knobs=knobs,
        notes=notes,
    )


def _build_knob_surfaces(settings: Settings) -> list[OptimizationKnobSurface]:
    optimization = settings.optimization
    rollout = settings.phase4.policy_rollout
    canary = settings.phase4.canary_routing
    shadow = settings.phase4.shadow_routing
    admission = settings.phase4.admission_control
    circuit = settings.phase4.circuit_breakers
    hybrid = settings.phase7.hybrid_execution
    return [
        OptimizationKnobSurface(
            knob_id="default_routing_policy",
            config_path="default_routing_policy",
            knob_type=OptimizationKnobType.ENUM,
            current_value=settings.default_routing_policy.value,
            allowed_values=[policy.value for policy in optimization.allowlisted_routing_policies],
            mutable_at_runtime=False,
            notes=["baseline compatibility policy for live routing"],
        ),
        OptimizationKnobSurface(
            knob_id="policy_rollout_mode",
            config_path="phase4.policy_rollout.mode",
            knob_type=OptimizationKnobType.ENUM,
            current_value=rollout.mode.value,
            allowed_values=[mode.value for mode in optimization.allowlisted_rollout_modes],
            mutable_at_runtime=True,
            notes=["runtime rollout controller may override this safely through the admin surface"],
        ),
        OptimizationKnobSurface(
            knob_id="policy_rollout_canary_percentage",
            config_path="phase4.policy_rollout.canary_percentage",
            knob_type=OptimizationKnobType.FLOAT,
            current_value=rollout.canary_percentage,
            min_value=0.0,
            max_value=optimization.max_rollout_canary_percentage,
            mutable_at_runtime=True,
            notes=["caps later policy promotion experiments to bounded slices"],
        ),
        OptimizationKnobSurface(
            knob_id="shadow_sampling_rate",
            config_path="phase4.shadow_routing.default_sampling_rate",
            knob_type=OptimizationKnobType.FLOAT,
            current_value=shadow.default_sampling_rate,
            min_value=0.0,
            max_value=optimization.max_shadow_sampling_rate,
            mutable_at_runtime=False,
            notes=["non-binding observational traffic only"],
        ),
        OptimizationKnobSurface(
            knob_id="canary_default_percentage",
            config_path="phase4.canary_routing.default_percentage",
            knob_type=OptimizationKnobType.FLOAT,
            current_value=canary.default_percentage,
            min_value=0.0,
            max_value=optimization.max_rollout_canary_percentage,
            mutable_at_runtime=False,
            notes=["applies to backend canaries rather than scorer rollout"],
        ),
        OptimizationKnobSurface(
            knob_id="admission_global_concurrency_cap",
            config_path="phase4.admission_control.global_concurrency_cap",
            knob_type=OptimizationKnobType.INTEGER,
            current_value=admission.global_concurrency_cap,
            min_value=1,
            max_value=optimization.max_global_concurrency_cap,
            mutable_at_runtime=False,
            notes=["kept explicit so offline tuning does not ignore overload posture"],
        ),
        OptimizationKnobSurface(
            knob_id="circuit_open_cooldown_seconds",
            config_path="phase4.circuit_breakers.open_cooldown_seconds",
            knob_type=OptimizationKnobType.FLOAT,
            current_value=circuit.open_cooldown_seconds,
            min_value=0.0,
            max_value=optimization.max_circuit_open_cooldown_seconds,
            mutable_at_runtime=False,
            notes=["backend protection remains outside the HTTP layer"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_prefer_local",
            config_path="phase7.hybrid_execution.prefer_local",
            knob_type=OptimizationKnobType.BOOLEAN,
            current_value=hybrid.prefer_local,
            mutable_at_runtime=False,
            notes=["local-first default should remain inspectable to optimizers"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_spillover_enabled",
            config_path="phase7.hybrid_execution.spillover_enabled",
            knob_type=OptimizationKnobType.BOOLEAN,
            current_value=hybrid.spillover_enabled,
            mutable_at_runtime=True,
            notes=["runtime operator controls may disable remote spillover immediately"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_max_remote_share_percent",
            config_path="phase7.hybrid_execution.max_remote_share_percent",
            knob_type=OptimizationKnobType.FLOAT,
            current_value=hybrid.max_remote_share_percent,
            min_value=0.0,
            max_value=optimization.max_remote_share_percent,
            mutable_at_runtime=True,
            notes=["bounded remote share guardrail for hybrid execution"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_remote_request_budget_per_minute",
            config_path="phase7.hybrid_execution.remote_request_budget_per_minute",
            knob_type=OptimizationKnobType.INTEGER,
            current_value=hybrid.remote_request_budget_per_minute,
            min_value=1 if hybrid.remote_request_budget_per_minute is not None else None,
            max_value=optimization.max_remote_request_budget_per_minute,
            mutable_at_runtime=True,
            notes=["null means no explicit per-minute budget is configured"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_remote_concurrency_cap",
            config_path="phase7.hybrid_execution.remote_concurrency_cap",
            knob_type=OptimizationKnobType.INTEGER,
            current_value=hybrid.remote_concurrency_cap,
            min_value=1 if hybrid.remote_concurrency_cap is not None else None,
            max_value=optimization.max_remote_concurrency_cap,
            mutable_at_runtime=True,
            notes=["protects remote capacity and cost posture"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_remote_cooldown_seconds",
            config_path="phase7.hybrid_execution.remote_cooldown_seconds",
            knob_type=OptimizationKnobType.FLOAT,
            current_value=hybrid.remote_cooldown_seconds,
            min_value=0.0,
            max_value=optimization.max_remote_cooldown_seconds,
            mutable_at_runtime=True,
            notes=["cooldown should remain explicit after transport instability"],
        ),
        OptimizationKnobSurface(
            knob_id="hybrid_allowed_remote_environments",
            config_path="phase7.hybrid_execution.allowed_remote_environments",
            knob_type=OptimizationKnobType.STRING_LIST,
            current_value=list(hybrid.allowed_remote_environments),
            mutable_at_runtime=False,
            notes=["keeps remote enablement scoped to named deployment environments"],
        ),
    ]


def _benchmark_runner_knobs(run_config: BenchmarkRunConfig) -> list[BenchmarkConfigKnob]:
    knobs: list[BenchmarkConfigKnob] = [
        BenchmarkConfigKnob(
            knob_id="benchmark.concurrency",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="run_config.concurrency",
            value=run_config.concurrency,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.timeout_seconds",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="run_config.timeout_seconds",
            value=run_config.timeout_seconds,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.canary_percentage",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="run_config.canary_percentage",
            value=run_config.canary_percentage,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.shadow_sampling_rate",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="run_config.shadow_sampling_rate",
            value=run_config.shadow_sampling_rate,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.trace_capture_mode",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="run_config.trace_capture_mode",
            value=run_config.trace_capture_mode.value,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.warmup.enabled",
            category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
            config_path="run_config.warmup.enabled",
            value=run_config.warmup.enabled,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.warmup.request_count",
            category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
            config_path="run_config.warmup.request_count",
            value=run_config.warmup.request_count,
            source_scope="benchmark_runner",
        ),
        BenchmarkConfigKnob(
            knob_id="benchmark.warmup.concurrency",
            category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
            config_path="run_config.warmup.concurrency",
            value=run_config.warmup.concurrency,
            source_scope="benchmark_runner",
        ),
    ]
    if run_config.replay_mode is not None:
        knobs.append(
            BenchmarkConfigKnob(
                knob_id="benchmark.replay_mode",
                category=BenchmarkConfigKnobCategory.SCHEDULING,
                config_path="run_config.replay_mode",
                value=run_config.replay_mode.value,
                source_scope="benchmark_runner",
            )
        )
    if run_config.session_affinity_ttl_seconds is not None:
        knobs.append(
            BenchmarkConfigKnob(
                knob_id="benchmark.session_affinity_ttl_seconds",
                category=BenchmarkConfigKnobCategory.ROUTING,
                config_path="run_config.session_affinity_ttl_seconds",
                value=run_config.session_affinity_ttl_seconds,
                source_scope="benchmark_runner",
            )
        )
    return knobs


def _settings_knobs(settings: Settings) -> list[BenchmarkConfigKnob]:
    phase4 = settings.phase4
    hybrid = settings.phase7.hybrid_execution
    optimization = settings.optimization
    knobs: list[BenchmarkConfigKnob] = [
        BenchmarkConfigKnob(
            knob_id="routing.default_policy",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="default_routing_policy",
            value=settings.default_routing_policy.value,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.global_concurrency_cap",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.global_concurrency_cap",
            value=phase4.admission_control.global_concurrency_cap,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.global_queue_size",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.global_queue_size",
            value=phase4.admission_control.global_queue_size,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.default_concurrency_cap",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.default_concurrency_cap",
            value=phase4.admission_control.default_concurrency_cap,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.default_queue_size",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.default_queue_size",
            value=phase4.admission_control.default_queue_size,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="admission.queue_timeout_seconds",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.admission_control.queue_timeout_seconds",
            value=phase4.admission_control.queue_timeout_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="circuit.failure_threshold",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.circuit_breakers.failure_threshold",
            value=phase4.circuit_breakers.failure_threshold,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="circuit.open_cooldown_seconds",
            category=BenchmarkConfigKnobCategory.SCHEDULING,
            config_path="phase4.circuit_breakers.open_cooldown_seconds",
            value=phase4.circuit_breakers.open_cooldown_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="rollout.mode",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.policy_rollout.mode",
            value=phase4.policy_rollout.mode.value,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="rollout.canary_percentage",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.policy_rollout.canary_percentage",
            value=phase4.policy_rollout.canary_percentage,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="canary.default_percentage",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.canary_routing.default_percentage",
            value=phase4.canary_routing.default_percentage,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="shadow.default_sampling_rate",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.shadow_routing.default_sampling_rate",
            value=phase4.shadow_routing.default_sampling_rate,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="affinity.ttl_seconds",
            category=BenchmarkConfigKnobCategory.ROUTING,
            config_path="phase4.session_affinity.ttl_seconds",
            value=phase4.session_affinity.ttl_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.prefer_local",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.prefer_local",
            value=hybrid.prefer_local,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.spillover_enabled",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.spillover_enabled",
            value=hybrid.spillover_enabled,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.max_remote_share_percent",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.max_remote_share_percent",
            value=hybrid.max_remote_share_percent,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.remote_request_budget_per_minute",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.remote_request_budget_per_minute",
            value=hybrid.remote_request_budget_per_minute,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.remote_concurrency_cap",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.remote_concurrency_cap",
            value=hybrid.remote_concurrency_cap,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.remote_cooldown_seconds",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.remote_cooldown_seconds",
            value=hybrid.remote_cooldown_seconds,
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="hybrid.allowed_remote_environments",
            category=BenchmarkConfigKnobCategory.HYBRID_EXECUTION,
            config_path="phase7.hybrid_execution.allowed_remote_environments",
            value=list(hybrid.allowed_remote_environments),
            source_scope="control_plane",
        ),
        BenchmarkConfigKnob(
            knob_id="search.allowlisted_routing_policies",
            category=BenchmarkConfigKnobCategory.SEARCH_SPACE,
            config_path="optimization.allowlisted_routing_policies",
            value=[policy.value for policy in optimization.allowlisted_routing_policies],
            source_scope="optimization_surface",
        ),
        BenchmarkConfigKnob(
            knob_id="search.allowlisted_rollout_modes",
            category=BenchmarkConfigKnobCategory.SEARCH_SPACE,
            config_path="optimization.allowlisted_rollout_modes",
            value=[mode.value for mode in optimization.allowlisted_rollout_modes],
            source_scope="optimization_surface",
        ),
    ]
    for model in sorted(settings.local_models, key=lambda item: item.alias):
        knobs.extend(
            [
                BenchmarkConfigKnob(
                    knob_id=f"serving.{model.alias}.configured_priority",
                    category=BenchmarkConfigKnobCategory.SERVING,
                    config_path=f"local_models[{model.alias}].configured_priority",
                    value=model.configured_priority,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"serving.{model.alias}.configured_weight",
                    category=BenchmarkConfigKnobCategory.SERVING,
                    config_path=f"local_models[{model.alias}].configured_weight",
                    value=model.configured_weight,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"worker_launch.{model.alias}.worker_transport",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"local_models[{model.alias}].worker_transport",
                    value=model.worker_transport.value,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"worker_launch.{model.alias}.warmup_enabled",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"local_models[{model.alias}].warmup.enabled",
                    value=model.warmup.enabled,
                    source_scope=model.alias,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"worker_launch.{model.alias}.warmup_eager",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"local_models[{model.alias}].warmup.eager",
                    value=model.warmup.eager,
                    source_scope=model.alias,
                ),
            ]
        )
    for preset in sorted(optimization.worker_launch_presets, key=lambda item: item.preset_name):
        prefix = f"worker_preset.{preset.preset_name}"
        knobs.extend(
            [
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.scope",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=f"optimization.worker_launch_presets[{preset.preset_name}].scope",
                    value=preset.scope.value,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.warmup_mode",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}].warmup_mode"
                    ),
                    value=preset.warmup_mode,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.concurrency_limit",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}]"
                        ".concurrency_limit"
                    ),
                    value=preset.concurrency_limit,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.supports_streaming",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}]"
                        ".supports_streaming"
                    ),
                    value=preset.supports_streaming,
                    source_scope=preset.preset_name,
                ),
                BenchmarkConfigKnob(
                    knob_id=f"{prefix}.stream_chunk_size",
                    category=BenchmarkConfigKnobCategory.WORKER_LAUNCH,
                    config_path=(
                        f"optimization.worker_launch_presets[{preset.preset_name}]"
                        ".stream_chunk_size"
                    ),
                    value=preset.stream_chunk_size,
                    source_scope=preset.preset_name,
                ),
            ]
        )
    return knobs


def _fingerprint_benchmark_knobs(
    *,
    profile_id: str,
    knobs: list[BenchmarkConfigKnob],
    notes: list[str],
) -> BenchmarkConfigFingerprint:
    payload = {
        "profile_id": profile_id,
        "knobs": [knob.model_dump(mode="json", exclude_none=True) for knob in knobs],
        "notes": notes,
    }
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return BenchmarkConfigFingerprint(
        algorithm="sha256_canonical_json",
        value=hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest(),
    )
