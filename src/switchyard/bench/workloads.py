"""Deterministic synthetic workload generation for local benchmarking."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from switchyard.schemas.benchmark import (
    WorkloadGenerationConfig,
    WorkloadItem,
    WorkloadPattern,
    WorkloadScenario,
    WorkloadScenarioFamily,
)
from switchyard.schemas.routing import RoutingPolicy, WorkloadShape

_SHORT_CHAT_TOPICS = (
    "latency budgets",
    "typed telemetry",
    "backend failover",
    "portable adapters",
    "benchmark reproducibility",
)

_LONG_PROMPT_CONTEXTS = (
    "local inference control plane design",
    "route-level observability tradeoffs",
    "deterministic benchmarking for model gateways",
    "adapter portability across device classes",
)

_REPEATED_PREFIXES = (
    "Shared context: customer tier is gold and retry budget is strict.",
    "Shared context: route health data should remain backend-agnostic.",
    "Shared context: compare policies without changing request shape.",
)

_BURST_ACTIONS = (
    "Summarize the incoming request in one sentence.",
    "Classify the request as interactive or batch.",
    "Extract the highest-risk routing concern.",
)

_PHASE4_SESSION_TOPICS = (
    "continue the troubleshooting thread",
    "keep the backend warm for follow-up turns",
    "preserve session-local context across retries",
)

_HYBRID_TOPICS = (
    "spill over only when local latency is under pressure",
    "keep local-first posture unless the queue is saturated",
    "compare remote help against explicit budget guardrails",
)

_REAL_CLOUD_VALIDATION_TOPICS = (
    "verify the first rented GPU worker path against the local baseline",
    "capture provider and runtime identity without hiding mixed evidence",
    "confirm queueing and spend posture before widening canaries",
)


def build_workload_manifest(
    *,
    family: WorkloadScenarioFamily,
    model_alias: str,
    request_count: int,
    seed: int,
    workload_shape: WorkloadShape = WorkloadShape.INTERACTIVE,
    policy: RoutingPolicy = RoutingPolicy.BALANCED,
) -> WorkloadScenario:
    """Build a deterministic workload manifest for a small built-in scenario family."""

    workload_generation = _generation_config_for_family(family=family, seed=seed)
    rng = random.Random(seed)
    items = [
        _build_item(
            family=family,
            model_alias=model_alias,
            index=index,
            request_count=request_count,
            seed=seed,
            rng=rng,
        )
        for index in range(request_count)
    ]
    return WorkloadScenario(
        name=f"{family.value}_{seed}_{request_count}",
        model=model_alias,
        model_alias=model_alias,
        family=family,
        policy=policy,
        workload_shape=workload_shape,
        request_count=request_count,
        prompt_template=None,
        workload_generation=workload_generation,
        items=items,
        scenario_seed=seed,
    )


def default_workload_manifest_path(base_dir: Path, scenario: WorkloadScenario) -> Path:
    """Return the default output path for a workload manifest."""

    return base_dir / f"{scenario.name}.json"


def _generation_config_for_family(
    *,
    family: WorkloadScenarioFamily,
    seed: int,
) -> WorkloadGenerationConfig:
    if family is WorkloadScenarioFamily.REPEATED_PREFIX:
        return WorkloadGenerationConfig(
            pattern=WorkloadPattern.REPEATED_PREFIX,
            seed=seed,
            shared_prefix=_REPEATED_PREFIXES[seed % len(_REPEATED_PREFIXES)],
        )
    if family is WorkloadScenarioFamily.CONCURRENCY_BURST:
        return WorkloadGenerationConfig(
            pattern=WorkloadPattern.BURSTY,
            seed=seed,
            burst_size=max(2, (seed % 3) + 2),
        )
    if family in {
        WorkloadScenarioFamily.QUEUE_SATURATION,
        WorkloadScenarioFamily.TENANT_CONTENTION,
        WorkloadScenarioFamily.HYBRID_SPILLOVER,
        WorkloadScenarioFamily.REMOTE_BUDGET_GUARDRAIL,
    }:
        return WorkloadGenerationConfig(
            pattern=WorkloadPattern.BURSTY,
            seed=seed,
            burst_size=max(3, (seed % 4) + 3),
        )
    if family is WorkloadScenarioFamily.REMOTE_COLD_WARM:
        return WorkloadGenerationConfig(
            pattern=WorkloadPattern.REPEATED_PREFIX,
            seed=seed,
            shared_prefix=(
                "Shared context: compare remote warm-path reuse against cold-start costs."
            ),
        )
    if family is WorkloadScenarioFamily.REAL_CLOUD_VALIDATION:
        return WorkloadGenerationConfig(
            pattern=WorkloadPattern.REPEATED_PREFIX,
            seed=seed,
            shared_prefix=(
                "Shared context: validate the first real cloud worker with explicit "
                "observed-versus-configured evidence boundaries."
            ),
        )
    return WorkloadGenerationConfig(seed=seed)


def _build_item(
    *,
    family: WorkloadScenarioFamily,
    model_alias: str,
    index: int,
    request_count: int,
    seed: int,
    rng: random.Random,
) -> WorkloadItem:
    if family is WorkloadScenarioFamily.MIXED:
        concrete_family = _mixed_family_for_index(seed=seed, index=index)
        return _build_item(
            family=concrete_family,
            model_alias=model_alias,
            index=index,
            request_count=request_count,
            seed=seed,
            rng=rng,
        )

    if family is WorkloadScenarioFamily.SHORT_CHAT:
        prompt = _short_chat_prompt(index=index, rng=rng)
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            metadata={
                "family": family.value,
                "variant": "chat_turn",
                "target_model_alias": model_alias,
            },
        )

    if family is WorkloadScenarioFamily.LONG_PROMPT:
        prompt = _long_prompt(index=index, rng=rng)
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            metadata={
                "family": family.value,
                "variant": "long_context",
                "target_model_alias": model_alias,
                "prompt_chars": str(len(prompt)),
            },
        )

    if family is WorkloadScenarioFamily.REPEATED_PREFIX:
        shared_prefix = _REPEATED_PREFIXES[seed % len(_REPEATED_PREFIXES)]
        prefix_id = f"prefix-{seed % len(_REPEATED_PREFIXES):02d}"
        body = (
            f"Request {index + 1}: explain how this context should affect routing in two lines."
        )
        prompt = f"{shared_prefix}\n{body}"
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            shared_prefix=shared_prefix,
            metadata={
                "family": family.value,
                "variant": "shared_prefix",
                "target_model_alias": model_alias,
                "prefix_group": prefix_id,
                "cache_candidate": "true",
            },
        )

    if family is WorkloadScenarioFamily.QUEUE_SATURATION:
        burst_size = max(3, (seed % 4) + 3)
        burst_index = (index % burst_size) + 1
        burst_group = (index // burst_size) + 1
        prompt = (
            f"Saturation burst {burst_group}, request {index + 1}: "
            "answer in two lines and prioritize latency over detail."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            burst_index=burst_index,
            burst_size=burst_size,
            metadata={
                "family": family.value,
                "variant": "saturation_burst",
                "target_model_alias": model_alias,
                "tenant_id": "default",
                "request_class": "latency_sensitive",
                "burst_group": str(burst_group),
                "expected_signal": "admission_control",
            },
        )

    if family is WorkloadScenarioFamily.TENANT_CONTENTION:
        burst_size = max(3, (seed % 4) + 3)
        burst_index = (index % burst_size) + 1
        burst_group = (index // burst_size) + 1
        tenant_id = "tenant-priority" if index % 3 != 2 else "tenant-standard"
        request_class = "latency_sensitive" if tenant_id == "tenant-priority" else "bulk"
        prompt = (
            f"Tenant contention window {burst_group}, request {index + 1}: "
            "respond with one routing recommendation."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            burst_index=burst_index,
            burst_size=burst_size,
            metadata={
                "family": family.value,
                "variant": "tenant_contention",
                "target_model_alias": model_alias,
                "tenant_id": tenant_id,
                "request_class": request_class,
                "burst_group": str(burst_group),
                "tenant_bucket": "hot" if tenant_id == "tenant-priority" else "cold",
            },
        )

    if family is WorkloadScenarioFamily.BACKEND_FLAKINESS:
        prompt = (
            f"Flakiness probe {index + 1}: summarize how the control plane should "
            "handle repeated backend failures."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            metadata={
                "family": family.value,
                "variant": "failure_probe",
                "target_model_alias": model_alias,
                "tenant_id": "default",
                "request_class": "standard",
                "failure_probe": "true",
                "expected_signal": "circuit_breaker",
            },
        )

    if family is WorkloadScenarioFamily.SESSION_STICKINESS:
        session_id = f"session-{seed:02d}-{index // 3:02d}"
        turn_number = (index % 3) + 1
        topic = _PHASE4_SESSION_TOPICS[index % len(_PHASE4_SESSION_TOPICS)]
        prompt = (
            f"Conversation turn {turn_number} for {session_id}: {topic}. "
            "Reply in one short paragraph."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            metadata={
                "family": family.value,
                "variant": "sticky_session",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-chat",
                "request_class": "standard",
                "session_id": session_id,
                "conversation_turn": str(turn_number),
                "expected_signal": "session_affinity",
            },
        )

    if family is WorkloadScenarioFamily.CANARY_ROLLOUT:
        session_id = f"canary-session-{seed:02d}-{index // 2:02d}"
        prompt = (
            f"Canary candidate request {index + 1}: explain rollout safety checks in two lines."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            metadata={
                "family": family.value,
                "variant": "rollout_candidate",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-rollout",
                "request_class": "latency_sensitive",
                "session_id": session_id,
                "rollout_bucket_key": session_id,
                "expected_signal": "canary_routing",
            },
        )

    if family is WorkloadScenarioFamily.SHADOW_TRAFFIC:
        session_id = f"shadow-session-{seed:02d}-{index:02d}"
        prompt = (
            f"Shadow sample {index + 1}: summarize the answer in one sentence without "
            "changing the primary response."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            metadata={
                "family": family.value,
                "variant": "shadow_candidate",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-shadow",
                "request_class": "standard",
                "session_id": session_id,
                "shadow_opt_in": "true",
                "expected_signal": "shadow_traffic",
            },
        )

    if family is WorkloadScenarioFamily.HYBRID_SPILLOVER:
        burst_size = max(3, (seed % 4) + 3)
        burst_index = (index % burst_size) + 1
        burst_group = (index // burst_size) + 1
        topic = _HYBRID_TOPICS[index % len(_HYBRID_TOPICS)]
        prompt = (
            f"Hybrid spillover wave {burst_group}, request {index + 1}: {topic}. "
            "Explain the serving tradeoff in two lines."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            burst_index=burst_index,
            burst_size=burst_size,
            metadata={
                "family": family.value,
                "variant": "hybrid_spillover",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-hybrid",
                "request_class": "latency_sensitive",
                "burst_group": str(burst_group),
                "expected_signal": "hybrid_spillover",
                "hybrid_policy_target": "burst_to_remote",
                "injected_execution_path": (
                    "local_only" if burst_index == 1 else "hybrid_spillover"
                ),
                "injected_remote_temperature": "cold" if burst_group == 1 else "warm",
                "injected_network_penalty_ms": str(25 + (burst_group * 5)),
                "injected_modeled_cost": str(round(0.02 * burst_index, 3)),
                "injected_budget_outcome": "within_budget",
            },
        )

    if family is WorkloadScenarioFamily.REMOTE_COLD_WARM:
        shared_prefix = (
            "Shared context: exercise remote worker cold start versus warm reuse honestly."
        )
        remote_temperature = "cold" if index < max(1, request_count // 2) else "warm"
        prompt = (
            f"Remote temperature probe {index + 1}: compare the {remote_temperature} path "
            "against the local fallback in two lines."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=f"{shared_prefix}\n{prompt}",
            shared_prefix=shared_prefix,
            metadata={
                "family": family.value,
                "variant": "remote_temperature",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-remote-temp",
                "request_class": "standard",
                "expected_signal": "remote_cold_warm",
                "hybrid_policy_target": "burst_to_remote",
                "injected_execution_path": "hybrid_spillover",
                "injected_remote_temperature": remote_temperature,
                "injected_network_penalty_ms": "40" if remote_temperature == "cold" else "12",
                "injected_cold_start_penalty_ms": (
                    "120" if remote_temperature == "cold" else "0"
                ),
                "injected_modeled_cost": "0.08",
                "injected_budget_outcome": "within_budget",
            },
        )

    if family is WorkloadScenarioFamily.REMOTE_BUDGET_GUARDRAIL:
        burst_size = max(3, (seed % 4) + 3)
        burst_index = (index % burst_size) + 1
        burst_group = (index // burst_size) + 1
        exhausted = index >= max(1, request_count // 2)
        prompt = (
            f"Remote budget guardrail {burst_group}, request {index + 1}: explain whether "
            "the control plane should spend remote budget or stay local."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=prompt,
            burst_index=burst_index,
            burst_size=burst_size,
            metadata={
                "family": family.value,
                "variant": "remote_budget_guardrail",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-budget",
                "request_class": "bulk" if exhausted else "latency_sensitive",
                "burst_group": str(burst_group),
                "expected_signal": "remote_budget_guardrail",
                "hybrid_policy_target": "remote_disabled" if exhausted else "burst_to_remote",
                "injected_execution_path": "remote_blocked" if exhausted else "hybrid_spillover",
                "injected_remote_temperature": "warm",
                "injected_network_penalty_ms": "18",
                "injected_modeled_cost": "0.05",
                "injected_budget_outcome": "exhausted" if exhausted else "within_budget",
            },
        )

    if family is WorkloadScenarioFamily.REAL_CLOUD_VALIDATION:
        shared_prefix = (
            "Shared context: confirm that the first real cloud worker run is honest, "
            "bounded, and comparable to the local path."
        )
        topic = _REAL_CLOUD_VALIDATION_TOPICS[index % len(_REAL_CLOUD_VALIDATION_TOPICS)]
        validation_phase = "baseline" if index == 0 else "cloud_probe"
        prompt = (
            f"Cloud validation request {index + 1}: {topic}. "
            "Respond in two lines with the main operator takeaway."
        )
        return _workload_item(
            family=family,
            model_alias=model_alias,
            index=index,
            seed=seed,
            prompt=f"{shared_prefix}\n{prompt}",
            shared_prefix=shared_prefix,
            metadata={
                "family": family.value,
                "variant": "real_cloud_validation",
                "target_model_alias": model_alias,
                "tenant_id": "tenant-cloud-validation",
                "request_class": "latency_sensitive",
                "expected_signal": "real_cloud_validation",
                "validation_phase": validation_phase,
                "comparison_baseline": "local_only",
                "expected_evidence_class": (
                    "unsupported" if validation_phase == "baseline" else "observed"
                ),
                "requires_observed_cloud": (
                    "false" if validation_phase == "baseline" else "true"
                ),
                "observed_provider_hint": "aws",
                "observed_region_hint": "us-east-1",
            },
        )

    burst_size = max(2, (seed % 3) + 2)
    burst_index = (index % burst_size) + 1
    burst_group = (index // burst_size) + 1
    prompt = _concurrency_burst_prompt(index=index, burst_group=burst_group, rng=rng)
    return _workload_item(
        family=family,
        model_alias=model_alias,
        index=index,
        seed=seed,
        prompt=prompt,
        burst_index=burst_index,
        burst_size=burst_size,
        metadata={
            "family": family.value,
            "variant": "burst",
            "target_model_alias": model_alias,
            "burst_group": str(burst_group),
            "burst_window": f"window-{burst_group:03d}",
        },
    )


def _workload_item(
    *,
    family: WorkloadScenarioFamily,
    model_alias: str,
    index: int,
    seed: int,
    prompt: str,
    metadata: dict[str, str],
    shared_prefix: str | None = None,
    burst_index: int | None = None,
    burst_size: int | None = None,
) -> WorkloadItem:
    stable_id = _stable_item_id(
        family=family,
        model_alias=model_alias,
        index=index,
        seed=seed,
        prompt=prompt,
    )
    return WorkloadItem(
        item_id=stable_id,
        family=family,
        prompt=prompt,
        metadata=metadata,
        shared_prefix=shared_prefix,
        burst_index=burst_index,
        burst_size=burst_size,
    )


def _stable_item_id(
    *,
    family: WorkloadScenarioFamily,
    model_alias: str,
    index: int,
    seed: int,
    prompt: str,
) -> str:
    source = f"{family.value}|{model_alias}|{seed}|{index}|{prompt}"
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]
    return f"{family.value}-{seed:08d}-{index:04d}-{digest}"


def _short_chat_prompt(*, index: int, rng: random.Random) -> str:
    topic = rng.choice(_SHORT_CHAT_TOPICS)
    return f"User turn {index + 1}: explain {topic} in one short paragraph."


def _long_prompt(*, index: int, rng: random.Random) -> str:
    context = rng.choice(_LONG_PROMPT_CONTEXTS)
    repeated_clause = (
        "Keep the design portable, typed, measurable, and easy to compare across runs. "
    )
    return (
        f"Scenario {index + 1}: review the following context about {context}. "
        + repeated_clause * 8
        + "End with two concise recommendations."
    )


def _concurrency_burst_prompt(*, index: int, burst_group: int, rng: random.Random) -> str:
    action = rng.choice(_BURST_ACTIONS)
    return (
        f"Burst group {burst_group}, request {index + 1}: {action} "
        "Keep the answer under three lines."
    )


def _mixed_family_for_index(*, seed: int, index: int) -> WorkloadScenarioFamily:
    families = (
        WorkloadScenarioFamily.SHORT_CHAT,
        WorkloadScenarioFamily.LONG_PROMPT,
        WorkloadScenarioFamily.REPEATED_PREFIX,
        WorkloadScenarioFamily.CONCURRENCY_BURST,
        WorkloadScenarioFamily.HYBRID_SPILLOVER,
        WorkloadScenarioFamily.REMOTE_COLD_WARM,
        WorkloadScenarioFamily.REAL_CLOUD_VALIDATION,
    )
    rng = random.Random(seed + index)
    return rng.choice(families)
