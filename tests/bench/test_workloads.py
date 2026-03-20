from __future__ import annotations

import json
from pathlib import Path

import pytest

from switchyard.bench.workloads import build_workload_manifest, default_workload_manifest_path
from switchyard.schemas.benchmark import WorkloadScenarioFamily
from switchyard.schemas.routing import WorkloadShape


@pytest.mark.parametrize(
    "family",
    [
        WorkloadScenarioFamily.SHORT_CHAT,
        WorkloadScenarioFamily.LONG_PROMPT,
        WorkloadScenarioFamily.REPEATED_PREFIX,
        WorkloadScenarioFamily.CONCURRENCY_BURST,
        WorkloadScenarioFamily.QUEUE_SATURATION,
        WorkloadScenarioFamily.TENANT_CONTENTION,
        WorkloadScenarioFamily.BACKEND_FLAKINESS,
        WorkloadScenarioFamily.SESSION_STICKINESS,
        WorkloadScenarioFamily.CANARY_ROLLOUT,
        WorkloadScenarioFamily.SHADOW_TRAFFIC,
        WorkloadScenarioFamily.HYBRID_SPILLOVER,
        WorkloadScenarioFamily.REMOTE_COLD_WARM,
        WorkloadScenarioFamily.REMOTE_BUDGET_GUARDRAIL,
        WorkloadScenarioFamily.REAL_CLOUD_VALIDATION,
        WorkloadScenarioFamily.MIXED,
    ],
)
def test_workload_manifest_supports_all_required_families(
    family: WorkloadScenarioFamily,
) -> None:
    scenario = build_workload_manifest(
        family=family,
        model_alias="chat-shared",
        request_count=4,
        seed=11,
    )

    assert scenario.family is family
    assert scenario.model_alias == "chat-shared"
    assert scenario.request_count == 4
    assert len(scenario.items) == 4
    assert all(item.metadata["target_model_alias"] == "chat-shared" for item in scenario.items)


def test_workload_manifest_is_deterministic_for_same_seed() -> None:
    first = build_workload_manifest(
        family=WorkloadScenarioFamily.MIXED,
        model_alias="chat-shared",
        request_count=5,
        seed=7,
    )
    second = build_workload_manifest(
        family=WorkloadScenarioFamily.MIXED,
        model_alias="chat-shared",
        request_count=5,
        seed=7,
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_workload_manifest_changes_for_different_seed() -> None:
    first = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=3,
        seed=1,
    )
    second = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=3,
        seed=2,
    )

    assert first.model_dump(mode="json") != second.model_dump(mode="json")


def test_repeated_prefix_manifest_carries_shared_prefix_metadata() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.REPEATED_PREFIX,
        model_alias="chat-shared",
        request_count=2,
        seed=5,
    )

    assert scenario.workload_generation.shared_prefix is not None
    assert all(
        item.shared_prefix == scenario.workload_generation.shared_prefix
        for item in scenario.items
    )
    assert all(item.prompt.startswith(f"{item.shared_prefix}\n") for item in scenario.items)
    assert all(item.metadata["cache_candidate"] == "true" for item in scenario.items)
    assert all("prefix_group" in item.metadata for item in scenario.items)


def test_concurrency_burst_manifest_carries_burst_metadata() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.CONCURRENCY_BURST,
        model_alias="chat-shared",
        request_count=5,
        seed=4,
        workload_shape=WorkloadShape.BATCH,
    )

    assert scenario.workload_shape is WorkloadShape.BATCH
    assert scenario.workload_generation.burst_size >= 2
    assert any(item.burst_index is not None for item in scenario.items)
    assert all(item.family is WorkloadScenarioFamily.CONCURRENCY_BURST for item in scenario.items)
    assert all("burst_window" in item.metadata for item in scenario.items)


def test_queue_saturation_manifest_carries_admission_pressure_metadata() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.QUEUE_SATURATION,
        model_alias="chat-shared",
        request_count=4,
        seed=8,
    )

    assert scenario.workload_generation.pattern.value == "bursty"
    assert all(item.metadata["tenant_id"] == "default" for item in scenario.items)
    assert all(item.metadata["request_class"] == "latency_sensitive" for item in scenario.items)
    assert all(item.metadata["expected_signal"] == "admission_control" for item in scenario.items)
    assert any(item.burst_index is not None for item in scenario.items)


def test_tenant_contention_manifest_spreads_requests_across_tenants() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.TENANT_CONTENTION,
        model_alias="chat-shared",
        request_count=6,
        seed=8,
    )

    tenant_ids = {item.metadata["tenant_id"] for item in scenario.items}

    assert tenant_ids == {"tenant-priority", "tenant-standard"}
    assert {item.metadata["request_class"] for item in scenario.items} == {
        "latency_sensitive",
        "bulk",
    }
    assert all("tenant_bucket" in item.metadata for item in scenario.items)


def test_session_stickiness_manifest_reuses_session_ids_across_turns() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.SESSION_STICKINESS,
        model_alias="chat-shared",
        request_count=6,
        seed=3,
    )

    session_ids = [item.metadata["session_id"] for item in scenario.items]

    assert session_ids[0] == session_ids[1] == session_ids[2]
    assert session_ids[3] == session_ids[4] == session_ids[5]
    assert session_ids[0] != session_ids[3]
    assert {item.metadata["conversation_turn"] for item in scenario.items[:3]} == {"1", "2", "3"}


def test_canary_and_shadow_manifests_carry_control_plane_matching_metadata() -> None:
    canary = build_workload_manifest(
        family=WorkloadScenarioFamily.CANARY_ROLLOUT,
        model_alias="chat-shared",
        request_count=4,
        seed=5,
    )
    shadow = build_workload_manifest(
        family=WorkloadScenarioFamily.SHADOW_TRAFFIC,
        model_alias="chat-shared",
        request_count=2,
        seed=5,
    )

    assert all(item.metadata["tenant_id"] == "tenant-rollout" for item in canary.items)
    assert all(
        item.metadata["rollout_bucket_key"].startswith("canary-session-")
        for item in canary.items
    )
    assert all(item.metadata["expected_signal"] == "canary_routing" for item in canary.items)
    assert all(item.metadata["tenant_id"] == "tenant-shadow" for item in shadow.items)
    assert all(item.metadata["shadow_opt_in"] == "true" for item in shadow.items)
    assert all(item.metadata["expected_signal"] == "shadow_traffic" for item in shadow.items)


def test_hybrid_spillover_manifest_carries_remote_pressure_metadata() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.HYBRID_SPILLOVER,
        model_alias="chat-shared",
        request_count=6,
        seed=5,
    )

    assert scenario.workload_generation.pattern.value == "bursty"
    assert all(
        item.metadata["expected_signal"] == "hybrid_spillover" for item in scenario.items
    )
    assert all(
        item.metadata["hybrid_policy_target"] == "burst_to_remote"
        for item in scenario.items
    )
    assert any(
        item.metadata["injected_execution_path"] == "hybrid_spillover"
        for item in scenario.items
    )


def test_remote_cold_warm_manifest_distinguishes_temperatures() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.REMOTE_COLD_WARM,
        model_alias="chat-shared",
        request_count=4,
        seed=6,
    )

    temperatures = {item.metadata["injected_remote_temperature"] for item in scenario.items}

    assert scenario.workload_generation.pattern.value == "repeated_prefix"
    assert temperatures == {"cold", "warm"}
    assert all(item.metadata["expected_signal"] == "remote_cold_warm" for item in scenario.items)


def test_remote_budget_guardrail_manifest_marks_budget_exhaustion() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.REMOTE_BUDGET_GUARDRAIL,
        model_alias="chat-shared",
        request_count=6,
        seed=6,
    )

    assert any(
        item.metadata["injected_budget_outcome"] == "exhausted" for item in scenario.items
    )
    assert any(
        item.metadata["injected_execution_path"] == "remote_blocked"
        for item in scenario.items
    )
    assert all(
        item.metadata["expected_signal"] == "remote_budget_guardrail"
        for item in scenario.items
    )


def test_real_cloud_validation_manifest_marks_observed_cloud_expectations() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.REAL_CLOUD_VALIDATION,
        model_alias="chat-shared",
        request_count=4,
        seed=12,
    )

    assert scenario.workload_generation.pattern.value == "repeated_prefix"
    assert scenario.workload_generation.shared_prefix is not None
    assert scenario.items[0].metadata["validation_phase"] == "baseline"
    assert scenario.items[0].metadata["expected_evidence_class"] == "unsupported"
    assert any(
        item.metadata["requires_observed_cloud"] == "true" for item in scenario.items[1:]
    )
    assert all(
        item.metadata["expected_signal"] == "real_cloud_validation"
        for item in scenario.items
    )


def test_default_workload_manifest_path_uses_scenario_name(tmp_path: Path) -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.SHORT_CHAT,
        model_alias="chat-shared",
        request_count=2,
        seed=9,
    )

    assert default_workload_manifest_path(tmp_path, scenario) == tmp_path / f"{scenario.name}.json"


def test_workload_manifest_serializes_to_json_shape() -> None:
    scenario = build_workload_manifest(
        family=WorkloadScenarioFamily.LONG_PROMPT,
        model_alias="chat-shared",
        request_count=1,
        seed=3,
    )

    payload = json.loads(scenario.model_dump_json())

    assert payload["family"] == "long_prompt"
    assert payload["items"][0]["item_id"].startswith("long_prompt-")
    assert payload["items"][0]["metadata"]["family"] == "long_prompt"
