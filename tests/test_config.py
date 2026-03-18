from pathlib import Path

from pydantic import ValidationError
from pytest import MonkeyPatch

from switchyard.config import AppEnvironment, Settings
from switchyard.schemas.backend import (
    BackendType,
    DeploymentProfile,
    WorkerRegistrationState,
    WorkerTransportType,
)
from switchyard.schemas.benchmark import TraceCaptureMode
from switchyard.schemas.routing import RoutingPolicy


def test_settings_loads_valid_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHYARD_ENV", "test")
    monkeypatch.setenv("SWITCHYARD_GATEWAY_PORT", "9000")
    monkeypatch.setenv("SWITCHYARD_DEFAULT_ROUTING_POLICY", "latency_first")
    monkeypatch.setenv("SWITCHYARD_METRICS_ENABLED", "true")
    monkeypatch.setenv("SWITCHYARD_TRACE_CAPTURE_MODE", "metadata_only")
    monkeypatch.setenv("SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH", "artifacts/traces/test.jsonl")

    settings = Settings()

    assert settings.env is AppEnvironment.TEST
    assert settings.gateway_port == 9000
    assert settings.default_routing_policy is RoutingPolicy.LATENCY_FIRST
    assert settings.metrics_enabled is True
    assert settings.trace_capture_mode is TraceCaptureMode.METADATA_ONLY
    assert str(settings.trace_capture_output_path) == "artifacts/traces/test.jsonl"
    assert settings.phase4.feature_toggles.admission_control_enabled is False


def test_settings_rejects_invalid_port(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHYARD_GATEWAY_PORT", "70000")

    try:
        Settings()
    except ValidationError as exc:
        assert "gateway_port" in str(exc)
    else:
        raise AssertionError("Settings should reject ports outside the valid range")


def test_settings_loads_local_model_configs(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_LOCAL_MODELS",
        (
            '[{"alias":"chat-default","model_identifier":"mlx-community/Qwen","backend_type":"mlx_lm",'
            '"serving_target":"chat-shared","configured_priority":10,"configured_weight":2.5,'
            '"deployment_profile":"compose","environment":"dev","worker_transport":"http",'
            '"image_tag":"switchyard/mlx-worker:dev","build_metadata":{"git_sha":"abcdef1"},'
            '"instances":[{"instance_id":"mlx-local-1","base_url":"http://127.0.0.1:9001","locality":"compose","transport":"http","source_of_truth":"registered","tags":["local","canary"],"registration":{"state":"registered"}}],'
            '"generation_defaults":{"max_output_tokens":256,"temperature":0.2},'
            '"warmup":{"enabled":true,"eager":true,"timeout_seconds":30}}]'
        ),
    )
    monkeypatch.setenv("SWITCHYARD_DEFAULT_MODEL_ALIAS", "chat-default")

    settings = Settings()

    assert settings.default_model_alias == "chat-default"
    assert len(settings.local_models) == 1
    assert settings.local_models[0].alias == "chat-default"
    assert settings.local_models[0].serving_target == "chat-shared"
    assert settings.local_models[0].backend_type is BackendType.MLX_LM
    assert settings.local_models[0].deployment_profile is DeploymentProfile.COMPOSE
    assert settings.local_models[0].environment == "dev"
    assert settings.local_models[0].worker_transport is WorkerTransportType.HTTP
    assert settings.local_models[0].image_tag == "switchyard/mlx-worker:dev"
    assert settings.local_models[0].build_metadata["git_sha"] == "abcdef1"
    assert settings.local_models[0].configured_priority == 10
    assert settings.local_models[0].configured_weight == 2.5
    assert settings.local_models[0].instances[0].instance_id == "mlx-local-1"
    assert settings.local_models[0].instances[0].base_url == "http://127.0.0.1:9001"
    assert settings.local_models[0].instances[0].locality == "compose"
    assert settings.local_models[0].instances[0].transport is WorkerTransportType.HTTP
    assert settings.local_models[0].instances[0].source_of_truth.value == "registered"
    assert settings.local_models[0].instances[0].tags == ("local", "canary")
    assert (
        settings.local_models[0].instances[0].registration.state
        is WorkerRegistrationState.REGISTERED
    )
    assert settings.local_models[0].generation_defaults.max_output_tokens == 256
    assert settings.local_models[0].warmup.enabled is True


def test_settings_loads_multiple_backends_for_one_serving_target(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_LOCAL_MODELS",
        (
            '[{"alias":"mlx-chat","serving_target":"chat-shared","model_identifier":"mlx-community/Qwen","backend_type":"mlx_lm"},'
            '{"alias":"metal-chat","serving_target":"chat-shared","model_identifier":"NousResearch/Meta-Llama-3","backend_type":"vllm_metal"}]'
        ),
    )
    monkeypatch.setenv("SWITCHYARD_DEFAULT_MODEL_ALIAS", "chat-shared")

    settings = Settings()

    assert settings.default_model_alias == "chat-shared"
    assert len(settings.local_models) == 2
    assert [model.serving_target for model in settings.local_models] == [
        "chat-shared",
        "chat-shared",
    ]
    assert [model.backend_type for model in settings.local_models] == [
        BackendType.MLX_LM,
        BackendType.VLLM_METAL,
    ]


def test_settings_load_phase4_control_plane_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_PHASE4",
        (
            '{"admission_control":{"enabled":true,"default_concurrency_cap":8,'
            '"global_concurrency_cap":32,"global_queue_size":64,'
            '"default_queue_size":16,"request_timeout_seconds":20.0,'
            '"queue_timeout_seconds":3.0,'
            '"per_tenant_limits":[{"tenant_id":"tenant-a","concurrency_cap":4,"queue_size":2}]},'
            '"circuit_breakers":{"enabled":true,"failure_threshold":3,"open_cooldown_seconds":15.0},'
            '"session_affinity":{"enabled":true,"ttl_seconds":120.0},'
            '"canary_routing":{"enabled":true,"default_percentage":5.0,'
            '"policies":[{"policy_name":"chat-rollout","serving_target":"chat-shared","enabled":true,'
            '"allocations":[{"backend_name":"metal-chat","percentage":5.0}]}]},'
            '"shadow_routing":{"enabled":true,"default_sampling_rate":0.1,'
            '"policies":[{"policy_name":"chat-shadow","enabled":true,"target_backend":"metal-chat",'
            '"sampling_rate":0.1}]}}'
        ),
    )

    settings = Settings()

    assert settings.phase4.admission_control.enabled is True
    assert settings.phase4.admission_control.global_concurrency_cap == 32
    assert settings.phase4.admission_control.global_queue_size == 64
    assert settings.phase4.admission_control.default_concurrency_cap == 8
    assert settings.phase4.admission_control.per_tenant_limits[0].tenant_id == "tenant-a"
    assert settings.phase4.circuit_breakers.failure_threshold == 3
    assert settings.phase4.session_affinity.ttl_seconds == 120.0
    assert settings.phase4.canary_routing.default_percentage == 5.0
    assert settings.phase4.shadow_routing.default_sampling_rate == 0.1
    assert settings.phase4.feature_toggles.canary_routing_enabled is True


def test_settings_load_phase7_hybrid_and_remote_worker_config(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_PHASE7",
        (
            '{"hybrid_execution":{"enabled":true,"prefer_local":true,'
            '"spillover_enabled":true,"require_healthy_local_backends":true,'
            '"max_remote_share_percent":20.0,"remote_request_budget_per_minute":180,'
            '"remote_concurrency_cap":12,"remote_kill_switch_enabled":false,'
            '"remote_cooldown_seconds":45.0,"allow_high_priority_remote_escalation":true,'
            '"allowed_remote_environments":["staging","prod-remote"],'
            '"per_tenant_remote_spillover":[{"tenant_id":"tenant-a","remote_enabled":false,'
            '"allow_high_priority_bypass":true}]},'
            '"remote_workers":{"secure_registration_required":true,'
            '"dynamic_registration_enabled":true,"auth_mode":"static_token",'
            '"heartbeat_timeout_seconds":45.0,"stale_eviction_seconds":120.0,'
            '"registration_token_name":"SWITCHYARD_WORKER_TOKEN","allow_static_instances":false}}'
        ),
    )

    settings = Settings()

    assert settings.phase7.hybrid_execution.enabled is True
    assert settings.phase7.hybrid_execution.spillover_enabled is True
    assert settings.phase7.hybrid_execution.max_remote_share_percent == 20.0
    assert settings.phase7.hybrid_execution.remote_concurrency_cap == 12
    assert settings.phase7.hybrid_execution.remote_cooldown_seconds == 45.0
    assert settings.phase7.hybrid_execution.allowed_remote_environments == (
        "staging",
        "prod-remote",
    )
    assert settings.phase7.hybrid_execution.per_tenant_remote_spillover[0].tenant_id == "tenant-a"
    assert settings.phase7.hybrid_execution.per_tenant_remote_spillover[0].remote_enabled is False
    assert settings.phase7.remote_workers.secure_registration_required is True
    assert settings.phase7.remote_workers.dynamic_registration_enabled is True
    assert settings.phase7.remote_workers.auth_mode.value == "static_token"
    assert settings.phase7.remote_workers.heartbeat_timeout_seconds == 45.0
    assert settings.phase7.remote_workers.stale_eviction_seconds == 120.0
    assert settings.phase7.remote_workers.registration_token_name == "SWITCHYARD_WORKER_TOKEN"
    assert settings.phase7.remote_workers.allow_static_instances is False


def test_settings_reject_duplicate_phase7_remote_environments(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_PHASE7",
        '{"hybrid_execution":{"allowed_remote_environments":["prod","prod"]}}',
    )

    try:
        Settings()
    except ValidationError as exc:
        assert "allowed_remote_environments" in str(exc)
    else:
        raise AssertionError("Settings should reject duplicate Phase 7 remote environments")


def test_settings_reject_duplicate_phase7_tenant_spillover_rules(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_PHASE7",
        (
            '{"hybrid_execution":{"per_tenant_remote_spillover":['
            '{"tenant_id":"tenant-a","remote_enabled":true},'
            '{"tenant_id":"tenant-a","remote_enabled":false}]}}'
        ),
    )

    try:
        Settings()
    except ValidationError as exc:
        assert "per_tenant_remote_spillover" in str(exc)
    else:
        raise AssertionError("Settings should reject duplicate Phase 7 tenant spillover rules")


def test_settings_reject_duplicate_phase4_tenant_limits(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_PHASE4",
        (
            '{"admission_control":{"enabled":true,"per_tenant_limits":['
            '{"tenant_id":"tenant-a","concurrency_cap":2,"queue_size":1},'
            '{"tenant_id":"tenant-a","concurrency_cap":4,"queue_size":2}]}}'
        ),
    )

    try:
        Settings()
    except ValidationError as exc:
        assert "per_tenant_limits" in str(exc)
    else:
        raise AssertionError("Settings should reject duplicate Phase 4 tenant limits")


def test_settings_load_topology_layers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_TOPOLOGY",
        (
            '{"active_environment":"dev","deployment_profile":"control_plane_container",'
            '"default_transport":"http","control_plane_image":{"image_tag":"switchyard/control-plane:dev"},'
            '"layers":[{"name":"dev","deployment_profile":"compose","default_transport":"http",'
            '"gateway_base_url":"http://switchyard-gateway:8000","metadata":{"namespace":"switchyard-dev"}},'
            '{"name":"kind","deployment_profile":"kind","default_transport":"http"}]}'
        ),
    )

    settings = Settings()

    assert settings.topology.active_environment == "dev"
    assert settings.topology.deployment_profile is DeploymentProfile.CONTROL_PLANE_CONTAINER
    assert settings.topology.default_transport is WorkerTransportType.HTTP
    assert settings.topology.control_plane_image is not None
    assert settings.topology.control_plane_image.image_tag == "switchyard/control-plane:dev"
    assert settings.topology.layers[0].gateway_base_url == "http://switchyard-gateway:8000"
    assert settings.topology.layers[0].metadata["namespace"] == "switchyard-dev"
    assert settings.topology.layers[1].deployment_profile is DeploymentProfile.KIND


def test_settings_reject_duplicate_topology_layer_names(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_TOPOLOGY",
        (
            '{"layers":[{"name":"dev","deployment_profile":"compose"},'
            '{"name":"dev","deployment_profile":"kind"}]}'
        ),
    )

    try:
        Settings()
    except ValidationError as exc:
        assert "topology.layers" in str(exc)
    else:
        raise AssertionError("Settings should reject duplicate topology layer names")


def test_settings_reject_instance_paths_without_leading_slash(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(
        "SWITCHYARD_LOCAL_MODELS",
        (
            '[{"alias":"chat-default","model_identifier":"mlx-community/Qwen","backend_type":"mlx_lm",'
            '"instances":[{"instance_id":"mlx-local-1","base_url":"http://127.0.0.1:9001",'
            '"chat_completions_path":"v1/chat/completions"}]}]'
        ),
    )

    try:
        Settings()
    except ValidationError as exc:
        assert "chat_completions_path" in str(exc)
    else:
        raise AssertionError("Settings should reject instance paths without a leading slash")


def test_phase5_compose_m4pro_example_env_loads(monkeypatch: MonkeyPatch) -> None:
    _load_env_file(monkeypatch, Path("docs/examples/phase5_compose_m4pro.env"))

    settings = Settings()

    assert settings.default_model_alias == "chat-shared"
    assert len(settings.local_models) == 2
    assert settings.local_models[0].worker_transport is WorkerTransportType.HTTP
    assert settings.local_models[0].instances[0].base_url == "http://host.docker.internal:8101"
    assert settings.local_models[1].instances[0].base_url == "http://host.docker.internal:8102"


def test_phase5_compose_smoke_example_env_loads(monkeypatch: MonkeyPatch) -> None:
    _load_env_file(monkeypatch, Path("docs/examples/phase5_compose_smoke.env"))

    settings = Settings()

    assert settings.env is AppEnvironment.TEST
    assert settings.default_model_alias == "chat-smoke"
    assert len(settings.local_models) == 1
    assert settings.local_models[0].backend_type is BackendType.MOCK
    assert settings.local_models[0].worker_transport is WorkerTransportType.HTTP


def test_phase5_kind_m4pro_example_env_loads(monkeypatch: MonkeyPatch) -> None:
    _load_env_file(monkeypatch, Path("docs/examples/phase5_kind_m4pro.env"))

    settings = Settings()

    assert settings.default_model_alias == "chat-shared"
    assert len(settings.local_models) == 2
    assert settings.local_models[0].deployment_profile is DeploymentProfile.KIND
    assert settings.local_models[0].instances[0].base_url == "http://host.docker.internal:8101"
    assert settings.local_models[1].instances[0].base_url == "http://host.docker.internal:8102"


def test_phase5_kind_smoke_example_env_loads(monkeypatch: MonkeyPatch) -> None:
    _load_env_file(monkeypatch, Path("docs/examples/phase5_kind_smoke.env"))

    settings = Settings()

    assert settings.env is AppEnvironment.TEST
    assert settings.default_model_alias == "chat-smoke"
    assert len(settings.local_models) == 1
    assert settings.local_models[0].deployment_profile is DeploymentProfile.KIND
    assert settings.local_models[0].backend_type is BackendType.MOCK


def test_kind_overlay_env_files_match_documented_examples() -> None:
    assert Path("infra/kind/overlays/m4pro/settings.env").read_text(encoding="utf-8") == Path(
        "docs/examples/phase5_kind_m4pro.env"
    ).read_text(encoding="utf-8")
    assert Path("infra/kind/overlays/smoke/settings.env").read_text(encoding="utf-8") == Path(
        "docs/examples/phase5_kind_smoke.env"
    ).read_text(encoding="utf-8")


def _load_env_file(monkeypatch: MonkeyPatch, path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        monkeypatch.setenv(key, value)
