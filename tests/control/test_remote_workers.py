from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from switchyard.config import RemoteWorkerLifecycleSettings
from switchyard.control.remote_workers import (
    RemoteWorkerRegistrationError,
    RemoteWorkerRegistryService,
    build_signed_enrollment_token,
)
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendHealth,
    BackendHealthState,
    BackendLoadState,
    BackendNetworkEndpoint,
    BackendType,
    CapacitySnapshot,
    DeviceClass,
    EngineType,
    GPUDeviceMetadata,
    RequestFeatureSupport,
    RuntimeIdentity,
    WorkerLifecycleState,
    WorkerTransportType,
)
from switchyard.schemas.worker import (
    RemoteWorkerAuthMode,
    RemoteWorkerDeregisterRequest,
    RemoteWorkerHeartbeatRequest,
    RemoteWorkerRegistrationRequest,
)


def _registration_request(
    *,
    worker_id: str = "worker-1",
    ready: bool = False,
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.REGISTERING,
) -> RemoteWorkerRegistrationRequest:
    return RemoteWorkerRegistrationRequest(
        worker_id=worker_id,
        worker_name="remote-a",
        backend_type=BackendType.VLLM_CUDA,
        model_identifier="meta-llama/Llama-3.1-8B-Instruct",
        serving_targets=["chat-shared"],
        endpoint=BackendNetworkEndpoint(
            base_url=f"https://{worker_id}.internal",
            transport=WorkerTransportType.HTTPS,
        ),
        capabilities=BackendCapabilities(
            backend_type=BackendType.VLLM_CUDA,
            engine_type=EngineType.VLLM_CUDA,
            device_class=DeviceClass.NVIDIA_GPU,
            runtime=RuntimeIdentity(
                runtime_family="vllm_cuda",
                runtime_label="vllm_cuda",
                runtime_version="0.6.5",
                engine_type=EngineType.VLLM_CUDA,
                backend_type=BackendType.VLLM_CUDA,
            ),
            gpu=GPUDeviceMetadata(
                accelerator_type="gpu",
                vendor="nvidia",
                model="L4",
                count=1,
                memory_per_device_gib=24.0,
                cuda_version="12.4",
            ),
            model_ids=["meta-llama/Llama-3.1-8B-Instruct"],
            serving_targets=["chat-shared"],
            max_context_tokens=8192,
            supports_streaming=True,
            concurrency_limit=8,
            request_features=RequestFeatureSupport(
                supports_streaming=True,
                supports_response_format_json=True,
                limitations=["no multimodal inputs"],
            ),
        ),
        device_class=DeviceClass.NVIDIA_GPU,
        runtime=RuntimeIdentity(
            runtime_family="vllm_cuda",
            runtime_label="vllm_cuda",
            runtime_version="0.6.5",
            engine_type=EngineType.VLLM_CUDA,
            backend_type=BackendType.VLLM_CUDA,
        ),
        gpu=GPUDeviceMetadata(
            accelerator_type="gpu",
            vendor="nvidia",
            model="L4",
            count=1,
            memory_per_device_gib=24.0,
            cuda_version="12.4",
        ),
        ready=ready,
        lifecycle_state=lifecycle_state,
        observed_capacity=CapacitySnapshot(
            concurrency_limit=8,
            active_requests=0,
            queue_depth=0,
            tokens_per_second=120.0,
            gpu_utilization_percent=65.0,
        ),
    )


def test_remote_worker_registry_tracks_lifecycle_and_ready_state() -> None:
    now = datetime(2026, 3, 17, tzinfo=UTC)

    def now_fn() -> datetime:
        return now

    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            heartbeat_timeout_seconds=30.0,
        ),
        now_fn=now_fn,
    )

    registered = service.register(_registration_request(), token=None)
    assert registered.lifecycle_state is WorkerLifecycleState.REGISTERING
    assert registered.ready is False
    assert registered.lease_token is not None

    now = now + timedelta(seconds=5)
    heartbeat = service.heartbeat(
        RemoteWorkerHeartbeatRequest(
            worker_id="worker-1",
            lifecycle_state=WorkerLifecycleState.WARMING,
            ready=False,
            active_requests=1,
            queue_depth=0,
            health=BackendHealth(
                state=BackendHealthState.HEALTHY,
                load_state=BackendLoadState.WARMING,
            ),
        ),
        lease_token=registered.lease_token,
    )
    assert heartbeat.lifecycle_state is WorkerLifecycleState.WARMING

    now = now + timedelta(seconds=5)
    ready = service.heartbeat(
        RemoteWorkerHeartbeatRequest(
            worker_id="worker-1",
            ready=True,
            active_requests=2,
            queue_depth=1,
            health=BackendHealth(
                state=BackendHealthState.HEALTHY,
                load_state=BackendLoadState.READY,
                latency_ms=8.0,
            ),
        ),
        lease_token=registered.lease_token,
    )
    assert ready.lifecycle_state is WorkerLifecycleState.READY
    assert ready.ready is True

    snapshot = service.snapshot()
    assert snapshot.ready_worker_count == 1
    assert snapshot.live_worker_count == 1
    assert snapshot.workers[0].heartbeat_count == 3
    assert snapshot.workers[0].lifecycle_state is WorkerLifecycleState.READY
    assert snapshot.workers[0].instance.registration.lifecycle_state is WorkerLifecycleState.READY
    assert snapshot.workers[0].runtime is not None
    assert snapshot.workers[0].runtime.runtime_label == "vllm_cuda"
    assert snapshot.workers[0].gpu is not None
    assert snapshot.workers[0].gpu.vendor == "nvidia"
    assert snapshot.workers[0].observed_capacity is not None
    assert snapshot.workers[0].observed_capacity.tokens_per_second == 120.0


def test_remote_worker_registry_marks_lost_and_evicts_stale_workers() -> None:
    now = datetime(2026, 3, 17, tzinfo=UTC)

    def now_fn() -> datetime:
        return now

    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            heartbeat_timeout_seconds=30.0,
            stale_eviction_seconds=20.0,
        ),
        now_fn=now_fn,
    )
    registered = service.register(
        _registration_request(ready=True, lifecycle_state=WorkerLifecycleState.READY),
        token=None,
    )

    now = now + timedelta(seconds=31)
    stale_snapshot = service.snapshot()
    assert stale_snapshot.stale_worker_count == 1
    assert stale_snapshot.lost_worker_count == 1
    assert stale_snapshot.workers[0].lifecycle_state is WorkerLifecycleState.LOST

    now = now + timedelta(seconds=21)
    cleanup = service.cleanup_stale_workers()
    assert cleanup.evicted_worker_ids == ["worker-1"]
    assert cleanup.remaining_worker_count == 0
    assert any(event.event_type.value == "evicted" for event in service.snapshot().recent_events)
    assert registered.lease_token is not None


def test_remote_worker_registry_supports_graceful_drain_and_retire() -> None:
    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(dynamic_registration_enabled=True)
    )
    registered = service.register(
        _registration_request(ready=True, lifecycle_state=WorkerLifecycleState.READY),
        token=None,
    )
    assert registered.lease_token is not None

    draining = service.heartbeat(
        RemoteWorkerHeartbeatRequest(
            worker_id="worker-1",
            lifecycle_state=WorkerLifecycleState.DRAINING,
            ready=False,
            active_requests=0,
            queue_depth=0,
        ),
        lease_token=registered.lease_token,
    )
    assert draining.lifecycle_state is WorkerLifecycleState.DRAINING
    assert draining.ready is False

    retired = service.deregister(
        RemoteWorkerDeregisterRequest(worker_id="worker-1", reason="rolling deploy"),
        lease_token=registered.lease_token,
    )
    assert retired.lifecycle_state is WorkerLifecycleState.RETIRED
    assert retired.live is False


def test_remote_worker_registry_snapshot_orders_recent_events_newest_first() -> None:
    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(dynamic_registration_enabled=True)
    )
    registered = service.register(_registration_request(), token=None)
    assert registered.lease_token is not None

    service.heartbeat(
        RemoteWorkerHeartbeatRequest(
            worker_id="worker-1",
            ready=True,
            active_requests=0,
            queue_depth=0,
            health=BackendHealth(
                state=BackendHealthState.HEALTHY,
                load_state=BackendLoadState.READY,
            ),
        ),
        lease_token=registered.lease_token,
    )

    snapshot = service.snapshot()
    event_types = [event.event_type.value for event in snapshot.recent_events]
    assert event_types[0] == "heartbeat"
    assert event_types[-1] == "registered"


def test_remote_worker_registry_supports_operator_quarantine_and_canary_only() -> None:
    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(dynamic_registration_enabled=True)
    )
    service.register(_registration_request(), token=None)

    quarantined = service.set_quarantined(
        "worker-1",
        enabled=True,
        reason="transport failures",
    )
    assert quarantined.lifecycle_state is WorkerLifecycleState.UNHEALTHY
    snapshot = service.snapshot()
    assert "quarantined" in snapshot.workers[0].instance.tags

    canary_only = service.set_canary_only(
        "worker-1",
        enabled=True,
        reason="operator canary bucket",
    )
    assert canary_only.worker_id == "worker-1"
    snapshot = service.snapshot()
    assert "canary-only" in snapshot.workers[0].instance.tags


def test_remote_worker_registry_requires_valid_static_token_and_lease_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHYARD_REMOTE_REGISTRATION_TOKEN", "secret-token")
    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            secure_registration_required=True,
            auth_mode=RemoteWorkerAuthMode.STATIC_TOKEN,
            registration_token_name="SWITCHYARD_REMOTE_REGISTRATION_TOKEN",
        )
    )

    with pytest.raises(RemoteWorkerRegistrationError):
        service.register(_registration_request(), token="wrong-token")

    registered = service.register(_registration_request(), token="secret-token")
    assert registered.token_verified is True
    assert registered.lease_token is not None

    with pytest.raises(RemoteWorkerRegistrationError):
        service.heartbeat(
            RemoteWorkerHeartbeatRequest(worker_id="worker-1"),
            lease_token="bad-lease",
        )


def test_remote_worker_registry_accepts_signed_enrollment_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 3, 17, tzinfo=UTC)

    def now_fn() -> datetime:
        return now

    monkeypatch.setenv("SWITCHYARD_ENROLLMENT_SECRET", "secret-value")
    service = RemoteWorkerRegistryService(
        RemoteWorkerLifecycleSettings(
            dynamic_registration_enabled=True,
            secure_registration_required=True,
            auth_mode=RemoteWorkerAuthMode.SIGNED_ENROLLMENT,
            enrollment_secret_name="SWITCHYARD_ENROLLMENT_SECRET",
        ),
        now_fn=now_fn,
    )
    request = _registration_request(worker_id="worker-signed")
    token = build_signed_enrollment_token(
        request=request,
        secret="secret-value",
        expires_at=now + timedelta(minutes=5),
    )

    registered = service.register(request, token=token)
    assert registered.token_verified is True

    with pytest.raises(RemoteWorkerRegistrationError):
        service.register(request, token="v1.999.bad-signature")
