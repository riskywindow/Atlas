"""In-memory remote worker registration and lifecycle tracking."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from switchyard.config import RemoteWorkerLifecycleSettings
from switchyard.schemas.backend import (
    BackendHealth,
    BackendHealthState,
    BackendInstance,
    BackendRegistrationMetadata,
    WorkerLifecycleState,
    WorkerRegistrationState,
)
from switchyard.schemas.worker import (
    RegisteredRemoteWorkerRecord,
    RegisteredRemoteWorkerSnapshot,
    RemoteWorkerAuthMode,
    RemoteWorkerCleanupResponse,
    RemoteWorkerDeregisterRequest,
    RemoteWorkerHeartbeatRequest,
    RemoteWorkerLifecycleEvent,
    RemoteWorkerLifecycleEventType,
    RemoteWorkerRegistrationRequest,
    RemoteWorkerRegistrationResponse,
)


class RemoteWorkerRegistrationError(RuntimeError):
    """Raised when remote worker lifecycle requests are invalid."""


@dataclass(slots=True)
class _RegisteredWorkerState:
    request: RemoteWorkerRegistrationRequest
    registered_at: datetime
    last_heartbeat_at: datetime
    expires_at: datetime
    deregistered_at: datetime | None
    lifecycle_state: WorkerLifecycleState
    ready: bool
    active_requests: int
    queue_depth: int
    heartbeat_count: int
    health: BackendHealth | None
    metadata: dict[str, str]
    operator_tags: set[str]
    enrollment_verified: bool
    lease_token: str


class RemoteWorkerRegistryService:
    """Tracks explicit remote worker lifecycle state for Phase 7."""

    def __init__(
        self,
        settings: RemoteWorkerLifecycleSettings,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._now = now_fn or (lambda: datetime.now(UTC))
        self._workers: dict[str, _RegisteredWorkerState] = {}
        self._events: deque[RemoteWorkerLifecycleEvent] = deque(
            maxlen=settings.max_lifecycle_events
        )

    def register(
        self,
        request: RemoteWorkerRegistrationRequest,
        *,
        token: str | None,
    ) -> RemoteWorkerRegistrationResponse:
        self._validate_registration_enabled()
        enrollment_verified = self._validate_enrollment_token(request=request, token=token)
        now = self._now()
        lifecycle_state = self._normalize_lifecycle_state(
            request.lifecycle_state,
            ready=request.ready,
            health=request.health,
        )
        lease_token = secrets.token_urlsafe(24)
        state = _RegisteredWorkerState(
            request=request.model_copy(deep=True),
            registered_at=now,
            last_heartbeat_at=now,
            expires_at=now + timedelta(seconds=self._settings.heartbeat_timeout_seconds),
            deregistered_at=None,
            lifecycle_state=lifecycle_state,
            ready=request.ready,
            active_requests=request.active_requests,
            queue_depth=request.queue_depth,
            heartbeat_count=1,
            health=request.health.model_copy(deep=True) if request.health is not None else None,
            metadata=dict(request.metadata),
            operator_tags=set(),
            enrollment_verified=enrollment_verified,
            lease_token=lease_token,
        )
        self._workers[request.worker_id] = state
        self._record_event(
            RemoteWorkerLifecycleEventType.REGISTERED,
            worker_id=request.worker_id,
            lifecycle_state=lifecycle_state,
            detail=f"worker '{request.worker_name}' registered",
            metadata={
                "environment": request.environment,
                "transport": request.endpoint.transport.value,
            },
        )
        return self._response_for(state, include_lease_token=True)

    def heartbeat(
        self,
        request: RemoteWorkerHeartbeatRequest,
        *,
        lease_token: str | None,
    ) -> RemoteWorkerRegistrationResponse:
        state = self._require_worker(request.worker_id)
        self._validate_lease_token(state=state, lease_token=lease_token)
        if state.lifecycle_state is WorkerLifecycleState.RETIRED:
            msg = f"worker '{request.worker_id}' has already retired"
            raise RemoteWorkerRegistrationError(msg)
        previous_state = state.lifecycle_state
        now = self._now()
        state.last_heartbeat_at = now
        state.expires_at = now + timedelta(seconds=self._settings.heartbeat_timeout_seconds)
        state.active_requests = request.active_requests
        state.queue_depth = request.queue_depth
        state.heartbeat_count += 1
        state.health = request.health.model_copy(deep=True) if request.health is not None else None
        state.metadata = {**state.metadata, **dict(request.metadata)}
        if request.ready is not None:
            state.ready = request.ready
        if request.lifecycle_state is not None:
            state.lifecycle_state = self._normalize_lifecycle_state(
                request.lifecycle_state,
                ready=state.ready,
                health=state.health,
            )
        elif state.lifecycle_state not in {
            WorkerLifecycleState.DRAINING,
            WorkerLifecycleState.RETIRED,
        }:
            state.lifecycle_state = self._normalize_lifecycle_state(
                state.lifecycle_state,
                ready=state.ready,
                health=state.health,
            )
        if state.lifecycle_state is not previous_state:
            self._record_event(
                RemoteWorkerLifecycleEventType.STATE_CHANGED,
                worker_id=request.worker_id,
                lifecycle_state=state.lifecycle_state,
                detail=(
                    f"lifecycle changed from {previous_state.value} "
                    f"to {state.lifecycle_state.value}"
                ),
            )
        self._record_event(
            RemoteWorkerLifecycleEventType.HEARTBEAT,
            worker_id=request.worker_id,
            lifecycle_state=state.lifecycle_state,
            metadata={
                "active_requests": str(state.active_requests),
                "queue_depth": str(state.queue_depth),
                "ready": str(state.ready).lower(),
            },
        )
        return self._response_for(state)

    def deregister(
        self,
        request: RemoteWorkerDeregisterRequest,
        *,
        lease_token: str | None,
    ) -> RemoteWorkerRegistrationResponse:
        state = self._require_worker(request.worker_id)
        self._validate_lease_token(state=state, lease_token=lease_token)
        now = self._now()
        state.deregistered_at = now
        state.last_heartbeat_at = now
        state.expires_at = now
        state.ready = False
        state.active_requests = 0
        state.queue_depth = 0
        state.lifecycle_state = (
            WorkerLifecycleState.RETIRED
            if request.retire
            else WorkerLifecycleState.DRAINING
        )
        self._record_event(
            RemoteWorkerLifecycleEventType.DEREGISTERED,
            worker_id=request.worker_id,
            lifecycle_state=state.lifecycle_state,
            detail=request.reason or "worker requested graceful deregistration",
        )
        return self._response_for(state)

    def cleanup_stale_workers(self) -> RemoteWorkerCleanupResponse:
        now = self._now()
        evicted: list[str] = []
        cutoff = timedelta(seconds=self._settings.stale_eviction_seconds)
        for worker_id, state in list(self._workers.items()):
            record = self._to_record(worker_id=worker_id, state=state, now=now)
            if record.lifecycle_state not in {
                WorkerLifecycleState.LOST,
                WorkerLifecycleState.RETIRED,
            }:
                continue
            anchor = state.deregistered_at or state.expires_at
            if now - anchor < cutoff:
                continue
            evicted.append(worker_id)
            self._workers.pop(worker_id, None)
            self._record_event(
                RemoteWorkerLifecycleEventType.EVICTED,
                worker_id=worker_id,
                lifecycle_state=record.lifecycle_state,
                detail="worker evicted from registry after lifecycle retention window",
            )
        return RemoteWorkerCleanupResponse(
            evicted_worker_ids=evicted,
            remaining_worker_count=len(self._workers),
        )

    def mark_draining(
        self,
        worker_id: str,
        *,
        reason: str | None = None,
    ) -> RemoteWorkerRegistrationResponse:
        state = self._require_worker(worker_id)
        state.lifecycle_state = WorkerLifecycleState.DRAINING
        state.ready = False
        self._record_event(
            RemoteWorkerLifecycleEventType.STATE_CHANGED,
            worker_id=worker_id,
            lifecycle_state=state.lifecycle_state,
            detail=reason or "operator marked worker draining",
        )
        return self._response_for(state)

    def set_quarantined(
        self,
        worker_id: str,
        *,
        enabled: bool,
        reason: str | None = None,
    ) -> RemoteWorkerRegistrationResponse:
        state = self._require_worker(worker_id)
        if enabled:
            state.operator_tags.add("quarantined")
            state.lifecycle_state = WorkerLifecycleState.UNHEALTHY
            state.ready = False
        else:
            state.operator_tags.discard("quarantined")
            if state.lifecycle_state is WorkerLifecycleState.UNHEALTHY:
                state.lifecycle_state = self._normalize_lifecycle_state(
                    WorkerLifecycleState.WARMING,
                    ready=state.ready,
                    health=state.health,
                )
        self._record_event(
            RemoteWorkerLifecycleEventType.STATE_CHANGED,
            worker_id=worker_id,
            lifecycle_state=state.lifecycle_state,
            detail=reason
            or (
                "operator quarantined worker"
                if enabled
                else "operator cleared quarantine"
            ),
            metadata={"tag": "quarantined", "enabled": str(enabled).lower()},
        )
        return self._response_for(state)

    def set_canary_only(
        self,
        worker_id: str,
        *,
        enabled: bool,
        reason: str | None = None,
    ) -> RemoteWorkerRegistrationResponse:
        state = self._require_worker(worker_id)
        if enabled:
            state.operator_tags.add("canary-only")
        else:
            state.operator_tags.discard("canary-only")
        self._record_event(
            RemoteWorkerLifecycleEventType.STATE_CHANGED,
            worker_id=worker_id,
            lifecycle_state=state.lifecycle_state,
            detail=reason
            or (
                "operator marked worker canary-only"
                if enabled
                else "operator cleared canary-only"
            ),
            metadata={"tag": "canary-only", "enabled": str(enabled).lower()},
        )
        return self._response_for(state)

    def snapshot(self) -> RegisteredRemoteWorkerSnapshot:
        now = self._now()
        workers = [
            self._to_record(worker_id=worker_id, state=state, now=now)
            for worker_id, state in sorted(self._workers.items())
        ]
        return RegisteredRemoteWorkerSnapshot(
            secure_registration_required=self._settings.secure_registration_required,
            auth_mode=(
                self._settings.auth_mode
                if self._settings.secure_registration_required
                else RemoteWorkerAuthMode.NONE
            ),
            dynamic_registration_enabled=self._settings.dynamic_registration_enabled,
            heartbeat_timeout_seconds=self._settings.heartbeat_timeout_seconds,
            stale_eviction_seconds=self._settings.stale_eviction_seconds,
            registration_token_name=self._settings.registration_token_name,
            worker_count=len(workers),
            stale_worker_count=sum(1 for worker in workers if worker.stale),
            ready_worker_count=sum(1 for worker in workers if worker.ready),
            live_worker_count=sum(1 for worker in workers if worker.live),
            draining_worker_count=sum(
                1
                for worker in workers
                if worker.lifecycle_state is WorkerLifecycleState.DRAINING
            ),
            unhealthy_worker_count=sum(
                1
                for worker in workers
                if worker.lifecycle_state is WorkerLifecycleState.UNHEALTHY
            ),
            lost_worker_count=sum(
                1 for worker in workers if worker.lifecycle_state is WorkerLifecycleState.LOST
            ),
            retired_worker_count=sum(
                1
                for worker in workers
                if worker.lifecycle_state is WorkerLifecycleState.RETIRED
            ),
            workers=workers,
            # Keep operator-facing lifecycle events in reverse chronological order so the
            # latest registration, heartbeat, or cleanup action is visible first.
            recent_events=list(reversed(self._events)),
        )

    def _to_record(
        self,
        *,
        worker_id: str,
        state: _RegisteredWorkerState,
        now: datetime,
    ) -> RegisteredRemoteWorkerRecord:
        live = state.deregistered_at is None and now <= state.expires_at
        lifecycle_state = state.lifecycle_state
        if (
            state.deregistered_at is not None
            and lifecycle_state is not WorkerLifecycleState.RETIRED
        ):
            lifecycle_state = WorkerLifecycleState.RETIRED
        elif not live and lifecycle_state is not WorkerLifecycleState.RETIRED:
            lifecycle_state = WorkerLifecycleState.LOST

        ready = state.ready and live and lifecycle_state is WorkerLifecycleState.READY
        health = state.health.model_copy(deep=True) if state.health is not None else None
        if health is None:
            health = BackendHealth(
                state=BackendHealthState.DEGRADED,
                detail="heartbeat not reported yet",
            )
        if lifecycle_state is WorkerLifecycleState.UNHEALTHY:
            health.state = BackendHealthState.DEGRADED
        elif lifecycle_state in {WorkerLifecycleState.LOST, WorkerLifecycleState.RETIRED}:
            health.state = BackendHealthState.UNAVAILABLE
            health.detail = (
                "worker retired"
                if lifecycle_state is WorkerLifecycleState.RETIRED
                else "heartbeat expired"
            )

        request = state.request
        instance = BackendInstance(
            instance_id=worker_id,
            endpoint=request.endpoint.model_copy(deep=True),
            source_of_truth=request.source_of_truth,
            backend_type=request.backend_type,
            device_class=request.device_class,
            model_identifier=request.model_identifier,
            locality=request.locality,
            locality_class=request.locality_class,
            execution_mode=request.execution_mode,
            placement=request.placement.model_copy(deep=True),
            cost_profile=request.cost_profile.model_copy(deep=True),
            readiness_hints=request.readiness_hints.model_copy(deep=True),
            trust=request.trust.model_copy(deep=True),
            network_characteristics=request.network_characteristics.model_copy(deep=True),
            tags=sorted(set(request.tags + ["remote", "registered"])),
            registration=BackendRegistrationMetadata(
                state=(
                    WorkerRegistrationState.STALE
                    if lifecycle_state is WorkerLifecycleState.LOST
                    else WorkerRegistrationState.REGISTERED
                ),
                lifecycle_state=lifecycle_state,
                registered_at=state.registered_at,
                last_heartbeat_at=state.last_heartbeat_at,
                expires_at=state.expires_at,
                deregistered_at=state.deregistered_at,
                heartbeat_count=state.heartbeat_count,
                source="dynamic_registration",
            ),
            health=health,
            last_seen_at=state.last_heartbeat_at if live else None,
            image_metadata=request.image_metadata.model_copy(deep=True)
            if request.image_metadata is not None
            else None,
            metadata={
                **request.metadata,
                **state.metadata,
                "worker_name": request.worker_name,
                "ready": str(ready).lower(),
            },
        )
        instance.tags = sorted(set(instance.tags) | state.operator_tags)
        return RegisteredRemoteWorkerRecord(
            worker_id=worker_id,
            worker_name=request.worker_name,
            environment=request.environment,
            serving_targets=list(request.serving_targets),
            backend_name=f"remote-worker:{request.worker_name}",
            backend_type=request.backend_type,
            lifecycle_state=lifecycle_state,
            registration_state=instance.registration.state,
            registered_at=state.registered_at,
            last_heartbeat_at=state.last_heartbeat_at,
            expires_at=state.expires_at,
            deregistered_at=state.deregistered_at,
            stale=lifecycle_state is WorkerLifecycleState.LOST,
            live=live,
            ready=ready,
            active_requests=state.active_requests,
            queue_depth=state.queue_depth,
            heartbeat_count=state.heartbeat_count,
            capabilities=request.capabilities.model_copy(deep=True),
            token_verified=state.enrollment_verified,
            instance=instance,
            metadata={
                **request.metadata,
                **state.metadata,
            },
        )

    def _response_for(
        self,
        state: _RegisteredWorkerState,
        *,
        include_lease_token: bool = False,
    ) -> RemoteWorkerRegistrationResponse:
        record = self._to_record(
            worker_id=state.request.worker_id,
            state=state,
            now=self._now(),
        )
        return RemoteWorkerRegistrationResponse(
            worker_id=state.request.worker_id,
            lifecycle_state=record.lifecycle_state,
            registration_state=record.registration_state,
            registered_at=state.registered_at,
            last_heartbeat_at=state.last_heartbeat_at,
            expires_at=state.expires_at,
            ready=record.ready,
            live=record.live,
            secure_registration_required=self._settings.secure_registration_required,
            auth_mode=(
                self._settings.auth_mode
                if self._settings.secure_registration_required
                else RemoteWorkerAuthMode.NONE
            ),
            token_verified=state.enrollment_verified,
            lease_token=state.lease_token if include_lease_token else None,
        )

    def _require_worker(self, worker_id: str) -> _RegisteredWorkerState:
        state = self._workers.get(worker_id)
        if state is None:
            msg = f"worker '{worker_id}' is not registered"
            raise RemoteWorkerRegistrationError(msg)
        return state

    def _validate_registration_enabled(self) -> None:
        if not self._settings.dynamic_registration_enabled:
            msg = "dynamic remote worker registration is disabled"
            raise RemoteWorkerRegistrationError(msg)

    def _validate_lease_token(
        self,
        *,
        state: _RegisteredWorkerState,
        lease_token: str | None,
    ) -> None:
        if not secrets.compare_digest(lease_token or "", state.lease_token):
            msg = f"lease token is invalid for worker '{state.request.worker_id}'"
            raise RemoteWorkerRegistrationError(msg)

    def _validate_enrollment_token(
        self,
        *,
        request: RemoteWorkerRegistrationRequest,
        token: str | None,
    ) -> bool:
        if not self._settings.secure_registration_required:
            return False
        if self._settings.auth_mode is RemoteWorkerAuthMode.STATIC_TOKEN:
            return self._validate_static_token(token)
        if self._settings.auth_mode is RemoteWorkerAuthMode.SIGNED_ENROLLMENT:
            return self._validate_signed_enrollment_token(request=request, token=token)
        return False

    def _validate_static_token(self, token: str | None) -> bool:
        token_name = self._settings.registration_token_name
        assert token_name is not None
        expected = os.getenv(token_name)
        if not expected:
            msg = f"registration token env var '{token_name}' is not set"
            raise RemoteWorkerRegistrationError(msg)
        if not secrets.compare_digest(token or "", expected):
            self._record_event(
                RemoteWorkerLifecycleEventType.AUTH_REJECTED,
                worker_id="unknown",
                detail="static registration token rejected",
            )
            raise RemoteWorkerRegistrationError("registration token is invalid")
        return True

    def _validate_signed_enrollment_token(
        self,
        *,
        request: RemoteWorkerRegistrationRequest,
        token: str | None,
    ) -> bool:
        secret_name = self._settings.enrollment_secret_name
        assert secret_name is not None
        secret = os.getenv(secret_name)
        if not secret:
            msg = f"enrollment secret env var '{secret_name}' is not set"
            raise RemoteWorkerRegistrationError(msg)
        if token is None:
            raise RemoteWorkerRegistrationError("signed enrollment token is required")
        try:
            version, expires_at_raw, signature = token.split(".", 2)
        except ValueError as exc:
            raise RemoteWorkerRegistrationError("signed enrollment token is malformed") from exc
        if version != "v1":
            raise RemoteWorkerRegistrationError("signed enrollment token version is invalid")
        try:
            expires_at = int(expires_at_raw)
        except ValueError as exc:
            raise RemoteWorkerRegistrationError(
                "signed enrollment token expiry is invalid"
            ) from exc
        now_ts = int(self._now().timestamp())
        if now_ts > expires_at:
            raise RemoteWorkerRegistrationError("signed enrollment token has expired")
        payload = (
            f"{request.worker_id}:{request.worker_name}:{request.endpoint.base_url}:"
            f"{request.model_identifier}:{expires_at}"
        ).encode()
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        if not secrets.compare_digest(signature, expected):
            self._record_event(
                RemoteWorkerLifecycleEventType.AUTH_REJECTED,
                worker_id=request.worker_id,
                detail="signed enrollment token rejected",
            )
            raise RemoteWorkerRegistrationError("signed enrollment token is invalid")
        return True

    def _normalize_lifecycle_state(
        self,
        lifecycle_state: WorkerLifecycleState,
        *,
        ready: bool,
        health: BackendHealth | None,
    ) -> WorkerLifecycleState:
        if lifecycle_state in {WorkerLifecycleState.DRAINING, WorkerLifecycleState.RETIRED}:
            return lifecycle_state
        if health is not None and health.state is BackendHealthState.UNAVAILABLE:
            return WorkerLifecycleState.UNHEALTHY
        if ready:
            return WorkerLifecycleState.READY
        if lifecycle_state is WorkerLifecycleState.REGISTERING:
            return WorkerLifecycleState.REGISTERING
        return WorkerLifecycleState.WARMING

    def _record_event(
        self,
        event_type: RemoteWorkerLifecycleEventType,
        *,
        worker_id: str,
        lifecycle_state: WorkerLifecycleState | None = None,
        detail: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._events.append(
            RemoteWorkerLifecycleEvent(
                event_type=event_type,
                worker_id=worker_id,
                lifecycle_state=lifecycle_state,
                detail=detail,
                metadata=metadata or {},
            )
        )


def build_signed_enrollment_token(
    *,
    request: RemoteWorkerRegistrationRequest,
    secret: str,
    expires_at: datetime,
) -> str:
    """Build a deterministic signed enrollment token for tests and local tooling."""

    expiry = int(expires_at.timestamp())
    payload = (
        f"{request.worker_id}:{request.worker_name}:{request.endpoint.base_url}:"
        f"{request.model_identifier}:{expiry}"
    ).encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"v1.{expiry}.{signature}"
