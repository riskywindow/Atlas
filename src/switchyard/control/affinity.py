"""In-process session-affinity helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from switchyard.config import SessionAffinitySettings
from switchyard.schemas.admin import SessionAffinityRuntimeSummary
from switchyard.schemas.routing import AffinityDisposition, SessionAffinityKey, StickyRouteRecord


class SessionAffinityClock(Protocol):
    """Clock abstraction for deterministic affinity tests."""

    def now(self) -> datetime: ...


@dataclass(frozen=True, slots=True)
class UtcSessionAffinityClock:
    """Default wall-clock implementation."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class SessionAffinityLookup:
    """Result of looking up a sticky route."""

    disposition: AffinityDisposition
    sticky_route: StickyRouteRecord | None = None
    reason: str | None = None


class SessionAffinityService:
    """Bounded in-memory sticky-route store for multi-turn requests."""

    def __init__(
        self,
        settings: SessionAffinitySettings,
        *,
        clock: SessionAffinityClock | None = None,
    ) -> None:
        self._settings = settings
        self._clock = clock or UtcSessionAffinityClock()
        self._records: OrderedDict[tuple[str, str, str], StickyRouteRecord] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def lookup(self, affinity_key: SessionAffinityKey) -> SessionAffinityLookup:
        """Return the current sticky route when it is still valid."""

        if not self.enabled:
            return SessionAffinityLookup(
                disposition=AffinityDisposition.NOT_REQUESTED,
                reason="session affinity is disabled",
            )
        store_key = _store_key(affinity_key)
        record = self._records.get(store_key)
        if record is None:
            self._prune_expired()
            return SessionAffinityLookup(disposition=AffinityDisposition.MISSED)
        if record.expires_at <= self._clock.now():
            self._records.pop(store_key, None)
            return SessionAffinityLookup(
                disposition=AffinityDisposition.EXPIRED,
                reason=f"sticky route for backend '{record.backend_name}' expired",
            )
        self._records.move_to_end(store_key)
        return SessionAffinityLookup(
            disposition=AffinityDisposition.REUSED,
            sticky_route=record,
        )

    def bind(
        self,
        affinity_key: SessionAffinityKey,
        *,
        backend_name: str,
        reason: str | None = None,
    ) -> StickyRouteRecord:
        """Create or refresh a sticky route with bounded retention."""

        if not self.enabled:
            msg = "session affinity is disabled"
            raise RuntimeError(msg)
        self._prune_expired()
        store_key = _store_key(affinity_key)
        if store_key in self._records:
            self._records.pop(store_key)
        while len(self._records) >= self._settings.max_sessions:
            self._records.popitem(last=False)
        bound_at = self._clock.now()
        record = StickyRouteRecord(
            affinity_key=affinity_key,
            backend_name=backend_name,
            bound_at=bound_at,
            expires_at=bound_at + timedelta(seconds=self._settings.ttl_seconds),
            reason=reason,
        )
        self._records[store_key] = record
        return record

    def __len__(self) -> int:
        self._prune_expired()
        return len(self._records)

    def inspect_state(self) -> SessionAffinityRuntimeSummary:
        """Return a bounded runtime summary for the sticky-route cache."""

        self._prune_expired()
        bindings_by_target: dict[str, int] = {}
        for record in self._records.values():
            serving_target = record.affinity_key.serving_target
            bindings_by_target[serving_target] = bindings_by_target.get(serving_target, 0) + 1
        return SessionAffinityRuntimeSummary(
            enabled=self.enabled,
            ttl_seconds=self._settings.ttl_seconds,
            max_sessions=self._settings.max_sessions,
            active_bindings=len(self._records),
            bindings_by_target=bindings_by_target,
        )

    def _prune_expired(self) -> None:
        if not self._records:
            return
        now = self._clock.now()
        expired_keys = [
            store_key
            for store_key, record in self._records.items()
            if record.expires_at <= now
        ]
        for store_key in expired_keys:
            self._records.pop(store_key, None)


def _store_key(affinity_key: SessionAffinityKey) -> tuple[str, str, str]:
    return (
        affinity_key.tenant_id,
        affinity_key.session_id,
        affinity_key.serving_target,
    )
