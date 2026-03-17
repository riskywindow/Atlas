"""Bounded in-process prefix-locality tracking for routing signals."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from switchyard.schemas.admin import (
    PrefixLocalityRuntimeSummary,
    TrackedPrefixRuntimeSummary,
)
from switchyard.schemas.routing import PrefixHotness, PrefixLocalitySignal, RequestFeatureVector

_SIGNAL_VERSION = "phase6.v1"
_FINGERPRINT_ALGORITHM = "sha256_truncated_16_hex"
_COLLISION_SCOPE = "serving_target+locality_key+prefix_fingerprint"


class PrefixLocalityClock(Protocol):
    """Clock abstraction for deterministic locality tests."""

    def now(self) -> datetime: ...


@dataclass(frozen=True, slots=True)
class UtcPrefixLocalityClock:
    """Default wall-clock implementation."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(slots=True)
class _PrefixEntry:
    serving_target: str
    locality_key: str
    prefix_fingerprint: str
    created_at: datetime
    last_seen_at: datetime
    request_count: int = 0
    backend_counts: OrderedDict[str, int] = field(default_factory=OrderedDict)
    instance_counts: OrderedDict[str, int] = field(default_factory=OrderedDict)


class PrefixLocalityService:
    """Bounded recent-prefix tracker for explainable routing signals."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_prefixes: int = 256,
        max_backends_per_prefix: int = 8,
        max_instances_per_prefix: int = 16,
        clock: PrefixLocalityClock | None = None,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_prefixes = max_prefixes
        self._max_backends_per_prefix = max_backends_per_prefix
        self._max_instances_per_prefix = max_instances_per_prefix
        self._clock = clock or UtcPrefixLocalityClock()
        self._entries: OrderedDict[tuple[str, str, str], _PrefixEntry] = OrderedDict()

    def inspect(
        self,
        *,
        serving_target: str,
        request_features: RequestFeatureVector,
        candidate_backends: list[str],
        sticky_backend_name: str | None,
        session_affinity_enabled: bool,
    ) -> PrefixLocalitySignal:
        """Return a decision-time locality summary without mutating tracker state."""

        self._prune_expired()
        if request_features.prefix_fingerprint is None:
            return PrefixLocalitySignal(
                signal_version=_SIGNAL_VERSION,
                serving_target=serving_target,
                locality_key=request_features.locality_key,
                session_affinity_enabled=session_affinity_enabled,
                session_affinity_backend=sticky_backend_name,
            )

        entry = self._entries.get(
            (
                serving_target,
                request_features.locality_key,
                request_features.prefix_fingerprint,
            )
        )
        if entry is not None:
            self._entries.move_to_end(
                (
                    serving_target,
                    request_features.locality_key,
                    request_features.prefix_fingerprint,
                )
            )

        recent_request_count = 0 if entry is None else entry.request_count
        recent_backend_counts = _sorted_counts(
            {} if entry is None else dict(entry.backend_counts)
        )
        recent_instance_counts = _sorted_counts(
            {} if entry is None else dict(entry.instance_counts)
        )
        preferred_backend, preferred_backend_request_count = _top_count(recent_backend_counts)
        preferred_instance_id, preferred_instance_request_count = _top_count(
            recent_instance_counts
        )
        candidate_local_backend, candidate_local_backend_request_count = _top_candidate(
            recent_backend_counts,
            candidates=candidate_backends,
        )
        affinity_conflict = (
            session_affinity_enabled
            and sticky_backend_name is not None
            and candidate_local_backend is not None
            and sticky_backend_name != candidate_local_backend
        )
        repeated_prefix_detected = recent_request_count > 0

        return PrefixLocalitySignal(
            signal_version=_SIGNAL_VERSION,
            serving_target=serving_target,
            locality_key=request_features.locality_key,
            prefix_fingerprint=request_features.prefix_fingerprint,
            repeated_prefix_detected=repeated_prefix_detected,
            recent_request_count=recent_request_count,
            hotness=_hotness(recent_request_count),
            cache_opportunity=(
                repeated_prefix_detected and request_features.repeated_prefix_candidate
            ),
            likely_benefits_from_locality=candidate_local_backend_request_count > 0,
            preferred_backend=preferred_backend,
            preferred_backend_request_count=preferred_backend_request_count,
            preferred_instance_id=preferred_instance_id,
            preferred_instance_request_count=preferred_instance_request_count,
            candidate_local_backend=candidate_local_backend,
            candidate_local_backend_request_count=candidate_local_backend_request_count,
            recent_backend_counts=recent_backend_counts,
            recent_instance_counts=recent_instance_counts,
            session_affinity_enabled=session_affinity_enabled,
            session_affinity_backend=sticky_backend_name,
            affinity_conflict=affinity_conflict,
            last_seen_at=None if entry is None else entry.last_seen_at,
        )

    def observe_request(
        self,
        *,
        serving_target: str,
        request_features: RequestFeatureVector,
    ) -> None:
        """Track one routed request for repeated-prefix detection."""

        if request_features.prefix_fingerprint is None:
            return
        self._prune_expired()
        key = (
            serving_target,
            request_features.locality_key,
            request_features.prefix_fingerprint,
        )
        now = self._clock.now()
        entry = self._entries.get(key)
        if entry is None:
            while len(self._entries) >= self._max_prefixes:
                self._entries.popitem(last=False)
            entry = _PrefixEntry(
                serving_target=serving_target,
                locality_key=request_features.locality_key,
                prefix_fingerprint=request_features.prefix_fingerprint,
                created_at=now,
                last_seen_at=now,
            )
            self._entries[key] = entry
        else:
            self._entries.move_to_end(key)
            entry.last_seen_at = now
        entry.request_count += 1

    def observe_execution(
        self,
        *,
        serving_target: str,
        request_features: RequestFeatureVector,
        backend_name: str | None,
        backend_instance_id: str | None,
    ) -> None:
        """Track successful execution locality for future cache-aware routing."""

        if request_features.prefix_fingerprint is None or backend_name is None:
            return
        self._prune_expired()
        key = (
            serving_target,
            request_features.locality_key,
            request_features.prefix_fingerprint,
        )
        entry = self._entries.get(key)
        if entry is None:
            now = self._clock.now()
            while len(self._entries) >= self._max_prefixes:
                self._entries.popitem(last=False)
            entry = _PrefixEntry(
                serving_target=serving_target,
                locality_key=request_features.locality_key,
                prefix_fingerprint=request_features.prefix_fingerprint,
                created_at=now,
                last_seen_at=now,
                request_count=1,
            )
            self._entries[key] = entry
        else:
            self._entries.move_to_end(key)
        _increment_bounded(
            entry.backend_counts,
            backend_name,
            limit=self._max_backends_per_prefix,
        )
        if backend_instance_id is not None:
            _increment_bounded(
                entry.instance_counts,
                backend_instance_id,
                limit=self._max_instances_per_prefix,
            )

    def inspect_state(self) -> PrefixLocalityRuntimeSummary:
        """Return a bounded runtime summary for diagnostics and artifact capture."""

        self._prune_expired()
        by_target: dict[str, int] = {}
        hot_prefixes: list[TrackedPrefixRuntimeSummary] = []
        for entry in self._entries.values():
            by_target[entry.serving_target] = by_target.get(entry.serving_target, 0) + 1
            if _hotness(entry.request_count) is PrefixHotness.COLD:
                continue
            preferred_backend, _ = _top_count(dict(entry.backend_counts))
            preferred_instance_id, _ = _top_count(dict(entry.instance_counts))
            hot_prefixes.append(
                TrackedPrefixRuntimeSummary(
                    serving_target=entry.serving_target,
                    locality_key=entry.locality_key,
                    prefix_fingerprint=entry.prefix_fingerprint,
                    recent_request_count=entry.request_count,
                    hotness=_hotness(entry.request_count),
                    preferred_backend=preferred_backend,
                    preferred_instance_id=preferred_instance_id,
                    last_seen_at=entry.last_seen_at,
                )
            )
        hot_prefixes = sorted(
            hot_prefixes,
            key=lambda item: (
                -item.recent_request_count,
                item.serving_target,
                item.prefix_fingerprint,
            ),
        )[:10]
        return PrefixLocalityRuntimeSummary(
            enabled=True,
            ttl_seconds=self._ttl_seconds,
            max_prefixes=self._max_prefixes,
            active_prefixes=len(self._entries),
            hot_prefixes=len(hot_prefixes),
            tracked_serving_targets=by_target,
            hottest_prefixes=hot_prefixes,
            prefix_fingerprint_algorithm=_FINGERPRINT_ALGORITHM,
            prefix_plaintext_retained=False,
            collision_scope=_COLLISION_SCOPE,
        )

    def _prune_expired(self) -> None:
        if not self._entries:
            return
        cutoff = self._clock.now() - timedelta(seconds=self._ttl_seconds)
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.last_seen_at <= cutoff
        ]
        for key in expired:
            self._entries.pop(key, None)


def _hotness(request_count: int) -> PrefixHotness:
    if request_count >= 3:
        return PrefixHotness.HOT
    if request_count >= 1:
        return PrefixHotness.WARM
    return PrefixHotness.COLD


def _sorted_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_count(counts: dict[str, int]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    name, count = next(iter(_sorted_counts(counts).items()))
    return name, count


def _top_candidate(
    counts: dict[str, int],
    *,
    candidates: list[str],
) -> tuple[str | None, int]:
    filtered = {name: count for name, count in counts.items() if name in set(candidates)}
    return _top_count(filtered)


def _increment_bounded(
    counts: OrderedDict[str, int],
    key: str,
    *,
    limit: int,
) -> None:
    current = counts.pop(key, 0) + 1
    counts[key] = current
    if len(counts) <= limit:
        return
    lowest_key = min(counts.items(), key=lambda item: (item[1], item[0]))[0]
    counts.pop(lowest_key, None)
