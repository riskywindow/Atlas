"""Runtime rollout controls for canary-only cloud backends."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from switchyard.config import CloudTrafficRolloutSettings
from switchyard.schemas.admin import (
    CloudRolloutDecisionRuntimeEntry,
    CloudRolloutRuntimeSummary,
    CloudRolloutUpdateRequest,
)
from switchyard.schemas.routing import RequestContext


@dataclass(frozen=True, slots=True)
class CloudRolloutDecision:
    """Outcome of evaluating one canary-only cloud backend for a request."""

    allowed: bool
    disposition: str
    reason: str | None = None
    bucket_percentage: float | None = None


class CloudTrafficRolloutService:
    """Bounded runtime controller for real cloud traffic rollout."""

    def __init__(self, settings: CloudTrafficRolloutSettings) -> None:
        self._enabled = settings.enabled
        self._canary_percentage = settings.canary_percentage
        self._kill_switch_enabled = settings.kill_switch_enabled
        self._auto_quarantine_failure_threshold = settings.auto_quarantine_failure_threshold
        self._recent_decisions: deque[CloudRolloutDecisionRuntimeEntry] = deque(
            maxlen=settings.max_recent_decisions
        )
        self._consecutive_failures_by_backend: dict[str, int] = {}
        self._last_updated_at: datetime | None = None

    def evaluate_canary_only_candidate(
        self,
        *,
        context: RequestContext,
        serving_target: str,
        backend_name: str,
        explicitly_selected_backend: str | None = None,
    ) -> CloudRolloutDecision:
        """Gate a canary-only backend behind explicit rollout controls."""

        if context.internal_backend_pin == backend_name:
            return self._record(
                context=context,
                serving_target=serving_target,
                backend_name=backend_name,
                allowed=True,
                disposition="explicit_pin",
                reason="explicit backend pin bypassed cloud rollout gating",
            )
        if self._kill_switch_enabled:
            return self._record(
                context=context,
                serving_target=serving_target,
                backend_name=backend_name,
                allowed=False,
                disposition="kill_switch",
                reason="cloud rollout kill switch blocked canary-only backend",
            )
        if not self._enabled or self._canary_percentage <= 0.0:
            return self._record(
                context=context,
                serving_target=serving_target,
                backend_name=backend_name,
                allowed=False,
                disposition="disabled",
                reason="cloud rollout is disabled for canary-only backend",
            )
        if explicitly_selected_backend == backend_name:
            return self._record(
                context=context,
                serving_target=serving_target,
                backend_name=backend_name,
                allowed=True,
                disposition="explicit_canary",
                reason="backend remained eligible because deterministic canary selected it",
            )
        bucket_percentage = _bucket_percentage(
            serving_target=serving_target,
            backend_name=backend_name,
            bucket_key=context.session_id or context.request_id,
        )
        allowed = bucket_percentage < self._canary_percentage
        return self._record(
            context=context,
            serving_target=serving_target,
            backend_name=backend_name,
            allowed=allowed,
            disposition="selected" if allowed else "blocked",
            reason=(
                "cloud rollout selected canary-only backend"
                if allowed
                else "cloud rollout kept canary-only backend out of primary traffic"
            ),
            bucket_percentage=bucket_percentage,
        )

    def inspect_state(self) -> CloudRolloutRuntimeSummary:
        """Return current rollout posture plus bounded recent decisions."""

        recent_decisions = list(self._recent_decisions)
        return CloudRolloutRuntimeSummary(
            enabled=self._enabled,
            canary_percentage=self._canary_percentage,
            kill_switch_enabled=self._kill_switch_enabled,
            auto_quarantine_failure_threshold=self._auto_quarantine_failure_threshold,
            recent_decisions=recent_decisions,
            recent_allowed_count=sum(1 for entry in recent_decisions if entry.allowed),
            recent_blocked_count=sum(1 for entry in recent_decisions if not entry.allowed),
            consecutive_failures_by_backend=dict(
                sorted(self._consecutive_failures_by_backend.items())
            ),
            last_updated_at=self._last_updated_at,
            notes=_notes(
                enabled=self._enabled,
                canary_percentage=self._canary_percentage,
                kill_switch_enabled=self._kill_switch_enabled,
                auto_quarantine_failure_threshold=self._auto_quarantine_failure_threshold,
            ),
        )

    def update(self, request: CloudRolloutUpdateRequest) -> CloudRolloutRuntimeSummary:
        """Mutate runtime rollout posture."""

        if request.enabled is not None:
            self._enabled = request.enabled
        if request.canary_percentage is not None:
            self._canary_percentage = request.canary_percentage
        if request.kill_switch_enabled is not None:
            self._kill_switch_enabled = request.kill_switch_enabled
        self._last_updated_at = datetime.now(UTC)
        return self.inspect_state()

    def record_backend_success(self, backend_name: str) -> None:
        """Clear tracked remote-transport failure state after a success."""

        self._consecutive_failures_by_backend.pop(backend_name, None)

    def record_backend_failure(self, backend_name: str) -> bool:
        """Track repeated remote-transport failures and signal quarantine thresholds."""

        next_count = self._consecutive_failures_by_backend.get(backend_name, 0) + 1
        self._consecutive_failures_by_backend[backend_name] = next_count
        threshold = self._auto_quarantine_failure_threshold
        return threshold is not None and next_count >= threshold

    def _record(
        self,
        *,
        context: RequestContext,
        serving_target: str,
        backend_name: str,
        allowed: bool,
        disposition: str,
        reason: str | None,
        bucket_percentage: float | None = None,
    ) -> CloudRolloutDecision:
        self._recent_decisions.appendleft(
            CloudRolloutDecisionRuntimeEntry(
                request_id=context.request_id,
                serving_target=serving_target,
                backend_name=backend_name,
                allowed=allowed,
                disposition=disposition,
                bucket_percentage=(
                    None if bucket_percentage is None else round(bucket_percentage, 3)
                ),
                notes=[] if reason is None else [reason],
            )
        )
        return CloudRolloutDecision(
            allowed=allowed,
            disposition=disposition,
            reason=reason,
            bucket_percentage=bucket_percentage,
        )


def _bucket_percentage(
    *,
    serving_target: str,
    backend_name: str,
    bucket_key: str,
) -> float:
    digest = sha256(f"{serving_target}:{backend_name}:{bucket_key}".encode()).digest()
    scaled = int.from_bytes(digest[:8], "big") / float(2**64)
    return scaled * 100.0


def _notes(
    *,
    enabled: bool,
    canary_percentage: float,
    kill_switch_enabled: bool,
    auto_quarantine_failure_threshold: int | None,
) -> list[str]:
    notes: list[str] = []
    if kill_switch_enabled:
        notes.append("cloud rollout kill switch is active")
    elif not enabled or canary_percentage <= 0.0:
        notes.append("canary-only cloud backends stay blocked from primary traffic")
    else:
        notes.append("canary-only cloud backends are gated by deterministic request bucketing")
    if auto_quarantine_failure_threshold is not None:
        notes.append(
            "registered cloud workers auto-quarantine after repeated transport failures"
        )
    return notes
