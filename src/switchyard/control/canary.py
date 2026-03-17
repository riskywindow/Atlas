"""Deterministic local canary-routing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from switchyard.config import CanaryRoutingSettings
from switchyard.schemas.admin import CanaryRoutingRuntimeSummary
from switchyard.schemas.routing import (
    CanaryPolicy,
    RequestContext,
    RolloutDisposition,
)


@dataclass(frozen=True, slots=True)
class CanarySelection:
    """Outcome of evaluating one canary policy for a request."""

    policy: CanaryPolicy
    disposition: RolloutDisposition
    selected_backend: str | None = None
    reason: str | None = None


class CanaryRoutingService:
    """Evaluate deterministic weighted rollout rules for a serving target."""

    def __init__(self, settings: CanaryRoutingSettings) -> None:
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def match_policy(self, *, serving_target: str) -> CanaryPolicy | None:
        if not self.enabled:
            return None
        for policy in self._settings.policies:
            if policy.enabled and policy.serving_target == serving_target:
                return policy
        return None

    def select(
        self,
        *,
        context: RequestContext,
        policy: CanaryPolicy,
    ) -> CanarySelection:
        """Select a rollout disposition and candidate backend for a request."""

        bucket_key = context.session_id or context.request_id
        bucket = _bucket_percentage(policy_name=policy.policy_name, bucket_key=bucket_key)
        cumulative = 0.0
        for allocation in policy.allocations:
            cumulative += allocation.percentage
            if bucket < cumulative:
                return CanarySelection(
                    policy=policy,
                    disposition=RolloutDisposition.CANARY,
                    selected_backend=allocation.backend_name,
                    reason=(
                        f"canary policy '{policy.policy_name}' selected backend "
                        f"'{allocation.backend_name}' at bucket={bucket:.3f}"
                    ),
                )

        return CanarySelection(
            policy=policy,
            disposition=RolloutDisposition.BASELINE,
            selected_backend=policy.baseline_backend,
            reason=f"canary policy '{policy.policy_name}' kept baseline at bucket={bucket:.3f}",
        )

    def inspect_state(self) -> CanaryRoutingRuntimeSummary:
        """Return active canary configuration for admin inspection."""

        return CanaryRoutingRuntimeSummary(
            enabled=self.enabled,
            default_percentage=self._settings.default_percentage,
            policies=list(self._settings.policies),
        )


def _bucket_percentage(*, policy_name: str, bucket_key: str) -> float:
    digest = sha256(f"{policy_name}:{bucket_key}".encode()).digest()
    scaled = int.from_bytes(digest[:8], "big") / float(2**64)
    return scaled * 100.0
