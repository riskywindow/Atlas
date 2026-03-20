"""Alias-scoped operator overrides for remote routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from switchyard.schemas.admin import AliasRoutingOverrideState


@dataclass(slots=True)
class AliasRoutingOverrideService:
    """Small in-memory store for alias-scoped backend pins and disables."""

    _overrides: dict[str, AliasRoutingOverrideState] = field(default_factory=dict)

    def get(self, serving_target: str) -> AliasRoutingOverrideState | None:
        return self._overrides.get(serving_target)

    def list_overrides(self) -> list[AliasRoutingOverrideState]:
        return [self._overrides[key] for key in sorted(self._overrides)]

    def update(
        self,
        *,
        serving_target: str,
        pinned_backend: str | None,
        disabled_backends: list[str],
        reason: str | None,
    ) -> AliasRoutingOverrideState:
        disabled: list[str] = sorted(set(disabled_backends))
        if pinned_backend is not None and pinned_backend in disabled:
            msg = "pinned_backend must not also appear in disabled_backends"
            raise ValueError(msg)
        state = AliasRoutingOverrideState(
            serving_target=serving_target,
            pinned_backend=pinned_backend,
            disabled_backends=disabled,
            updated_at=datetime.now(UTC),
            reason=reason,
        )
        self._overrides[serving_target] = state
        return state

    def clear(self, serving_target: str) -> None:
        self._overrides.pop(serving_target, None)
