"""Adapter registry for backend adapters."""

from __future__ import annotations

from builtins import list as builtin_list

from switchyard.adapters.base import BackendAdapter


class AdapterRegistry:
    """In-memory registry of backend adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BackendAdapter] = {}

    def register(self, adapter: BackendAdapter) -> None:
        """Register an adapter by its unique name."""

        if adapter.name in self._adapters:
            msg = f"adapter '{adapter.name}' is already registered"
            raise ValueError(msg)
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> BackendAdapter:
        """Return a registered adapter by name."""

        try:
            return self._adapters[name]
        except KeyError as exc:
            msg = f"adapter '{name}' is not registered"
            raise KeyError(msg) from exc

    def list(self) -> builtin_list[BackendAdapter]:
        """Return adapters in registration order."""

        return builtin_list(self._adapters.values())

    def names(self) -> builtin_list[str]:
        """Return registered adapter names in registration order."""

        return builtin_list(self._adapters)
