import pytest

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry


def test_registry_registers_and_lists_adapters() -> None:
    registry = AdapterRegistry()
    first = MockBackendAdapter(name="mock-a")
    second = MockBackendAdapter(name="mock-b")

    registry.register(first)
    registry.register(second)

    assert registry.get("mock-a") is first
    assert registry.names() == ["mock-a", "mock-b"]
    assert registry.list() == [first, second]


def test_registry_rejects_duplicate_names() -> None:
    registry = AdapterRegistry()
    registry.register(MockBackendAdapter(name="mock-a"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(MockBackendAdapter(name="mock-a"))


def test_registry_raises_for_missing_adapter() -> None:
    registry = AdapterRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")
