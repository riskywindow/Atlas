from switchyard import __version__
from switchyard.adapters import AdapterRegistry, MockBackendAdapter
from switchyard.config import Settings
from switchyard.schemas import (
    BackendType,
    BenchmarkRunArtifact,
    ChatCompletionRequest,
    RoutingPolicy,
)


def test_package_smoke() -> None:
    settings = Settings()

    assert __version__ == "0.1.0"
    assert settings.service_name == "switchyard-gateway"
    assert ChatCompletionRequest.__name__ == "ChatCompletionRequest"
    assert RoutingPolicy.BALANCED.value == "balanced"
    assert BackendType.MOCK.value == "mock"
    assert BenchmarkRunArtifact.__name__ == "BenchmarkRunArtifact"
    assert AdapterRegistry.__name__ == "AdapterRegistry"
    assert MockBackendAdapter.__name__ == "MockBackendAdapter"
