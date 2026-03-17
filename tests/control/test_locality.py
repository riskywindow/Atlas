from datetime import UTC, datetime, timedelta

from switchyard.control.locality import PrefixLocalityService
from switchyard.schemas.routing import PrefixHotness, RequestFeatureVector


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def build_features(*, fingerprint: str, locality_key: str) -> RequestFeatureVector:
    return RequestFeatureVector(
        message_count=1,
        user_message_count=1,
        prompt_character_count=64,
        prompt_token_estimate=10,
        max_output_tokens=128,
        expected_total_tokens=138,
        repeated_prefix_candidate=True,
        prefix_character_count=32,
        prefix_fingerprint=fingerprint,
        locality_key=locality_key,
    )


def test_prefix_locality_expires_after_ttl() -> None:
    clock = FakeClock()
    service = PrefixLocalityService(ttl_seconds=5.0, max_prefixes=4, clock=clock)
    features = build_features(fingerprint="feedfacecafebeef", locality_key="loc-0001")

    service.observe_request(serving_target="chat-default", request_features=features)
    warm = service.inspect(
        serving_target="chat-default",
        request_features=features,
        candidate_backends=["mock-a"],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )
    clock.advance(6.0)
    cold = service.inspect(
        serving_target="chat-default",
        request_features=features,
        candidate_backends=["mock-a"],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )

    assert warm.repeated_prefix_detected is True
    assert warm.hotness is PrefixHotness.WARM
    assert cold.repeated_prefix_detected is False
    assert cold.hotness is PrefixHotness.COLD


def test_prefix_locality_evicts_oldest_prefix_when_capacity_is_reached() -> None:
    clock = FakeClock()
    service = PrefixLocalityService(ttl_seconds=60.0, max_prefixes=2, clock=clock)

    service.observe_request(
        serving_target="chat-default",
        request_features=build_features(fingerprint="prefix-1", locality_key="loc-0001"),
    )
    clock.advance(1.0)
    service.observe_request(
        serving_target="chat-default",
        request_features=build_features(fingerprint="prefix-2", locality_key="loc-0002"),
    )
    clock.advance(1.0)
    service.observe_request(
        serving_target="chat-default",
        request_features=build_features(fingerprint="prefix-3", locality_key="loc-0003"),
    )

    first = service.inspect(
        serving_target="chat-default",
        request_features=build_features(fingerprint="prefix-1", locality_key="loc-0001"),
        candidate_backends=[],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )
    latest = service.inspect(
        serving_target="chat-default",
        request_features=build_features(fingerprint="prefix-3", locality_key="loc-0003"),
        candidate_backends=[],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )

    assert first.repeated_prefix_detected is False
    assert latest.repeated_prefix_detected is True
    assert service.inspect_state().active_prefixes == 2


def test_prefix_locality_scopes_collision_by_locality_key() -> None:
    service = PrefixLocalityService(ttl_seconds=60.0, max_prefixes=4)

    service.observe_request(
        serving_target="chat-default",
        request_features=build_features(fingerprint="same-digest", locality_key="loc-000a"),
    )
    signal = service.inspect(
        serving_target="chat-default",
        request_features=build_features(fingerprint="same-digest", locality_key="loc-000b"),
        candidate_backends=[],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )

    assert signal.repeated_prefix_detected is False
    assert service.inspect_state().collision_scope == (
        "serving_target+locality_key+prefix_fingerprint"
    )


def test_prefix_locality_signal_is_deterministic_and_tracks_warm_backend() -> None:
    service = PrefixLocalityService(ttl_seconds=60.0, max_prefixes=4)
    features = build_features(fingerprint="warm-digest", locality_key="loc-warm")

    service.observe_request(serving_target="chat-default", request_features=features)
    service.observe_execution(
        serving_target="chat-default",
        request_features=features,
        backend_name="mock-a",
        backend_instance_id="instance-a",
    )
    first = service.inspect(
        serving_target="chat-default",
        request_features=features,
        candidate_backends=["mock-a", "mock-b"],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )
    second = service.inspect(
        serving_target="chat-default",
        request_features=features,
        candidate_backends=["mock-a", "mock-b"],
        sticky_backend_name=None,
        session_affinity_enabled=False,
    )

    assert first == second
    assert first.likely_benefits_from_locality is True
    assert first.candidate_local_backend == "mock-a"
    assert first.preferred_instance_id == "instance-a"
