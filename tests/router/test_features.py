from __future__ import annotations

from switchyard.router.features import (
    extract_request_feature_vector,
    routing_feature_runtime_summary,
)
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import (
    HistoryDepthBucket,
    InputLengthBucket,
    RequestClass,
    RequestContext,
    TenantTier,
    WorkloadShape,
    WorkloadTag,
)


def test_request_feature_vector_is_deterministic() -> None:
    request = ChatCompletionRequest(
        model="chat-shared",
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    "Shared context: route health data should remain backend-agnostic.\n"
                    "Explain how to preserve cache locality."
                ),
            )
        ],
        max_output_tokens=128,
    )
    context = RequestContext(
        request_id="req-features-1",
        workload_shape=WorkloadShape.INTERACTIVE,
        request_class=RequestClass.LATENCY_SENSITIVE,
        tenant_id="tenant-a",
        session_id="session-a",
    )

    first = extract_request_feature_vector(request, context)
    second = extract_request_feature_vector(request, context)

    assert first == second
    assert first.repeated_prefix_candidate is True
    assert first.prefix_fingerprint is not None
    assert first.session_affinity_expected is True
    assert first.expected_total_tokens == first.prompt_token_estimate + 128
    assert first.feature_version == "phase6.v2"
    assert first.input_length_bucket is InputLengthBucket.TINY
    assert first.history_depth_bucket is HistoryDepthBucket.SINGLE_TURN
    assert WorkloadTag.REPEATED_PREFIX in first.workload_tags
    assert WorkloadTag.LATENCY_SENSITIVE in first.workload_tags
    assert WorkloadTag.SESSION_CONTINUATION in first.workload_tags


def test_request_feature_vector_changes_locality_key_when_prefix_changes() -> None:
    first_request = ChatCompletionRequest(
        model="chat-shared",
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    "Shared context: customer tier is gold and retry budget is strict.\n"
                    "Summarize the routing tradeoff."
                ),
            )
        ],
    )
    second_request = ChatCompletionRequest(
        model="chat-shared",
        messages=[
            ChatMessage(
                role=ChatRole.USER,
                content=(
                    "Shared context: compare policies without changing request shape.\n"
                    "Summarize the routing tradeoff."
                ),
            )
        ],
    )
    context = RequestContext(request_id="req-features-2")

    first = extract_request_feature_vector(first_request, context)
    second = extract_request_feature_vector(second_request, context)

    assert first.prefix_fingerprint != second.prefix_fingerprint
    assert first.locality_key != second.locality_key
    assert "Shared context" not in (first.prefix_fingerprint or "")


def test_request_feature_vector_classifies_long_context_and_burst_candidates() -> None:
    request = ChatCompletionRequest(
        model="chat-shared",
        stream=True,
        messages=[
            ChatMessage(role=ChatRole.SYSTEM, content="You are a concise assistant."),
            ChatMessage(role=ChatRole.USER, content="context " * 400),
            ChatMessage(role=ChatRole.ASSISTANT, content="previous answer"),
            ChatMessage(role=ChatRole.USER, content="continue"),
        ],
        max_output_tokens=512,
    )
    context = RequestContext(
        request_id="req-features-3",
        workload_shape=WorkloadShape.BATCH,
        request_class=RequestClass.BULK,
        tenant_tier=TenantTier.PRIORITY,
        session_id="session-burst",
    )

    features = extract_request_feature_vector(request, context)

    assert features.input_length_bucket in {
        InputLengthBucket.LONG,
        InputLengthBucket.VERY_LONG,
    }
    assert features.history_depth_bucket is HistoryDepthBucket.SHORT_HISTORY
    assert features.conversation_continuation is True
    assert WorkloadTag.LONG_CONTEXT in features.workload_tags
    assert WorkloadTag.BURST_CANDIDATE in features.workload_tags
    assert WorkloadTag.BULK in features.workload_tags
    assert WorkloadTag.STREAMING in features.workload_tags
    assert WorkloadTag.PRIORITY_TENANT in features.workload_tags


def test_routing_feature_runtime_summary_is_stable() -> None:
    summary = routing_feature_runtime_summary()

    assert summary.feature_version == "phase6.v2"
    assert summary.prefix_plaintext_retained is False
    assert summary.prefix_fingerprint_algorithm == "sha256_truncated_16_hex"
    assert WorkloadTag.SHORT_CHAT in summary.workload_tags
