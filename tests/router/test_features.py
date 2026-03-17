from __future__ import annotations

from switchyard.router.features import extract_request_feature_vector
from switchyard.schemas.chat import ChatCompletionRequest, ChatMessage, ChatRole
from switchyard.schemas.routing import RequestClass, RequestContext, WorkloadShape


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
