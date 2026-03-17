"""Deterministic request feature extraction for routing artifacts."""

from __future__ import annotations

import hashlib

from switchyard.schemas.admin import RoutingFeatureRuntimeSummary
from switchyard.schemas.chat import ChatCompletionRequest, ChatRole
from switchyard.schemas.routing import (
    HistoryDepthBucket,
    InputLengthBucket,
    RequestClass,
    RequestContext,
    RequestFeatureVector,
    TenantTier,
    WorkloadTag,
)

FEATURE_VERSION = "phase6.v2"
_DEFAULT_MAX_OUTPUT_TOKENS = 256
_MIN_PREFIX_LENGTH = 24
_MAX_PREFIX_LENGTH = 128


def routing_feature_runtime_summary() -> RoutingFeatureRuntimeSummary:
    """Return the stable runtime-visible feature contract."""

    return RoutingFeatureRuntimeSummary(
        feature_version=FEATURE_VERSION,
        input_length_buckets=list(InputLengthBucket),
        history_depth_buckets=list(HistoryDepthBucket),
        workload_tags=list(WorkloadTag),
        prefix_fingerprint_algorithm="sha256_truncated_16_hex",
        prefix_plaintext_retained=False,
    )


def extract_request_feature_vector(
    request: ChatCompletionRequest,
    context: RequestContext,
) -> RequestFeatureVector:
    """Build a stable request feature vector without backend-specific assumptions."""

    message_contents = [message.content.strip() for message in request.messages if message.content]
    prompt_text = "\n".join(message_contents).strip()
    prompt_token_estimate = sum(len(message.content.split()) for message in request.messages)
    max_output_tokens = request.max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS
    prefix = _extract_prefix_candidate(message_contents)
    prefix_fingerprint = None if prefix is None else _fingerprint(prefix, length=16)
    input_length_bucket = _input_length_bucket(
        prompt_token_estimate=prompt_token_estimate,
        prompt_character_count=len(prompt_text),
    )
    history_depth_bucket = _history_depth_bucket(message_count=len(request.messages))
    conversation_continuation = _conversation_continuation(
        message_count=len(request.messages),
        assistant_message_count=sum(
            message.role is ChatRole.ASSISTANT for message in request.messages
        ),
        session_id=context.session_id,
    )
    workload_tags = _workload_tags(
        input_length_bucket=input_length_bucket,
        request=request,
        context=context,
        repeated_prefix_candidate=prefix is not None and len(prompt_text) > len(prefix),
        conversation_continuation=conversation_continuation,
    )

    return RequestFeatureVector(
        feature_version=FEATURE_VERSION,
        message_count=len(request.messages),
        system_message_count=sum(message.role is ChatRole.SYSTEM for message in request.messages),
        user_message_count=sum(message.role is ChatRole.USER for message in request.messages),
        assistant_message_count=sum(
            message.role is ChatRole.ASSISTANT for message in request.messages
        ),
        tool_message_count=sum(message.role is ChatRole.TOOL for message in request.messages),
        prompt_character_count=len(prompt_text),
        prompt_token_estimate=prompt_token_estimate,
        max_output_tokens=max_output_tokens,
        expected_total_tokens=prompt_token_estimate + max_output_tokens,
        input_length_bucket=input_length_bucket,
        history_depth_bucket=history_depth_bucket,
        workload_tags=workload_tags,
        stream=request.stream,
        request_class=context.request_class,
        tenant_tier=context.tenant_tier,
        internal_backend_pinned=context.internal_backend_pin is not None,
        conversation_continuation=conversation_continuation,
        repeated_prefix_candidate=prefix is not None and len(prompt_text) > len(prefix),
        prefix_character_count=0 if prefix is None else len(prefix),
        prefix_fingerprint=prefix_fingerprint,
        locality_key=_build_locality_key(
            request=request,
            context=context,
            prefix_fingerprint=prefix_fingerprint,
        ),
        session_affinity_expected=context.session_id is not None,
    )


def _extract_prefix_candidate(message_contents: list[str]) -> str | None:
    if not message_contents:
        return None
    leading_line = message_contents[0].splitlines()[0].strip()
    if len(leading_line) < _MIN_PREFIX_LENGTH:
        return None
    return leading_line[:_MAX_PREFIX_LENGTH]


def _input_length_bucket(
    *,
    prompt_token_estimate: int,
    prompt_character_count: int,
) -> InputLengthBucket:
    if prompt_token_estimate <= 16 and prompt_character_count <= 120:
        return InputLengthBucket.TINY
    if prompt_token_estimate <= 64 and prompt_character_count <= 480:
        return InputLengthBucket.SHORT
    if prompt_token_estimate <= 256 and prompt_character_count <= 2_000:
        return InputLengthBucket.MEDIUM
    if prompt_token_estimate <= 1_024 and prompt_character_count <= 8_000:
        return InputLengthBucket.LONG
    return InputLengthBucket.VERY_LONG


def _history_depth_bucket(*, message_count: int) -> HistoryDepthBucket:
    if message_count <= 2:
        return HistoryDepthBucket.SINGLE_TURN
    if message_count <= 6:
        return HistoryDepthBucket.SHORT_HISTORY
    return HistoryDepthBucket.DEEP_HISTORY


def _conversation_continuation(
    *,
    message_count: int,
    assistant_message_count: int,
    session_id: str | None,
) -> bool:
    return session_id is not None or assistant_message_count > 0 or message_count > 2


def _workload_tags(
    *,
    input_length_bucket: InputLengthBucket,
    request: ChatCompletionRequest,
    context: RequestContext,
    repeated_prefix_candidate: bool,
    conversation_continuation: bool,
) -> list[WorkloadTag]:
    tags: list[WorkloadTag] = []

    if input_length_bucket in {InputLengthBucket.TINY, InputLengthBucket.SHORT}:
        tags.append(WorkloadTag.SHORT_CHAT)
    if input_length_bucket in {InputLengthBucket.LONG, InputLengthBucket.VERY_LONG}:
        tags.append(WorkloadTag.LONG_CONTEXT)
    if repeated_prefix_candidate:
        tags.append(WorkloadTag.REPEATED_PREFIX)
    if conversation_continuation:
        tags.append(WorkloadTag.SESSION_CONTINUATION)
    if request.stream:
        tags.append(WorkloadTag.STREAMING)
    if context.request_class is RequestClass.LATENCY_SENSITIVE:
        tags.append(WorkloadTag.LATENCY_SENSITIVE)
    if context.request_class is RequestClass.BULK:
        tags.append(WorkloadTag.BULK)
        tags.append(WorkloadTag.BURST_CANDIDATE)
    if context.tenant_tier is TenantTier.PRIORITY:
        tags.append(WorkloadTag.PRIORITY_TENANT)

    return list(dict.fromkeys(tags))


def _build_locality_key(
    *,
    request: ChatCompletionRequest,
    context: RequestContext,
    prefix_fingerprint: str | None,
) -> str:
    material = "|".join(
        [
            request.model,
            context.workload_shape.value,
            context.request_class.value,
            context.tenant_tier.value,
            "stream" if request.stream else "nonstream",
            prefix_fingerprint or "no-prefix",
            context.session_id or "no-session",
        ]
    )
    return _fingerprint(material, length=20)


def _fingerprint(value: str, *, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
