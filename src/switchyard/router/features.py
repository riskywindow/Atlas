"""Deterministic request feature extraction for routing artifacts."""

from __future__ import annotations

import hashlib

from switchyard.schemas.chat import ChatCompletionRequest, ChatRole
from switchyard.schemas.routing import RequestContext, RequestFeatureVector

_DEFAULT_MAX_OUTPUT_TOKENS = 256
_MIN_PREFIX_LENGTH = 24
_MAX_PREFIX_LENGTH = 128


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

    return RequestFeatureVector(
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
        stream=request.stream,
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
