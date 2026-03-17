"""Optional request trace capture for gateway benchmarking and replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from switchyard.schemas.benchmark import (
    CapturedTraceRecord,
    ControlPlaneReportMetadata,
    ExecutionTarget,
    TraceCaptureMode,
)
from switchyard.schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from switchyard.schemas.routing import AdmissionDecision, RequestContext, RouteDecision
from switchyard.telemetry import estimate_token_count


class TraceCaptureSink(Protocol):
    """Pluggable destination for captured request traces."""

    async def write(self, record: CapturedTraceRecord) -> None: ...


@dataclass(slots=True)
class NullTraceCaptureSink:
    """No-op sink used when capture is disabled."""

    async def write(self, record: CapturedTraceRecord) -> None:
        return None


@dataclass(slots=True)
class JsonlTraceCaptureSink:
    """Append-only JSONL trace sink for local benchmarking."""

    output_path: Path

    async def write(self, record: CapturedTraceRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record.model_dump(mode="json", exclude_none=True),
                    sort_keys=True,
                )
            )
            handle.write("\n")


@dataclass(slots=True)
class TraceCaptureService:
    """Trace capture policy and normalization helpers."""

    mode: TraceCaptureMode
    sink: TraceCaptureSink

    @property
    def enabled(self) -> bool:
        return self.mode is not TraceCaptureMode.OFF

    async def capture_chat_completion(
        self,
        *,
        request_timestamp: datetime,
        request_id: str,
        logical_alias: str,
        execution_target: ExecutionTarget,
        request_context: RequestContext,
        route_decision: RouteDecision | None,
        chosen_backend: str | None,
        stream: bool,
        status_code: int,
        latency_ms: float,
        ttft_ms: float | None,
        output_tokens: int | None,
        fallback_used: bool,
        error: str | None,
        error_category: str | None,
        request_payload: ChatCompletionRequest,
        response_payload: ChatCompletionResponse | str | None,
        admission_decision: AdmissionDecision | None = None,
    ) -> None:
        if not self.enabled:
            return

        record = CapturedTraceRecord(
            record_id=f"trace_{request_id}",
            request_timestamp=request_timestamp,
            captured_at=datetime.now(UTC),
            request_id=request_id,
            execution_target=execution_target,
            logical_alias=logical_alias,
            tenant_id=request_context.tenant_id,
            request_class=request_context.request_class,
            session_id=request_context.session_id,
            route_decision=route_decision,
            chosen_backend=chosen_backend,
            stream=stream,
            fallback_used=fallback_used,
            status_code=status_code,
            latency_ms=round(latency_ms, 3),
            ttft_ms=None if ttft_ms is None else round(ttft_ms, 3),
            output_tokens=output_tokens,
            error=error,
            error_category=error_category,
            capture_mode=self.mode,
            control_plane_metadata=None,
            normalized_request_payload=_normalize_request_payload(request_payload, self.mode),
            normalized_response_payload=_normalize_response_payload(response_payload, self.mode),
            metadata={"tenant_tier": request_context.tenant_tier.value},
        )
        if route_decision is not None and route_decision.telemetry_metadata is not None:
            record.control_plane_metadata = ControlPlaneReportMetadata(
                tenant_id=request_context.tenant_id,
                session_id=request_context.session_id,
                admission_decision=route_decision.admission_decision,
                circuit_breaker_state=route_decision.circuit_breaker_state,
                sticky_route=route_decision.sticky_route,
                canary_policy=route_decision.canary_policy,
                shadow_policy=route_decision.shadow_policy,
                telemetry_metadata=route_decision.telemetry_metadata,
            )
        elif admission_decision is not None:
            record.control_plane_metadata = ControlPlaneReportMetadata(
                tenant_id=request_context.tenant_id,
                session_id=request_context.session_id,
                admission_decision=admission_decision,
            )
        await self.sink.write(record)


def build_trace_capture_service(
    *,
    mode: TraceCaptureMode,
    output_path: Path,
) -> TraceCaptureService:
    """Build a trace capture service from runtime settings."""

    if mode is TraceCaptureMode.OFF:
        return TraceCaptureService(mode=mode, sink=NullTraceCaptureSink())
    return TraceCaptureService(mode=mode, sink=JsonlTraceCaptureSink(output_path=output_path))


def _normalize_request_payload(
    request: ChatCompletionRequest,
    mode: TraceCaptureMode,
) -> dict[str, object]:
    if mode is TraceCaptureMode.METADATA_ONLY:
        return {
            "model": request.model,
            "message_count": len(request.messages),
            "roles": [message.role.value for message in request.messages],
            "stream": request.stream,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "user_present": request.user is not None,
        }

    if mode is TraceCaptureMode.REDACTED_CONTENT:
        return {
            "model": request.model,
            "messages": [_redacted_message_payload(message) for message in request.messages],
            "stream": request.stream,
            "max_output_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

    return request.model_dump(mode="json", exclude_none=True)


def _normalize_response_payload(
    response: ChatCompletionResponse | str | None,
    mode: TraceCaptureMode,
) -> dict[str, object] | None:
    if response is None:
        return None
    if mode is TraceCaptureMode.METADATA_ONLY:
        if isinstance(response, ChatCompletionResponse):
            return {
                "backend_name": response.backend_name,
                "choice_count": len(response.choices),
                "finish_reasons": [choice.finish_reason.value for choice in response.choices],
                "usage": response.usage.model_dump(mode="json"),
            }
        return {
            "streamed_text_chars": len(response),
            "estimated_output_tokens": estimate_token_count(response),
        }
    if mode is TraceCaptureMode.REDACTED_CONTENT:
        if isinstance(response, ChatCompletionResponse):
            return {
                "backend_name": response.backend_name,
                "choices": [
                    {
                        "index": choice.index,
                        "finish_reason": choice.finish_reason.value,
                        "message": _redacted_message_payload(choice.message),
                    }
                    for choice in response.choices
                ],
                "usage": response.usage.model_dump(mode="json"),
            }
        return {
            "content": f"[redacted chars={len(response)}]",
            "estimated_output_tokens": estimate_token_count(response),
        }
    if isinstance(response, ChatCompletionResponse):
        return response.model_dump(mode="json")
    return {"content": response}


def _redacted_message_payload(message: ChatMessage) -> dict[str, object]:
    return {
        "role": message.role.value,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "content": f"[redacted chars={len(message.content)}]",
    }


def chunk_text_from_stream(chunks: list[ChatCompletionChunk]) -> str:
    """Reconstruct text content from streamed response chunks."""

    fragments: list[str] = []
    for chunk in chunks:
        for choice in chunk.choices:
            content = choice.delta.content
            if content is not None:
                fragments.append(content)
    return "".join(fragments)
