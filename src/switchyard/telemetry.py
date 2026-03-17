"""Telemetry helpers for local-friendly request and backend measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from starlette.applications import Starlette


@dataclass(slots=True)
class TelemetryState:
    """Local-friendly in-memory telemetry state for tests and development."""

    request_count: int = 0
    request_latency_ms: list[float] = field(default_factory=list)
    route_decision_count: int = 0
    admission_records: list[AdmissionRecord] = field(default_factory=list)
    route_decision_records: list[RouteDecisionRecord] = field(default_factory=list)
    route_attempt_records: list[RouteAttemptRecord] = field(default_factory=list)
    backend_health_snapshots: list[dict[str, Any]] = field(default_factory=list)
    backend_execution_records: list[BackendExecutionRecord] = field(default_factory=list)
    backend_warmup_records: list[BackendWarmupRecord] = field(default_factory=list)
    shadow_execution_records: list[ShadowExecutionRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BackendLabels:
    """Stable labels used across observability records."""

    backend_name: str
    backend_type: str
    model: str
    model_identifier: str
    execution_mode: str = "in_process"
    worker_transport: str = "in_process"


@dataclass(frozen=True, slots=True)
class BackendExecutionRecord:
    """Measured request execution details for a backend invocation."""

    route: str
    method: str
    status_code: int
    streaming: bool
    backend_name: str
    backend_type: str
    model: str
    model_identifier: str
    execution_mode: str
    worker_transport: str
    total_latency_ms: float
    ttft_ms: float | None
    output_tokens: int
    tokens_per_second: float | None


@dataclass(frozen=True, slots=True)
class BackendWarmupRecord:
    """Measured warmup details for an adapter/model pair."""

    backend_name: str
    backend_type: str
    model: str
    model_identifier: str
    execution_mode: str
    worker_transport: str
    readiness_state: str
    warmup_latency_ms: float
    success: bool


@dataclass(frozen=True, slots=True)
class RouteDecisionRecord:
    """Measured routing decision details before backend execution."""

    request_id: str
    tenant_id: str
    session_id: str | None
    requested_model: str
    serving_target: str
    policy: str
    chosen_backend: str
    considered_backends: list[str]
    candidate_backend_count: int
    fallback_backends: list[str]
    fallback_occurred: bool
    rejected_backends: dict[str, str]
    admission_limited_backends: dict[str, str]
    protected_backends: dict[str, str]
    degraded_backends: list[str]
    route_reason: str
    route_latency_ms: float


@dataclass(frozen=True, slots=True)
class AdmissionRecord:
    """Measured admission-control outcome for one request."""

    request_id: str
    tenant_id: str
    request_class: str
    state: str
    reason_code: str | None
    queue_depth: int
    queue_wait_ms: float | None
    status_code: int


@dataclass(frozen=True, slots=True)
class RouteAttemptRecord:
    """Measured route execution attempt for a concrete backend."""

    request_id: str
    policy: str
    backend_name: str
    attempt_number: int
    selected_by_router: bool
    outcome: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ShadowExecutionRecord:
    """Measured best-effort shadow execution details."""

    primary_request_id: str
    shadow_request_id: str
    policy_name: str
    target_kind: str
    configured_target: str
    resolved_backend_name: str | None
    requested_model: str
    launched_at: datetime
    success: bool
    latency_ms: float
    error: str | None = None


def estimate_token_count(text: str) -> int:
    """Estimate token count using whitespace tokenization for local metrics."""

    return len(text.split())


def compute_tokens_per_second(*, output_tokens: int, total_latency_ms: float) -> float | None:
    """Return throughput using total request latency when available."""

    if output_tokens <= 0 or total_latency_ms <= 0:
        return None
    return round(output_tokens / (total_latency_ms / 1000), 6)


class Telemetry:
    """Thin wrapper over local telemetry state and optional OpenTelemetry instruments."""

    def __init__(self, service_name: str, *, enabled: bool = False) -> None:
        self.service_name = service_name
        self.enabled = enabled
        self.state = TelemetryState()
        self._instrumented_apps: set[int] = set()

        resource = Resource.create({"service.name": service_name})
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self._meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(self._meter_provider)
        meter = metrics.get_meter(service_name)
        self._requests_total = meter.create_counter("switchyard_requests_total")
        self._request_latency_ms = meter.create_histogram("switchyard_request_latency_ms")
        self._route_decisions_total = meter.create_counter("switchyard_route_decisions_total")
        self._admission_decisions_total = meter.create_counter(
            "switchyard_admission_decisions_total"
        )
        self._admission_wait_ms = meter.create_histogram("switchyard_admission_wait_ms")
        self._admission_queue_depth = meter.create_histogram("switchyard_admission_queue_depth")
        self._route_attempts_total = meter.create_counter("switchyard_route_attempts_total")
        self._route_fallbacks_total = meter.create_counter("switchyard_route_fallbacks_total")
        self._route_decision_latency_ms = meter.create_histogram("switchyard_route_latency_ms")
        self._backend_health = meter.create_up_down_counter("switchyard_backend_health")
        self._backend_latency_ms = meter.create_histogram("switchyard_backend_request_latency_ms")
        self._backend_ttft_ms = meter.create_histogram("switchyard_backend_ttft_ms")
        self._backend_output_tokens = meter.create_histogram("switchyard_backend_output_tokens")
        self._backend_tokens_per_second = meter.create_histogram(
            "switchyard_backend_tokens_per_second"
        )
        self._backend_warmup_latency_ms = meter.create_histogram(
            "switchyard_backend_warmup_latency_ms"
        )
        self._shadow_executions_total = meter.create_counter("switchyard_shadow_executions_total")
        self._shadow_latency_ms = meter.create_histogram("switchyard_shadow_latency_ms")

    def instrument_fastapi(self, app: Starlette) -> None:
        """Instrument a FastAPI app once when telemetry is enabled."""

        if not self.enabled or id(app) in self._instrumented_apps:
            return
        FastAPIInstrumentor.instrument_app(app)
        self._instrumented_apps.add(id(app))

    def record_request(
        self,
        *,
        route: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record request counters and latency."""

        attributes = {"route": route, "method": method, "status_code": status_code}
        self.state.request_count += 1
        self.state.request_latency_ms.append(latency_ms)
        self._requests_total.add(1, attributes=attributes)
        self._request_latency_ms.record(latency_ms, attributes=attributes)

    def record_route_decision(
        self,
        *,
        request_id: str,
        tenant_id: str,
        session_id: str | None,
        requested_model: str,
        serving_target: str,
        policy: str,
        backend_name: str,
        considered_backends: list[str],
        fallback_backends: list[str],
        rejected_backends: dict[str, str],
        admission_limited_backends: dict[str, str],
        protected_backends: dict[str, str],
        degraded_backends: list[str],
        route_reason: str,
        route_latency_ms: float,
    ) -> RouteDecisionRecord:
        """Record a route decision."""

        record = RouteDecisionRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            session_id=session_id,
            requested_model=requested_model,
            serving_target=serving_target,
            policy=policy,
            chosen_backend=backend_name,
            considered_backends=considered_backends,
            candidate_backend_count=len(considered_backends),
            fallback_backends=fallback_backends,
            fallback_occurred=bool(fallback_backends),
            rejected_backends=rejected_backends,
            admission_limited_backends=admission_limited_backends,
            protected_backends=protected_backends,
            degraded_backends=degraded_backends,
            route_reason=route_reason,
            route_latency_ms=round(route_latency_ms, 6),
        )
        self.state.route_decision_count += 1
        self.state.route_decision_records.append(record)
        attributes = {
            "policy": policy,
            "backend_name": backend_name,
            "requested_model": requested_model,
            "serving_target": serving_target,
            "tenant_id": tenant_id,
            "session_affinity": bool(session_id),
            "candidate_backend_count": len(considered_backends),
            "fallback_occurred": bool(fallback_backends),
            "admission_limited_backend_count": len(admission_limited_backends),
            "protected_backend_count": len(protected_backends),
            "degraded_backend_count": len(degraded_backends),
            "route_reason": route_reason,
        }
        self._route_decisions_total.add(
            1,
            attributes=attributes,
        )
        self._route_decision_latency_ms.record(route_latency_ms, attributes=attributes)
        return record

    def record_admission_decision(
        self,
        *,
        request_id: str,
        tenant_id: str,
        request_class: str,
        state: str,
        reason_code: str | None,
        queue_depth: int,
        queue_wait_ms: float | None,
        status_code: int,
    ) -> AdmissionRecord:
        """Record admission-control state and queue timing."""

        record = AdmissionRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            request_class=request_class,
            state=state,
            reason_code=reason_code,
            queue_depth=queue_depth,
            queue_wait_ms=None if queue_wait_ms is None else round(queue_wait_ms, 6),
            status_code=status_code,
        )
        self.state.admission_records.append(record)
        attributes = {
            "tenant_id": tenant_id,
            "request_class": request_class,
            "state": state,
            "reason_code": reason_code or "",
            "status_code": status_code,
        }
        self._admission_decisions_total.add(1, attributes=attributes)
        self._admission_queue_depth.record(queue_depth, attributes=attributes)
        if queue_wait_ms is not None:
            self._admission_wait_ms.record(queue_wait_ms, attributes=attributes)
        return record

    def record_route_attempt(
        self,
        *,
        request_id: str,
        policy: str,
        backend_name: str,
        attempt_number: int,
        selected_by_router: bool,
        outcome: str,
        error: str | None = None,
    ) -> RouteAttemptRecord:
        """Record a concrete route execution attempt."""

        record = RouteAttemptRecord(
            request_id=request_id,
            policy=policy,
            backend_name=backend_name,
            attempt_number=attempt_number,
            selected_by_router=selected_by_router,
            outcome=outcome,
            error=error,
        )
        self.state.route_attempt_records.append(record)
        attributes = {
            "policy": policy,
            "backend_name": backend_name,
            "attempt_number": attempt_number,
            "selected_by_router": selected_by_router,
            "outcome": outcome,
        }
        self._route_attempts_total.add(1, attributes=attributes)
        if not selected_by_router:
            self._route_fallbacks_total.add(1, attributes=attributes)
        return record

    def record_backend_execution(
        self,
        *,
        route: str,
        method: str,
        status_code: int,
        streaming: bool,
        labels: BackendLabels,
        total_latency_ms: float,
        ttft_ms: float | None,
        output_tokens: int,
    ) -> BackendExecutionRecord:
        """Record backend execution metrics for streaming and non-streaming requests."""

        tokens_per_second = compute_tokens_per_second(
            output_tokens=output_tokens,
            total_latency_ms=total_latency_ms,
        )
        record = BackendExecutionRecord(
            route=route,
            method=method,
            status_code=status_code,
            streaming=streaming,
            backend_name=labels.backend_name,
            backend_type=labels.backend_type,
            model=labels.model,
            model_identifier=labels.model_identifier,
            execution_mode=labels.execution_mode,
            worker_transport=labels.worker_transport,
            total_latency_ms=round(total_latency_ms, 6),
            ttft_ms=None if ttft_ms is None else round(ttft_ms, 6),
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
        )
        self.state.backend_execution_records.append(record)
        attributes = {
            "route": route,
            "method": method,
            "status_code": status_code,
            "streaming": streaming,
            "backend_name": labels.backend_name,
            "backend_type": labels.backend_type,
            "model": labels.model,
            "model_identifier": labels.model_identifier,
            "execution_mode": labels.execution_mode,
            "worker_transport": labels.worker_transport,
        }
        self._backend_latency_ms.record(total_latency_ms, attributes=attributes)
        self._backend_output_tokens.record(output_tokens, attributes=attributes)
        if ttft_ms is not None:
            self._backend_ttft_ms.record(ttft_ms, attributes=attributes)
        if tokens_per_second is not None:
            self._backend_tokens_per_second.record(tokens_per_second, attributes=attributes)
        return record

    def record_backend_warmup(
        self,
        *,
        labels: BackendLabels,
        readiness_state: str,
        warmup_latency_ms: float,
        success: bool,
    ) -> BackendWarmupRecord:
        """Record backend warmup timing and resulting readiness state."""

        record = BackendWarmupRecord(
            backend_name=labels.backend_name,
            backend_type=labels.backend_type,
            model=labels.model,
            model_identifier=labels.model_identifier,
            execution_mode=labels.execution_mode,
            worker_transport=labels.worker_transport,
            readiness_state=readiness_state,
            warmup_latency_ms=round(warmup_latency_ms, 6),
            success=success,
        )
        self.state.backend_warmup_records.append(record)
        self._backend_warmup_latency_ms.record(
            warmup_latency_ms,
            attributes={
                "backend_name": labels.backend_name,
                "backend_type": labels.backend_type,
                "model": labels.model,
                "model_identifier": labels.model_identifier,
                "execution_mode": labels.execution_mode,
                "worker_transport": labels.worker_transport,
                "readiness_state": readiness_state,
                "success": success,
            },
        )
        return record

    def record_shadow_execution(
        self,
        *,
        primary_request_id: str,
        shadow_request_id: str,
        policy_name: str,
        target_kind: str,
        configured_target: str,
        resolved_backend_name: str | None,
        requested_model: str,
        launched_at: datetime,
        success: bool,
        latency_ms: float,
        error: str | None,
    ) -> ShadowExecutionRecord:
        """Record shadow traffic launch and outcome details."""

        record = ShadowExecutionRecord(
            primary_request_id=primary_request_id,
            shadow_request_id=shadow_request_id,
            policy_name=policy_name,
            target_kind=target_kind,
            configured_target=configured_target,
            resolved_backend_name=resolved_backend_name,
            requested_model=requested_model,
            launched_at=launched_at,
            success=success,
            latency_ms=round(latency_ms, 6),
            error=error,
        )
        self.state.shadow_execution_records.append(record)
        attributes = {
            "policy_name": policy_name,
            "target_kind": target_kind,
            "configured_target": configured_target,
            "resolved_backend_name": resolved_backend_name or "",
            "requested_model": requested_model,
            "success": success,
        }
        self._shadow_executions_total.add(1, attributes=attributes)
        self._shadow_latency_ms.record(latency_ms, attributes=attributes)
        return record

    def record_backend_health_snapshot(
        self,
        *,
        backend_name: str,
        health_state: str,
        latency_ms: float | None,
    ) -> None:
        """Record a backend health snapshot."""

        snapshot = {
            "backend_name": backend_name,
            "health_state": health_state,
            "latency_ms": latency_ms,
        }
        self.state.backend_health_snapshots.append(snapshot)
        self._backend_health.add(
            1,
            attributes={
                "backend_name": backend_name,
                "health_state": health_state,
            },
        )

    def render_prometheus_text(self) -> str:
        """Render a small Prometheus-friendly snapshot from local state."""

        fallback_attempt_count = sum(
            not record.selected_by_router for record in self.state.route_attempt_records
        )
        lines = [
            "# HELP switchyard_requests_total Total HTTP requests handled.",
            "# TYPE switchyard_requests_total counter",
            f"switchyard_requests_total {self.state.request_count}",
            "# HELP switchyard_route_decisions_total Total route decisions recorded.",
            "# TYPE switchyard_route_decisions_total counter",
            f"switchyard_route_decisions_total {self.state.route_decision_count}",
            "# HELP switchyard_admission_decisions_total Total admission decisions recorded.",
            "# TYPE switchyard_admission_decisions_total counter",
            f"switchyard_admission_decisions_total {len(self.state.admission_records)}",
            "# HELP switchyard_route_attempts_total Total route execution attempts recorded.",
            "# TYPE switchyard_route_attempts_total counter",
            f"switchyard_route_attempts_total {len(self.state.route_attempt_records)}",
            "# HELP switchyard_route_fallbacks_total Total fallback attempts recorded.",
            "# TYPE switchyard_route_fallbacks_total counter",
            f"switchyard_route_fallbacks_total {fallback_attempt_count}",
        ]

        if self.state.request_latency_ms:
            lines.extend(
                [
                    "# HELP switchyard_request_latency_ms_avg Average HTTP request latency.",
                    "# TYPE switchyard_request_latency_ms_avg gauge",
                    (
                        "switchyard_request_latency_ms_avg "
                        f"{round(mean(self.state.request_latency_ms), 6)}"
                    ),
                ]
            )

        for index, record in enumerate(self.state.backend_execution_records):
            lines.append(
                _prometheus_metric_line(
                    "switchyard_backend_request_latency_ms",
                    record.total_latency_ms,
                    labels={
                        "index": str(index),
                        "backend_name": record.backend_name,
                        "backend_type": record.backend_type,
                        "model": record.model,
                        "model_identifier": record.model_identifier,
                        "streaming": str(record.streaming).lower(),
                        "status_code": str(record.status_code),
                    },
                )
            )
            lines.append(
                _prometheus_metric_line(
                    "switchyard_backend_output_tokens",
                    record.output_tokens,
                    labels={
                        "index": str(index),
                        "backend_name": record.backend_name,
                        "model": record.model,
                    },
                )
            )
            if record.ttft_ms is not None:
                lines.append(
                    _prometheus_metric_line(
                        "switchyard_backend_ttft_ms",
                        record.ttft_ms,
                        labels={
                            "index": str(index),
                            "backend_name": record.backend_name,
                            "model": record.model,
                        },
                    )
                )
            if record.tokens_per_second is not None:
                lines.append(
                    _prometheus_metric_line(
                        "switchyard_backend_tokens_per_second",
                        record.tokens_per_second,
                        labels={
                            "index": str(index),
                            "backend_name": record.backend_name,
                            "model": record.model,
                        },
                    )
                )

        for index, record in enumerate(self.state.route_decision_records):
            lines.append(
                _prometheus_metric_line(
                    "switchyard_route_decision_latency_ms",
                    record.route_latency_ms,
                    labels={
                        "index": str(index),
                        "request_id": record.request_id,
                        "requested_model": record.requested_model,
                        "serving_target": record.serving_target,
                        "policy": record.policy,
                        "chosen_backend": record.chosen_backend,
                        "candidate_backend_count": str(record.candidate_backend_count),
                        "fallback_occurred": str(record.fallback_occurred).lower(),
                        "route_reason": record.route_reason,
                    },
                )
            )

        for index, record in enumerate(self.state.backend_warmup_records):
            lines.append(
                _prometheus_metric_line(
                    "switchyard_backend_warmup_latency_ms",
                    record.warmup_latency_ms,
                    labels={
                        "index": str(index),
                        "backend_name": record.backend_name,
                        "backend_type": record.backend_type,
                        "model": record.model,
                        "model_identifier": record.model_identifier,
                        "readiness_state": record.readiness_state,
                        "success": str(record.success).lower(),
                    },
                )
            )

        for index, record in enumerate(self.state.shadow_execution_records):
            lines.append(
                _prometheus_metric_line(
                    "switchyard_shadow_latency_ms",
                    record.latency_ms,
                    labels={
                        "index": str(index),
                        "primary_request_id": record.primary_request_id,
                        "policy_name": record.policy_name,
                        "target_kind": record.target_kind,
                        "configured_target": record.configured_target,
                        "resolved_backend_name": record.resolved_backend_name or "",
                        "success": str(record.success).lower(),
                    },
                )
            )

        return "\n".join(lines) + "\n"

    def state_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the local telemetry state."""

        return {
            "request_count": self.state.request_count,
            "request_latency_ms": self.state.request_latency_ms,
            "route_decision_count": self.state.route_decision_count,
            "admission_records": [asdict(record) for record in self.state.admission_records],
            "route_decision_records": [
                asdict(record) for record in self.state.route_decision_records
            ],
            "route_attempt_records": [
                asdict(record) for record in self.state.route_attempt_records
            ],
            "backend_health_snapshots": self.state.backend_health_snapshots,
            "backend_execution_records": [
                asdict(record) for record in self.state.backend_execution_records
            ],
            "backend_warmup_records": [
                asdict(record) for record in self.state.backend_warmup_records
            ],
            "shadow_execution_records": [
                asdict(record) for record in self.state.shadow_execution_records
            ],
        }


def configure_telemetry(service_name: str, *, enabled: bool = False) -> Telemetry:
    """Create the telemetry wrapper."""

    return Telemetry(service_name=service_name, enabled=enabled)


def _prometheus_metric_line(
    name: str,
    value: float | int,
    *,
    labels: dict[str, str],
) -> str:
    encoded_labels = ",".join(
        f'{key}="{_escape_prometheus_label_value(label_value)}"'
        for key, label_value in sorted(labels.items())
    )
    return f"{name}{{{encoded_labels}}} {value}"


def _escape_prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
