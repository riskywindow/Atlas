"""Operator-focused hybrid runtime insights and recent route examples."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from switchyard.schemas.admin import (
    CloudRouteEvidenceRuntimeSummary,
    HybridOperatorRuntimeSummary,
    HybridRouteExample,
    PlacementDistributionRuntimeSummary,
    RemoteTransportErrorRuntimeEntry,
)
from switchyard.schemas.backend import BackendInstance
from switchyard.schemas.benchmark import CloudEvidenceSource
from switchyard.schemas.routing import RouteDecision, RoutingPolicy


@dataclass(slots=True)
class HybridOperatorService:
    """In-memory operator-facing state for recent hybrid routing behavior."""

    max_recent_routes: int = 50
    max_recent_transport_errors: int = 20
    remote_enabled_override: bool | None = None
    _recent_route_examples: deque[HybridRouteExample] = field(init=False)
    _recent_transport_errors: deque[RemoteTransportErrorRuntimeEntry] = field(init=False)

    def __post_init__(self) -> None:
        self._recent_route_examples = deque(maxlen=self.max_recent_routes)
        self._recent_transport_errors = deque(maxlen=self.max_recent_transport_errors)

    def set_remote_enabled_override(self, enabled: bool | None) -> None:
        self.remote_enabled_override = enabled

    def record_route_example(
        self,
        *,
        decision: RouteDecision,
        executed_backend: str | None,
        remote_candidate_count: int,
        final_outcome: str,
        fallback_used: bool,
    ) -> None:
        route_reason_codes: list[str] = []
        notes: list[str] = []
        if decision.explanation is not None:
            route_reason_codes = [
                code.value for code in decision.explanation.selection_reason_codes
            ]
            notes.extend(decision.explanation.selected_reason)
        if decision.annotations is not None:
            notes.extend(decision.annotations.notes)
        admission_reason_code = (
            None
            if decision.admission_decision is None
            or decision.admission_decision.reason_code is None
            else decision.admission_decision.reason_code.value
        )
        deployment = decision.selected_deployment
        observed_instance = _observed_instance(decision=decision)
        placement_provider = (
            None
            if observed_instance is None or observed_instance.placement.provider is None
            else observed_instance.placement.provider
        )
        placement_region = (
            None
            if observed_instance is None or observed_instance.placement.region is None
            else observed_instance.placement.region
        )
        placement_zone = (
            None
            if observed_instance is None or observed_instance.placement.zone is None
            else observed_instance.placement.zone
        )
        if deployment is not None:
            if placement_provider is None:
                placement_provider = deployment.placement.provider
            if placement_region is None:
                placement_region = deployment.placement.region
            if placement_zone is None:
                placement_zone = deployment.placement.zone
        placement_evidence_source = _placement_evidence_source(decision=decision)
        relative_cost_index = (
            None
            if observed_instance is None
            else observed_instance.cost_profile.relative_cost_index
        )
        if relative_cost_index is None and deployment is not None:
            relative_cost_index = deployment.cost_profile.relative_cost_index
        cost_evidence_source = _cost_evidence_source(decision=decision)
        self._recent_route_examples.appendleft(
            HybridRouteExample(
                request_id=decision.request_id,
                recorded_at=datetime.now(UTC),
                policy=decision.policy.value,
                tenant_id=(
                    "default"
                    if decision.telemetry_metadata is None
                    else decision.telemetry_metadata.tenant_id
                ),
                chosen_backend=decision.backend_name,
                executed_backend=executed_backend,
                execution_path=_execution_path_for_example(
                    policy=decision.policy,
                    backend_name=executed_backend or decision.backend_name,
                    admission_reason_code=admission_reason_code,
                ),
                fallback_used=fallback_used,
                final_outcome=final_outcome,
                route_reason_codes=route_reason_codes,
                admission_reason_code=admission_reason_code,
                remote_candidate_count=remote_candidate_count,
                placement_provider=placement_provider,
                placement_region=placement_region,
                placement_zone=placement_zone,
                placement_evidence_source=placement_evidence_source,
                relative_cost_index=relative_cost_index,
                cost_evidence_source=cost_evidence_source,
                notes=notes[:5],
            )
        )

    def record_remote_transport_error(
        self,
        *,
        request_id: str,
        backend_name: str,
        error: str,
        cooldown_triggered: bool,
    ) -> None:
        self._recent_transport_errors.appendleft(
            RemoteTransportErrorRuntimeEntry(
                request_id=request_id,
                backend_name=backend_name,
                error=error,
                cooldown_triggered=cooldown_triggered,
            )
        )

    def inspect_state(
        self,
        *,
        remote_effectively_enabled: bool,
    ) -> HybridOperatorRuntimeSummary:
        placement_counts = Counter(
            example.execution_path for example in self._recent_route_examples
        )
        provider_counts = Counter(
            example.placement_provider
            for example in self._recent_route_examples
            if example.execution_path == "remote" and example.placement_provider is not None
        )
        estimated_costs = [
            example.relative_cost_index
            for example in self._recent_route_examples
            if example.relative_cost_index is not None
            and example.cost_evidence_source != CloudEvidenceSource.OBSERVED_RUNTIME.value
        ]
        notes: list[str] = []
        if self.remote_enabled_override is not None:
            notes.append("remote routing enablement is currently operator-overridden")
        if self._recent_transport_errors:
            notes.append("recent remote transport errors are retained for cooldown triage")
        return HybridOperatorRuntimeSummary(
            remote_enabled_override=self.remote_enabled_override,
            remote_effectively_enabled=remote_effectively_enabled,
            recent_route_examples=list(self._recent_route_examples),
            recent_route_example_count=len(self._recent_route_examples),
            recent_placement_distribution=PlacementDistributionRuntimeSummary(
                sample_size=len(self._recent_route_examples),
                local_count=placement_counts.get("local", 0),
                remote_count=placement_counts.get("remote", 0),
                remote_blocked_count=placement_counts.get("remote_blocked", 0),
                unknown_count=placement_counts.get("unknown", 0),
            ),
            recent_cloud_evidence=CloudRouteEvidenceRuntimeSummary(
                sample_size=len(self._recent_route_examples),
                observed_placement_count=sum(
                    1
                    for example in self._recent_route_examples
                    if example.placement_evidence_source
                    == CloudEvidenceSource.OBSERVED_RUNTIME.value
                ),
                estimated_placement_count=sum(
                    1
                    for example in self._recent_route_examples
                    if example.placement_evidence_source is not None
                    and example.placement_evidence_source
                    != CloudEvidenceSource.OBSERVED_RUNTIME.value
                ),
                observed_cost_count=sum(
                    1
                    for example in self._recent_route_examples
                    if example.cost_evidence_source == CloudEvidenceSource.OBSERVED_RUNTIME.value
                ),
                estimated_cost_count=sum(
                    1
                    for example in self._recent_route_examples
                    if example.cost_evidence_source is not None
                    and example.cost_evidence_source
                    != CloudEvidenceSource.OBSERVED_RUNTIME.value
                ),
                remote_provider_counts=dict(provider_counts),
                total_estimated_relative_cost_index=(
                    None if not estimated_costs else round(sum(estimated_costs), 6)
                ),
            ),
            recent_remote_transport_errors=list(self._recent_transport_errors),
            notes=notes,
        )


def _execution_path_for_example(
    *,
    policy: RoutingPolicy,
    backend_name: str,
    admission_reason_code: str | None,
) -> str:
    if admission_reason_code in {"remote_budget_exhausted", "remote_spillover_not_permitted"}:
        return "remote_blocked"
    if backend_name.startswith("remote-worker:"):
        if policy in {
            RoutingPolicy.BURST_TO_REMOTE,
            RoutingPolicy.LOCAL_PREFERRED,
            RoutingPolicy.LATENCY_SLO,
            RoutingPolicy.QUALITY_ON_DEMAND,
            RoutingPolicy.REMOTE_PREFERRED_IF_LOCAL_UNHEALTHY,
        }:
            return "remote"
        return "remote"
    if policy in {RoutingPolicy.LOCAL_ONLY, RoutingPolicy.REMOTE_DISABLED}:
        return "local"
    return "local"


def _observed_instance(*, decision: RouteDecision) -> BackendInstance | None:
    deployment = decision.selected_deployment
    if (
        deployment is None
        or decision.execution_observation is None
        or decision.execution_observation.backend_instance_id is None
    ):
        return None
    for instance in deployment.instances:
        if instance.instance_id == decision.execution_observation.backend_instance_id:
            return instance
    return None


def _placement_evidence_source(*, decision: RouteDecision) -> str | None:
    observed_instance = _observed_instance(decision=decision)
    if (
        observed_instance is not None
        and (
            observed_instance.placement.provider is not None
            or observed_instance.placement.region is not None
            or observed_instance.placement.zone is not None
        )
    ):
        return CloudEvidenceSource.OBSERVED_RUNTIME.value
    deployment = decision.selected_deployment
    if deployment is None:
        return None
    if (
        deployment.placement.provider is not None
        or deployment.placement.region is not None
        or deployment.placement.zone is not None
    ):
        return CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE.value
    return None


def _cost_evidence_source(*, decision: RouteDecision) -> str | None:
    observed_instance = _observed_instance(decision=decision)
    if (
        observed_instance is not None
        and (
            observed_instance.cost_profile.relative_cost_index is not None
            or observed_instance.cost_profile.budget_bucket is not None
            or observed_instance.cost_profile.currency is not None
        )
    ):
        return CloudEvidenceSource.OBSERVED_RUNTIME.value
    deployment = decision.selected_deployment
    if (
        deployment is None
        or deployment.cost_profile.relative_cost_index is None
    ):
        return None
    return CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE.value
