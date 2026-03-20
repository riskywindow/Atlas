"""Operator-focused hybrid runtime insights and recent route examples."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from switchyard.schemas.admin import (
    AliasCompatibilityRuntimeEntry,
    AliasRoutingOverrideState,
    CloudRouteEvidenceRuntimeSummary,
    CloudWorkerControlRuntimeEntry,
    HybridBudgetRuntimeSummary,
    HybridExecutionRuntimeSummary,
    HybridOperatorRuntimeSummary,
    HybridRouteExample,
    PlacementDistributionRuntimeSummary,
    RemoteTransportErrorRuntimeEntry,
)
from switchyard.schemas.backend import BackendInstance
from switchyard.schemas.benchmark import CloudEvidenceSource
from switchyard.schemas.routing import RouteDecision, RoutingPolicy
from switchyard.schemas.worker import RegisteredRemoteWorkerRecord, RegisteredRemoteWorkerSnapshot


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
        budget_bucket = (
            None if observed_instance is None else observed_instance.cost_profile.budget_bucket
        )
        if budget_bucket is None and deployment is not None:
            budget_bucket = deployment.cost_profile.budget_bucket
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
                budget_bucket=budget_bucket,
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
        spillover_runtime: HybridExecutionRuntimeSummary | None = None,
        remote_worker_snapshot: RegisteredRemoteWorkerSnapshot | None = None,
        known_serving_targets: list[str] | None = None,
        alias_overrides: list[AliasRoutingOverrideState] | None = None,
    ) -> HybridOperatorRuntimeSummary:
        placement_counts = Counter(
            example.execution_path for example in self._recent_route_examples
        )
        provider_counts = Counter(
            example.placement_provider
            for example in self._recent_route_examples
            if example.execution_path == "remote" and example.placement_provider is not None
        )
        observed_budget_bucket_counts = Counter(
            example.budget_bucket
            for example in self._recent_route_examples
            if example.budget_bucket is not None
            and example.cost_evidence_source == CloudEvidenceSource.OBSERVED_RUNTIME.value
        )
        estimated_budget_bucket_counts = Counter(
            example.budget_bucket
            for example in self._recent_route_examples
            if example.budget_bucket is not None
            and example.cost_evidence_source is not None
            and example.cost_evidence_source != CloudEvidenceSource.OBSERVED_RUNTIME.value
        )
        observed_costs = [
            example.relative_cost_index
            for example in self._recent_route_examples
            if example.relative_cost_index is not None
            and example.cost_evidence_source == CloudEvidenceSource.OBSERVED_RUNTIME.value
        ]
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
        resolved_alias_overrides = [] if alias_overrides is None else alias_overrides
        resolved_remote_workers = (
            [] if remote_worker_snapshot is None else remote_worker_snapshot.workers
        )
        budget_state = _budget_state(
            spillover_runtime=spillover_runtime,
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
                observed_budget_bucket_counts=dict(observed_budget_bucket_counts),
                estimated_budget_bucket_counts=dict(estimated_budget_bucket_counts),
                total_observed_relative_cost_index=(
                    None if not observed_costs else round(sum(observed_costs), 6)
                ),
                total_estimated_relative_cost_index=(
                    None if not estimated_costs else round(sum(estimated_costs), 6)
                ),
            ),
        )
        cloud_workers = [
            _cloud_worker_entry(worker)
            for worker in resolved_remote_workers
        ]
        alias_compatibility = _alias_compatibility_entries(
            workers=resolved_remote_workers,
            serving_targets=known_serving_targets or [],
            alias_overrides=resolved_alias_overrides,
        )
        recent_cloud_evidence = CloudRouteEvidenceRuntimeSummary(
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
            observed_budget_bucket_counts=dict(observed_budget_bucket_counts),
            estimated_budget_bucket_counts=dict(estimated_budget_bucket_counts),
            total_observed_relative_cost_index=(
                None if not observed_costs else round(sum(observed_costs), 6)
            ),
            total_estimated_relative_cost_index=(
                None if not estimated_costs else round(sum(estimated_costs), 6)
            ),
        )
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
            recent_cloud_evidence=recent_cloud_evidence,
            budget_state=budget_state,
            cloud_workers=cloud_workers,
            alias_compatibility=alias_compatibility,
            alias_overrides=resolved_alias_overrides,
            recent_remote_transport_errors=list(self._recent_transport_errors),
            notes=notes,
        )


def _budget_state(
    *,
    spillover_runtime: HybridExecutionRuntimeSummary | None,
    recent_cloud_evidence: CloudRouteEvidenceRuntimeSummary,
) -> HybridBudgetRuntimeSummary:
    if spillover_runtime is None:
        return HybridBudgetRuntimeSummary(
            recent_observed_budget_bucket_counts=(
                recent_cloud_evidence.observed_budget_bucket_counts
            ),
            recent_estimated_budget_bucket_counts=(
                recent_cloud_evidence.estimated_budget_bucket_counts
            ),
            total_observed_relative_cost_index=(
                recent_cloud_evidence.total_observed_relative_cost_index
            ),
            total_estimated_relative_cost_index=(
                recent_cloud_evidence.total_estimated_relative_cost_index
            ),
        )
    return HybridBudgetRuntimeSummary(
        remote_request_budget_per_minute=spillover_runtime.remote_request_budget_per_minute,
        remote_budget_window_started_at=spillover_runtime.remote_budget_window_started_at,
        remote_budget_requests_used=spillover_runtime.remote_budget_requests_used,
        remote_budget_requests_remaining=spillover_runtime.remote_budget_requests_remaining,
        remote_in_flight_requests=spillover_runtime.remote_in_flight_requests,
        remote_concurrency_cap=spillover_runtime.remote_concurrency_cap,
        cooldown_active=spillover_runtime.cooldown_active,
        cooldown_until=spillover_runtime.cooldown_until,
        recent_observed_budget_bucket_counts=recent_cloud_evidence.observed_budget_bucket_counts,
        recent_estimated_budget_bucket_counts=recent_cloud_evidence.estimated_budget_bucket_counts,
        total_observed_relative_cost_index=(
            recent_cloud_evidence.total_observed_relative_cost_index
        ),
        total_estimated_relative_cost_index=(
            recent_cloud_evidence.total_estimated_relative_cost_index
        ),
        notes=list(spillover_runtime.notes),
    )


def _cloud_worker_entry(worker: RegisteredRemoteWorkerRecord) -> CloudWorkerControlRuntimeEntry:
    deployment = worker.deployment
    placement = deployment.placement if deployment is not None else worker.instance.placement
    return CloudWorkerControlRuntimeEntry(
        worker_id=worker.worker_id,
        backend_name=worker.backend_name,
        serving_targets=list(worker.serving_targets),
        lifecycle_state=worker.lifecycle_state.value,
        last_heartbeat_at=worker.last_heartbeat_at,
        runtime=worker.runtime.model_copy(deep=True) if worker.runtime is not None else None,
        gpu=worker.gpu.model_copy(deep=True) if worker.gpu is not None else None,
        provider=placement.provider,
        region=placement.region,
        zone=placement.zone,
        ready=worker.ready,
        usable=worker.usable,
        quarantined=worker.quarantined,
        canary_only="canary-only" in worker.tags,
        draining=worker.lifecycle_state.value == "draining",
        active_requests=worker.active_requests,
        queue_depth=worker.queue_depth,
        eligibility_reasons=list(worker.eligibility_reasons),
        tags=list(worker.tags),
    )


def _alias_compatibility_entries(
    *,
    workers: list[RegisteredRemoteWorkerRecord],
    serving_targets: list[str],
    alias_overrides: list[AliasRoutingOverrideState],
) -> list[AliasCompatibilityRuntimeEntry]:
    overrides_by_target = {override.serving_target: override for override in alias_overrides}
    targets = set(serving_targets)
    for worker in workers:
        targets.update(worker.serving_targets)
    entries: list[AliasCompatibilityRuntimeEntry] = []
    for serving_target in sorted(targets):
        override = overrides_by_target.get(serving_target)
        eligible_remote_backends: list[str] = []
        ineligible_remote_backends: dict[str, list[str]] = {}
        notes: list[str] = []
        matching_workers = [
            worker for worker in workers if serving_target in worker.serving_targets
        ]
        for worker in matching_workers:
            reasons = list(worker.eligibility_reasons)
            if override is not None and worker.backend_name in override.disabled_backends:
                reasons.append("operator disabled backend for this serving target")
            if worker.usable and not reasons:
                eligible_remote_backends.append(worker.backend_name)
                continue
            if not reasons:
                reasons.append("worker is not currently usable")
            ineligible_remote_backends[worker.backend_name] = sorted(set(reasons))
        if not matching_workers:
            notes.append("no registered remote cloud worker currently serves this target")
        if override is not None and override.pinned_backend is not None:
            notes.append(f"operator pinned backend={override.pinned_backend}")
        if override is not None and override.disabled_backends:
            notes.append("operator disabled one or more backends for this target")
        entries.append(
            AliasCompatibilityRuntimeEntry(
                serving_target=serving_target,
                eligible_remote_backends=sorted(set(eligible_remote_backends)),
                ineligible_remote_backends=ineligible_remote_backends,
                pinned_backend=None if override is None else override.pinned_backend,
                disabled_backends=[] if override is None else list(override.disabled_backends),
                notes=notes,
            )
        )
    return entries


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
        or (
            deployment.cost_profile.relative_cost_index is None
            and deployment.cost_profile.budget_bucket is None
            and deployment.cost_profile.currency is None
        )
    ):
        return None
    return CloudEvidenceSource.DEPLOYMENT_METADATA_ESTIMATE.value
