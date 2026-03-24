from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.adapters.registry import AdapterRegistry
from switchyard.config import PolicyRolloutSettings, Settings
from switchyard.control.policy_rollout import PolicyRolloutService
from switchyard.gateway.app import create_app
from switchyard.optimization import build_baseline_optimization_config_profile
from switchyard.router.policies import CandidateAssessment, PolicyEvaluation
from switchyard.schemas.backend import (
    BackendCapabilities,
    BackendStatusSnapshot,
    BackendType,
    DeviceClass,
)
from switchyard.schemas.benchmark import CounterfactualObjective, RecommendationConfidence
from switchyard.schemas.chat import ChatCompletionRequest
from switchyard.schemas.optimization import (
    ForgeCandidateKind,
    ForgeEvidenceSourceKind,
    OptimizationArtifactEvidenceKind,
    OptimizationArtifactSourceType,
    OptimizationCampaignArtifact,
    OptimizationCampaignComparisonArtifact,
    OptimizationCampaignMetadata,
    OptimizationCandidateComparisonArtifact,
    OptimizationCandidateConfigurationArtifact,
    OptimizationComparisonOperator,
    OptimizationConstraintAssessment,
    OptimizationConstraintDimension,
    OptimizationConstraintStrength,
    OptimizationEvidenceRecord,
    OptimizationKnobChange,
    OptimizationPromotionDecision,
    OptimizationPromotionDisposition,
    OptimizationRecommendationDisposition,
    OptimizationRecommendationLabel,
    OptimizationRecommendationReasonCode,
    OptimizationRecommendationSummary,
    OptimizationTrialArtifact,
    OptimizationTrialIdentity,
)
from switchyard.schemas.routing import (
    PolicyReference,
    PolicyRolloutMode,
    RequestContext,
    RouteSelectionReasonCode,
    RoutingPolicy,
)


class FixedPolicy:
    compatibility_policy: RoutingPolicy | None

    def __init__(self, policy_id: str) -> None:
        self.policy_reference = PolicyReference(
            policy_id=policy_id,
            policy_version="phase6.test",
        )
        self.compatibility_policy = RoutingPolicy.BALANCED

    def evaluate(
        self,
        *,
        request: ChatCompletionRequest,
        context: RequestContext,
        candidates: Sequence[BackendStatusSnapshot],
    ) -> PolicyEvaluation:
        del request, context
        assessments = sorted(
            [
                CandidateAssessment(
                    snapshot=snapshot,
                    score=100.0 if snapshot.name == "alpha" else 1.0,
                    eligible=True,
                    rationale=[f"selected_by={self.policy_reference.policy_id}"],
                    reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
                )
                for snapshot in candidates
            ],
            key=lambda assessment: (-(assessment.score or float("-inf")), assessment.snapshot.name),
        )
        return PolicyEvaluation(
            policy_reference=self.policy_reference,
            assessments=assessments,
            selected_backend=assessments[0].snapshot.name,
            selected_reason_codes=[RouteSelectionReasonCode.POLICY_SCORE],
            selected_reason=[f"selected_by={self.policy_reference.policy_id}"],
        )


def build_adapter(*, name: str, latency_ms: float) -> MockBackendAdapter:
    return MockBackendAdapter(
        name=name,
        simulated_latency_ms=latency_ms,
        capability_metadata=BackendCapabilities(
            backend_type=BackendType.MOCK,
            device_class=DeviceClass.CPU,
            model_ids=["mock-chat"],
            serving_targets=["mock-chat"],
            max_context_tokens=8192,
            concurrency_limit=1,
        ),
    )


def test_admin_policy_rollout_endpoints_control_runtime_state() -> None:
    registry = AdapterRegistry()
    registry.register(build_adapter(name="alpha", latency_ms=20.0))
    registry.register(build_adapter(name="beta", latency_ms=5.0))
    rollout = PolicyRolloutService(
        PolicyRolloutSettings(
            candidate_policy_id="adaptive-admin-v1",
            max_recent_decisions=5,
        ),
        candidate_policies=[FixedPolicy("adaptive-admin-v1")],
    )
    app = create_app(
        registry=registry,
        policy_rollout=rollout,
    )
    client = TestClient(app)

    update = client.post(
        "/admin/policy-rollout",
        json={
            "mode": "report_only",
            "kill_switch_enabled": False,
            "learning_frozen": True,
        },
    )
    assert update.status_code == 200
    assert update.json()["mode"] == "report_only"
    assert update.json()["learning_frozen"] is True

    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-policy-admin"},
        json={
            "model": "mock-chat",
            "messages": [{"role": "user", "content": "inspect rollout"}],
        },
    )
    assert response.status_code == 200

    runtime = client.get("/admin/runtime")
    assert runtime.status_code == 200
    payload = runtime.json()
    assert payload["policy_rollout"]["mode"] == "report_only"
    assert payload["policy_rollout"]["candidate_policy"]["policy_id"] == "adaptive-admin-v1"
    assert payload["policy_rollout"]["recent_decisions"][0]["request_id"] == "req-policy-admin"

    exported = client.get("/admin/policy-rollout/export")
    assert exported.status_code == 200
    assert exported.json()["recent_decisions"][0]["request_id"] == "req-policy-admin"

    reset = client.post("/admin/policy-rollout/reset")
    assert reset.status_code == 200
    assert reset.json()["recent_decisions"] == []

    imported = client.post(
        "/admin/policy-rollout/import",
        json=exported.json(),
    )
    assert imported.status_code == 200
    assert imported.json()["recent_decisions"][0]["request_id"] == "req-policy-admin"
    assert imported.json()["learning_frozen"] is True


def test_admin_forge_stage_a_endpoint_exports_campaign_snapshot() -> None:
    registry = AdapterRegistry()
    registry.register(build_adapter(name="alpha", latency_ms=20.0))
    settings = Settings()
    settings.default_routing_policy = RoutingPolicy.LOCAL_PREFERRED
    settings.phase4.policy_rollout.mode = PolicyRolloutMode.REPORT_ONLY
    settings.phase4.policy_rollout.candidate_policy_id = "adaptive-admin-v2"
    app = create_app(
        registry=registry,
        settings=settings,
    )
    client = TestClient(app)

    response = client.get("/admin/forge/stage-a")

    assert response.status_code == 200
    payload = response.json()
    assert payload["optimization_profile_id"] == "phase9-stage-a-baseline"
    assert payload["active_routing_policy"] == "local_preferred"
    assert payload["active_rollout_mode"] == "report_only"
    assert payload["promotion"]["rollout_mode"] == "canary"
    assert payload["automatic_promotion_enabled"] is False
    assert payload["trial_lineage"][0]["trial_role"] == "baseline"
    assert payload["trial_lineage"][1]["candidate_kind"] == "rollout_policy"
    assert payload["trial_lineage"][1]["required_evaluation_sources"] == [
        "replayed_benchmark",
        "replayed_trace",
        "counterfactual_simulation",
    ]
    assert any(
        trial["candidate_kind"] == "routing_policy" for trial in payload["trial_lineage"][1:]
    )


def test_admin_forge_stage_a_campaign_inspection_endpoint_summarizes_trials() -> None:
    settings = Settings()
    registry = AdapterRegistry()
    registry.register(build_adapter(name="alpha", latency_ms=20.0))
    app = create_app(registry=registry, settings=settings)
    client = TestClient(app)
    campaign_artifact = build_campaign_artifact(settings=settings)
    comparison_artifact = build_comparison_artifact(campaign_artifact.trials[0])

    response = client.post(
        "/admin/forge/stage-a/campaigns/inspect",
        json={
            "campaign_artifacts": [campaign_artifact.model_dump(mode="json")],
            "comparison_artifacts": [comparison_artifact.model_dump(mode="json")],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["campaigns"][0]["campaign_artifact_id"] == "campaign-artifact-phase9-001"
    assert payload["campaigns"][0]["recommendation_status_counts"] == {"promote_candidate": 1}
    assert payload["campaigns"][0]["remote_budget_involved"] is True
    assert payload["campaigns"][0]["helped_workload_families"] == ["repeated_prefix"]
    trial = payload["campaigns"][0]["trials"][0]
    assert trial["recommendation_disposition"] == "promote_candidate"
    assert trial["evidence_kinds"] == ["observed", "replayed"]
    assert trial["helped_workload_families"] == ["repeated_prefix"]
    assert trial["hurt_workload_families"] == ["burst_parallel"]
    assert trial["remote_budget_constraint_outcomes"] == ["remote-share-cap:satisfied"]
    assert trial["diff_entries"][0]["knob_id"] == "default_routing_policy"
    assert trial["provenance"]["baseline_config_profile_id"] == _baseline_config_profile_id(
        settings
    )


def test_admin_forge_promotion_endpoints_apply_and_reset_trial() -> None:
    registry = AdapterRegistry()
    registry.register(build_adapter(name="alpha", latency_ms=20.0))
    settings = Settings()
    app = create_app(registry=registry, settings=settings)
    client = TestClient(app)

    propose_response = client.post(
        "/admin/forge/stage-a/promotion/propose",
        json={
            "trial_artifact": build_promotable_trial_artifact(
                settings=settings,
                canary_percentage=12.5,
            ).model_dump(mode="json"),
        },
    )
    assert propose_response.status_code == 200
    proposed = propose_response.json()
    assert proposed["lifecycle_state"] == "proposed"
    rollout_artifact_id = proposed["rollout_artifact_id"]
    assert rollout_artifact_id is not None

    approve_response = client.post(
        "/admin/forge/stage-a/promotion/approve",
        json={"rollout_artifact_id": rollout_artifact_id},
    )
    assert approve_response.status_code == 200
    assert approve_response.json()["lifecycle_state"] == "approved"

    apply_response = client.post(
        "/admin/forge/stage-a/promotion/apply",
        json={
            "rollout_artifact_id": rollout_artifact_id,
            "canary_percentage": 12.5,
        },
    )

    assert apply_response.status_code == 200
    apply_payload = apply_response.json()
    assert apply_payload["applied"] is True
    assert apply_payload["active_config_profile_id"] == "phase9-local-preferred"
    assert apply_payload["lifecycle_state"] == "canary_active"
    assert apply_payload["rollout_mode"] == "canary"
    assert apply_payload["canary_percentage"] == 12.5

    compare_response = client.post(
        "/admin/forge/stage-a/promotion/compare",
        json={
            "rollout_artifact_id": rollout_artifact_id,
            "comparison_artifact": build_comparison_artifact(
                build_promotable_trial_artifact(
                    settings=settings,
                    canary_percentage=12.5,
                )
            ).model_dump(mode="json"),
        },
    )
    assert compare_response.status_code == 200
    compare_payload = compare_response.json()
    assert compare_payload["lifecycle_state"] == "compared"
    assert compare_payload["comparison"]["rank"] == 1

    promote_response = client.post(
        "/admin/forge/stage-a/promotion/promote-default",
        json={
            "rollout_artifact_id": rollout_artifact_id,
            "reason": "operator reviewed the canary evidence",
        },
    )
    assert promote_response.status_code == 200
    promote_payload = promote_response.json()
    assert promote_payload["lifecycle_state"] == "promoted_default"
    assert promote_payload["rollout_mode"] == "active_guarded"

    runtime = client.get("/admin/runtime")
    assert runtime.status_code == 200
    runtime_payload = runtime.json()
    assert runtime_payload["policy_rollout"]["mode"] == "active_guarded"
    assert runtime_payload["policy_rollout"]["candidate_policy"]["policy_id"] == (
        "phase9-local-preferred"
    )

    promotion = client.get("/admin/forge/stage-a/promotion")
    assert promotion.status_code == 200
    assert promotion.json()["lifecycle_state"] == "promoted_default"

    reset_response = client.post(
        "/admin/forge/stage-a/promotion/reset",
        json={
            "rollout_artifact_id": rollout_artifact_id,
            "reason": "restore baseline",
        },
    )
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["applied"] is False
    assert reset_payload["lifecycle_state"] == "rolled_back"
    assert (
        reset_payload["active_config_profile_id"] == (reset_payload["baseline_config_profile_id"])
    )


def build_promotable_trial_artifact(
    *,
    settings: Settings,
    config_profile_id: str = "phase9-local-preferred",
    routing_policy: RoutingPolicy = RoutingPolicy.LOCAL_PREFERRED,
    canary_percentage: float = 10.0,
) -> OptimizationTrialArtifact:
    trial_identity = OptimizationTrialIdentity(
        trial_id=f"trial-{config_profile_id}",
        candidate_id=f"routing-policy:{routing_policy.value}",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=config_profile_id,
        routing_policy=routing_policy,
    )
    candidate_configuration = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id=f"candidate-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        candidate=trial_identity,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        config_profile_id=config_profile_id,
        knob_changes=[
            OptimizationKnobChange(
                knob_id="default_routing_policy",
                config_path="default_routing_policy",
                baseline_value=RoutingPolicy.BALANCED.value,
                candidate_value=routing_policy.value,
            )
        ],
    )
    return OptimizationTrialArtifact(
        trial_artifact_id=f"trial-artifact-{config_profile_id}",
        campaign_id="campaign-phase9-001",
        baseline_candidate_configuration_id="candidate-phase9-baseline",
        candidate_configuration=candidate_configuration,
        trial_identity=trial_identity,
        evidence_records=[
            OptimizationEvidenceRecord(
                evidence_id=f"evidence-observed-{config_profile_id}",
                evidence_kind=OptimizationArtifactEvidenceKind.OBSERVED,
                source_type=OptimizationArtifactSourceType.BENCHMARK_RUN,
                source_artifact_id="benchmark-phase9-001",
            ),
            OptimizationEvidenceRecord(
                evidence_id=f"evidence-replayed-{config_profile_id}",
                evidence_kind=OptimizationArtifactEvidenceKind.REPLAYED,
                source_type=OptimizationArtifactSourceType.REPLAY_PLAN,
                source_artifact_id="replay-phase9-001",
            ),
        ],
        constraint_assessments=[
            OptimizationConstraintAssessment(
                constraint_id="remote-share-cap",
                dimension=OptimizationConstraintDimension.REMOTE_SHARE_PERCENT,
                strength=OptimizationConstraintStrength.HARD,
                operator=OptimizationComparisonOperator.LTE,
                threshold_value=25.0,
                evaluated_value=18.0,
                satisfied=True,
                evidence_kinds=[OptimizationArtifactEvidenceKind.OBSERVED],
            )
        ],
        recommendation_summary=OptimizationRecommendationSummary(
            recommendation_summary_id=f"recommendation-{config_profile_id}",
            disposition=OptimizationRecommendationDisposition.PROMOTE_CANDIDATE,
            recommendation_label=OptimizationRecommendationLabel.PROMOTION_ELIGIBLE,
            confidence=RecommendationConfidence.MEDIUM,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=config_profile_id,
            evidence_kinds=[
                OptimizationArtifactEvidenceKind.OBSERVED,
                OptimizationArtifactEvidenceKind.REPLAYED,
            ],
            benefited_workload_families=["repeated_prefix"],
            regressed_workload_families=["burst_parallel"],
            reason_codes=[OptimizationRecommendationReasonCode.PRIMARY_OBJECTIVE_IMPROVED],
            rationale=["candidate improved the primary objective"],
        ),
        promotion_decision=OptimizationPromotionDecision(
            promotion_decision_id=f"promotion-{config_profile_id}",
            disposition=OptimizationPromotionDisposition.RECOMMEND_CANARY,
            candidate_configuration_id=candidate_configuration.candidate_configuration_id,
            config_profile_id=config_profile_id,
            rollout_mode=PolicyRolloutMode.CANARY,
            canary_percentage=canary_percentage,
        ),
    )


def build_campaign_artifact(*, settings: Settings) -> OptimizationCampaignArtifact:
    baseline_trial_identity = OptimizationTrialIdentity(
        trial_id="trial-baseline",
        candidate_id="routing-policy:balanced",
        candidate_kind=ForgeCandidateKind.ROUTING_POLICY,
        config_profile_id=_baseline_config_profile_id(settings),
        routing_policy=RoutingPolicy.BALANCED,
    )
    baseline_candidate = OptimizationCandidateConfigurationArtifact(
        candidate_configuration_id="candidate-phase9-baseline",
        campaign_id="campaign-phase9-001",
        candidate=baseline_trial_identity,
        baseline_config_profile_id=_baseline_config_profile_id(settings),
        config_profile_id=_baseline_config_profile_id(settings),
    )
    trial = build_promotable_trial_artifact(settings=settings)
    assert trial.recommendation_summary is not None
    assert trial.promotion_decision is not None
    return OptimizationCampaignArtifact(
        campaign_artifact_id="campaign-artifact-phase9-001",
        campaign=OptimizationCampaignMetadata(
            campaign_id="campaign-phase9-001",
            optimization_profile_id="phase9-stage-a-baseline",
            objective=CounterfactualObjective.LATENCY,
            evidence_sources=[
                ForgeEvidenceSourceKind.OBSERVED_RUNTIME,
                ForgeEvidenceSourceKind.REPLAYED_BENCHMARK,
            ],
            required_evidence_sources=[ForgeEvidenceSourceKind.REPLAYED_BENCHMARK],
            default_workload_set_ids=["phase9-cache-locality"],
        ),
        baseline_candidate_configuration=baseline_candidate,
        candidate_configurations=[trial.candidate_configuration],
        trials=[trial],
        evidence_records=list(trial.evidence_records),
        recommendation_summaries=[trial.recommendation_summary],
        promotion_decisions=[trial.promotion_decision],
    )


def build_comparison_artifact(
    trial: OptimizationTrialArtifact,
) -> OptimizationCampaignComparisonArtifact:
    recommendation = trial.recommendation_summary
    assert recommendation is not None
    return OptimizationCampaignComparisonArtifact(
        comparison_artifact_id=f"comparison-{trial.campaign_id}",
        campaign_id=trial.campaign_id,
        baseline_candidate_configuration_id=trial.baseline_candidate_configuration_id,
        candidate_comparisons=[
            OptimizationCandidateComparisonArtifact(
                candidate_configuration_id=(
                    trial.candidate_configuration.candidate_configuration_id
                ),
                trial_artifact_id=trial.trial_artifact_id,
                config_profile_id=trial.candidate_configuration.config_profile_id,
                rank=1,
                pareto_optimal=True,
                dominated=False,
                recommendation_summary=recommendation,
                notes=["artifact-backed canary comparison favored the candidate"],
            )
        ],
    )


def _baseline_config_profile_id(settings: Settings) -> str:
    return build_baseline_optimization_config_profile(settings).config_profile_id
