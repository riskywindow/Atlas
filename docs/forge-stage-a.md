# Forge Stage A: Evidence-Driven Autotuning

Forge Stage A is the first optimization layer in Switchyard.  It does not generate
code, synthesize kernels, or auto-promote untested configurations into production.
Instead it provides a structured, artifact-backed workflow for:

1. declaring what is safely tunable,
2. generating and pruning candidate configurations,
3. evaluating candidates offline against benchmark and replay evidence,
4. ranking candidates with explicit multi-objective comparison,
5. surfacing honest recommendations with typed trust caveats,
6. promoting reviewed candidates through bounded canary rollout,
7. rolling back to the baseline at any point.

The entire flow is read-only until an operator explicitly proposes, approves, and
applies a promotion.

## Why Forge Stage A Exists

Switchyard already has explainable routing, hybrid local/remote guardrails, and
reproducible benchmark artifacts.  What was missing was a principled way to ask:

> *Given the evidence we have, should we change the active routing policy or
> hybrid budget posture, and can we do that safely?*

Forge Stage A answers that question with typed artifacts rather than ad hoc
judgment.  It keeps recommendations conservative and reversible so that operators
never have to trust a black-box decision.

## The Optimization Surface

Forge Stage A exposes a typed knob surface through `OptimizationProfile`.  Each
knob declares its current value, allowed domain, and whether it can be mutated at
runtime.

### Safely Tunable Knobs

| Knob group | Examples | Runtime mutable |
|---|---|---|
| Routing policy | `default_routing_policy` | Yes (via canary rollout) |
| Policy rollout | `policy_rollout_mode`, `canary_percentage` | Yes |
| Hybrid execution | `hybrid_spillover_enabled`, `hybrid_max_remote_share_percent`, `remote_request_budget_per_minute`, `remote_concurrency_cap` | Yes (via spillover controller) |
| Shadow routing | `shadow_sampling_rate` | Yes |

### What Is NOT Tunable in Stage A

- Kernel parameters, compilation flags, or runtime code generation.
- Backend-specific adapter internals (MLX-LM, vLLM-Metal, CUDA).
- Model loading or quantization choices.
- Anything that requires a control-plane restart or infrastructure change.

These are explicitly listed in `OptimizationProfile.excluded_dimensions`.

### Inspecting the Current Surface

```bash
uv run switchyard-control-plane export-optimization-profile | python -m json.tool
```

This emits the resolved `OptimizationProfile` as JSON: knobs, objectives,
constraints, workload sets, campaign metadata, and promotion guardrails.

## Evidence Semantics

Forge Stage A keeps four kinds of evidence explicitly separate in every typed
surface.  They are never collapsed into a single score.

| Kind | Meaning | Trust level |
|---|---|---|
| **Observed** | A real request was executed and the outcome was directly measured. | Highest |
| **Replayed** | A captured trace was replayed through the gateway and the result recorded. | High (but may not reflect current conditions) |
| **Simulated** | An offline simulation projected what would happen under a different policy. | Medium (counterfactual, not observed) |
| **Estimated** | A predictor or heuristic filled in a value that was not directly measurable. | Low (explicitly marked) |

These labels appear in:
- `OptimizationEvidenceRecord.evidence_kind`
- `OptimizationArtifactEvidenceKind` tags on every trial and recommendation
- `OptimizationEvidenceMixSummary` in comparison outputs
- `ForgeHonestyWarningSummary` in inspection views

### What Claims Should NOT Be Made From Replay-Only Evidence

Replay evidence reflects how the system *would have routed* a historical workload
under a different policy.  It does not prove:

- that the candidate policy will perform the same way under current topology,
- that remote workers will be available with the same latency characteristics,
- that the budget posture has not changed since the replay was captured,
- that workload patterns will remain similar.

When a promotion recommendation is backed only by replayed or simulated evidence
(no observed evidence), the system:
1. sets `recommendation_label` to `REVIEW_ONLY` rather than `PROMOTION_ELIGIBLE`,
2. adds the `OBSERVED_EVIDENCE_MISSING` reason code,
3. triggers a honesty warning in the inspection view.

Operators should treat replay-only recommendations as decision support, not as
proof of improvement.

## Running a Local-Only Optimization Campaign

A local campaign generates candidates, evaluates them offline against benchmark
artifacts, and produces ranked recommendations without touching live routing.

### Step 1: Run a Benchmark

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --markdown-report
```

### Step 2: Compare Policies Offline

```bash
uv run python -m switchyard.bench.cli compare-offline-policies \
  --artifact-path artifacts/benchmarks/<run-id>.json \
  --routing-policy balanced \
  --routing-policy latency_first \
  --markdown-report
```

### Step 3: Inspect the Campaign Snapshot

```bash
uv run switchyard-control-plane export-forge-stage-a-campaign | python -m json.tool
```

### Step 4: Run the Offline Campaign Executor

The campaign executor is invoked programmatically (see `execute_forge_stage_a_campaign`
in `switchyard.bench.campaigns`).  It takes benchmark artifacts as input and
produces:

- `OptimizationCampaignArtifact` — the authoritative campaign result
- `OptimizationCampaignComparisonArtifact` — ranked candidate comparisons
- `PolicyRecommendationReportArtifact` — human-readable recommendation report

### Step 5: Inspect the Result

```bash
uv run switchyard-control-plane inspect-forge-stage-a-campaign \
  --artifact-path artifacts/campaigns/<campaign-artifact>.json \
  --comparison-artifact-path artifacts/campaigns/<comparison-artifact>.json
```

This prints a markdown summary with:
- per-trial recommendation disposition and confidence,
- objective deltas and constraint outcomes,
- workload-family impacts,
- evidence kinds and honesty warnings.

## Running a Hybrid Campaign with Bounded Remote Budget

For campaigns that include remote/cloud workers, Forge Stage A enforces explicit
budget constraints.

### Budget Constraints

The optimization profile exports these as typed constraints:

- `REMOTE_SHARE_PERCENT` — maximum percentage of traffic that may go to remote workers
- `REMOTE_REQUEST_BUDGET_PER_MINUTE` — hard cap on remote requests per minute
- `REMOTE_CONCURRENCY_CAP` — maximum concurrent remote requests

Candidates that violate these constraints are:
1. **pruned** before execution (pre-execution hard constraint check), or
2. **rejected** during comparison (hard constraint violation in trial assessment).

### Honesty Checks

When inspecting a campaign against the *current* environment, the honesty
assessment warns if:

- the current remote budget is lower than what the campaign assumed,
- the current remote share cap is lower than what trials evaluated,
- workers from the campaign are no longer in the topology (topology drift),
- the evidence is older than the staleness threshold.

These warnings appear in the `honesty_warnings` field of
`ForgeCampaignInspectionSummary` and are surfaced through the inspection CLI and
admin endpoints.

## Inspecting Recommendations and Promotions

### Recommendation Dispositions

Each trial recommendation carries one of:

| Disposition | Meaning |
|---|---|
| `PROMOTE_CANDIDATE` | Evidence supports promotion through a bounded canary |
| `KEEP_BASELINE` | Hard constraints failed or the candidate is dominated |
| `NEED_MORE_EVIDENCE` | Required evidence sources were not present |
| `INVALIDATE_TRIAL` | Campaign conditions changed and the trial should not be trusted |
| `NO_CHANGE` | No evidence justified changing the baseline |

### Recommendation Labels

| Label | Meaning |
|---|---|
| `PROMOTION_ELIGIBLE` | Can proceed to canary if reviewed |
| `REVIEW_ONLY` | Needs operator judgment before any action |
| `REJECTED` | Hard constraint violation or dominated |

### Admin Endpoints

```bash
# Current Forge campaign snapshot
curl -s http://127.0.0.1:8000/admin/forge/stage-a | python -m json.tool

# Current promotion/rollout state
curl -s http://127.0.0.1:8000/admin/forge/stage-a/promotion | python -m json.tool

# Inspect a campaign artifact (POST with artifact body)
curl -sS http://127.0.0.1:8000/admin/forge/stage-a/campaigns/inspect \
  -H 'content-type: application/json' \
  -d @artifacts/campaigns/inspection-request.json | python -m json.tool
```

## Canary and Rollback Workflow

Forge Stage A promotion follows an explicit, multi-step lifecycle:

```text
PROPOSED -> APPROVED -> CANARY_ACTIVE -> COMPARED -> PROMOTED_DEFAULT
    |           |            |              |
    +-----------+------------+--------------+---> REJECTED
                             |              |
                             +--------------+---> ROLLED_BACK
```

Every transition is recorded as a `ForgePromotionLifecycleEvent`.

### Step 1: Propose

```bash
uv run switchyard-control-plane propose-forge-stage-a-promotion \
  --artifact-path artifacts/campaigns/<trial-or-campaign>.json \
  --gateway-base-url http://127.0.0.1:8000
```

This registers the trial as a promotion proposal without changing runtime state.

### Step 2: Approve

```bash
uv run switchyard-control-plane approve-forge-stage-a-promotion \
  --rollout-artifact-id <rollout-id> \
  --gateway-base-url http://127.0.0.1:8000
```

### Step 3: Apply as Canary

```bash
uv run switchyard-control-plane apply-forge-stage-a-promotion \
  --rollout-artifact-id <rollout-id> \
  --canary-percentage 10.0 \
  --gateway-base-url http://127.0.0.1:8000
```

This activates the candidate routing policy as a bounded canary.  The canary
percentage is capped by `OptimizationSettings.max_rollout_canary_percentage`.

### Step 4: Compare

Run a benchmark or replay against the live canary, then attach the comparison:

```bash
uv run switchyard-control-plane compare-forge-stage-a-promotion \
  --rollout-artifact-id <rollout-id> \
  --artifact-path artifacts/campaigns/<comparison-artifact>.json \
  --gateway-base-url http://127.0.0.1:8000
```

### Step 5: Promote or Roll Back

Promote to default (requires comparison evidence):

```bash
uv run switchyard-control-plane promote-default-forge-stage-a-promotion \
  --rollout-artifact-id <rollout-id> \
  --reason "canary showed 12% latency improvement across mixed workloads" \
  --gateway-base-url http://127.0.0.1:8000
```

Or roll back at any point:

```bash
uv run switchyard-control-plane reset-forge-stage-a-promotion \
  --rollout-artifact-id <rollout-id> \
  --gateway-base-url http://127.0.0.1:8000
```

Or reject the proposal entirely:

```bash
uv run switchyard-control-plane reject-forge-stage-a-promotion \
  --rollout-artifact-id <rollout-id> \
  --reason "observed regression under tenant-contention workload" \
  --gateway-base-url http://127.0.0.1:8000
```

Rollback and rejection restore the pre-promotion runtime state automatically.

## Trust Boundaries

### What Forge Stage A Can Do

- Generate, prune, and rank candidate configurations.
- Compare candidates against offline evidence with explicit multi-objective tradeoffs.
- Surface recommendations with typed evidence posture and honesty warnings.
- Apply one reviewed candidate as a bounded canary with explicit rollback.
- Promote a compared canary to the default policy through operator review.

### What Forge Stage A Cannot Do

- Auto-promote without operator review (unless explicitly disabled, which is not
  recommended).
- Generate or modify backend runtime code, kernels, or model weights.
- Tune parameters that require infrastructure changes or control-plane restarts.
- Guarantee that replay-only evidence predicts production behavior.
- Override hybrid budget guardrails or cloud rollout gates.

### Where Evidence Stops Being Trustworthy

The campaign honesty assessment (`assess_campaign_honesty`) checks for:

| Warning kind | When it fires |
|---|---|
| `BUDGET_BOUND_EXCEEDED` | Campaign assumed higher remote budget than currently configured |
| `TOPOLOGY_DRIFT` | Workers from campaign evidence are missing or new workers appeared |
| `STALE_EVIDENCE` | Evidence windows are older than the configured staleness threshold |
| `NARROW_WORKLOAD_COVERAGE` | Campaign covers too few workload families (overfit risk) |
| `EVIDENCE_INCONSISTENCY` | Estimated evidence outweighs observed evidence in trial inputs |
| `OBSERVED_EVIDENCE_MISSING` | Promotion recommendations exist with no observed runtime backing |

These warnings appear in the inspection view and should be reviewed before acting
on any recommendation.

## Preparing for Forge Stage B

Phase 9 explicitly prepares for later Forge Stage B without implementing it.

### What Stage B May Add

- Adaptive policy parameter tuning (scorer weights, thresholds, feature selection).
- Runtime-profile optimization with live A/B evidence.
- Multi-gateway coordination for rollout state.
- Kernel-level or compilation-level tuning behind backend adapter boundaries.

### What Stage A Preserves for Stage B

- **Typed evidence lineage**: every campaign, trial, and recommendation carries
  explicit evidence records with source types and evidence kinds.
- **Serializable optimization profiles**: the knob surface, objectives, constraints,
  and workload sets are all Pydantic models that can be extended.
- **Config profile diffs**: every promotion produces a typed diff showing what
  changed from the baseline.
- **Topology lineage**: campaigns record which workers and endpoints were present
  when evidence was collected.
- **Honesty assessment**: Stage B can reuse the same trust-boundary checks to
  validate whether Stage A results remain applicable.

### What Stage A Does NOT Build for Stage B

- No kernel code generation infrastructure.
- No compiler or quantization pipeline.
- No live learned-policy feedback loop.
- No multi-service optimization coordinator.

The boundary is intentional: Stage A proves that the typed surfaces, evidence
semantics, and promotion lifecycle work before Stage B adds richer automation.

## Key Source Files

| File | Purpose |
|---|---|
| `src/switchyard/schemas/optimization.py` | Typed schemas for profiles, campaigns, trials, recommendations |
| `src/switchyard/schemas/forge.py` | Promotion lifecycle, inspection, and honesty warning schemas |
| `src/switchyard/optimization.py` | Profile builder and config-profile materializer |
| `src/switchyard/bench/campaigns.py` | Offline campaign executor and inspection |
| `src/switchyard/bench/campaign_comparison.py` | Multi-objective candidate ranking |
| `src/switchyard/bench/candidate_generation.py` | Candidate generation and pruning |
| `src/switchyard/bench/campaign_honesty.py` | Honesty assessment and staleness marking |
| `src/switchyard/control/forge_promotion.py` | Bounded promotion lifecycle service |
| `src/switchyard/gateway/routes.py` | Admin endpoints for Forge inspection and promotion |
| `src/switchyard/control_plane/cli.py` | CLI commands for Forge workflows |
