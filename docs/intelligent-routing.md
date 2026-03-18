# Intelligent Routing

Phase 6 adds a typed routing-intelligence layer on top of the Phase 5 control plane.
The goal is to make routing smarter without making it opaque. The route path, the
evidence path, and the rollout path remain separate and inspectable.

## What Phase 6 Adds

- deterministic request and workload feature extraction before scoring,
- repeated-prefix and locality-aware signals that rely on hashes and counters rather
  than stored prompt text,
- historical performance summaries and transparent candidate-estimate inputs,
- a richer scorer and policy abstraction with per-candidate reasoning and shadow
  scoring,
- offline simulation that distinguishes observed outcomes from estimates,
- a conservative adaptive policy with abstention and guardrails,
- local-first rollout controls for shadow, report-only, canary, and guarded-active
  modes,
- evidence-based recommendation reports derived from authoritative artifacts.

## Request And Workload Features

`switchyard.router.features` extracts deterministic request features before routing.
These features are serializable and appear in route explanations, benchmark artifacts,
and replay flows.

Current feature families include:

- message counts and prompt-size estimates,
- input-length and history-depth buckets,
- request class and tenant tier,
- repeated-prefix candidates,
- locality keys and prefix fingerprints,
- workload tags such as `short_chat`, `long_context`, `repeated_prefix`, and
  `latency_sensitive`.

Trust boundary:

- the control plane stores hashes and coarse buckets for locality work,
- raw prompt text is not retained as a locality signal,
- routing logic stays backend-agnostic and does not inspect Apple-specific runtime state.

## Prefix And Locality Signals

`switchyard.control.locality.PrefixLocalityService` tracks recent repeated-prefix
activity by serving target and locality key. It emits a `PrefixLocalitySignal` with:

- hotness (`cold`, `warm`, `hot`),
- cache opportunity,
- whether locality is likely to help,
- a preferred backend or instance when recent evidence exists,
- affinity conflicts when locality and session stickiness disagree.

These signals are hints, not hidden routing authority. They are recorded in route
artifacts so operators can see when locality influenced a decision or when the system
intentionally ignored it.

## Historical Performance Summaries

`switchyard.bench.history` aggregates benchmark and replay records into typed historical
summaries. Candidate estimates are derived from:

- model alias,
- backend name and optional instance,
- request class,
- input-size and workload-tag buckets,
- locality-related signals when available,
- optional tenant scoping.

The predictor falls back through broader slices when narrow slices do not have enough
samples. That fallback is visible in the estimate rationale.

## Scorer And Policy Abstraction

`switchyard.router.policies` exposes a small scorer contract:

- candidate enumeration stays in the router,
- each scorer evaluates candidates and returns per-candidate scores or rejections,
- every evaluation carries policy IDs, versions, reason codes, and rationale,
- older fixed policies still run through compatibility wrappers,
- shadow scorers stay non-binding.

This is intentionally simple. The router stays a plain Python service rather than a
policy framework with hidden lifecycle rules.

## Offline Simulation

`switchyard.bench.simulation` evaluates policies against benchmark artifacts and
captured traces. Simulation outputs are explicit about evidence quality:

- `direct_observation`
- `predictor_estimate`
- `low_confidence_estimate`
- `unsupported`

Simulation limitations are part of the contract:

- it is not counterfactual ground truth,
- unsupported candidates stay unsupported,
- thin slices remain low confidence,
- guarded policies may block otherwise higher-scoring recommendations.

Use it for decision support, not for automatic promotion.

Example:

```bash
uv run python -m switchyard.bench.cli compare-offline-policies \
  --trace-path traces/captured.jsonl \
  --routing-policy balanced \
  --candidate-policy adaptive-safe:balanced \
  --markdown-report
```

## Adaptive Policy Design

The first adaptive policy is intentionally transparent and conservative.

It uses historical estimates rather than an opaque online learner. It supports:

- fallback to a fixed compatibility policy,
- minimum sample thresholds,
- confidence-aware abstention,
- bounded deterministic exploration,
- optional tenant scoping,
- explicit abstention, fallback, estimate, and exploration reason codes.

Guardrails currently include:

- abstain when there is not enough evidence,
- abstain when the top candidates are too close,
- avoid degraded backends,
- reject candidates above the predicted error-rate limit,
- ignore exploration in deterministic evaluation mode.

The adaptive policy is opt-in and does not replace baseline routing by default.

## Safe Rollout Modes

`switchyard.control.policy_rollout.PolicyRolloutService` provides local-first controls
for intelligent policies:

- `disabled`
- `shadow_only`
- `report_only`
- `canary`
- `active_guarded`

The rollout controller also supports:

- kill switch,
- learning freeze,
- reset,
- export/import,
- bounded recent-decision history,
- runtime inspection of active policy, shadow policy, abstentions, exploration, and
  guardrail triggers.

Example commands against a gateway with a registered candidate policy:

Inspect runtime state:

```bash
curl -s http://127.0.0.1:8000/admin/policy-rollout | python -m json.tool
```

Enable shadow scoring:

```bash
curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"shadow_only"}' \
  | python -m json.tool
```

Enable adaptive-policy canary mode:

```bash
curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"canary","canary_percentage":10.0}' \
  | python -m json.tool
```

Disable rollout cleanly:

```bash
curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"disabled","kill_switch_enabled":true}' \
  | python -m json.tool
```

## Policy Recommendation Reports

`switchyard.bench.recommendations` turns benchmark and simulation artifacts into
human-readable guidance. Recommendation reports may identify:

- a preferred policy for an alias,
- a preferred policy for a request class,
- a case where the adaptive policy should remain shadow-only,
- a strong or weak backend or instance for repeated-prefix traffic,
- a case where the evidence is too thin to recommend change.

Every recommendation includes:

- evidence windows,
- sample sizes,
- workload buckets,
- a recommendation or explicit no-change result,
- caveats and confidence notes,
- notable regressions or counterexamples.

Example:

```bash
uv run python -m switchyard.bench.cli recommend-policies \
  benchmarks/20260317T000000Z_balanced.json \
  benchmarks/20260317T001500Z_policy-comparison.json \
  --markdown-report
```

## Phase 6 And Future Cloud Expansion

Phase 6 is still Mac-first, but the intelligence layer is intentionally portable.

- Request features, historical summaries, and policy explanations are runtime-agnostic.
- Worker inventory is typed so instance-aware signals can later span cloud workers.
- Offline simulation and recommendation reports do not require Apple GPU access.
- The adaptive policy depends on artifact-backed evidence rather than Apple-specific
  telemetry.

This is also the preparation work for later Forge-style tuning:

- route decisions are now serializable enough to compare policy changes safely,
- recommendation reports make limitations explicit,
- rollout controls keep future learned or optimized policies bounded and reversible.
