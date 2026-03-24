# Architecture

Switchyard is in Phase 9 on top of the Phase 6, Phase 7, and Phase 8 control-plane
foundation: two real Mac-native backend families, logical alias routing, explicit
worker inventory, conservative health-aware fallback, reproducible benchmark and trace
workflows, typed routing intelligence, remote-worker lifecycle posture, hybrid
local/remote guardrails, operator-visible budget state, and the first real
Linux/NVIDIA worker path. Phase 9 adds Forge Stage A evidence-driven autotuning
surfaces on top of that base while keeping observed evidence distinct from replayed,
simulated, or estimated evidence and keeping promotion bounded, explainable, and
reversible. The core design goal is still portability: Apple-specific and CUDA-specific
runtime details stop at the adapter/runtime boundary so the same contracts can host
local Apple workers and real cloud workers.

## Core Terms

### Logical Model Alias

The logical alias, also called the serving target, is what clients send in
`ChatCompletionRequest.model`.

Examples:
- `chat-mlx`
- `chat-metal`
- `chat-shared`

The logical alias is the public serving contract. Clients should not need to know whether
the request is ultimately served by MLX-LM, vLLM-Metal, a future CUDA worker, or a remote
OpenAI-like adapter.

### Backend Deployment

A backend deployment is one concrete adapter instance registered in the control plane.

Examples:
- `mlx-lm:chat-mlx`
- `vllm-metal:chat-metal`
- future `vllm-cuda:chat-cuda`

Each deployment declares:
- backend type,
- device class,
- health and readiness,
- streaming support,
- context window,
- quality and performance hints,
- configured priority and weight,
- the logical serving targets it can satisfy.

### Worker Instance

A worker instance is one addressable serving endpoint behind a backend deployment.

Examples:
- `mlx-local-1` at `http://127.0.0.1:9001`
- `vllm-metal-local-1` at `http://127.0.0.1:9002`
- future remote worker addresses reachable from a portable control plane

Worker instances matter in Phase 5 because deployment topology must be explicit and
explainable in config, runtime inspection, and benchmark artifacts rather than inferred
only from deployment names.

### Why The Distinction Matters

A single logical alias may map to one deployment or many.

Single-backend alias:
- `chat-mlx` -> `mlx-lm:chat-mlx`

Dual-backend alias:
- `chat-shared` -> `mlx-lm:chat-mlx`
- `chat-shared` -> `vllm-metal:chat-metal`

This lets Switchyard keep a simple public API while still routing, benchmarking, and
failing over across multiple concrete implementations.

## Main Components

- `switchyard.schemas`
  Typed request, response, backend, routing, and benchmark contracts.
- `switchyard.adapters`
  Shared adapter boundary plus concrete `MLXLMAdapter`, `VLLMMetalAdapter`, and the mock
  backend.
- `switchyard.runtime`
  Provider/runtime boundaries for version-sensitive backend logic such as MLX-LM and
  vLLM-Metal.
- `switchyard.adapters.registry`
  Maps logical serving targets to one or more concrete backend deployments and exposes
  target-level snapshots.
- `switchyard.router`
  Pure Python policy selection with deterministic candidate explanations.
- `switchyard.control`
  Local in-process admission control, circuit breakers, session affinity, canary
  routing, and best-effort shadow orchestration.
- `switchyard.gateway`
  FastAPI entrypoint that builds request context, asks the router for a decision, executes
  the chosen backend, and handles safe fallback.
- `switchyard.telemetry` and `switchyard.logging`
  Local-friendly metrics and structured logs for route decisions, attempts, execution,
  warmup, and fallback.
- `switchyard.bench`
  Synthetic and live gateway benchmarks that serialize route choices, overload
  outcomes, rollout decisions, and performance outcomes into reproducible artifacts and
  derived reports.  Also includes the Forge Stage A offline campaign executor,
  multi-objective candidate ranking, candidate generation, and campaign honesty
  assessment.
- `switchyard.optimization`
  Profile builder that resolves current settings into a serializable optimization
  profile of knobs, objectives, constraints, and promotion guardrails.

## Control-Plane To Worker Boundary

Phase 7 keeps the Phase 5 execution boundary and extends it with remote worker
lifecycle, hybrid placement controls, and operator inspection.

```text
client
  -> FastAPI gateway
       -> request context, admission, routing, affinity, rollout, spillover, telemetry
       -> adapter registry and backend-instance inventory
       -> remote worker registry and hybrid budget state
       -> local adapter call OR HTTP worker protocol
  -> worker process
       -> health / readiness / capabilities / warmup / generate / stream
       -> backend runtime (MLX-LM, vLLM-Metal today; remote GPU workers later)
```

Rules at this boundary:

- the control plane reasons about capabilities, health, policy, and inventory, not Apple
  GPU implementation details,
- host-native Apple workers are still first-class, but they are now explicitly
  addressable worker processes,
- remote workers are first-class topology members with the same inventory and artifact
  contracts,
- the same boundary is what later enables remote CUDA or cloud GPU workers.

The current internal worker protocol surface is intentionally small:

- `GET /healthz`
- `GET /internal/worker/ready`
- `GET /internal/worker/capabilities`
- `POST /internal/worker/warmup`
- `POST /internal/worker/generate`
- `POST /internal/worker/generate/stream`

This keeps the control plane portable while preserving public-path parity where practical.

## Phase 5 Control-Plane Layers

Phase 5 keeps the request path layered rather than burying policy in route handlers.

- request classification
  The gateway parses tenant ID, request class, session ID, and optional internal
  backend pins from explicit Switchyard headers.
- admission control
  An in-process bounded admission service decides admit, queue, or reject before
  execution begins.
- routing and eligibility
  The router resolves the logical alias, applies health checks, policy scoring, local
  constraints, and backend protection state.
- backend protection
  A local circuit breaker records repeated invocation failures and temporarily protects
  open backends from more traffic.
- stickiness and rollout
  Session affinity can reuse a healthy sticky backend. Canary routing can deterministically
  redirect a bounded percentage of eligible traffic to a candidate backend.
- primary execution and fallback
  The gateway executes the selected backend and may perform one conservative fallback
  when it is still safe to do so.
- observational shadowing
  Best-effort shadow traffic is launched separately and never changes the primary
  user-visible response.
- runtime inspection and artifacts
  Runtime state is exposed through a read-only admin endpoint, while benchmark/replay
  artifacts remain the historical source of truth for deployed topology and routing
  behavior.

## Phase 7 Hybrid Control Path

Phase 7 adds three explicit layers on top of the earlier local-first control plane:

- remote worker lifecycle
  Dynamic registration, heartbeat, draining, quarantine, and stale-worker cleanup stay
  outside the router and are visible through admin surfaces.
- hybrid spillover guardrails
  Remote enablement, budget, concurrency, cooldown, and tenant restrictions are enforced
  by a dedicated spillover service.
- operator runtime insight
  Recent hybrid route examples, remote transport failures, and effective remote posture
  are retained separately from benchmark artifacts for day-to-day operations.

This keeps the system explainable:

- routing still selects candidates from typed topology,
- lifecycle state is explicit rather than hidden in transport failures,
- operator overrides do not rewrite artifact truth.

## Benchmark, Trace, And Report Placement

Phase 3 adds four closely related subsystems around the existing control plane:

- workload generation,
- benchmark execution,
- optional trace capture and replay,
- artifact-derived reporting.

They fit into the architecture like this:

- workload generation lives in `switchyard.bench.workloads`
  It builds deterministic workload manifests from typed scenario families and seeds.
- benchmark execution lives in `switchyard.bench.artifacts` and `switchyard.bench.cli`
  It drives either local adapters or the normal gateway path and writes authoritative
  benchmark artifacts.
- trace capture lives in `switchyard.gateway.trace_capture`
  It is an opt-in gateway concern that records replay-oriented JSONL traces without
  changing routing behavior.
- trace replay lives in `switchyard.bench.artifacts` and `switchyard.bench.cli`
  It turns captured traces into typed replay plans and executes them through the same
  gateway-facing benchmark path.
- reporting lives in `switchyard.bench.artifacts` and `switchyard.bench.cli`
  It renders markdown from benchmark artifacts after the fact rather than inventing a
  separate analytics store.

This split matters because benchmark and replay workflows should remain backend-agnostic
extensions of the control plane, not a second system with its own contracts.

## Optimization-Ready Control Surface

Phase 7 also exposes a typed optimization-ready surface for later Forge Stage A work.
This stays separate from the live request path.

- `switchyard.config.OptimizationSettings` defines allowlisted routing policies,
  rollout modes, evidence thresholds, bounded tuning ranges, and worker-launch presets.
- `switchyard.optimization.build_optimization_profile` resolves the current settings
  into a serializable profile of knobs, current values, and promotion constraints.
- `BenchmarkRunConfig` and `ReplayPlan` now carry an immutable config snapshot and
  deterministic fingerprint for experiment truthfulness.
- `switchyard-control-plane export-optimization-profile` emits that profile as JSON for
  offline tuning, replay, and simulation workflows.

This keeps later optimization work reviewable and reproducible:

- tuning inputs are typed and diffable,
- local-first and spillover constraints remain explicit,
- operator review is still required before promotion.

## Forge Stage A: Evidence-Driven Autotuning

Phase 9 activates Forge Stage A on top of the optimization-ready surface.  Forge Stage A
does not generate code, synthesize kernels, or auto-promote untested configurations.
Instead it provides a structured, artifact-backed workflow for generating candidate
configurations, evaluating them offline, ranking them with explicit tradeoffs, and
promoting reviewed candidates through bounded canary rollout.

### Three-Layer Model

1. **Offline Campaign Evaluation** — campaigns generate, prune, and evaluate candidates
   against benchmark and replay evidence without touching live routing.  The output is a
   set of typed artifacts: `OptimizationCampaignArtifact`, comparison rankings, and
   per-trial recommendation summaries.

2. **Honesty Assessment** — before acting on recommendations, the system validates them
   against current environment state: budget bounds, topology drift, evidence staleness,
   workload coverage, and evidence consistency.  Warnings are typed and surfaced in
   operator inspection views.

3. **Bounded Promotion Lifecycle** — promotion follows an explicit multi-step lifecycle
   (`PROPOSED -> APPROVED -> CANARY_ACTIVE -> COMPARED -> PROMOTED_DEFAULT`) with
   rollback at every stage.

### Evidence Semantics

Evidence kinds are explicitly separate in every surface:

- **Observed**: real request executed and outcome directly measured.
- **Replayed**: captured trace replayed through the gateway.
- **Simulated**: offline simulation projected what would happen.
- **Estimated**: predictor or heuristic filled in a value.

These labels appear in campaign trials, recommendation summaries, comparison outputs,
and honesty warnings.  They are never collapsed into a single score.

### Campaign Honesty

The honesty assessment (`assess_campaign_honesty`) checks for:

- budget bounds exceeded (campaign assumed higher remote budget than currently configured),
- topology drift (workers from campaign evidence are missing or new workers appeared),
- stale evidence (evidence older than the configured staleness threshold),
- narrow workload coverage (campaign covers too few workload families),
- evidence inconsistency (estimated evidence outweighs observed evidence),
- observed evidence missing (promotion recommendations with no observed runtime backing).

These warnings appear in `ForgeCampaignInspectionSummary` and must be reviewed before
acting on any recommendation.

### Promotion Lifecycle

```text
PROPOSED -> APPROVED -> CANARY_ACTIVE -> COMPARED -> PROMOTED_DEFAULT
    |           |            |              |
    +-----------+------------+--------------+---> REJECTED
                             |              |
                             +--------------+---> ROLLED_BACK
```

Key properties:

- promotion requires explicit operator review by default,
- canary percentage is capped by the optimization profile,
- runtime rollback restores the pre-promotion state atomically,
- hybrid guardrail knobs (spillover, budget, concurrency) can be part of the promoted
  profile and are also rolled back,
- every lifecycle transition is recorded as an auditable event.

### Key Source Files

| File | Purpose |
|---|---|
| `schemas/optimization.py` | Typed schemas for profiles, campaigns, trials, recommendations |
| `schemas/forge.py` | Promotion lifecycle, inspection, and honesty warning schemas |
| `optimization.py` | Profile builder and config-profile materializer |
| `bench/campaigns.py` | Offline campaign executor and inspection |
| `bench/campaign_comparison.py` | Multi-objective candidate ranking |
| `bench/candidate_generation.py` | Candidate generation and pruning |
| `bench/campaign_honesty.py` | Honesty assessment and staleness marking |
| `control/forge_promotion.py` | Bounded promotion lifecycle service |

For the full Forge Stage A runbook, see [docs/forge-stage-a.md](forge-stage-a.md).

## Request And Routing Flow

1. A request enters the gateway.
2. Middleware assigns or propagates `x-request-id`.
3. The handler builds `RequestContext` from request metadata and internal headers such as:
   - `x-switchyard-routing-policy`
   - `x-switchyard-workload-shape`
   - `x-switchyard-tenant-id`
   - `x-switchyard-request-class`
   - `x-switchyard-session-id`
   - `x-switchyard-internal-backend-pin`
4. The admission-control service decides whether the request is admitted immediately,
   queued briefly, or rejected with an explicit overload decision.
5. `RouterService` resolves the logical model alias to eligible backend deployments using
   the registry.
6. Router policy scoring ranks candidates while consulting circuit-breaker, session-affinity,
   and canary-routing state and returns a typed `RouteDecision` plus a structured
   `RouteExplanation`.
7. The gateway executes the chosen backend either:
   - directly through an in-process adapter, or
   - over the internal worker protocol to a network-addressable worker endpoint.
8. If execution is unsafe or impossible on the first backend, the gateway may perform one
   conservative fallback attempt on another eligible backend.
9. If the primary execution succeeds, session affinity may be rebound to the backend that
   actually served the request.
10. If a shadow policy matches, a best-effort background shadow copy may be launched.
11. The gateway records route and execution telemetry, then returns the response or stream.

## Phase 7 Remote Placement Decision Path

Remote placement is still a normal routing decision, but it now passes through explicit
guardrails:

1. The registry resolves local and remote candidate workers for the logical alias.
2. Lifecycle state, health, and readiness determine whether a remote worker is even
   eligible.
3. The router scores candidates without embedding hardware-specific logic.
4. `RemoteSpilloverControlService` evaluates remote budget, concurrency, cooldown,
   allowed environments, and tenant restrictions.
5. The gateway either executes the chosen remote worker through the standard worker
   protocol or keeps the request local if a guardrail blocks remote execution.
6. Operator-facing hybrid summaries and benchmark artifacts both record the outcome.

The important rule is that remote execution remains attributable:

- if remote execution was chosen, the route decision can explain why,
- if it was blocked, the reason code is serializable,
- if a remote worker failed in transport, the operator surface retains that event.

## Phase 8 Real Cloud Worker Path

Phase 8 adds one concrete real-cloud runtime path without changing the generic worker
contract.

The split is intentional:

- logical alias remains the client-facing contract,
- `remote-worker:<worker_name>` remains the control-plane deployment identity,
- `vllm_cuda` remains the concrete Linux/NVIDIA runtime label at the worker boundary.

That gives Switchyard one honest real-cloud path while keeping the control plane
portable:

- the worker/runtime layer knows about `vllm`, CUDA, GPU count, and Linux bring-up,
- the router and gateway still only consume typed capability, health, placement, spend,
  and lifecycle signals,
- packaging for the rented-GPU worker stays isolated from the default Mac-first control
  plane workspace.

## Phase 8 Alias Resolution Flow

Phase 8 makes the local-plus-remote alias story explicit.

1. The client sends one logical alias such as `chat-shared`.
2. The registry resolves every compatible deployment for that alias:
   - `mlx-lm:...`
   - `vllm-metal:...`
   - `remote-worker:...`
3. Registered worker lifecycle state further constrains whether a remote deployment is
   usable:
   - ready,
   - quarantined,
   - draining,
   - canary-only,
   - stale or lost.
4. The router scores only the remaining eligible candidates.
5. Hybrid guardrails then bound whether remote execution is actually allowed:
   - cloud rollout,
   - remote enabled/disabled posture,
   - spillover budget,
   - remote concurrency,
   - cooldown,
   - tenant/environment restrictions.
6. The gateway executes the chosen path and records why remote routing happened or why it
   was blocked.

The important property is that alias compatibility is explicit:

- one alias can span local and remote implementations,
- adding a real cloud worker does not require client API changes,
- a blocked or quarantined cloud worker still remains visible in topology truth.

## Phase 8 Observed Cloud Signal Path

Phase 8 keeps the observed-cloud path explicit from worker runtime to artifacts.

1. The real remote worker reports or registers typed runtime identity, GPU metadata,
   placement, readiness, and cost-profile metadata.
2. The control plane stores that data in runtime inventory and operator views.
3. Routing and execution consume the observed signals when present:
   - health/readiness,
   - queue depth and active requests,
   - placement and network posture,
   - budget bucket and relative cost metadata.
4. Route examples and execution observations record whether cloud execution really
   happened.
5. Benchmark and replay artifacts preserve evidence source rather than flattening
   observed and estimated cloud metadata together.

This is why Phase 8 can answer different questions honestly:

- "was a remote cloud worker eligible?"
- "did a remote cloud worker actually serve this request?"
- "was the placement or spend metadata directly observed, configured, mocked, or
  estimated?"

## Phase 6 Route-Decision Path

Phase 6 makes the decision path explicit instead of mixing live routing with historical
analysis:

1. The gateway creates `RequestContext`.
2. `switchyard.router.features` extracts deterministic request and workload features.
3. The router resolves eligible backend deployments and concrete worker instances.
4. Session affinity, circuit-breaker state, and backend eligibility are applied.
5. `PrefixLocalityService` emits repeated-prefix and locality signals.
6. The primary scorer evaluates candidates and returns per-candidate scores, rejections,
   reason codes, and rationale.
7. Optional shadow scorers run without affecting the selected route.
8. Optional rollout controls may keep the candidate in shadow, report-only, canary, or
   guarded-active mode.
9. The router returns a typed `RouteDecision` and `RouteExplanation`.

This path is intentionally explainable:

- the route decision is still synchronous and local,
- the scorer output is serializable,
- shadow scoring is explicit and non-binding,
- rollout controls are separate from model execution,
- the gateway fallback path still owns execution retries.

## Phase 6 Evidence Path

The evidence path is separate from the request path and stays rooted in authoritative
artifacts.

Sources:

- benchmark run artifacts,
- replay artifacts,
- captured traces,
- runtime topology snapshots recorded into artifacts.

Derived evidence:

- deterministic request features,
- locality signals,
- historical performance summaries,
- candidate route estimates,
- offline simulation artifacts,
- policy recommendation reports.

Trust boundaries:

- runtime logs and metrics help operators, but they are not authoritative evidence for
  recommendation or promotion,
- simulation must distinguish direct observations from predictor estimates,
- low-confidence and unsupported cases stay visible instead of being folded into one
  score.

## Phase 6 Feedback Loop

The feedback loop is now explicit and conservative:

1. Run benchmark or replay workloads and write artifacts.
2. Summarize historical performance from those artifacts.
3. Simulate fixed or adaptive policies offline.
4. Produce recommendation reports from simulation and benchmark evidence.
5. Use rollout controls to keep candidates in shadow, report-only, canary, or guarded
   active mode.
6. Inspect runtime policy state and benchmark outcomes.
7. Repeat with new evidence before widening rollout.

Switchyard does not auto-promote policies in Phase 6. The loop is decision support for
operators, not autonomous routing governance.

## Router And Control-Plane Policy Flow

Router v1 is deterministic and inspectable, not a learned scheduler.

Current policy modes:
- `latency_first`
- `balanced`
- `quality_first`
- `local_only`

The router considers signals that are backend-agnostic:
- health and readiness,
- admission and queue state,
- breaker protection state,
- session-affinity eligibility,
- canary rollout policy and deterministic bucketing,
- route eligibility for the logical target,
- streaming compatibility,
- context-window compatibility,
- configured priority and weight,
- quality hints,
- performance hints and observed latency when available,
- simple request-shape hints such as prompt and output size.

The router produces:
- the selected backend deployment,
- the remaining ordered fallback candidates,
- per-candidate rationale,
- explicit rejection reasons,
- typed annotations for overload, breaker, affinity, rollout, and shadow decisions,
- a compact explanation string suitable for logs and artifacts.

The router does not know or care about Apple-specific APIs such as Metal kernels. It only
consumes typed capability and health signals.

## Where Failover Happens

Failover happens in the gateway execution layer, not in the router.

That split is intentional:
- the router chooses an ordered plan,
- the executor decides whether it is still safe to advance to the next candidate.

Current failover policy:
- retry budget is one fallback attempt,
- fallback is allowed when the chosen backend is unhealthy before execution,
- fallback is allowed when execution fails before any response has been produced,
- for streaming, fallback is allowed only before the first visible token or chunk,
- no mid-stream migration is attempted.

This keeps the fallback mechanism backend-agnostic and avoids duplicate or misleading
response framing.

## Observability Model

Route-level observability remains first-class in Phase 5.

Telemetry and benchmark artifacts capture:
- logical alias,
- chosen backend,
- candidate backend count,
- routing policy,
- tenant ID and request class,
- admission outcome and queue wait timing,
- breaker phase and breaker-trigger reason,
- session-affinity hit/miss/failover signals,
- canary policy and rollout disposition,
- shadow policy and shadow disposition,
- compact route reason,
- fallback occurrence,
- request latency,
- TTFT when available,
- output tokens,
- tokens per second.

Phase 5 keeps two governance rules on top of that:
- benchmark and trace workflows stay backend-agnostic,
- reports derive from authoritative benchmark artifacts rather than ad hoc logs.

The local-friendly default is still enough for development:
- structured JSON logs,
- `/metrics`,
- `/admin/runtime`,
- `/admin/hybrid`,
- `/admin/hybrid/cloud-rollout`,
- `/admin/remote-workers`,
- `/admin/forge/stage-a`,
- `/admin/forge/stage-a/promotion`,
- `/admin/forge/stage-a/campaigns/inspect`,
- JSON benchmark artifacts,
- optional markdown reports derived from those artifacts.

Trace capture is a separate, opt-in concern. It should stay bounded and privacy-conscious
rather than becoming a default logging behavior.

## Benchmark And Replay Data Flow

The benchmark and replay path intentionally follows the same control-plane boundaries:

1. a workload manifest or captured trace set becomes typed benchmark input,
2. the benchmark runner resolves an execution target:
   - logical alias,
   - routing policy override,
   - pinned backend,
3. requests execute through the normal gateway/control-plane route where practical,
4. per-request outcomes are written into a benchmark artifact,
5. optional comparisons read those artifacts or rerun the same source inputs,
6. markdown reports are rendered from the artifact data.

The important design rule is that markdown is not authoritative. JSON artifacts are.

## Trace Capture Boundary

Trace capture belongs at the gateway boundary because that is the narrowest place where
Switchyard can observe:

- request timestamp and request ID,
- logical alias,
- policy or backend override,
- route decision,
- chosen backend,
- stream intent,
- timing and fallback summary,
- normalized request and response payloads.

The capture service is intentionally pluggable:

- today: local JSONL sink,
- later: another sink if Phase 5 needs one.

The capture policy is also intentionally explicit:

- `off`
- `metadata_only`
- `redacted_content`
- `full_content`

This keeps privacy-sensitive behavior out of the default request path while still making
replay possible when an operator opts in.

Prometheus, Grafana, or a larger telemetry stack remain optional.

## Runtime Boundaries

Real backend imports stay behind runtime/provider boundaries:

- `MLXLMChatRuntime`
- `VLLMMetalChatRuntime`

Those layers own:
- optional imports,
- model loading and connection setup,
- backend-specific generation calls,
- streaming translation,
- capability declaration,
- version-sensitive glue code.

The gateway, router, and registry interact only with adapter contracts and typed backend
metadata.

## Portability Forward

Phase 3 is intentionally structured so future non-Mac backends can fit the same boundary.

Expected future peers:
- `vllm_cuda`
- remote worker adapters
- remote OpenAI-like inference endpoints

To add one of those later, the higher-level expectation is the same:
- implement the shared adapter contract,
- declare backend capabilities honestly,
- register the deployment under one or more logical serving targets,
- let the router reason over generic signals instead of hardware details.

That is why the system is Mac-first now without becoming Mac-locked later.
