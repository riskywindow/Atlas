# Phase 6

Phase 6 moves Switchyard from a deployment-aware Phase 5 control plane into an
explainable routing-intelligence phase. The goal is not to replace the existing typed
router with opaque adaptation. The goal is to extend the existing routing, trace, and
benchmark foundation with deterministic features, richer route-decision artifacts,
historical evidence, offline policy comparison, and guarded adaptive behavior.

## Goals

- Preserve the current typed and portable serving boundary:
  - host-native Apple-Silicon workers remain behind a network-addressable worker
    protocol,
  - the control plane stays backend-agnostic,
  - CI-friendly tests still run without Apple GPU access.
- Add richer route-decision artifacts:
  - route decisions should carry deterministic request and workload features,
  - explanations should stay serializable and diffable,
  - benchmark and replay artifacts should remain useful for policy comparison.
- Make cache and locality signals explicit:
  - repeated-prefix candidates should be derivable deterministically,
  - cache/locality hints should be visible in artifacts and explanations,
  - the router should not depend on hidden runtime-only heuristics.
- Add historical evidence in typed form:
  - policy comparisons should be able to summarize historical latency, throughput, and
    failure behavior,
  - summaries should be suitable for offline simulation and regression analysis.
- Introduce explainable policy/scorer boundaries:
  - policy selection and scoring should be inspectable,
  - scorer outputs should be explainable,
  - shadow scoring should be possible without changing the user-visible outcome.
- Add offline simulation before adaptive runtime behavior:
  - policies should be compared against captured or synthetic traces,
  - simulation evidence should be archivable,
  - offline comparisons should remain portable to later remote and cloud workers.
- Keep future portability open:
  - later `vllm_cuda` workers should fit the same control-plane contracts,
  - later Forge-style optimization should consume typed evidence rather than hidden
    routing state.

## Definition Of Done

- The repo keeps a clean Python workspace with linting, typing, and tests.
- The control plane still supports at least two real backend families:
  - `mlx_lm`
  - `vllm_metal`
- The gateway still serves `GET /healthz`, `GET /readyz`, and
  `POST /v1/chat/completions` with health-aware fallback.
- Routing logic, scoring logic, admission, protection, affinity, rollout, and adaptive
  behavior remain outside the HTTP layer.
- Route decisions and benchmark artifacts carry deterministic request and workload
  features, including repeated-prefix and locality signals where relevant.
- Routing intelligence remains explainable:
  - route selection reasons are inspectable,
  - scorer outputs are attributable,
  - shadow scoring never becomes the hidden source of truth for the user-visible path.
- Authoritative artifacts remain the source of truth:
  - benchmark, replay, and simulation artifacts stay authoritative,
  - reports and runbooks derive from artifacts rather than reconstructing behavior from
    ad hoc logs.
- Offline simulation evidence and online runtime truth stay distinct:
  - offline artifacts describe what should have happened under a policy,
  - runtime traces and route decisions describe what actually happened in production or
    local execution.
- Adaptive routing is safe-by-default and reversible:
  - bounded rollout or shadow-only paths come first,
  - guardrails can disable or bypass adaptation,
  - operators can fall back to deterministic baseline behavior without refactoring the
    system.
- The Phase 6 design remains portable to later cloud workers and Forge-style
  optimization workflows.

## Non-Goals

- No giant refactor of the routing stack.
- No opaque learned router in the live request path.
- No removal of deterministic baseline routing.
- No control-plane coupling to Apple-only runtime details.
- No rebuild of deployment packaging that Phase 5 already established.
- No cloud-platform buildout, CUDA runtime integration, or production-scale Kubernetes
  platform in this phase.
- No reporting layer that supersedes benchmark, replay, or simulation artifacts as the
  authoritative record.

## Rules

### Explainable Routing Rule

Routing intelligence must stay explainable. If a policy, scorer, or adaptive rule
changes route choice, the reason should be representable in typed artifacts and route
explanations. Hidden heuristics that cannot be surfaced in logs, traces, or artifacts
are out of scope for Phase 6.

### Artifact Source Of Truth Rule

Authoritative artifacts remain the source of truth. Benchmark artifacts, replay
artifacts, and simulation artifacts are the canonical records for comparative analysis.
Derived markdown, dashboards, and operator notes should point back to those artifacts.

### Offline Versus Online Truth Rule

Offline simulation evidence and online runtime truth are not the same thing:

- offline simulation shows what a policy would have done against a captured or synthetic
  workload under controlled assumptions,
- online runtime truth shows what actually happened for a routed request under live
  health, load, and deployment conditions.

Phase 6 should preserve both kinds of evidence without conflating them.

### Safe Adaptive Routing Rule

Adaptive routing must be safe-by-default and reversible. New adaptive behavior should
start in shadow or bounded mode, respect explicit guardrails, and remain easy to disable
in favor of a deterministic baseline policy.

### Portability Rule

Phase 6 should remain compatible with later cloud workers and Forge-style optimization.
Typed request features, scorer inputs, policy outputs, and evidence artifacts should be
portable enough to support future `vllm_cuda`, remote GPU workers, and optimization
loops that consume routing evidence without rewriting the control plane.

## Status

Phase 5 established the portable control-plane and worker-topology foundation that
Phase 6 depends on:

- explicit worker endpoints and inventory,
- host-native worker serving for Mac-native backends,
- deployment-aware benchmark and replay artifacts,
- containerized control-plane paths,
- typed runtime and deployment diagnostics.

The first Phase 6 slices should stay small:

- deterministic request and workload feature extraction,
- richer route-decision artifacts,
- typed historical summaries,
- explainable scorer boundaries,
- offline simulation before live adaptive behavior.

## Artifact Evolution

Phase 6 evidence capture extends the authoritative artifact family in a backward-aware
way. New artifacts may use `switchyard.benchmark.v3` when they include richer policy,
topology, request-feature, shadow-decision, and runtime-observation fields. Older
`v2` artifacts should still remain loadable when those newer fields are absent.
