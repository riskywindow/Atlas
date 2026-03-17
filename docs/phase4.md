# Phase 4

Phase 4 moves Switchyard from a replayable local control plane into a more resilient
serving fabric. The goal is to add explicit overload protection and progressive-delivery
behavior without breaking the Mac-first, backend-agnostic boundary established in the
earlier phases.

## Goals

- Preserve the existing typed control-plane foundation:
  - two real Mac-native backend families,
  - backend-agnostic routing,
  - health-aware fallback,
  - replayable benchmarking and trace capture with safe defaults.
- Add bounded admission control so overload behavior is explicit, deterministic, and
  testable.
- Add per-tenant limits without hardcoding Apple-specific runtime assumptions into the
  router or gateway.
- Add circuit-breaker-style backend protection so unhealthy deployments can be excluded
  before they cause visible instability.
- Add session affinity primitives for multi-turn chat while keeping the public API
  backend-agnostic.
- Add shadow traffic and canary routing with safe, opt-in controls.
- Strengthen observability so overload, degradation, and rollout behavior are visible in
  structured logs, telemetry, and benchmark artifacts.

## Definition Of Done

- The repo keeps a clean Python workspace with linting, typing, and tests.
- The shared contracts still support at least two real backend adapter paths:
  - `mlx_lm`
  - `vllm_metal`
- Admission control and per-tenant limits are represented explicitly in typed routing
  context or decisions and are testable without running the HTTP server.
- Backend protection behavior is explicit, bounded, and observable rather than inferred
  from opaque exceptions.
- Session affinity can influence routing for multi-turn chat without coupling the control
  plane to hardware-specific logic.
- Shadow traffic stays opt-in and never affects the primary user-visible response.
- Canary or rollout behavior is bounded and explainable from logs and machine-readable
  artifacts.
- Route-level observability captures overload, fallback, degradation, and rollout
  signals in a way that benchmarks and replay tooling can consume.
- Lightweight runtime inspection tooling should expose current control-plane state
  without requiring developers to infer it only from prior log output.
- Apple-specific imports remain lazy and optional so CI-friendly tests do not require
  Apple GPU hardware.
- The design remains portable to future `vllm_cuda`, remote, and cloud-backed workers.

## Non-Goals

- No giant refactor or package split.
- No request-path Kubernetes, Ray, or multi-service scheduler buildout.
- No cloud orchestration layer in this phase.
- No hidden overload shedding based only on backend-specific side effects.
- No default-on shadow traffic.
- No rollout mechanism that cannot be reconstructed from logs or artifacts after the
  fact.

## Mac-First Constraint

Switchyard remains Mac-first in Phase 4. Real local inference should continue to run
host-native on Apple Silicon macOS, and Apple-specific runtime details must remain at
the adapter/runtime boundary rather than leaking into routing, admission, affinity,
shadowing, rollout, or reporting logic.

## Overload Rule

Overload behavior must be explicit and testable.

- Admission limits should be represented with typed inputs and decisions.
- Backend protection should produce clear rejection or degradation signals.
- Tests should be able to exercise overload and protection behavior without Apple GPU
  hardware.
- Operators should be able to understand why requests were admitted, rejected, or
  rerouted.
- HTTP overload responses should be explicit. The current local-first default is `429 Too Many Requests`
  for admission-control rejection before execution begins.

## Shadow Traffic Rule

Shadow traffic must be opt-in and must never affect primary user-visible responses.

- Primary execution decides the user-visible response.
- Shadow execution is observational only.
- Shadow routing must stay bounded and safe by default.
- Failures in shadow paths must not mutate the primary result path.

## Rollout Explainability Rule

Rollout behavior must be explainable from logs and artifacts.

- Canary or weighted rollout decisions should emit structured route signals.
- Benchmark and replay artifacts should retain enough context to reconstruct why a
  backend was selected, skipped, or shadowed.
- Debugging rollout behavior should not require reading backend-specific source code or
  inferring behavior from missing logs.

## Typed Config Surface

Phase 4 control-plane settings are now intended to be explicit in configuration rather
than implied by hidden defaults.

- `SWITCHYARD_PHASE4.admission_control`
  Includes default concurrency caps, bounded queue sizes, timeout values, and per-tenant
  overrides.
- `SWITCHYARD_PHASE4.circuit_breakers`
  Includes failure thresholds and cooldown defaults.
- `SWITCHYARD_PHASE4.session_affinity`
  Includes sticky-route TTL and capacity bounds. Session affinity remains local and
  in-process in this phase; it must stay bounded, expire predictably, and fail over
  with explicit reasons when the sticky backend is no longer safe to use.
- `SWITCHYARD_PHASE4.canary_routing`
  Includes default canary percentages and named weighted policies. Canary selection
  should be deterministic enough to test, prefer session-stable bucketing when a session
  key is present, and automatically bypass unhealthy or ineligible rollout candidates.
- `SWITCHYARD_PHASE4.shadow_routing`
  Includes default shadow sampling and named shadow policies. Shadowing remains explicit
  and opt-in, can target either another backend or another alias, and must record its
  own outcome without mutating the primary user-visible response path.

These settings are typed, backend-agnostic, and intentionally conservative by default.
The benchmark artifact schema remains `switchyard.benchmark.v2` for now; Phase 4 report
and trace metadata are added as backward-compatible optional fields rather than forcing a
format break immediately.

## Audit Notes

- The repo already had the main Phase 3 building blocks needed to enter Phase 4:
  typed schemas, backend-agnostic routing, health-aware fallback, two real backend
  families, and replayable benchmark/report plumbing.
- The smallest obvious blockers were repo-alignment issues:
  Phase 4 did not have its own phase doc, and some high-level docs still described the
  repo primarily through a Phase 3 lens.
- This pass keeps the change intentionally small and focused on audit/alignment work
  rather than broad implementation changes.
