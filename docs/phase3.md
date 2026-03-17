# Phase 3

Phase 3 moves Switchyard from a multi-backend local control plane into a replayable,
comparative evaluation environment. The goal is to make benchmark and trace workflows
reproducible, inspectable, and portable without turning the project into a heavy
distributed system too early.

## Goals

- Preserve the Phase 2 control-plane foundation:
  - two real Mac-native backend paths,
  - backend-agnostic routing,
  - health-aware fallback,
  - route-level observability.
- Add reproducible workload generation for benchmark and replay flows.
- Enrich benchmark artifacts so runs are easy to diff, archive, compare, and report on.
- Support trace replay against:
  - logical aliases,
  - routing policies,
  - pinned backend deployments.
- Support A/B-style comparative runs across the same request shape and workload seed.
- Support repeated-prefix and bursty workload scenarios without hardcoding backend
  assumptions into the benchmark layer.
- Generate human-readable reports from the same authoritative artifact data used by
  programmatic analysis.

## Definition Of Done

- The repo keeps a clean Python workspace with linting, typing, and tests.
- The Phase 2 backend boundary remains intact with at least two real adapter paths:
  - `mlx_lm`
  - `vllm_metal`
- Reproducible workload generation is represented with typed, serializable config and can
  express at least:
  - uniform synthetic requests,
  - repeated-prefix scenarios,
  - bursty scenarios.
- Benchmark artifacts remain the source of truth for comparative analysis and report
  generation.
- Benchmark and replay workflows can compare aliases, routing policies, and pinned
  backends without coupling logic to Apple-specific runtime details.
- Optional request-trace capture is off by default, bounded when enabled, and documented
  as a privacy-conscious debugging or benchmarking tool rather than a universal default.
- Reports can be generated in both machine-readable and markdown form from authoritative
  benchmark artifacts.
- Apple-specific imports remain lazy and optional so CI-friendly tests do not require
  Apple GPU hardware.
- The design remains portable to future `vllm_cuda` and remote/cloud-backed workers.

## Non-Goals

- No giant architecture rewrite or package split.
- No Kubernetes, Ray, or multi-service scheduler in the request path.
- No cloud worker buildout in this phase.
- No learned router or speculative cost optimizer yet.
- No trace capture that silently stores full prompts or outputs by default.
- No reporting pipeline that derives conclusions from ad hoc logs instead of benchmark
  artifacts.

## Mac-First Constraint

Switchyard remains Mac-first in Phase 3. Real local inference should continue to run
host-native on Apple Silicon macOS, and Apple-specific runtime details must remain at
the adapter/runtime boundary rather than leaking into routing, benchmark, trace, or
report logic.

## Backend-Agnostic Benchmark And Trace Rule

Benchmark generation, artifact schemas, trace replay, and report generation must stay
backend-agnostic. Those workflows should reason about logical aliases, deployments,
policies, workload shape, health, latency, and quality signals rather than directly
embedding backend-specific runtime mechanics.

This is what keeps the Phase 3 work portable to:
- future `vllm_cuda`,
- remote worker adapters,
- remote OpenAI-like backends.

## Trace Capture Rule

Trace capture must be opt-in and privacy-conscious.

- It must be disabled by default.
- Enabling it should be an explicit operator choice.
- Captured data should be bounded and intentional.
- Safe defaults should prefer metadata and routing context over full raw payload capture.

The point of trace capture in Phase 3 is reproducible debugging and evaluation, not
blanket retention.

## Artifact Authority Rule

Reports must derive from authoritative benchmark artifacts rather than ad hoc logs.

Structured logs and telemetry are useful for local debugging and operations, but
comparative benchmark summaries, markdown reports, and machine-readable reports should be
generated from typed artifact data so the output remains reproducible, diffable, and
auditable.

## Artifact Versioning

Phase 3 makes artifact versioning explicit.

- Phase 2-style benchmark artifacts are treated as the older compatible shape.
- The current Phase 3 schema version is `switchyard.benchmark.v2`.
- New typed benchmark, replay, comparison, and report metadata should serialize that
  version explicitly so downstream tooling can branch intentionally when schema changes
  are introduced later.

## Audit Notes

- The existing repo already has most of the technical Phase 2 baseline needed to enter
  Phase 3: two real backend families, typed schemas, backend-agnostic routing,
  health-aware fallback, and benchmark artifact generation.
- The obvious blockers for a clean Phase 3 start were repo-alignment issues:
  `docs/phase3.md` was missing and some high-level docs still described the system
  primarily as a Phase 2 artifact.
- This pass intentionally keeps the change small and focused on audit/alignment work
  rather than a broad implementation rewrite.
