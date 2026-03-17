# Progress

## 2026-03-16

- Advanced the repo contract from Phase 2 into early Phase 3 by updating `AGENTS.md`
  and `README.md` around replayable benchmarking, workload generation, report output,
  and the portability/safe-default constraints for future trace capture and replay.
- Extended the benchmark schema layer with typed deterministic workload generation
  settings so synthetic runs can describe repeated-prefix and bursty scenarios without
  coupling the control plane to hardware-specific behavior.
- Wired the synthetic benchmark helpers and CLI to accept workload-pattern controls,
  keep them serialized in benchmark artifacts, and optionally emit Markdown reports
  alongside the existing JSON artifacts.
- Added tests covering workload-generation validation, deterministic prompt shaping, and
  Markdown report generation for both single-run and comparative synthetic benchmarks.
- Audited the existing repo against the intended Phase 1 outcomes and confirmed the
  Phase 0 implementation already provides the needed foundation.
- Updated `AGENTS.md` so the active project contract is now Phase 1: first real local
  backend, streaming, runtime metrics, and benchmark extension on top of the existing
  control plane.
- Added `docs/phase1.md` with Phase 1 goals, definition of done, non-goals, the
  Mac-first constraint, the adapter-boundary rule for future non-Mac backends, and a
  short audit summary.
- Updated `README.md` so top-level docs no longer describe the repo as Phase 0 only.
- Extended the shared contracts for Phase 1 with streaming chat schemas, richer backend
  status models, backend-agnostic local model config, and a mock adapter that supports
  status, warmup, and deterministic streaming.
- Added a small internal runtime boundary under `switchyard.runtime`, including a lazy
  MLX-LM provider and chat runtime with optional-dependency-safe model loading, health,
  warmup, buffered fallback streaming, and unit tests driven by fake module imports.
- Added the first real adapter path with an `MLXLMAdapter`, config-driven adapter
  registration from `Settings.local_models`, and gateway wiring that conditionally
  registers configured MLX backends without breaking explicit mock-registry tests.
- Added adapter, factory, and gateway tests using fake runtimes so the MLX path is
  exercised end to end without requiring Apple GPU hardware or the `mlx-lm` package.
- Added Phase 1 streaming support through the existing `/v1/chat/completions` route
  using SSE framing, `[DONE]` termination, and the same router-plus-adapter decision path
  for both mock and MLX-backed adapters.
- Added streaming integration tests covering SSE chunk framing, termination behavior,
  request ID propagation, backend metadata in streamed chunks, and a configured fake-MLX
  gateway path.

## 2026-03-15

- Created the initial Phase 0 single-workspace Python skeleton for Switchyard.
- Added project tooling config for `uv`, `ruff`, `mypy`, and `pytest`.
- Added placeholder package layout, environment examples, and a smoke test.
- Implemented the Phase 0 shared config and schema layer for chat, backend, routing, and benchmark domains.
- Added schema validation and serialization tests covering valid and invalid cases across the major model groups.
- Implemented the Phase 0 backend adapter contract, in-memory registry, and deterministic mock backend.
- Added adapter tests covering registry behavior, health reporting, simulated latency, and deterministic response generation.
- Implemented the Phase 0 pure-Python router service with deterministic policy scoring and typed route decisions.
- Added router tests covering unhealthy filtering, policy-specific choices, deterministic tie-breaking, and no-route failures.
- Implemented the initial FastAPI gateway with explicit dependency wiring, request ID propagation, typed errors, and mock-backed chat completions.
- Added gateway integration tests for health, readiness, validation errors, backend-unavailable errors, and deterministic chat completion responses.
- Added structlog-based structured logging and local-friendly telemetry scaffolding with OpenTelemetry initialization.
- Instrumented the gateway to record request, route-decision, and backend-health telemetry and to emit structured logs with request IDs and chosen backends.
- Implemented the first benchmark artifact path with a lightweight synthetic runner, reproducible JSON writer, and Typer CLI entrypoint.
- Added benchmark tests covering synthetic artifact generation, stable JSON serialization, and the CLI artifact-writing path.
- Polished Phase 0 docs with an updated README, architecture note, environment example, and initial ADR.
- Added a small Makefile for the common local development, gateway, and benchmark commands.
- Added optional local infra compose scaffolding for Postgres, Redis, and an OpenTelemetry Collector, with a short usage note.
- Removed one unused config knob (`SWITCHYARD_LOG_JSON`) so the documented settings surface matches the actual runtime behavior.
