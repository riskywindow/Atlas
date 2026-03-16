# Progress

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
