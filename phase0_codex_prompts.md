# Phase 0 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

A good rule: do not ask Codex to build all of Phase 0 in one shot. Keep it moving in clean, reviewable slices.

---

## Prompt 0 — bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it. This repo is Switchyard, a Mac-first, backend-agnostic inference fabric. We are only working on Phase 0 right now: foundation and contracts. Do not add real MLX-LM, vLLM-Metal, CUDA, Kubernetes, or cloud integrations yet unless explicitly asked. Prefer a single Python workspace, small vertical slices, typed code, tests, and clear docs. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 — create the project skeleton

```text
Set up the initial Phase 0 project skeleton for Switchyard.

Requirements:
- Use Python 3.12 and uv.
- Create a clean pyproject.toml with dev dependencies for FastAPI, Pydantic v2, pydantic-settings, structlog, OpenTelemetry, pytest, pytest-asyncio, pytest-cov, ruff, mypy, typer, and httpx.
- Create the src layout described in AGENTS.md using a single Python workspace.
- Add placeholder __init__.py files where needed.
- Add a minimal README.md, .python-version, and .env.example.
- Add basic ruff, mypy, and pytest configuration.
- Add a tiny docs/phase0.md file with a checklist matching AGENTS.md.

Acceptance criteria:
- `uv run ruff check .` passes.
- `uv run mypy src tests` passes.
- `uv run pytest` passes even if only with a smoke test.

Please keep the implementation minimal and clean.
```

---

## Prompt 2 — implement core config and schemas

```text
Implement the core shared config and schema layer for Phase 0.

Requirements:
- Add a typed config module using pydantic-settings.
- Implement schemas for chat, backend, routing, and benchmark domains as described in AGENTS.md.
- Keep the API OpenAI-like but not a copy-paste.
- Include enums/types for BackendType, DeviceClass, RoutingPolicy, and WorkloadShape.
- Add validation where it matters, but keep it practical.
- Add tests covering key schema validation and serialization paths.

Acceptance criteria:
- The schemas are importable and readable.
- Tests cover at least one valid and one invalid case per major schema group.
- Lint, type checks, and tests all pass.

Do not build any gateway or backend behavior yet.
```

---

## Prompt 3 — build the backend adapter contract and mock backend

```text
Implement the backend adapter layer for Phase 0.

Requirements:
- Create a BackendAdapter protocol or abstract base class matching AGENTS.md.
- Add a small adapter registry.
- Implement a deterministic MockBackendAdapter.
- The mock backend should be configurable for:
  - backend name,
  - simulated latency,
  - health state,
  - capability metadata,
  - fixed response template.
- The mock backend response should make it easy to verify which backend handled the request.
- Add thorough tests for registry behavior, health behavior, and deterministic response generation.

Acceptance criteria:
- The mock backend can generate a valid chat completion response from a valid request.
- The registry can register and retrieve adapters cleanly.
- Tests, lint, and typing all pass.

Do not add any real model-serving integration.
```

---

## Prompt 4 — implement the routing module

```text
Implement the Phase 0 routing module.

Requirements:
- Add a router service that is separate from the HTTP layer.
- Support at least these policies: local_only, latency_first, balanced, quality_first.
- The router should choose among registered backends using capability metadata and health information.
- For now, keep the scoring logic simple and deterministic.
- Return a typed RouteDecision that explains why the backend was chosen.
- Add unit tests covering healthy, unhealthy, and tie-break cases.

Acceptance criteria:
- Routing does not depend on FastAPI.
- RouteDecision includes enough metadata to log and benchmark later.
- Tests clearly show that policies produce different choices when appropriate.

Do not implement learned routing yet.
```

---

## Prompt 5 — build the FastAPI gateway skeleton

```text
Implement the initial FastAPI gateway for Switchyard.

Requirements:
- Add a FastAPI app with GET /healthz, GET /readyz, and POST /v1/chat/completions.
- The gateway should:
  - create or propagate a request ID,
  - call the router,
  - invoke the chosen backend,
  - return a valid chat completion response.
- Use dependency injection so the router and adapter registry are not hidden globals.
- Add typed error handling for invalid input and unavailable backends.
- Add integration tests using the mock backend.

Acceptance criteria:
- A test client can hit /v1/chat/completions and receive a deterministic mock response.
- Health endpoints work.
- The gateway stays thin and the routing logic stays outside route handlers.

Keep streaming out of scope for now unless it is trivial.
```

---

## Prompt 6 — add structured logging and telemetry scaffolding

```text
Add structured logging and telemetry scaffolding to Phase 0.

Requirements:
- Add a small logging module using structlog.
- Add a small telemetry module that initializes OpenTelemetry cleanly.
- Instrument the FastAPI gateway enough to attach request IDs and route metadata to logs.
- Add placeholder counters/histograms or telemetry wrapper functions for:
  - total requests,
  - request latency,
  - route decisions,
  - backend health snapshots.
- Keep exports simple and local-friendly.
- Add tests for any helper functions that are practical to test.

Acceptance criteria:
- The gateway emits structured logs with request ID and chosen backend.
- Telemetry initialization does not pollute business logic.
- Lint, typing, and tests all pass.

Do not build a full Prometheus/Grafana stack yet unless it is only config scaffolding.
```

---

## Prompt 7 — add benchmark artifact models and a tiny bench CLI

```text
Implement the first benchmark artifact path for Phase 0.

Requirements:
- Add benchmark artifact utilities and models if not already present.
- Add a small Typer CLI under src/switchyard/bench/cli.py.
- The CLI should be able to:
  - construct a small synthetic scenario,
  - issue a few requests to the local gateway or directly to the router/backend service layer,
  - collect per-request results,
  - write a reproducible JSON artifact to disk.
- Include run_id, timestamp, policy, backend info, request count, and summary statistics.
- Add tests for artifact serialization and at least one CLI-path helper.

Acceptance criteria:
- A developer can run one command and get a JSON benchmark artifact.
- The artifact format is clean and documented.
- All checks pass.

Keep this lightweight; do not build a big benchmark framework yet.
```

---

## Prompt 8 — improve docs and developer ergonomics

```text
Polish Phase 0 for a clean handoff.

Requirements:
- Update README.md with a short project overview, current scope, and local dev instructions.
- Add docs/architecture.md describing the Phase 0 architecture and future backend portability.
- Add at least one short ADR documenting a key decision from Phase 0.
- Add a simple Makefile or justfile with the most useful dev commands, but do not overengineer it.
- Ensure .env.example matches the actual config.

Acceptance criteria:
- A new contributor can set up and run the repo from the docs.
- Docs reflect the code as it exists.
- All checks still pass.
```

---

## Prompt 9 — optional compose scaffolding for infra services

```text
Add minimal local infra scaffolding for later phases, but keep it clearly optional.

Requirements:
- Add an infra/compose directory with a small docker-compose file or compose.yaml for optional services such as Postgres, Redis, and an OpenTelemetry Collector.
- Keep this scaffolding lightweight and clearly marked as not required for core Phase 0 development.
- Add a short doc explaining what is used now versus later.

Acceptance criteria:
- The compose file is coherent and documented.
- Core Phase 0 functionality still works without starting these services.

Do not wire production-grade infra yet.
```

---

## Prompt 10 — Phase 0 exit review

```text
Review the repo against AGENTS.md and the Phase 0 definition of done.

Tasks:
- Identify anything missing or weak in Phase 0.
- Tighten tests, docs, or naming where needed.
- Remove any accidental overengineering.
- If needed, make a final cleanup pass so the project feels crisp and coherent.

Deliverables:
- a concise summary of Phase 0 status,
- any remaining gaps,
- the top 5 recommended Phase 1 tasks.

Only make code changes if they clearly improve Phase 0 completeness or clarity.
```

---

## Optional single-shot prompt if you want Codex to plan before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 0 implementation plan before any major coding. Produce:
1. a proposed file tree,
2. the exact dependencies you want in pyproject.toml,
3. the core shared schemas you will implement first,
4. the tests you will write first,
5. a step-by-step execution plan for Prompts 1 through 8.
Do not change code yet.
```
