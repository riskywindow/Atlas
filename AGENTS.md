# AGENTS.md

## Project name
**Switchyard**

Switchyard is a **Mac-first, backend-agnostic inference fabric**. It starts as a local control plane for Apple Silicon and later grows into a hybrid local + cloud inference system with pluggable CUDA-backed workers.

The goal is **not** to build a thin LLM wrapper. The goal is to build a serious systems project that can:
- expose an OpenAI-like API,
- route requests across multiple inference backends,
- measure latency/throughput/error tradeoffs,
- replay workloads and compare policies,
- stay portable enough to support remote GPU workers later.

---

## Current phase
**Phase 5: portable worker inventory and deployment paths**

Phase 5 builds on the Phase 4 routing and benchmarking core with explicit
network-addressable worker endpoints, portable backend-instance inventory, stronger
isolation of Apple-specific backend dependencies from the control plane, and
deployment-aware local run paths. The system should still be Mac-first for real local
backends, but the control plane, contracts, and benchmark artifacts now need to model
future remote workers and mixed deployment topologies directly.

Phase 5 is explicitly about making the control plane deployable and portable without
overengineering. Containerization, Docker Compose, a `kind` path, and worker inventory
must show up in typed contracts, config, docs, and tests rather than living as implied
tribal knowledge.

### Definition of done for Phase 5
Phase 5 is complete when all of the following are true:
1. The repo keeps a clean Python workspace with linting, typing, and tests.
2. There are at least two real backend adapter paths behind the shared contracts:
   - `mlx_lm`
   - `vllm_metal`
3. Local model registration supports multiple configured backends without coupling the
   control plane to Apple-specific runtime details.
4. The FastAPI gateway still serves `GET /healthz`, `GET /readyz`, and
   `POST /v1/chat/completions`, now with health-aware fallback across routed candidates.
5. Router policy modes, overload admission, backend protection, and progressive-delivery
   decisions remain outside the HTTP layer and are benchmarkable without spinning up the
   API.
6. Structured logging and telemetry include route-level decision, overload, fallback,
   circuit-breaker, and degradation signals.
7. Session affinity can keep multi-turn chat on a stable serving path without coupling
   the control plane to hardware-specific logic.
8. Shadow traffic and canary routing stay explicit, bounded, and safe by default for
   local and CI workflows.
   Shadow traffic must never change the primary user-visible response path.
9. Typed backend inventory can describe multiple backend instances per deployment,
   including explicit network endpoints for future remote or containerized workers.
10. Apple-specific imports stay lazy and optional so CI-friendly tests do not require
   Apple GPU hardware and portable control-plane images do not pull those dependencies in
   by default.
11. The control plane has a containerized local deployment path, a Docker Compose stack,
   and a documented `kind` path that preserve the single-workspace developer model.
12. Benchmark and replay tooling can compare aliases, policies, pinned backends,
   deployment variants, and progressive-delivery variants with machine-readable
   artifacts and readable markdown reports.
13. Deployment docs and runbooks cover local host-native backends, containerized control
   plane operation, and future remote-worker extension points without assuming Apple GPU
   access in CI.

---

## Hard constraints

### Platform constraints
- Primary development machine: **Apple Silicon Mac (M4 Pro, 24GB RAM)**.
- During the Mac-first phase, **real model backends should run host-native on macOS**.
- Containerized services are fine for infra such as Postgres, Redis, Prometheus, Grafana, and the OpenTelemetry Collector.
- Do **not** add CUDA-only or Triton-only runtime dependencies in Phase 5.
- Keep the design explicitly portable so later phases can add `vllm_cuda` or other remote GPU workers.

### Scope constraints
- Phase 5 should preserve **two real Mac-native backend paths** behind the same adapter boundary:
  - `mlx_lm`
  - `vllm_metal`
- A single logical model alias may map to multiple backend implementations. Registration, routing, and benchmarks should preserve that abstraction instead of assuming one alias implies one runtime.
- A single logical backend deployment may map to multiple explicit worker instances with
  distinct network addresses. Inventory should stay typed and portable to future remote
  workers.
- Avoid premature multi-service complexity. In Phase 5, a **single Python workspace** is still preferred over many separate packages.
- Do not build a frontend UI yet.
- A small `kind` deployment path is in scope, but avoid a production-grade Kubernetes
  platform build-out.
- Do not introduce Ray into the request path in Phase 5.

### Quality constraints
- Prefer **small vertical slices** over giant speculative scaffolding.
- Every major module must have tests.
- New behavior should come with documentation or inline comments where needed.
- Keep abstractions honest. Do not invent interfaces that are obviously disconnected from expected usage.

---

## Architectural principles

1. **The control plane must not know hardware details.**
   The router should reason about capabilities, health, cost, and performance history, not about “Metal vs CUDA” directly.

2. **Backends are adapters.**
   Backends should implement a common contract and register their capabilities explicitly.

3. **Routing is a first-class subsystem.**
   Routing logic must live outside the HTTP layer and be testable without spinning up the API.

4. **Observability is not optional.**
   Every request should have a request ID and trace context. Logs must be structured.
   Overload, degradation, and rollout decisions must be explainable from logs and
   artifacts.

5. **Benchmarks are product features.**
   Benchmark artifact schemas and reproducibility matter from the beginning.

6. **Mac-first, not Mac-locked.**
   All Apple-specific logic must stay at the adapter boundary.

7. **Prefer simple, typed Python.**
   Avoid magical frameworks or overengineered metaprogramming.

---

## Recommended stack for Phase 5
- Python 3.12
- `uv` for environment and dependency management
- FastAPI
- Pydantic v2
- `pydantic-settings`
- `httpx` for internal HTTP calls and tests
- `structlog` for structured logs
- OpenTelemetry Python SDK + FastAPI instrumentation
- `pytest`, `pytest-asyncio`, `pytest-cov`
- `ruff`
- `mypy`
- `typer` for simple CLIs
- `orjson` if useful, but only if it improves clarity

Keep the stack minimal and coherent.

---

## Suggested repo layout
Use a **single Python project** with clear internal modules.

```text
switchyard/
  AGENTS.md
  README.md
  pyproject.toml
  .python-version
  .env.example
  src/
    switchyard/
      __init__.py
      config.py
      logging.py
      telemetry.py
      schemas/
        __init__.py
        chat.py
        backend.py
        routing.py
        benchmark.py
      adapters/
        __init__.py
        base.py
        mock.py
        registry.py
      router/
        __init__.py
        policies.py
        service.py
      gateway/
        __init__.py
        app.py
        dependencies.py
        routes.py
      bench/
        __init__.py
        cli.py
        artifacts.py
  tests/
    adapters/
    router/
    gateway/
    bench/
  docs/
    architecture.md
    phase0.md
    adr/
  infra/
    compose/
```

If the repo already exists, adapt this layout rather than forcing a rewrite.

---

## Core domain model to establish early

### Chat/request-response schemas
Create typed schemas for:
- `ChatMessage`
- `ChatCompletionRequest`
- `ChatCompletionChoice`
- `ChatCompletionResponse`
- `UsageStats`
- `ErrorResponse`

Keep these **OpenAI-like but not copy-pasted**. The API should feel familiar.

### Backend schemas
Create typed schemas for:
- `BackendType`
- `DeviceClass`
- `BackendCapabilities`
- `BackendHealth`
- `BackendStatusSnapshot`

Suggested values:
- `BackendType`: `mock`, `mlx_lm`, `vllm_metal`, `vllm_cuda`, `remote_openai_like`
- `DeviceClass`: `cpu`, `apple_gpu`, `nvidia_gpu`, `amd_gpu`, `remote`

### Routing schemas
Create typed schemas for:
- `RoutingPolicy`
- `RouteDecision`
- `RequestContext`
- `WorkloadShape`

Phase 5 routing decisions should keep enough information to support:
- route-level observability,
- health-aware fallback,
- bounded admission and backend protection,
- session affinity and progressive delivery,
- comparative policy benchmarking.

### Benchmark schemas
Create typed schemas for:
- `BenchmarkScenario`
- `BenchmarkRequestRecord`
- `BenchmarkSummary`
- `BenchmarkRunArtifact`

Artifacts should be reproducible and serializable.
Phase 5 may add replay-, overload-, deployment-, and comparison-oriented schemas as long
as the JSON remains easy to diff and archive.

---

## Backend adapter contract
Backends must present a small, clear async contract. Use a `Protocol` or abstract base class.

Recommended shape:

```python
class BackendAdapter(Protocol):
    name: str
    backend_type: BackendType

    async def health(self) -> BackendHealth: ...
    async def capabilities(self) -> BackendCapabilities: ...
    async def warmup(self, model_id: str | None = None) -> None: ...
    async def generate(
        self,
        request: ChatCompletionRequest,
        context: RequestContext,
    ) -> ChatCompletionResponse: ...
```

It is fine to extend this later with streaming, cancellation, embeddings, and snapshots.

### Mock backend requirements
The mock backend should be:
- deterministic,
- cheap to run,
- configurable,
- capable of simulating latency,
- capable of simulating health states,
- capable of simulating explicit worker-instance metadata when tests need deployment
  inventory,
- suitable for routing tests.

The mock backend should return responses that include enough metadata to verify which backend handled the request.

---

## Routing design rules
- The router must be a pure Python service/module that can be tested directly.
- The gateway should ask the router for a `RouteDecision`, then invoke the chosen backend.
- Phase 5 routing can still be simple and deterministic, but it must leave room for:
  - latency-first,
  - balanced,
  - quality-first,
  - local-only,
  - future cost-aware or learned routing,
  - bounded admission and backend protection,
  - session affinity, canaries, and shadow routing,
  - explicit instance-aware routing when worker inventory is available.
- Health-aware fallback should happen without baking hardware-specific assumptions into the control plane.

Do not bury routing rules inside FastAPI route handlers.

---

## Telemetry and logging rules
- Every request gets a request ID.
- Structured logs must include at least:
  - request ID
  - route policy
  - chosen backend
  - latency
  - status
- Telemetry should be initialized cleanly even if exports are no-ops locally.
- Prefer a thin telemetry wrapper over scattering metrics code everywhere.

Suggested metric names:
- `switchyard_requests_total`
- `switchyard_request_latency_ms`
- `switchyard_route_decisions_total`
- `switchyard_backend_health`

---

## Benchmark artifact rules
Benchmark artifacts should be easy to diff, inspect, and archive.

At minimum, a benchmark artifact should contain:
- `run_id`
- `timestamp`
- `git_sha` if available
- scenario/config metadata
- policy used
- backend(s) involved
- request count
- summary statistics
- per-request results or a path/reference to them

Store initial artifacts as JSON. Add Parquet later only if it clearly helps.

---

## Coding standards

### General
- Prefer readability over cleverness.
- Use explicit type hints.
- Use dataclasses or Pydantic only when they provide clear value.
- Avoid global mutable state where possible.

### Python
- Target Python 3.12.
- Use `pathlib` instead of raw path strings where reasonable.
- Use `async` in I/O-facing boundaries.
- Keep functions small and composable.

### Error handling
- Fail fast on invalid config.
- Return typed error responses from the gateway.
- Avoid swallowing exceptions silently.

### Testing
- Every module added in Phase 0 should have tests.
- Favor unit tests first.
- Add at least one API integration test for the gateway.
- Add at least one benchmark artifact serialization test.

### Documentation
- Keep `README.md` accurate.
- Add `docs/architecture.md` once the skeleton exists.
- When making important structural decisions, add a short ADR under `docs/adr/`.

---

## How agents should work in this repo
When asked to implement something, agents should follow this loop:

1. Read `AGENTS.md` and the relevant files.
2. Restate the task in concrete terms.
3. Make a short plan before editing.
4. Implement the smallest coherent vertical slice.
5. Run relevant checks.
6. Summarize:
   - files changed,
   - commands run,
   - what works now,
   - what remains.

### Required checks before claiming a task is done
At minimum, run the relevant subset of:
- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest`

If a command fails, explain why and fix it if practical.

---

## What not to do
- Do not hardcode Apple-specific assumptions into router logic.
- Do not add real cloud dependencies in Phase 0.
- Do not introduce a heavy frontend.
- Do not create unused abstractions “for later” unless they directly support a known next phase.
- Do not overcomplicate packaging.
- Do not skip tests for core logic.

---

## Phase 0 implementation priorities
When choosing between multiple valid next steps, prefer this order:
1. Tooling and workspace setup
2. Core schemas and config
3. Backend adapter contract
4. Mock backend
5. Router service and policies
6. Gateway skeleton
7. Telemetry scaffolding
8. Benchmark artifact format and CLI
9. Documentation and dev ergonomics

## Phase 5 implementation priorities
When choosing between multiple valid next steps, prefer this order:
1. Keep the shared contracts, routing core, and test baseline clean
2. Add explicit worker inventory and network endpoint modeling in small slices
3. Isolate Apple-specific runtime dependencies from the portable control-plane packaging
4. Add containerized control-plane, Docker Compose, and `kind` deployment paths with safe defaults
5. Extend benchmarks, replay, reports, and runbooks to cover deployment-aware execution
6. Preserve adapter portability for future `vllm_cuda`, cloud, and remote workers
7. Update documentation and local dev ergonomics

---

## Future phases to keep in mind, but not build yet
- MLX-LM backend
- vLLM-Metal backend
- trace replay runner
- richer benchmark scenarios
- local cluster deployment
- remote CUDA workers
- learned router
- Forge integration for autotuning and later kernel optimization

Design with those in mind, but do not jump ahead.

---

## North star
Switchyard should eventually feel like a **real inference control plane**:
- clean contracts,
- measurable behavior,
- backend portability,
- strong observability,
- reproducible experiments.

If a change makes the project feel more like a serious systems/research-engineering artifact, it is probably the right direction.
