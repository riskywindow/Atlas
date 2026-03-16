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
**Phase 0: foundation and contracts**

This phase should create the repo skeleton, shared contracts, mock backend, gateway skeleton, routing primitives, telemetry scaffolding, and benchmark artifact format.

### Definition of done for Phase 0
Phase 0 is complete when all of the following are true:
1. The repo has a clean Python workspace with linting, typing, and tests.
2. There is a shared domain model for requests, responses, route decisions, backend capabilities, backend health, and benchmark artifacts.
3. There is a `BackendAdapter` interface / protocol and a deterministic `MockBackendAdapter` implementation.
4. There is a FastAPI gateway with:
   - `GET /healthz`
   - `GET /readyz`
   - `POST /v1/chat/completions`
5. The gateway can serve a chat-completions-style response using the mock backend.
6. Routing logic exists as a separate module from the HTTP layer.
7. Structured logging and basic telemetry hooks exist.
8. A benchmark artifact can be written to disk in a reproducible JSON format.
9. The project can be booted locally with a short, documented dev flow.

---

## Hard constraints

### Platform constraints
- Primary development machine: **Apple Silicon Mac (M4 Pro, 24GB RAM)**.
- During the Mac-first phase, **real model backends should run host-native on macOS**.
- Containerized services are fine for infra such as Postgres, Redis, Prometheus, Grafana, and the OpenTelemetry Collector.
- Do **not** add CUDA-only or Triton-only runtime dependencies in Phase 0.
- Keep the design explicitly portable so later phases can add `vllm_cuda` or other remote GPU workers.

### Scope constraints
- Phase 0 should **not** integrate real MLX-LM or vLLM-Metal yet unless required for a tiny smoke-test stub.
- Use **mocked or simulated backends** for the foundation work.
- Avoid premature multi-service complexity. In Phase 0, a **single Python workspace** is preferred over many separate packages.
- Do not build a frontend UI yet.
- Do not build Kubernetes manifests yet unless needed for a tiny placeholder doc.
- Do not introduce Ray into the request path in Phase 0.

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

5. **Benchmarks are product features.**
   Benchmark artifact schemas and reproducibility matter from the beginning.

6. **Mac-first, not Mac-locked.**
   All Apple-specific logic must stay at the adapter boundary.

7. **Prefer simple, typed Python.**
   Avoid magical frameworks or overengineered metaprogramming.

---

## Recommended stack for Phase 0
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

## Suggested Phase 0 repo layout
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

### Benchmark schemas
Create typed schemas for:
- `BenchmarkScenario`
- `BenchmarkRequestRecord`
- `BenchmarkSummary`
- `BenchmarkRunArtifact`

Artifacts should be reproducible and serializable.

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
- suitable for routing tests.

The mock backend should return responses that include enough metadata to verify which backend handled the request.

---

## Routing design rules
- The router must be a pure Python service/module that can be tested directly.
- The gateway should ask the router for a `RouteDecision`, then invoke the chosen backend.
- Phase 0 routing can be simple and static, but it must leave room for:
  - latency-first,
  - balanced,
  - quality-first,
  - local-only,
  - future cost-aware or learned routing.

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
- Update PROGRESS.md every time with what was done
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
