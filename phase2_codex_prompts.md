# Phase 2 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

Keep the same discipline as earlier phases: do not ask Codex to build all of Phase 2 in one shot. Push it through small, reviewable vertical slices.

Phase 2 is the **second real backend + real routing** phase. The goal is to keep Switchyard Mac-first and backend-agnostic while adding a second serious local path using **vLLM-Metal** and upgrading the control plane so the same logical model target can route across multiple backend implementations.

Core Phase 2 outcomes:
- a second real backend exists (`vllm_metal`),
- one logical model alias can be backed by more than one local backend,
- router v1 can select between eligible backends,
- policy modes exist and are testable,
- health-aware fallback works,
- route-level metrics and traces exist,
- benchmark tooling can compare backends and routing policies,
- the control plane remains portable to future `vllm_cuda`, remote HTTP workers, and cloud bursting.

Recommended policy modes for Phase 2:
- `latency_first`
- `balanced`
- `quality_first`
- `local_only`
- optional explicit backend pin for testing/admin use

Non-goals for Phase 2:
- no CUDA workers yet,
- no rented cloud GPUs yet,
- no Triton/CUDA kernel work yet,
- no Ray or Kubernetes in the request path,
- no learned router yet,
- no giant UI,
- no fragile assumptions that CI has Apple GPU access,
- no broad refactor of the Phase 1 MLX path unless clearly needed.

A good theme for this phase: **one public API, multiple local engines, measurable routing behavior**.

---

## Prompt 0 — bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 2. The old “current phase” text in AGENTS.md can be updated as part of this work. Switchyard is still a Mac-first, backend-agnostic inference fabric. For Phase 2, the major additions are: a second real backend using vLLM-Metal, multi-backend model registration, router v1 policy modes, health-aware fallback, route-level observability, and comparative benchmark tooling. Keep the control plane portable to future vLLM-CUDA and remote workers. Use lazy optional imports for Mac-specific dependencies, keep CI-friendly tests that do not require Apple GPU hardware, avoid overengineering, and ship in small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 — Phase 2 kickoff and repo audit

```text
Inspect the current repo and prepare it for Phase 2.

Requirements:
- Review the current codebase against the intended Phase 2 outcomes.
- Update AGENTS.md so the project phase is now Phase 2 instead of Phase 1.
- Add or update docs/phase2.md with:
  - Phase 2 goals,
  - definition of done,
  - non-goals,
  - the Mac-first constraint,
  - the requirement that future non-Mac backends must fit the same adapter boundary,
  - the requirement that one logical model alias may map to multiple backend implementations.
- Identify any small Phase 1 gaps that would obviously block Phase 2 and patch only the smallest necessary blockers.
- Do not do a giant refactor.

Acceptance criteria:
- AGENTS.md and docs reflect the new phase accurately.
- The repo has a crisp Phase 2 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a rebuild.
```

---

## Prompt 2 — harden contracts for multi-backend routing

```text
Extend the current contracts so Switchyard can support multiple backend implementations behind one logical model target.

Requirements:
- Preserve the existing backend-agnostic design.
- Introduce or refine the distinction between:
  - logical model alias / serving target,
  - backend deployment / adapter instance,
  - backend capabilities,
  - route decision / route explanation.
- Add or refine typed models for backend metadata such as:
  - engine_type,
  - device_class,
  - readiness / warm state,
  - streaming support,
  - max context,
  - quality tier or quality hint,
  - performance hint,
  - cache-related capability flags if available,
  - configured priority / weight,
  - route eligibility state.
- Add a typed route-decision/explanation structure so routing choices can be logged and benchmarked deterministically.
- Keep the public API stable. If you need routing overrides for tests or admin workflows, use a clearly namespaced internal mechanism rather than polluting the base chat schema.
- Update tests for the new contracts.

Acceptance criteria:
- The repo has clean typed contracts for multi-backend selection.
- The distinction between logical model alias and concrete backend deployment is clear.
- Tests, lint, and type checks pass.

Do not implement the real routing logic yet unless a tiny placeholder is needed to keep the code compiling.
```

---

## Prompt 3 — add a vLLM-Metal provider/runtime abstraction

```text
Implement the smallest honest runtime/provider abstraction for vLLM-Metal integration.

Requirements:
- Add a thin internal runtime/provider layer for vLLM-Metal so the rest of Switchyard does not depend directly on raw vLLM-Metal calls everywhere.
- Keep the dependency optional and lazily imported.
- Design the runtime so it can later be paralleled by vLLM-CUDA with minimal higher-level changes.
- The runtime/provider should cover:
  - model loading or connection setup,
  - warmup,
  - health checks,
  - non-streaming generation primitives,
  - streaming generation primitives where honestly supported,
  - capability discovery or capability declaration.
- Isolate version-sensitive or backend-specific translation logic inside the provider layer.
- Use fakes / monkeypatching so tests do not require vLLM-Metal or Apple GPU hardware.
- Fail gracefully with clear error messages when the optional dependency is missing or misconfigured.

Acceptance criteria:
- There is a small, well-named vLLM-Metal provider/runtime abstraction with tests.
- The rest of the codebase is not polluted with ad hoc vLLM-specific import logic.
- Optional-dependency behavior is clear and safe.

Do not wire this fully into the HTTP path yet unless a tiny integration point is necessary.
```

---

## Prompt 4 — implement the vLLM-Metal adapter end to end for non-streaming requests

```text
Implement the second real backend: a vLLM-Metal adapter for Apple Silicon.

Requirements:
- Add a vLLM-Metal adapter that implements the existing BackendAdapter contract.
- The adapter should expose honest values for:
  - backend name,
  - backend type,
  - device class,
  - capabilities,
  - health,
  - warmup,
  - non-streaming generation.
- Keep all Apple-specific or vLLM-specific logic at the adapter/runtime boundary.
- Register the adapter conditionally based on config.
- Do not break the MLX-LM path or the mock backend path.
- Add tests that cover:
  - the adapter contract using a fake provider,
  - config/registration behavior,
  - a small end-to-end non-streaming flow where feasible.

Acceptance criteria:
- On a configured Apple Silicon machine, the new backend can answer a normal chat completion request through Switchyard.
- In CI or non-Mac environments, tests still pass without requiring vLLM-Metal.
- Existing MLX behavior still works.

Do not add router policy logic yet beyond whatever minimal wiring is needed to select the backend explicitly for testing.
```

---

## Prompt 5 — support one logical model alias backed by multiple local engines

```text
Upgrade the model registry / config / backend registration path so one logical model alias can be served by more than one backend implementation.

Requirements:
- Keep the public serving story simple: a client should target a logical model alias, not a hardcoded backend name.
- Add or refine config so a logical model alias can point to multiple candidate backend deployments, for example MLX-LM and vLLM-Metal.
- Keep explicit backend pinning available for tests/admin workflows, but do not make it the normal public path.
- Ensure the registry can answer questions like:
  - which backend deployments are eligible for this alias,
  - which are healthy,
  - which support streaming,
  - which are warm,
  - which are currently preferred or deprioritized.
- Preserve a simple path for single-backend aliases too.
- Add tests for registry behavior and config loading.

Acceptance criteria:
- The same logical model alias can resolve to multiple candidate backends.
- The code clearly separates alias resolution from route selection.
- Existing single-backend behavior still works.

Do not implement full policy scoring yet unless a tiny placeholder is needed.
```

---

## Prompt 6 — implement router v1 with explicit policy modes

```text
Implement router v1 so Switchyard can choose among eligible backends using explicit policy modes.

Requirements:
- Add policy modes at minimum:
  - latency_first,
  - balanced,
  - quality_first,
  - local_only.
- Support explicit backend pinning for tests/admin use via a clearly namespaced override path.
- The router should consider at least:
  - health/readiness,
  - backend eligibility for the request,
  - streaming compatibility,
  - configured priority/weight,
  - basic performance hints or observed metrics if available,
  - quality hints,
  - request shape hints such as prompt length or expected output length where easy to compute.
- Keep the scoring logic deterministic, inspectable, and easy to test.
- Emit a route-decision explanation object that records why a backend won or lost.
- Do not make the router depend on Apple specifics.
- Add tests for each policy mode and edge cases.

Acceptance criteria:
- Given a logical model alias with multiple healthy candidates, Switchyard can select a backend according to the chosen policy.
- The route decision is explainable in logs and artifacts.
- Tests, lint, and type checks pass.

Keep the router honest and relatively simple. This is policy routing v1, not a learned scheduler.
```

---

## Prompt 7 — add health-aware fallback and pre-first-token failover

```text
Implement health-aware fallback for Phase 2.

Requirements:
- If the chosen backend is unhealthy or fails before producing a response, the router/executor should be able to retry on an eligible alternate backend.
- Keep the retry budget conservative, for example a single fallback attempt.
- For streaming requests, only allow failover before the first token/chunk is emitted. Do not attempt magical mid-stream migration.
- Record fallback behavior clearly in logs, route explanations, and metrics.
- Avoid duplicate or confusing response framing.
- Keep the fallback mechanism backend-agnostic.
- Add tests for:
  - unhealthy-primary fallback,
  - exception-before-first-token fallback,
  - no-eligible-fallback behavior,
  - streaming pre-first-token fallback rules.

Acceptance criteria:
- A failed first-choice backend can trigger a clean fallback to another eligible backend where safe.
- The behavior is explainable and measurable.
- Existing happy-path behavior remains clean.

Do not add a full circuit-breaker framework unless a tiny local version is clearly justified.
```

---

## Prompt 8 — ensure the second backend works in the streaming path too

```text
Extend the streaming path so router v1 and the second backend work cleanly together.

Requirements:
- Make sure streaming requests can route through either MLX-LM or vLLM-Metal using the same logical alias and policy system.
- Reuse the same adapter and router boundaries as the non-streaming path.
- Keep request IDs, backend metadata, and route metadata consistent in the streaming path.
- If one backend has different streaming semantics, normalize them at the gateway/adapter boundary without lying about behavior.
- Keep tests deterministic using fakes where needed.
- Add at least one integration-style streaming test that exercises a multi-backend alias.

Acceptance criteria:
- A client can request stream=true and be served by either backend under router control.
- Streaming remains observable and debuggable.
- Tests and type checks pass.

Do not overengineer cancellation, SSE edge cases, or advanced backpressure unless required for correctness.
```

---

## Prompt 9 — add route-level metrics, route explanations, and comparative benchmark runs

```text
Upgrade observability and benchmark tooling so Phase 2 routing behavior is measurable.

Requirements:
- Extend metrics/logging to capture at least:
  - logical model alias,
  - chosen backend,
  - candidate backend count,
  - routing policy,
  - whether fallback occurred,
  - route-decision reason or compact explanation,
  - request latency,
  - TTFT if available,
  - output tokens,
  - tokens per second.
- Keep the observability path local-friendly and do not require a full monitoring stack to be useful.
- Extend the benchmark CLI so it can:
  - run the same workload against different routing policies,
  - compare a multi-backend alias versus an explicitly pinned backend,
  - emit a benchmark artifact that clearly shows route choices and outcome summaries.
- Reuse the existing benchmark artifact story rather than inventing a separate reporting system.
- Add tests for benchmark artifact helpers and route-explanation serialization.

Acceptance criteria:
- A developer can run comparative benchmarks and see which backend or routing policy was used and how it performed.
- Route decisions show up in both logs and benchmark artifacts.
- All checks pass.

Keep defaults lightweight enough for an M4 Pro with 24GB RAM.
```

---

## Prompt 10 — documentation and developer ergonomics for the dual-backend Mac path

```text
Polish the docs and developer ergonomics for Phase 2.

Requirements:
- Update README.md with a clear Mac-first Phase 2 setup flow.
- Document how to install optional MLX-LM and vLLM-Metal dependencies using the repo’s preferred dependency flow.
- Add example configs for:
  - an MLX-only alias,
  - a vLLM-Metal-only alias,
  - a dual-backend alias that can route between both.
- Add example commands for:
  - starting the gateway,
  - hitting the chat endpoint,
  - streaming a response,
  - selecting a routing policy,
  - pinning a backend for debugging,
  - running a comparative benchmark.
- Update docs/architecture.md to explain:
  - logical model alias vs backend deployment,
  - router v1 policy flow,
  - where failover happens,
  - how this design stays portable to future cloud/CUDA backends.
- Add at least one short ADR for a key Phase 2 decision, such as the alias-to-multiple-backends model or pre-first-token failover policy.
- Add troubleshooting notes for optional dependency issues and backend mismatch/misconfiguration.

Acceptance criteria:
- A new contributor on an Apple Silicon Mac can run either backend and a dual-backend alias from the docs.
- The docs make it obvious that the design is Mac-first now but portable later.
- All checks still pass.
```

---

## Prompt 11 — Phase 2 exit review

```text
Review the repo against AGENTS.md and the intended Phase 2 definition of done.

Tasks:
- Identify anything missing, weak, or too clever in the current Phase 2 implementation.
- Tighten tests, docs, config naming, error messages, and benchmark/report clarity where needed.
- Remove accidental overengineering.
- Verify that:
  - both real local backends still fit the same adapter contract,
  - one logical alias can route across both,
  - policy routing is deterministic and explainable,
  - fallback is safe and measurable,
  - the benchmark story is coherent,
  - the design is still clean for future vLLM-CUDA and remote backends.
- Make code changes only where they clearly improve completeness or clarity.

Deliverables:
- a concise Phase 2 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 3 tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 2 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add a second real backend and multi-backend routing,
2. any contract/config changes needed for alias-to-multiple-backends support,
3. the exact dependency and optional-extra changes you recommend,
4. the test strategy for CI without Apple GPU access,
5. the implementation order you would use,
6. the biggest risks or version-sensitivity points in vLLM-Metal integration,
7. the benchmark plan for comparing backend and routing-policy behavior.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
