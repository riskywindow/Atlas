# Phase 1 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

A good rule: do not ask Codex to build all of Phase 1 in one shot. Keep it moving in clean, reviewable slices.

Phase 1 is the **first real local backend** phase. The goal is to keep Switchyard backend-agnostic while adding a serious Mac-native path using **MLX-LM** on Apple Silicon.

Core Phase 1 outcomes:
- one real local backend works end to end,
- health and warmup are real,
- gateway + router can use the real backend,
- streaming exists,
- TTFT / total latency / output tokens / tokens-per-second are measured,
- benchmark artifacts can be produced from real local runs,
- the control plane remains cleanly portable to future `vllm_metal`, `vllm_cuda`, and remote workers.

Non-goals for Phase 1:
- no vLLM-Metal yet,
- no CUDA or Triton runtime work yet,
- no Kubernetes or Ray in the request path,
- no giant frontend,
- no overengineered model registry or cloud scheduler,
- no assumptions that CI has Apple GPU access.

---

## Prompt 0 — bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 1. The old “current phase” text in AGENTS.md can be updated as part of this work. Switchyard is still a Mac-first, backend-agnostic inference fabric. For Phase 1, our only real backend is MLX-LM on Apple Silicon. Keep the control plane backend-agnostic and portable to future non-Mac backends. Do not add vLLM-Metal, CUDA, Triton, Kubernetes, Ray, or cloud workers yet unless explicitly asked. Prefer the smallest honest integration, lazy optional imports for MLX dependencies, CI-friendly tests that do not require Apple GPU access, and small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 — Phase 1 kickoff and repo audit

```text
Inspect the current repo and prepare it for Phase 1.

Requirements:
- Review the current codebase against the intended Phase 1 outcomes.
- Update AGENTS.md so the project phase is now Phase 1 instead of Phase 0.
- Add or update docs/phase1.md with:
  - Phase 1 goals,
  - definition of done,
  - non-goals,
  - the Mac-first constraint,
  - the requirement that future non-Mac backends must fit the same adapter boundary.
- Do not rewrite working Phase 0 code unless there is a clear mismatch with Phase 1 needs.
- If the current repo has any obvious Phase 0 gaps that would block Phase 1, patch only the smallest necessary gap.

Acceptance criteria:
- AGENTS.md and docs reflect the new phase accurately.
- The repo has a crisp Phase 1 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a giant refactor.
```

---

## Prompt 2 — extend contracts for real backends and streaming

```text
Extend the Phase 0 contracts so Switchyard can support a real local backend in Phase 1.

Requirements:
- Preserve the existing backend-agnostic design.
- Extend the chat schemas to support streaming responses cleanly.
  - Add a chunk/event schema for streamed chat completions.
  - Keep the public API OpenAI-like but not a copy-paste.
- Extend the backend adapter contract so it can support:
  - warmup,
  - non-streaming generate,
  - streaming generate,
  - richer health/capability snapshots.
- Add or refine config models for local model configuration, such as:
  - model alias,
  - model identifier/path,
  - backend type,
  - optional generation defaults,
  - optional warmup behavior.
- Update the mock backend so it can still satisfy the new interfaces and support deterministic streaming for tests.
- Add tests for the new schemas and contract changes.

Acceptance criteria:
- The repo has clean typed contracts for both streaming and non-streaming generation.
- The mock backend still works and is testable.
- Lint, type checks, and tests all pass.

Do not integrate MLX-LM yet. This prompt is only about contracts, config, and keeping the abstractions honest.
```

---

## Prompt 3 — add an MLX provider/runtime abstraction

```text
Implement the smallest honest runtime abstraction for MLX-LM integration.

Requirements:
- Add a thin internal runtime/provider layer for MLX-LM so the rest of Switchyard does not directly depend on raw MLX-LM calls everywhere.
- Keep the dependency optional and lazily imported.
- Design the runtime so it can later be replaced or paralleled by vLLM-Metal, vLLM-CUDA, or remote HTTP backends without changing the router or gateway.
- The runtime/provider should cover:
  - model loading,
  - warmup,
  - health checks,
  - non-streaming generation primitives,
  - streaming generation primitives if MLX-LM supports them cleanly.
- If native token streaming is awkward or version-sensitive, create a minimal interface that can support streaming without lying about what the backend is actually doing.
- Use test doubles / monkeypatching so the unit tests do not require MLX-LM or Apple GPU hardware.
- Fail gracefully with a clear error message when MLX-LM is not installed or the backend is misconfigured.

Acceptance criteria:
- There is a small, well-named runtime/provider abstraction with tests.
- The runtime is optional-dependency-safe.
- The rest of the codebase is not polluted with MLX-specific import logic.

Do not wire the runtime into the HTTP gateway yet unless a tiny integration point is necessary.
```

---

## Prompt 4 — implement the MLX-LM adapter end to end for non-streaming requests

```text
Implement the first real backend: an MLX-LM adapter for Apple Silicon.

Requirements:
- Add an MLX-LM adapter that implements the existing BackendAdapter contract.
- The adapter should expose honest values for:
  - backend name,
  - backend type,
  - device class,
  - capabilities,
  - health,
  - warmup,
  - non-streaming generation.
- Keep all Apple-specific or MLX-specific logic at the adapter/runtime boundary.
- Register the adapter conditionally based on config.
- Do not break the mock backend path.
- Wire the real adapter into the registry/router/gateway so a configured local model can answer a normal POST /v1/chat/completions request.
- If no MLX backend is configured, the app should still boot cleanly and surface a clear readiness/health state.
- Add tests that cover:
  - the adapter contract using a fake provider,
  - config/registration behavior,
  - end-to-end non-streaming gateway behavior where feasible.

Acceptance criteria:
- On a configured Apple Silicon machine, one real local model can answer a normal chat completion request through Switchyard.
- In CI or non-Mac environments, tests still pass without requiring MLX-LM.
- Existing routing and mock-backend behavior still work.

Do not add vLLM or any second real backend yet.
```

---

## Prompt 5 — implement streaming end to end

```text
Add Phase 1 streaming support across the adapter, gateway, and tests.

Requirements:
- Support a streamed chat-completions path through Switchyard.
- Keep the HTTP API shape familiar for clients that expect an OpenAI-like streaming experience.
- The gateway should be able to route a streaming request to the chosen backend and emit well-formed streamed chunks.
- The mock backend should support deterministic streaming for tests.
- The MLX-LM adapter should support streaming in the smallest honest way the backend allows.
  - If native incremental token streaming is available and stable, use it.
  - If not, implement a clearly documented fallback that still exercises the control-plane streaming path without pretending to be something it is not.
- Add tests for:
  - streamed chunk framing,
  - termination behavior,
  - request IDs and backend metadata in the streaming path,
  - at least one integration-style streaming test.

Acceptance criteria:
- A client can request stream=true and receive a streamed response through Switchyard.
- The streaming path reuses the same routing and adapter boundaries as the non-streaming path.
- Tests and type checks pass.

Do not overengineer cancellation, SSE edge cases, or multi-backend streaming policy yet unless needed for correctness.
```

---

## Prompt 6 — add real request metrics and backend lifecycle metrics

```text
Add the Phase 1 metrics needed to make the real backend measurable.

Requirements:
- Keep observability simple, typed, and local-friendly.
- Measure and record at least:
  - time to first token (TTFT) where meaningful,
  - total request latency,
  - output token count,
  - tokens per second,
  - chosen backend,
  - model alias or model identifier,
  - warmup timing and readiness state.
- Make sure metrics work for both streaming and non-streaming requests where possible.
- Keep the instrumentation logic out of the core business logic as much as practical.
- Expose metrics in a way that works for local development. A lightweight Prometheus-friendly path is fine if it stays optional and clean.
- Improve structured logs so a single request can be traced through route selection, backend execution, and response summary.
- Add tests for helper functions and any deterministic metrics logic.

Acceptance criteria:
- A developer can inspect logs and/or metrics and see TTFT, total latency, output tokens, and backend choice.
- The metrics do not require a full monitoring stack to be useful locally.
- All checks pass.

Do not build a full production observability platform yet.
```

---

## Prompt 7 — extend the benchmark CLI for real local runs

```text
Upgrade the benchmark CLI so it can exercise the real MLX-LM backend in Phase 1.

Requirements:
- Reuse the Phase 0 benchmark artifact path rather than inventing a new benchmark system.
- Add a benchmark mode that can run against the local gateway and record real backend metrics.
- Include conservative default scenarios that are realistic on an M4 Pro with 24GB RAM.
- Capture at least:
  - run_id,
  - timestamp,
  - routing policy,
  - backend name/type,
  - model alias,
  - request count,
  - success/error counts,
  - latency summary,
  - TTFT summary if available,
  - output token totals,
  - tokens-per-second summary,
  - enough config/environment metadata to reproduce the run.
- Keep defaults lightweight so a developer can run a benchmark without frying the laptop.
- Add tests for the artifact format and benchmark helper logic.

Acceptance criteria:
- One command can produce a clean benchmark artifact from a real local backend run.
- The artifact format stays coherent with the existing Phase 0 benchmark story.
- All checks pass.

Do not add complex concurrency or distributed load generation yet. Keep it honest and Mac-friendly.
```

---

## Prompt 8 — improve developer ergonomics for the Mac-first path

```text
Polish the developer experience for Phase 1 on Apple Silicon.

Requirements:
- Update README.md with a clear Mac-first Phase 1 setup flow.
- Document how to install optional MLX-LM dependencies using uv extras or the repo’s preferred dependency flow.
- Document how to configure a local model without hardcoding a single required model in the codebase.
- Add example commands for:
  - starting the gateway,
  - hitting the chat endpoint,
  - streaming a response,
  - warming up the backend if applicable,
  - running a small benchmark.
- Add or improve a minimal Makefile or justfile if it helps, but keep it small.
- Add troubleshooting notes for common local issues such as missing optional dependencies, model load failures, and memory pressure.
- Update docs/architecture.md to explain how the MLX adapter fits into a future multi-backend design.
- Add at least one short ADR for a key Phase 1 decision, such as optional MLX dependencies or the chosen runtime boundary.

Acceptance criteria:
- A new contributor on an Apple Silicon Mac can get a real local backend working from the docs.
- The docs make it clear that Phase 1 is Mac-first but not Mac-locked.
- All checks still pass.
```

---

## Prompt 9 — optional local metrics/dashboard scaffolding

```text
Add optional local metrics/dashboard scaffolding for Phase 1 without making it mandatory.

Requirements:
- If it fits the current repo cleanly, add a minimal /metrics path or Prometheus scrape path.
- Optionally add a small Grafana dashboard JSON or docs snippet that visualizes:
  - request latency,
  - TTFT,
  - tokens per second,
  - request count by backend,
  - readiness / warmup state.
- Keep everything optional and clearly documented.
- Do not make the core Phase 1 workflow depend on Docker Compose or a running monitoring stack.

Acceptance criteria:
- Local observability is nicer, but the core developer flow still works with no extra services.
- The added files are clean, documented, and not overengineered.
```

---

## Prompt 10 — Phase 1 exit review

```text
Review the repo against AGENTS.md and the intended Phase 1 definition of done.

Tasks:
- Identify anything missing, weak, or too clever in the current Phase 1 implementation.
- Tighten tests, docs, config naming, and error messages where needed.
- Remove accidental overengineering.
- Make sure the adapter boundary is still clean enough for future vLLM-Metal and vLLM-CUDA backends.
- Verify that the mock backend path still works and remains useful for tests.
- Verify that the real local backend path is clearly documented and measurable.

Deliverables:
- a concise Phase 1 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 2 tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 1 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add a real MLX-LM backend,
2. any contract/schema changes needed for streaming,
3. the exact dependency and optional-extra changes you recommend,
4. the test strategy for CI without Apple GPU access,
5. the implementation order you would use,
6. the biggest risks or version-sensitivity points in MLX-LM integration.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
