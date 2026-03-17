# Phase 4 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

Keep the same discipline as earlier phases: do not ask Codex to build all of Phase 4 in one shot. Push it through small, reviewable vertical slices.

Phase 4 is the **control-plane behavior under load and partial failure** phase. The goal is to make Switchyard act more like a real inference control plane when the system is stressed, degraded, or gradually changing.

Core Phase 4 outcomes:
- bounded admission control and overload behavior exist,
- per-tenant limits and request classification are first-class,
- backend failure tracking and circuit-breaker-style routing exist,
- session affinity for multi-turn chat exists,
- shadow traffic exists for safe comparative evaluation,
- canary routing exists for controlled rollout to candidate backends,
- runtime state is observable and inspectable,
- the whole design stays Mac-first, backend-agnostic, and ready for later cloud workers.

Recommended scenario families to test in Phase 4:
- queue saturation,
- backend flakiness and recovery,
- partial backend degradation,
- session-affinity stickiness and failover,
- shadow mirroring,
- canary rollout distribution,
- per-tenant contention.

Non-goals for Phase 4:
- no cloud GPU workers yet,
- no distributed global rate limiter,
- no external identity provider or billing system,
- no giant frontend UI,
- no learned router yet,
- no Forge or kernel optimization work yet,
- no heavy background job framework unless a tiny one is clearly required,
- no hidden side effects in shadow traffic,
- no multi-region or HA control-plane claims.

A good theme for this phase: **graceful degradation, safe rollout, and observable control-plane decisions**.

---

## Prompt 0 - bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 4. The old current-phase text in AGENTS.md can be updated as part of this work. Switchyard is still a Mac-first, backend-agnostic inference fabric. For Phase 4, the major additions are: bounded admission control, per-tenant limits, circuit-breaker-style backend protection, session affinity for multi-turn chat, shadow traffic, canary routing, and stronger observability around overload and degradation. Keep the design portable to future vLLM-CUDA and cloud workers, keep tests CI-friendly without Apple GPU access, avoid overengineering, and ship in small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 - Phase 4 kickoff and repo audit

```text
Inspect the current repo and prepare it for Phase 4.

Requirements:
- Review the codebase against the intended Phase 4 outcomes.
- Update AGENTS.md so the project phase is now Phase 4 instead of Phase 3.
- Add or update docs/phase4.md with:
  - Phase 4 goals,
  - definition of done,
  - non-goals,
  - the Mac-first constraint,
  - the rule that overload behavior must be explicit and testable,
  - the rule that shadow traffic must be opt-in and must never affect primary user-visible responses,
  - the rule that rollout behavior must be explainable from logs and artifacts.
- Identify any tiny Phase 3 gaps that obviously block Phase 4 and patch only the smallest necessary blockers.
- Do not do a giant refactor.

Acceptance criteria:
- AGENTS.md and docs reflect Phase 4 accurately.
- The repo has a crisp Phase 4 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a rebuild.
```

---

## Prompt 2 - harden schemas and config for control-plane state

```text
Extend the current schemas and config so Phase 4 control-plane behavior becomes first-class and typed.

Requirements:
- Preserve existing Phase 3 artifact compatibility where reasonable. If a schema or config version bump is needed, make it explicit and documented.
- Add or refine typed models for:
  - tenant identity and request class,
  - admission decision,
  - queue snapshot,
  - limiter state,
  - circuit breaker state,
  - session affinity key and sticky-route record,
  - canary policy / weighted rollout rule,
  - shadow policy,
  - route annotations for overload, breaker, and affinity decisions,
  - relevant telemetry/report metadata.
- Extend configuration to support at minimum:
  - per-tenant concurrency caps,
  - bounded queue sizes,
  - default timeout / cooldown values,
  - canary percentages,
  - shadow sampling,
  - session affinity TTL,
  - feature toggles to disable advanced Phase 4 behavior.
- Keep the control plane hardware-agnostic. No schema should hardcode Apple-specific behavior.
- Add tests for validation and serialization.

Acceptance criteria:
- The repo has clean typed contracts for Phase 4 control-plane behavior.
- Config is explicit and documented.
- Existing code still compiles and tests pass.

Do not implement the full control-plane logic yet unless a tiny placeholder is needed to keep the code compiling.
```

---

## Prompt 3 - introduce tenant-aware request context and request classification

```text
Make tenant identity and request classification first-class in the gateway and routing context.

Requirements:
- Add a clear way for Switchyard to identify a tenant and optional request class without breaking the public OpenAI-like API.
- Prefer a small, explicit Switchyard-specific mechanism such as headers or a clearly namespaced extension field rather than changing core chat semantics.
- Support at minimum:
  - tenant ID,
  - request priority/class,
  - optional session/conversation key,
  - explicit backend pinning for admin/test workflows only.
- Ensure request context is propagated through routing, execution, benchmark artifacts, trace capture, and replay where appropriate.
- Add sensible defaults for local development so a missing tenant ID still works.
- Validate and sanitize incoming values.
- Add tests for parsing, validation, and propagation.

Acceptance criteria:
- Switchyard can classify requests by tenant and request class.
- The context propagates cleanly without breaking existing Phase 3 workflows.
- Tests pass and docs are updated where necessary.

Keep this focused on context propagation, not admission control itself.
```

---

## Prompt 4 - implement admission control and bounded queueing v1

```text
Implement admission control and bounded queueing as a first-class subsystem outside the HTTP layer.

Requirements:
- Create a small, testable admission-control service/module that decides whether a request is:
  - admitted immediately,
  - queued briefly,
  - or rejected.
- Support at minimum:
  - global caps,
  - per-tenant concurrency caps,
  - bounded queue length,
  - queue timeout / stale-request expiration,
  - clear reason codes for rejection.
- Keep the first implementation simple and local-friendly, for example in-process coordination rather than a distributed queue.
- Integrate the admission decision into the gateway/control-plane path without burying logic inside route handlers.
- Expose relevant timing and state in logs, metrics, and benchmark artifacts where appropriate.
- Return sensible HTTP status behavior for overload, such as 429 or 503, and make the choice explicit and documented.
- Add tests for saturation, fairness, queue expiry, and deterministic behavior under mocked timing.

Acceptance criteria:
- Under load, Switchyard behaves predictably instead of accepting unbounded work.
- Per-tenant caps are enforced.
- Queueing/rejection decisions are visible in logs/metrics.
- Tests pass.

Do not add a heavy external queuing system here.
```

---

## Prompt 5 - add backend failure tracking and circuit breakers

```text
Implement backend failure tracking and circuit-breaker-style protection.

Requirements:
- Add a circuit-breaker subsystem that can track backend failures and transition through explicit states such as:
  - closed,
  - open,
  - half_open.
- The router should avoid routing to open backends except for controlled probe behavior where appropriate.
- Make failure accounting explicit for at least:
  - transport or invocation failures,
  - timeout-like failures,
  - repeated backend errors,
  - recovery after cooldown.
- Keep time-dependent behavior testable by isolating time sources or using injectable clocks where practical.
- Ensure route decisions record breaker-related reasons so they show up in artifacts and logs.
- Add tests for state transitions, cooldowns, recovery, and interaction with existing health/fallback logic.

Acceptance criteria:
- Repeated backend failure causes Switchyard to stop hammering that backend.
- Recovery is gradual and explicit.
- Breaker behavior is observable and testable.
- Existing routing and fallback behavior still works.

Keep this practical. This is a local control-plane breaker, not a giant resilience framework.
```

---

## Prompt 6 - implement session affinity for multi-turn chat

```text
Implement session affinity so related multi-turn requests try to stay on the same backend when that is beneficial and safe.

Requirements:
- Add a Switchyard-specific session or conversation key mechanism that works alongside the OpenAI-like API.
- Implement sticky routing so repeated requests with the same session key prefer the same backend while it remains healthy and eligible.
- Keep session affinity bounded and configurable with TTL / expiration. Do not create an unbounded in-memory map.
- If the sticky backend becomes unhealthy, overloaded, or ineligible, fail over cleanly and record the reason.
- Ensure affinity interacts sensibly with backend pinning, canaries, and circuit breakers.
- Add tests for stickiness, expiry, failover, and interaction with routing policies.

Acceptance criteria:
- Multi-turn requests can remain on the same backend when appropriate.
- Session state expires predictably.
- Failover from sticky routing is explainable in logs/artifacts.
- Tests pass.

Do not build a full distributed session store yet. Keep it local and pluggable.
```

---

## Prompt 7 - implement shadow traffic v1

```text
Implement shadow traffic so selected requests can be mirrored to another backend or alias without affecting the primary response.

Requirements:
- Shadow traffic must be explicitly opt-in and off by default.
- Support a clear policy model for shadowing, such as by alias, backend, sampling percentage, or tenant.
- The primary response path must remain authoritative. Shadow execution must never change the user-visible response.
- Avoid accidental loops or recursive shadowing.
- Record shadow outcomes separately enough that they can be analyzed later, including:
  - shadow target,
  - launch timestamp,
  - latency,
  - success/failure,
  - route/backend details,
  - link back to the primary request ID.
- Keep the implementation lightweight and local-friendly. It is fine if shadow execution is best-effort.
- Integrate with existing Phase 3 artifact/report concepts where sensible.
- Add tests for opt-in behavior, non-interference with primary responses, and loop prevention.

Acceptance criteria:
- Switchyard can mirror selected traffic safely.
- Shadow activity is observable and linked to primary requests.
- Primary request handling remains stable even if shadow execution fails.
- Tests pass.

Do not build a large experiment platform. Keep shadowing practical and controlled.
```

---

## Prompt 8 - implement canary routing / weighted rollout v1

```text
Implement canary routing so a logical alias can gradually shift a controlled fraction of traffic to a candidate backend.

Requirements:
- Add a weighted rollout mechanism for a logical alias that can direct a configured percentage of eligible traffic to a candidate backend while the rest stays on the primary path.
- Use deterministic bucketing where practical so routing is stable and testable, for example by request ID or session key.
- Keep canary behavior health-aware so an unhealthy candidate backend is automatically deprioritized or bypassed.
- Ensure canary routing interacts sensibly with:
  - session affinity,
  - backend pinning,
  - shadow traffic,
  - circuit breakers,
  - local-only policy constraints.
- Route annotations and logs should make it obvious when a request was selected by canary logic.
- Add tests for percentage distribution, determinism, health-aware fallback, and interaction with affinity.

Acceptance criteria:
- A logical alias can split traffic between current and candidate backends.
- Canary selection is explainable and deterministic enough for testing.
- Candidate-backend failures do not destabilize primary traffic.
- Tests pass.

Keep this as a focused rollout mechanism, not a full feature-flag system.
```

---

## Prompt 9 - add admin/runtime inspection tools for Phase 4 state

```text
Add lightweight admin/runtime inspection tooling for the new Phase 4 control-plane state.

Requirements:
- Provide a small admin CLI, admin API, or clearly scoped inspection commands that let a developer inspect at least:
  - backend health,
  - circuit breaker state,
  - queue depth / admission stats,
  - tenant limiter state summary,
  - active canary config,
  - active shadow config,
  - session-affinity cache summary.
- Keep the first version local-dev friendly and safe. Do not expose powerful mutation endpoints casually.
- If you add state mutation, keep it tightly scoped and documented.
- Ensure the inspection view is consistent with runtime truth and does not invent fake status.
- Add tests for any API/CLI helpers you introduce.

Acceptance criteria:
- A developer can inspect key Phase 4 runtime state without digging through logs only.
- The tooling is lightweight and coherent with the current repo style.
- Tests pass.

Do not build a web dashboard here. A CLI or small admin API is enough.
```

---

## Prompt 10 - extend benchmarking and replay to cover overload, rollout, and shadow scenarios

```text
Build on the Phase 3 benchmarking/replay system so the new Phase 4 behavior can be exercised and explained.

Requirements:
- Extend workload generation, benchmark execution, or replay tooling as needed so a developer can exercise at minimum:
  - queue saturation,
  - per-tenant contention,
  - backend flakiness,
  - session stickiness,
  - canary rollout,
  - shadow traffic.
- Keep the benchmark artifact as the authoritative truth source.
- Ensure artifacts can record the new control-plane signals, such as:
  - admission outcomes,
  - queue wait times,
  - breaker state or breaker-trigger reason,
  - canary-selection reason,
  - shadow linkage,
  - session-affinity hit/miss and failover reason.
- Prefer small scenario additions over giant simulation frameworks.
- Add tests and at least one realistic local example configuration for an M4 Pro Mac.

Acceptance criteria:
- The new Phase 4 behavior can be exercised through artifacts and reports, not just logs.
- Benchmark and replay outputs make control-plane decisions explainable.
- Tests pass.

Do not turn this into a huge load-testing product. Keep it tightly aligned with Switchyard.
```

---

## Prompt 11 - docs and developer ergonomics for Phase 4 workflows

```text
Polish the docs and developer ergonomics for Phase 4.

Requirements:
- Update README.md with a clear Mac-first Phase 4 workflow.
- Add docs/control-plane.md or equivalent that explains:
  - tenant/request classification,
  - admission control,
  - per-tenant limits,
  - circuit breakers,
  - session affinity,
  - shadow traffic,
  - canary routing,
  - runtime inspection,
  - how Phase 4 integrates with Phase 3 benchmarking and replay.
- Add example commands for:
  - running the gateway under bounded admission control,
  - exercising a tenant cap,
  - observing breaker transitions with a flaky backend,
  - sending requests with a session key,
  - enabling safe shadow traffic,
  - enabling a canary rollout,
  - inspecting runtime state,
  - capturing benchmark artifacts for overload or rollout experiments.
- Add example local configs that are realistic for an M4 Pro Mac.
- Update docs/architecture.md to explain where admission control, breakers, session affinity, shadowing, and canaries fit into the system.
- Add at least one short ADR for a key Phase 4 decision, such as bounded in-process queues, deterministic canary bucketing, or local session-affinity storage.

Acceptance criteria:
- A new contributor can run the Phase 4 workflow from the docs.
- The docs make safety and overload behavior obvious.
- All checks still pass.
```

---

## Prompt 12 - Phase 4 exit review

```text
Review the repo against AGENTS.md and the intended Phase 4 definition of done.

Tasks:
- Identify anything missing, weak, or too clever in the current Phase 4 implementation.
- Tighten tests, docs, config naming, artifact clarity, failure handling, rollout UX, and inspection tooling where needed.
- Remove accidental overengineering.
- Verify that:
  - overload behavior is explicit and bounded,
  - per-tenant limits are actually enforced,
  - circuit breakers prevent repeated hammering of bad backends,
  - session affinity is useful but bounded,
  - shadow traffic is safe and non-interfering,
  - canary routing is deterministic enough to test and explain,
  - Phase 4 behavior shows up in artifacts and reports,
  - the design still stays clean for future CUDA/cloud phases.
- Make code changes only where they clearly improve completeness or clarity.

Deliverables:
- a concise Phase 4 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 5 tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 4 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add admission control, per-tenant limits, circuit breakers, session affinity, shadow traffic, canary routing, and runtime inspection,
2. any schema/config changes needed,
3. the simplest local-first design you recommend for queues, sticky-session state, and breaker state,
4. the test strategy for CI without Apple GPU access,
5. the implementation order you would use,
6. the main safety or correctness risks in shadow traffic and canary rollout and how to mitigate them,
7. how you would extend Phase 3 artifacts and reports so Phase 4 behavior is explainable.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
