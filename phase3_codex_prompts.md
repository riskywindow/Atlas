# Phase 3 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

Keep the same discipline as earlier phases: do not ask Codex to build all of Phase 3 in one shot. Push it through small, reviewable vertical slices.

Phase 3 is the **benchmarking, workload generation, and trace replay** phase. The goal is to make benchmarking a first-class product feature instead of a side utility.

Core Phase 3 outcomes:
- reproducible workload generation exists,
- benchmark artifacts are richer, versioned, and easy to compare,
- the gateway can optionally capture replayable request traces,
- recorded traces can be replayed against a logical alias, a routing policy, or a pinned backend,
- A/B comparative runs exist for policy and backend comparisons,
- repeated-prefix and bursty workload shapes are easy to test,
- markdown and machine-readable reports are generated from the same benchmark truth source,
- the whole workflow stays lightweight enough to run on an M4 Pro Mac.

Recommended benchmark scenario families for Phase 3:
- short chat,
- long prompt,
- repeated prefix,
- concurrency burst,
- mixed workload set.

Non-goals for Phase 3:
- no cloud GPU workers yet,
- no Ray or Kubernetes in the request path,
- no learned router yet,
- no giant web UI,
- no Forge or kernel optimization work yet,
- no advanced statistical framework unless it is tiny and clearly useful,
- no privacy-hostile trace capture defaults,
- no broad refactor of Phase 2 routing unless clearly required.

A good theme for this phase: **one control plane, reproducible workloads, explainable comparisons**.

---

## Prompt 0 - bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 3. The old current-phase text in AGENTS.md can be updated as part of this work. Switchyard is still a Mac-first, backend-agnostic inference fabric. For Phase 3, the major additions are: reproducible workload generation, richer benchmark artifacts, optional request-trace capture with safe defaults, trace replay against aliases/policies/pinned backends, A/B comparative runs, repeated-prefix and bursty workload scenarios, and markdown plus machine-readable report generation. Keep the design portable to future vLLM-CUDA and cloud workers, keep tests CI-friendly without Apple GPU access, avoid overengineering, and ship in small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 - Phase 3 kickoff and repo audit

```text
Inspect the current repo and prepare it for Phase 3.

Requirements:
- Review the codebase against the intended Phase 3 outcomes.
- Update AGENTS.md so the project phase is now Phase 3 instead of Phase 2.
- Add or update docs/phase3.md with:
  - Phase 3 goals,
  - definition of done,
  - non-goals,
  - the Mac-first constraint,
  - the requirement that benchmark and trace workflows stay backend-agnostic,
  - the rule that trace capture must be opt-in and privacy-conscious,
  - the requirement that reports derive from authoritative benchmark artifacts rather than ad hoc logs.
- Identify any tiny Phase 2 gaps that obviously block Phase 3 and patch only the smallest necessary blockers.
- Do not do a giant refactor.

Acceptance criteria:
- AGENTS.md and docs reflect Phase 3 accurately.
- The repo has a crisp Phase 3 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a rebuild.
```

---

## Prompt 2 - harden benchmark and trace domain models

```text
Extend the current schemas so benchmarking and replay become first-class, typed subsystems.

Requirements:
- Preserve existing Phase 2 artifact compatibility where reasonable. If a schema version bump is needed, make it explicit and documented.
- Add or refine typed models for:
  - benchmark run config,
  - workload scenario,
  - workload item,
  - execution target (logical alias, routing policy, pinned backend),
  - replay plan,
  - captured trace record,
  - comparison run summary,
  - report metadata,
  - environment snapshot.
- Ensure benchmark artifacts can record at least:
  - schema version,
  - timestamp,
  - git revision if available,
  - host/platform summary,
  - model alias,
  - selected policy or backend override,
  - scenario seed,
  - concurrency setting,
  - warmup settings,
  - route decision and fallback info per request,
  - timings, token counts, and error details per request.
- Keep the benchmark artifact as the source of truth, and treat markdown reports as derived views.
- Add tests for schema validation and serialization.

Acceptance criteria:
- The repo has clean typed contracts for Phase 3 benchmarking and replay.
- Artifact versioning is explicit.
- Existing code still compiles and tests pass.

Do not implement the full runner yet unless a tiny placeholder is needed to keep the code compiling.
```

---

## Prompt 3 - implement reproducible workload scenarios and generation

```text
Implement a workload-generation subsystem for synthetic benchmarking.

Requirements:
- Add a small, well-typed workload scenario library that can generate deterministic request sets from a seed.
- Support at minimum these scenario families:
  - short_chat,
  - long_prompt,
  - repeated_prefix,
  - concurrency_burst,
  - mixed.
- Keep prompts/templates lightweight and local-friendly. Do not require external datasets.
- Ensure generated workload items carry stable IDs and enough metadata for later comparison.
- Add a CLI entry point so a developer can generate workload manifests/artifacts to disk.
- Make sure the generated workloads can target a logical model alias while still being reusable for pinned-backend tests later.
- Add tests for determinism, schema validity, and edge cases.

Acceptance criteria:
- Given a seed, the same workload manifest is generated repeatedly.
- The workload generator supports multiple scenario families.
- A developer can generate a workload artifact from the CLI.
- Tests, lint, and type checks pass.

Keep this generator simple and honest. It is not a giant synthetic-data framework.
```

---

## Prompt 4 - build benchmark runner v1 on top of generated workloads

```text
Implement benchmark runner v1 using the generated workload scenarios.

Requirements:
- Extend the benchmark CLI so it can execute a workload manifest against:
  - a logical model alias,
  - a chosen routing policy,
  - or an explicitly pinned backend.
- Reuse the existing gateway/control-plane path where practical instead of inventing a separate benchmark-only execution path.
- Record per-request outcome data into the authoritative benchmark artifact.
- Capture at minimum:
  - request ID,
  - workload item ID,
  - route decision,
  - chosen backend,
  - fallback status,
  - total latency,
  - TTFT if available,
  - output tokens,
  - tokens per second,
  - error category if any.
- Include a configurable warmup phase before measured runs where appropriate.
- Keep defaults small enough to run comfortably on an M4 Pro.
- Add tests for artifact writing and a small end-to-end benchmark flow using mocks/fakes.

Acceptance criteria:
- A developer can generate a workload and execute it as a benchmark run from the CLI.
- The benchmark artifact contains rich per-request records and run metadata.
- Existing runtime paths still work.

Do not add trace replay yet beyond any tiny shared primitives that are clearly reusable.
```

---

## Prompt 5 - add safe, opt-in request trace capture

```text
Implement optional request trace capture in the gateway/control plane.

Requirements:
- Trace capture must be explicitly opt-in and off by default.
- Add clear capture modes, for example:
  - off,
  - metadata_only,
  - redacted_content,
  - full_content.
- The default safe mode should avoid storing raw prompt/response bodies unless explicitly requested.
- Do not store secrets, auth headers, or unrelated request metadata.
- Capture enough information for replay and analysis, including at least:
  - original request timestamp,
  - request ID,
  - logical alias,
  - policy or backend override if present,
  - route decision,
  - backend chosen,
  - stream flag,
  - timing summary,
  - error/fallback summary,
  - normalized request payload in the allowed capture mode.
- Use a simple local storage format such as JSONL or another repo-consistent format.
- Keep capture pluggable so later phases could redirect traces elsewhere.
- Add tests for capture modes, redaction behavior, and file writing.

Acceptance criteria:
- Switchyard can safely record replayable traces when enabled.
- Safe defaults are clear and documented.
- Trace capture does not break normal request handling.

Keep this focused. This is trace capture for benchmarking and replay, not a full analytics platform.
```

---

## Prompt 6 - implement trace replay engine v1

```text
Implement a trace replay engine that can replay captured requests through Switchyard.

Requirements:
- Add a replay CLI that can take captured traces and replay them against:
  - the same logical model alias,
  - a different routing policy,
  - or an explicitly pinned backend.
- Support at minimum these replay modes:
  - sequential,
  - fixed_concurrency,
  - preserve_order_without_original_timing.
- If original timestamps/inter-arrival data exist, structure the code so time-scaled replay can be added later without refactoring everything.
- Preserve traceability between original and replayed requests using correlation metadata.
- Record replay output using the same benchmark artifact story rather than inventing a parallel reporting system.
- Keep replay behavior backend-agnostic.
- Add tests for replay planning, ordering, and a small mock-backed replay run.

Acceptance criteria:
- A developer can replay captured traces against a policy or pinned backend.
- Replay runs produce comparable benchmark artifacts.
- The code stays clean for future time-based replay modes.

Do not attempt perfect SSE chunk-timing emulation. Preserve stream intent and the major request properties, but keep replay practical.
```

---

## Prompt 7 - add A/B comparative runs and diff summaries

```text
Implement comparative benchmark runs so the same workload or trace set can be evaluated under two execution targets.

Requirements:
- Add a comparison mode for at least these use cases:
  - policy A vs policy B,
  - pinned backend A vs pinned backend B,
  - logical alias routed normally vs alias pinned to a specific backend.
- Ensure both sides of the comparison use the same workload items or captured trace set.
- Add comparison summaries for at least:
  - request count,
  - success/error rate,
  - fallback rate,
  - p50/p95 latency,
  - TTFT if available,
  - tokens per second,
  - route distribution,
  - backend distribution,
  - notable per-scenario deltas.
- Keep the comparison output deterministic and easy to serialize.
- Reuse benchmark artifacts as inputs where possible instead of duplicating execution logic.
- Add tests for comparison-summary helpers and at least one end-to-end comparison flow using mocks/fakes.

Acceptance criteria:
- A developer can run the same workload or trace set against two targets and get a clear diff summary.
- Comparative output is both machine-readable and human-usable.
- Tests, lint, and type checks pass.

Keep the comparison framework modest. It should be useful and rigorous without becoming a research statistics package.
```

---

## Prompt 8 - make workload realism better: repeated-prefix and burst behavior

```text
Deepen workload realism by improving repeated-prefix and burst benchmarking.

Requirements:
- Extend workload generation and benchmark execution so repeated-prefix scenarios are first-class and measurable.
- Add a burst-oriented scenario that stresses queueing/routing behavior using short windows of higher concurrency.
- Ensure the benchmark artifact can summarize results by scenario family as well as overall totals.
- If a backend exposes cache-related hints or metrics, record them honestly; if not, represent that information as unavailable rather than making it up.
- Add a warmup or preconditioning path where helpful so repeated-prefix scenarios are not accidentally measuring only cold-start behavior.
- Keep the implementation portable across MLX-LM, vLLM-Metal, and future backends.
- Add tests for per-scenario aggregation and scenario-family reporting.

Acceptance criteria:
- Repeated-prefix and bursty workloads are easy to generate, run, and analyze.
- Benchmark outputs can be broken down by scenario family.
- The design stays backend-agnostic.

Do not turn this into a huge queuing simulator. Keep it concrete and useful.
```

---

## Prompt 9 - generate markdown reports from benchmark truth artifacts

```text
Implement report generation so Switchyard can produce human-readable benchmark summaries from authoritative artifacts.

Requirements:
- Add a report-generation command that takes one or more benchmark artifacts and emits a markdown report.
- The markdown report should include at minimum:
  - run metadata,
  - environment summary,
  - benchmark configuration,
  - scenario mix,
  - aggregate metrics,
  - per-scenario tables,
  - route/backend distributions,
  - fallback/error summary,
  - top takeaways or notable deltas for comparison runs.
- Keep the machine-readable artifact as the source of truth and avoid embedding logic only in the markdown layer.
- Make report generation work for both single benchmark runs and A/B comparison runs.
- Prefer compact, reviewable markdown over giant generated prose.
- Add tests for report rendering helpers.

Acceptance criteria:
- A developer can generate a readable markdown report from benchmark artifacts.
- The markdown output lines up with the machine-readable source data.
- Report generation is robust enough for local iteration.

Do not build a web dashboard here. Keep this as artifact-to-markdown reporting.
```

---

## Prompt 10 - docs and developer ergonomics for Phase 3 workflows

```text
Polish the docs and developer ergonomics for Phase 3.

Requirements:
- Update README.md with a clear Mac-first Phase 3 workflow.
- Add docs/benchmarking.md or equivalent that explains:
  - synthetic workload generation,
  - benchmark execution,
  - trace capture modes,
  - trace replay,
  - A/B comparison,
  - markdown report generation,
  - privacy/safety considerations for captured traces.
- Add example commands for:
  - generating a workload,
  - running a benchmark against a logical alias,
  - pinning a backend,
  - selecting a routing policy,
  - enabling safe trace capture,
  - replaying a trace,
  - running an A/B comparison,
  - generating a markdown report.
- Add example benchmark/replay configs that are realistic for an M4 Pro Mac.
- Update docs/architecture.md to explain where benchmark execution, trace capture, replay, and reporting fit into the system.
- Add at least one short ADR for a key Phase 3 decision, such as artifact-as-source-of-truth, trace capture modes, or replay-mode scope.

Acceptance criteria:
- A new contributor can run the Phase 3 workflow from the docs.
- The docs make privacy and reproducibility expectations obvious.
- All checks still pass.
```

---

## Prompt 11 - Phase 3 exit review

```text
Review the repo against AGENTS.md and the intended Phase 3 definition of done.

Tasks:
- Identify anything missing, weak, or too clever in the current Phase 3 implementation.
- Tighten tests, docs, config naming, artifact clarity, replay UX, and report wording where needed.
- Remove accidental overengineering.
- Verify that:
  - workload generation is deterministic,
  - benchmark artifacts are the authoritative truth source,
  - trace capture is safe by default,
  - trace replay is backend-agnostic,
  - A/B comparisons are clear and reproducible,
  - repeated-prefix and burst scenarios are actually usable,
  - the design still stays clean for future CUDA/cloud phases.
- Make code changes only where they clearly improve completeness or clarity.

Deliverables:
- a concise Phase 3 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 4 tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 3 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add reproducible workload generation, trace capture, replay, A/B comparison, and report generation,
2. any schema/versioning changes needed for benchmark artifacts and captured traces,
3. the storage format you recommend for traces and why,
4. the test strategy for CI without Apple GPU access,
5. the implementation order you would use,
6. the main privacy or safety risks in trace capture and how to mitigate them,
7. the benchmark plan for repeated-prefix, burst, and mixed workloads.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
