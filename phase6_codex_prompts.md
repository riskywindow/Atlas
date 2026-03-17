# Phase 6 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

Keep the same discipline as earlier phases: do not ask Codex to build all of Phase 6 in one shot. Push it through small, reviewable vertical slices.

Phase 6 is the **intelligent routing, simulation, and policy-learning** phase. The goal is to make Switchyard use its authoritative artifacts, traces, and topology information to make smarter routing decisions while staying transparent, testable, and safe. This phase should improve routing quality without turning the system into an opaque black box.

Core Phase 6 outcomes:
- authoritative artifacts capture enough route-decision context to support offline analysis, simulation, and policy comparison,
- request/workload feature extraction is explicit, typed, deterministic, and inspectable,
- repeated-prefix and cache/locality-aware signals exist and are usable by routing logic,
- historical performance summaries are queryable across aliases, instances, policies, and workload buckets,
- the routing system supports richer policy/scorer interfaces with explanations and shadow evaluation,
- an offline simulation / counterfactual-evaluation harness exists for comparing candidate policies,
- a transparent adaptive policy exists, such as a contextual bandit or similarly simple adaptive scorer,
- online use of adaptive policy logic is guarded by shadow mode, canaries, kill switches, and minimum-confidence rules,
- policy recommendation reports can be generated from historical artifacts,
- docs and runbooks explain trust boundaries, limitations, and how Phase 6 prepares for later cloud and Forge work.

Recommended scenario families to exercise in Phase 6:
- short versus long prompts,
- repeated-prefix multi-turn chats,
- bursty versus steady tenants,
- one backend being faster but less stable than another,
- worker warm versus cold states,
- route decisions that should prefer session affinity,
- canary and shadow evaluation of new policies,
- replaying older traces against newer routing logic,
- topology-aware comparisons after instance inventory changes.

Non-goals for Phase 6:
- no opaque deep-learning router in the critical path,
- no LLM in the routing critical path,
- no Forge or kernel/code-generation work yet,
- no cloud autoscaler or full GPU-market scheduler yet,
- no pretending counterfactual simulation is exact when the evidence is weak,
- no irreversible self-modifying control-plane behavior,
- no removal of explainability from route decisions,
- no requirement for large training datasets or rented GPUs,
- no giant data platform if small, typed, local-first components are enough,
- no breaking earlier routing modes unless a compatibility path is clearly provided.

A good theme for this phase: **make routing smarter, but keep every decision explainable and falsifiable**.

---

## Prompt 0 - bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 6. The old current-phase text in AGENTS.md can be updated as part of this work. Switchyard is now a Mac-first, backend-agnostic inference fabric with explicit worker topology, deployment-aware benchmarking/replay, and host-native Apple-Silicon workers behind a network-addressable worker protocol. For Phase 6, the major additions are: richer route-decision artifacts, deterministic request/workload feature extraction, repeated-prefix and cache/locality-aware routing signals, historical performance summaries, a policy/scorer interface with explanations and shadow scoring, an offline simulation harness for comparing routing policies, and a transparent adaptive policy with strong guardrails. Keep the design portable to later vLLM-CUDA and cloud GPU workers, keep tests CI-friendly without Apple GPU access, avoid opaque routing logic, and ship in small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 - Phase 6 kickoff and repo audit

```text
Inspect the current repo and prepare it for Phase 6.

Requirements:
- Review the codebase against the intended Phase 6 outcomes.
- Update AGENTS.md so the project phase is now Phase 6 instead of Phase 5.
- Add or update docs/phase6.md with:
  - Phase 6 goals,
  - definition of done,
  - non-goals,
  - the rule that routing intelligence must stay explainable,
  - the rule that authoritative artifacts remain the source of truth,
  - the distinction between offline simulation evidence and online runtime truth,
  - the rule that adaptive routing must be safe-by-default and reversible,
  - the rule that Phase 6 should remain compatible with later cloud workers and Forge-style optimization.
- Identify any tiny Phase 5 gaps that obviously block Phase 6 and patch only the smallest necessary blockers.
- Do not do a giant refactor.

Acceptance criteria:
- AGENTS.md and docs reflect Phase 6 accurately.
- The repo has a crisp Phase 6 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a rebuild.
```

---

## Prompt 2 - extend authoritative artifacts for route intelligence

```text
Extend Switchyard's authoritative artifacts and route-decision records so they can support offline analysis and simulation.

Requirements:
- Audit the current request, routing, benchmark, replay, and topology artifact schemas.
- Extend them in a typed and versioned way to capture at minimum:
  - deterministic request/workload feature snapshots,
  - candidate backends or instances considered by the router,
  - chosen route and explicit explanation or reason codes,
  - shadow-policy scores or decisions when present,
  - session-affinity context when relevant,
  - prefix/locality identifiers or summaries in a privacy-conscious form,
  - observed runtime outcomes such as queue delay, TTFT, latency, output tokens, error class, and backend instance used,
  - topology snapshot references so decisions can be tied to a concrete deployment state,
  - policy ID and policy-version metadata.
- Preserve backward compatibility where practical. If an artifact/schema version bump is required, make it explicit and documented.
- Add serialization/deserialization tests and migration coverage where needed.
- Keep artifact truthfulness high. If a field was not known at decision time, it must be distinguishable from fields learned after execution.

Acceptance criteria:
- Route-intelligence artifacts are rich enough to support later simulation and recommendation work.
- Schema changes are typed, documented, and tested.
- Existing tests still pass and artifact truth boundaries are explicit.

Do not implement adaptive policies yet. This prompt is about evidence capture and schema quality.
```

---

## Prompt 3 - add deterministic request feature extraction and workload tagging

```text
Implement a deterministic request-feature and workload-tagging layer for routing.

Requirements:
- Add a typed feature-extraction module that derives routing-relevant features from incoming requests and runtime context without calling an LLM.
- Features should stay backend-agnostic and may include, where appropriate:
  - input-length buckets,
  - estimated token counts,
  - message count / history depth,
  - session or conversation continuity markers,
  - streaming flag,
  - tenant or service-class markers if the repo already supports them,
  - request priority or canary/shadow status,
  - session-affinity eligibility,
  - repeated-prefix fingerprints or digests in a privacy-conscious form,
  - any other small, deterministic features that clearly help routing.
- Introduce typed workload tags or workload classes that are explainable and useful, such as short_chat, long_context, repeated_prefix, burst_candidate, or similar.
- Ensure the route context, runtime inspection surfaces, and authoritative artifacts can expose these features cleanly.
- Add tests for determinism, schema stability, privacy-conscious prefix fingerprinting, and representative workload classifications.

Acceptance criteria:
- Switchyard has an explicit, deterministic request-feature extraction path.
- Workload tags are explainable, stable, and visible in artifacts.
- Tests pass.

Do not make the router adaptive yet. This prompt is about feature groundwork and inspectability.
```

---

## Prompt 4 - add prefix-locality and cache-aware signals

```text
Add repeated-prefix and cache/locality-aware routing signals in a conservative, explainable way.

Requirements:
- Build a small, bounded component that tracks prefix-locality or repeated-prefix signals across recent requests.
- The design should be explicit about scope and retention. Use TTLs, bounded memory, and privacy-conscious digests rather than storing raw prompts.
- Support at minimum:
  - repeated-prefix detection,
  - per-alias or per-instance locality summaries,
  - cache-opportunity signals or prefix-hotness summaries,
  - visibility into whether a request likely benefits from staying near a warm backend instance.
- Ensure these signals can be included in route context, runtime diagnostics, and authoritative artifacts.
- Add tests for TTL behavior, bounded-state behavior, prefix collision handling assumptions, and determinism.
- If the repo already has session affinity, integrate carefully so prefix-locality and session-affinity signals do not fight each other silently.

Acceptance criteria:
- Switchyard has explicit prefix/locality-aware signals that later policies can use.
- The implementation is bounded, privacy-conscious, and well-tested.
- Artifacts and diagnostics can show the signals.

Do not yet change the main routing policy beyond exposing these signals where clearly useful.
```

---

## Prompt 5 - build historical performance summaries and simple predictors

```text
Build the first historical-performance layer that can support policy comparison and adaptive routing.

Requirements:
- Add a typed aggregation path over authoritative artifacts so Switchyard can summarize historical behavior by dimensions such as:
  - alias,
  - backend type,
  - backend instance,
  - policy,
  - workload tag or feature bucket,
  - warm/cold or locality-related context where available,
  - tenant or service class if already supported.
- Support transparent summary statistics, such as counts, error rates, EWMA values, percentiles if practical, and simple bucketed performance summaries for TTFT, total latency, throughput, and queue delay.
- Introduce a predictor or estimator interface that can produce simple expected-outcome estimates for candidate routes using historical evidence.
- Keep the predictor transparent. Prefer simple statistical estimators or lightweight regression over anything opaque or data-hungry.
- Make data sufficiency explicit. If there is not enough evidence for a prediction, the system should say so.
- Add tests for summary computation, bucket behavior, sparse-data handling, and predictor outputs.

Acceptance criteria:
- Switchyard can summarize historical routing outcomes in a queryable, typed way.
- A transparent predictor interface exists for later use by simulation and adaptive policies.
- Tests pass.

Do not add live adaptive routing yet. This prompt is about building honest historical evidence and simple estimators.
```

---

## Prompt 6 - midpoint bootstrap instruction
Paste this first in a **new** Codex session if context compaction becomes a problem around the middle of Phase 6.

```text
Read AGENTS.md first and follow it. The repo is in Phase 6 of Switchyard. Earlier phases already delivered: a Mac-first control plane, explicit worker topology, host-native Apple-Silicon workers behind a network-addressable worker protocol, deployment-aware benchmarking and replay, runtime inspection, canaries, shadow traffic, session affinity, and artifact-based evaluation. The first half of Phase 6 is expected to establish: richer route-intelligence artifacts, deterministic request-feature extraction, repeated-prefix/locality-aware signals, and historical performance summaries with simple predictor interfaces. Continue from there without redoing earlier phases. The remaining major work is: richer policy/scorer interfaces with explanations, offline simulation and counterfactual policy comparison, a transparent adaptive policy with strict guardrails, safe rollout controls, policy recommendation reporting, and final docs/runbooks. Keep the design explainable, CI-friendly without Apple GPU access, and portable to later cloud workers and Forge integration. For every task: inspect the current repo state, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 7 - upgrade the policy interface into a scorer/explanation framework

```text
Refine the routing-policy layer so Phase 6 intelligence can plug in cleanly.

Requirements:
- Audit the current router and policy abstractions.
- Evolve them into a richer policy/scorer framework that can support at minimum:
  - candidate enumeration,
  - per-candidate scoring,
  - explicit reason codes or explanations,
  - shadow scoring without affecting the chosen route,
  - policy IDs and versions,
  - safe compatibility for older fixed policies.
- Preserve or wrap existing routing modes so older policy behavior still works through the new abstraction where practical.
- Ensure score/explanation outputs can flow into runtime diagnostics and authoritative artifacts.
- Keep the framework easy to test and inspect. Avoid making the routing core overly abstract or clever.
- Add tests for compatibility, candidate scoring, reason-code emission, and shadow-policy behavior.

Acceptance criteria:
- Switchyard has a policy/scorer abstraction suitable for heuristic, predictive, and adaptive policies.
- Existing policy modes remain available through a clean compatibility path.
- Tests pass.

Do not implement the adaptive policy itself yet unless a tiny placeholder is needed.
```

---

## Prompt 8 - add an offline simulation and counterfactual policy-evaluation harness

```text
Build an offline simulation harness so candidate routing policies can be compared using authoritative artifacts.

Requirements:
- Design and implement a simulation/evaluation path that can load historical trace, replay, and benchmark artifacts and evaluate one or more candidate policies against them.
- Make the design honest about evidence quality. A simulation result must distinguish between:
  - directly observed outcomes,
  - predictor-based estimates,
  - unsupported or low-confidence cases.
- Support at minimum:
  - replaying workload features through candidate policies,
  - comparing fixed heuristic policies and future adaptive policies,
  - producing aggregate metrics by workload bucket, alias, instance, and tenant if relevant,
  - surfacing sample-size and confidence limitations,
  - emitting authoritative comparison artifacts and a human-readable report.
- Integrate with the Phase 5 topology-aware world so the simulation can reason about instance inventory and deployment context when the artifacts provide that information.
- Add tests for artifact loading, comparison logic, low-confidence cases, and report generation.

Acceptance criteria:
- Switchyard can compare candidate routing policies offline using historical evidence.
- The simulation/reporting path is explicit about uncertainty and unsupported cases.
- Tests pass.

Do not pretend this is perfect counterfactual truth. Keep the limitations visible in both code and reports.
```

---

## Prompt 9 - implement a transparent adaptive policy v1

```text
Implement the first transparent adaptive routing policy for Switchyard.

Requirements:
- Choose a simple, inspectable adaptive approach that fits the existing codebase, such as:
  - a contextual bandit,
  - a bucketed multi-armed bandit,
  - a policy-of-policies selector,
  - or another lightweight adaptive method that is clearly explainable.
- The adaptive policy must support at minimum:
  - a safe default or fallback heuristic,
  - minimum-sample thresholds,
  - confidence-aware abstention,
  - bounded or disable-able exploration,
  - alias-level or tenant-level scoping where practical,
  - reason codes that explain why a route was chosen or why the policy abstained.
- Prefer using the historical summary/predictor layer rather than inventing a separate opaque data path.
- Keep all state explicit and testable. If persistence is needed, make it typed and documented.
- Add strong tests for cold start, sparse data, unstable backends, abstention, exploration off, and deterministic evaluation modes.

Acceptance criteria:
- Switchyard has a transparent adaptive policy that can operate safely or abstain when evidence is weak.
- The design remains explainable and bounded.
- Tests pass.

Do not auto-promote this policy into the main live path by default. Safe rollout comes next.
```

---

## Prompt 10 - add safe rollout controls for intelligent and adaptive routing

```text
Integrate the new scoring and adaptive-routing capabilities into safe rollout controls.

Requirements:
- Add configuration, admin, or CLI controls so policies can run in modes such as:
  - disabled,
  - shadow-only,
  - report-only,
  - canary,
  - active with guardrails.
- Reuse and extend earlier shadow/canary machinery where appropriate instead of building a separate rollout system.
- Add explicit kill switches, freeze-learning controls, policy-state reset, and state export/import if stateful adaptive logic is used.
- Ensure runtime diagnostics can show:
  - active policy,
  - shadow policy,
  - recent policy decisions and abstentions,
  - exploration status,
  - last policy update or learning event,
  - obvious guardrail triggers.
- Add tests for mode transitions, kill-switch behavior, learning freeze, state reset, and compatibility with existing canary/shadow paths.

Acceptance criteria:
- Switchyard can evaluate and roll out intelligent/adaptive policies safely.
- Operators can inspect and disable the policy cleanly.
- Tests pass.

Do not build a giant policy-management platform. Keep the controls practical and local-first.
```

---

## Prompt 11 - build policy recommendation reports from historical evidence

```text
Add a policy recommendation layer that turns artifacts and simulation results into human-readable guidance.

Requirements:
- Build a CLI, report generator, or small admin path that can analyze recent authoritative artifacts and simulation outputs and recommend routing policy choices.
- Recommendations should be scoped and evidence-based. For example, they may recommend:
  - a preferred policy for a given alias,
  - a preferred policy for a workload class,
  - where the adaptive policy should remain shadow-only,
  - which instances/backends are strong or weak for repeated-prefix traffic,
  - where confidence is too low to recommend a change.
- Reports should include at minimum:
  - evidence windows,
  - sample sizes,
  - workload buckets,
  - recommended policy or no-change recommendation,
  - caveats and confidence notes,
  - notable regressions or counterexamples.
- Ensure recommendation output is derived from authoritative artifacts and simulation results rather than ad hoc runtime logs.
- Add tests for report generation, sparse-data handling, and no-recommendation cases.

Acceptance criteria:
- Switchyard can generate evidence-based routing-policy recommendations.
- Reports remain honest about uncertainty and sample size.
- Tests pass.

Do not let the system auto-apply recommendations yet. This prompt is about decision support, not autonomous promotion.
```

---

## Prompt 12 - docs, architecture updates, and ADRs for intelligent routing

```text
Polish the docs and developer ergonomics for Phase 6.

Requirements:
- Update README.md with a clear summary of what Phase 6 adds.
- Add docs/intelligent-routing.md or equivalent that explains:
  - request/workload feature extraction,
  - prefix/locality-aware signals,
  - historical performance summaries,
  - the scorer/policy abstraction,
  - offline simulation and its limitations,
  - the adaptive policy design and guardrails,
  - safe rollout modes,
  - policy recommendation reports,
  - how Phase 6 prepares for later cloud expansion and Forge work.
- Update docs/architecture.md so the route-decision path, evidence path, and feedback loop are explicit.
- Add at least one ADR for a key Phase 6 decision, such as:
  - why the adaptive policy is intentionally transparent and conservative,
  - why simulation must distinguish observed versus estimated outcomes,
  - or why raw prompt text is not stored for locality signals.
- Add practical example commands for:
  - running offline policy comparison,
  - enabling shadow scoring,
  - enabling or disabling adaptive policy canary mode,
  - generating a policy recommendation report,
  - inspecting runtime policy state.

Acceptance criteria:
- A new contributor can understand and operate the Phase 6 routing-intelligence stack from the docs.
- Trust boundaries and limitations are visible, not hidden.
- All checks still pass.
```

---

## Prompt 13 - Phase 6 exit review

```text
Review the repo against AGENTS.md and the intended Phase 6 definition of done.

Tasks:
- Identify anything missing, weak, too clever, or too opaque in the current Phase 6 implementation.
- Tighten tests, docs, config naming, artifact clarity, simulation honesty, and rollout ergonomics where needed.
- Remove accidental overengineering.
- Verify that:
  - route-intelligence artifacts are rich and versioned enough for offline analysis,
  - request/workload feature extraction is deterministic and visible,
  - prefix/locality-aware signals are bounded and privacy-conscious,
  - historical performance summaries and predictors are queryable and honest about data sufficiency,
  - the policy/scorer abstraction supports explanations and shadow evaluation,
  - offline simulation can compare policies while surfacing uncertainty,
  - the adaptive policy is transparent, guarded, and disable-able,
  - rollout controls are practical and safe,
  - policy recommendation reports are evidence-based,
  - the design remains portable to later cloud workers and later Forge integration.
- Make code changes only where they clearly improve completeness or clarity.

Deliverables:
- a concise Phase 6 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 7 tasks,
- the top 5 recommended cloud-extension tasks,
- the top 5 recommended Forge-integration tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 6 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add richer route-intelligence artifacts, deterministic request-feature extraction, prefix/locality-aware signals, historical performance summaries, a scorer/explanation framework, offline policy simulation, and a transparent adaptive policy with rollout guardrails,
2. the schema and artifact changes you recommend,
3. the simplest predictor and adaptive-policy design you recommend and why,
4. the implementation order you would use,
5. the test strategy for CI without Apple GPU access,
6. the main risks in offline policy comparison and adaptive rollout and how you would mitigate them,
7. how you would keep the design explainable and portable to later cloud workers and Forge work.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
