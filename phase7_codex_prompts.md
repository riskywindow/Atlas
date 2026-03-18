# Phase 7 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

Keep the same discipline as earlier phases: do not ask Codex to build all of Phase 7 in one shot. Push it through small, reviewable vertical slices.

Phase 7 is the **hybrid local/remote execution and cloud-ready extension** phase. The goal is to make Switchyard treat remote workers as first-class execution targets while preserving the Mac-first developer experience, the authoritative-artifact model, and the explainable routing stack built in Phase 6. Most of this phase should remain testable **without** renting a cloud GPU yet, using mock remote workers, fake latency/error injectors, and CI-friendly contract tests. Real CUDA-backed workers can be plugged in later without changing the control-plane contract.

Core Phase 7 outcomes:
- remote workers are first-class in topology, capability, cost, and health models,
- the control plane can talk to local and remote workers through explicit, typed, network-aware contracts,
- remote worker registration, heartbeats, readiness, and de-registration are secure and observable,
- routing supports hybrid policies such as local-preferred, burst-to-remote, latency-SLO, quality-on-demand, and remote-disabled,
- admission control and budgeting can constrain cloud spillover safely,
- benchmark, replay, and policy-comparison flows can evaluate local-only, remote-only, and hybrid routing strategies,
- observability and runtime inspection show where traffic went, why, and what it cost,
- deployment and packaging paths exist for Linux/NVIDIA remote workers while preserving Mac-first local development,
- the architecture remains backend-agnostic and ready for later vLLM-CUDA workers,
- Phase 7 leaves clean hooks for later Forge Stage A autotuning without doing kernel-generation work yet.

Recommended scenario families to exercise in Phase 7:
- local-only traffic with remote workers unavailable,
- local-preferred routing with overflow to remote during burst load,
- latency-sensitive traffic that should prefer remote under some conditions,
- repeated-prefix or session-affine traffic that should stay local unless guardrails force spillover,
- remote worker cold-start and warm-state comparisons,
- remote network degradation, timeout, or partial-failure scenarios,
- budget exhaustion or remote-disable scenarios,
- canarying a remote backend before routing real user traffic to it,
- replaying historical traces through local-only versus hybrid policies,
- topology changes where remote inventory appears or disappears during runtime.

Non-goals for Phase 7:
- no full cloud autoscaler or giant cluster-management platform yet,
- no hard requirement to rent GPUs during initial implementation,
- no region/spot-market scheduler that pretends to solve cloud economics completely,
- no Forge kernel/operator generation yet,
- no opaque cloud policy layer that bypasses Phase 6 explainability,
- no secrets baked into the repo or test suite,
- no replacement of local-first development with cloud-first workflows,
- no breaking the Apple-Silicon path just to make remote workers work,
- no giant frontend before the control-plane, benchmark, and operator surfaces are solid,
- no pretending benchmark/replay estimates are the same as real cloud runtime truth.

A good theme for this phase: **make Switchyard hybrid, but keep it measurable, portable, and honest**.

---

## Prompt 0 - bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 7. The old current-phase text in AGENTS.md can be updated as part of this work. Switchyard already has: a Mac-first control plane, explicit worker topology, host-native Apple-Silicon workers behind a network-addressable worker protocol, benchmark/replay artifacts, runtime inspection, admission control, canaries/shadowing, session affinity, circuit-breaker-style behavior, and Phase 6 explainable routing with richer route artifacts, deterministic request features, locality-aware signals, historical summaries, offline simulation, and guarded adaptive policies. Phase 7 adds hybrid local/remote execution and cloud-ready worker support. The major outcomes are: remote workers as first-class topology members, secure worker registration and lifecycle, hybrid routing policies and spillover guardrails, remote-aware benchmark/replay/reporting, operator surfaces for budgets and remote health, and deployment/package scaffolding for later Linux/NVIDIA vLLM-CUDA workers. Keep the design testable without renting GPUs yet, preserve the Mac-first local path, keep the architecture backend-agnostic, and ship in small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 - Phase 7 kickoff and repo alignment

```text
Inspect the current repo and prepare it for Phase 7.

Requirements:
- Review the codebase against the intended Phase 7 outcomes.
- Update AGENTS.md so the project phase is now Phase 7 instead of Phase 6.
- Add or update docs/phase7.md with:
  - Phase 7 goals,
  - definition of done,
  - non-goals,
  - the rule that local-first development remains the default,
  - the rule that remote/cloud support must be testable without real rented GPUs,
  - the rule that topology/cost/health truth must come from typed contracts and authoritative artifacts,
  - the rule that hybrid routing must remain explainable,
  - the rule that Phase 7 prepares for later vLLM-CUDA workers and later Forge Stage A autotuning.
- Identify any tiny Phase 6 gaps that obviously block Phase 7 and patch only the smallest necessary blockers.
- Do not do a giant refactor.

Acceptance criteria:
- AGENTS.md and docs reflect Phase 7 accurately.
- The repo has a crisp Phase 7 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a rebuild.
```

---

## Prompt 2 - extend topology and capability models for remote/cloud workers

```text
Extend Switchyard's typed topology, worker, and capability models so remote/cloud workers are first-class citizens.

Requirements:
- Audit the current topology, worker registry, instance inventory, capability store, and runtime-inspection schemas.
- Extend them in a typed and versioned way to support remote workers and later CUDA-backed execution.
- Add fields or structures as appropriate for concepts such as:
  - worker locality class (local_host, local_network, remote_cloud, etc.),
  - device class (apple_gpu, cpu, nvidia_gpu, amd_gpu, remote_unknown, etc.),
  - region/zone/provider tags where relevant,
  - control-plane reachability and transport metadata,
  - cost/budget profile metadata,
  - cold versus warm readiness hints,
  - trust/auth state,
  - expected network characteristics,
  - execution-mode labels such as host_native, remote_worker, or external_service.
- Preserve backward compatibility where practical. If a schema version bump is required, make it explicit and documented.
- Ensure these model changes flow into authoritative artifacts, runtime diagnostics, and topology snapshots.
- Add tests for serialization, compatibility, and sensible defaults.

Acceptance criteria:
- Remote/cloud workers can be described cleanly without special-case hacks.
- Topology snapshots and artifacts can represent hybrid deployments explicitly.
- Tests pass.

Do not implement real remote execution yet. This prompt is about typed models and future-proof topology truth.
```

---

## Prompt 3 - add a generic remote-worker transport path and contract-tested mock implementation

```text
Implement the first generic remote-worker transport path for Switchyard in a way that is testable without GPUs.

Requirements:
- Audit the existing worker protocol and transport assumptions.
- Add or refine a generic remote-worker transport/client path that can communicate with a network-addressable worker over explicit typed APIs.
- Support at minimum:
  - health/readiness checks,
  - capability discovery,
  - request execution and streaming where applicable,
  - cancellation or timeout propagation if the existing protocol supports it,
  - transport-level error classification,
  - metadata needed for tracing and request correlation.
- Create a mock or fake remote-worker server/runtime for CI and local development so this path can be exercised without Apple or cloud GPUs.
- Add contract tests that verify interoperability between the control plane and the mock remote worker.
- Keep this transport generic enough that a later Linux/NVIDIA vLLM-CUDA worker can sit behind it without changing the control plane.

Acceptance criteria:
- Switchyard can talk to a remote worker over an explicit, typed, testable transport.
- CI-friendly mock remote-worker coverage exists.
- Tests pass.

Do not add real cloud deployment yet. This prompt is about the remote execution contract and fake-but-realistic validation.
```

---

## Prompt 4 - midpoint bootstrap instruction for a new session
Paste this first in a **new** Codex session if context compaction becomes a problem after the first few Phase 7 tasks.

```text
Read AGENTS.md first and follow it. The repo is in Phase 7 of Switchyard. Earlier phases already delivered: Mac-first local workers, network-addressable worker protocol, benchmark/replay artifacts, admission control, runtime inspection, explainable routing, simulation, and guarded adaptive-policy support. The early Phase 7 work is expected to establish: Phase 7 docs/definition of done, remote/cloud-aware typed topology and capability models, and a generic remote-worker transport path with a CI-friendly mock implementation. Continue from there without redoing earlier phases. The remaining major work is: secure worker registration/lifecycle, hybrid routing policies and spillover guardrails, remote-aware benchmarking/replay, operator surfaces for budgets and remote health, deployment/package scaffolding for Linux/NVIDIA workers, and final docs/runbooks. Keep the design backend-agnostic, testable without rented GPUs, Mac-first for local dev, and portable to later vLLM-CUDA workers and Forge Stage A autotuning. For every task: inspect the current repo state, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 5 - implement secure remote-worker registration, heartbeats, and lifecycle state

```text
Implement the control-plane pieces needed for remote-worker registration and lifecycle management.

Requirements:
- Extend the worker-registration path so remote workers can register with explicit typed identity, capabilities, transport metadata, and lifecycle state.
- Add or refine support for:
  - authenticated registration or signed enrollment tokens,
  - heartbeat and lease/TTL behavior,
  - readiness versus liveness distinction,
  - graceful de-registration,
  - stale-worker eviction,
  - explicit lifecycle states such as registering, warming, ready, draining, unhealthy, lost, or retired.
- Keep the auth story practical for a local-first repo. It can be simple but must not be hand-wavy.
- Make lifecycle events visible in runtime diagnostics and authoritative artifacts where appropriate.
- Add tests for registration, heartbeat timeouts, stale-worker cleanup, invalid token or auth failures, and drain behavior.

Acceptance criteria:
- Remote workers can register and maintain an explicit lifecycle in the control plane.
- Lifecycle/auth behavior is visible and tested.
- Tests pass.

Do not build a giant service-mesh or PKI system. Keep the security model practical and explicit.
```

---

## Prompt 6 - add hybrid routing modes and remote-aware candidate scoring

```text
Extend the routing stack so Switchyard can make explainable local-versus-remote decisions.

Requirements:
- Audit the current policy/scorer framework and integrate remote-aware candidate scoring into it.
- Add or refine hybrid routing modes such as:
  - local_preferred,
  - burst_to_remote,
  - latency_slo,
  - quality_on_demand,
  - remote_disabled,
  - remote_preferred_if_local_unhealthy.
- Candidate scoring should be able to consider, where supported by current artifacts and topology:
  - predicted latency and queue delay,
  - session affinity,
  - prefix/locality signals,
  - worker health and warm/cold state,
  - network penalty or remoteness,
  - cost/budget pressure,
  - tenant/service-class rules,
  - confidence or evidence sufficiency.
- Keep routing decisions explainable with explicit reason codes and abstention/fallback behavior.
- Preserve compatibility with earlier fixed policies.
- Add tests for local/remote scoring, fallback behavior, sparse-data cases, and reason-code emission.

Acceptance criteria:
- Switchyard can score local and remote candidates through the existing explainable policy framework.
- Hybrid routing modes exist and are testable.
- Tests pass.

Do not implement cloud autoscaling yet. This prompt is about explainable candidate selection in a hybrid topology.
```

---

## Prompt 7 - integrate spillover guardrails, budgets, and remote-admission limits

```text
Add the operational guardrails that make burst-to-remote routing safe.

Requirements:
- Integrate remote-aware controls into admission control, bounded queueing, and policy rollout paths.
- Add support for concepts such as:
  - remote request budgets or spend buckets,
  - per-tenant remote spillover permissions,
  - concurrency caps for remote traffic,
  - budget exhaustion behavior,
  - remote-only disable/kill switches,
  - escalation paths for high-priority requests,
  - optional backoff or cooldown after remote instability.
- Ensure the system can distinguish between:
  - local admission failure with eligible remote spillover,
  - local-only tenants or requests,
  - remote budget exhaustion,
  - remote health failures.
- Make operator-visible reason codes and metrics for these decisions.
- Add tests for budget exhaustion, tenant restrictions, kill switches, cooldown behavior, and spillover eligibility.

Acceptance criteria:
- Remote spillover is bounded and policy-driven rather than implicit.
- Operators can see why a request stayed local, spilled remote, or was rejected.
- Tests pass.

Do not invent a full cloud billing engine. Keep the budget model simple, typed, and operationally useful.
```

---

## Prompt 8 - midpoint bootstrap instruction for another new session
Paste this first in a **new** Codex session if context compaction becomes a problem around the middle of Phase 7.

```text
Read AGENTS.md first and follow it. The repo is in Phase 7 of Switchyard. Earlier phases already delivered: Mac-first local workers, network-addressable worker protocol, benchmark/replay artifacts, admission control, explainable routing, simulation, and adaptive-policy guardrails. The current Phase 7 work is expected to already include: Phase 7 docs/definition of done, remote/cloud-aware typed topology and capability models, a generic remote-worker transport path with a mock implementation, secure-ish worker registration/lifecycle, hybrid routing modes, and remote spillover guardrails such as budgets and remote-admission limits. Continue from there without redoing earlier phases. The remaining major work is: remote-aware benchmark/replay/reporting, operator surfaces for remote health and budgets, deployment/package scaffolding for Linux/NVIDIA workers, optimization-ready config surfaces for later Forge Stage A, and final docs/runbooks. Keep the design explainable, testable without rented GPUs, and portable to later vLLM-CUDA workers. For every task: inspect the current repo state, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 9 - make benchmark, replay, and policy comparison remote-aware

```text
Extend benchmark/replay/reporting flows so they can evaluate local-only, remote-only, and hybrid behavior honestly.

Requirements:
- Audit current workload generation, replay artifacts, benchmark summaries, and policy-comparison tooling.
- Extend them so a run can capture and compare at minimum:
  - local-only policy behavior,
  - hybrid burst-to-remote behavior,
  - remote-disabled or budget-exhausted behavior,
  - remote cold versus warm paths,
  - simulated or observed network penalties,
  - cost/budget outcomes where modeled.
- Ensure authoritative artifacts distinguish:
  - observed runtime outcomes,
  - injected/mock remote conditions,
  - predictor-based estimates,
  - unsupported or low-confidence comparisons.
- Add scenario helpers or canned workloads that stress remote spillover and hybrid routing.
- Update report generation so a human can see where hybrid routing helped, hurt, or was unsupported by the evidence.
- Add tests for artifact correctness, comparison logic, and honest uncertainty handling.

Acceptance criteria:
- Switchyard can benchmark and replay hybrid-local/remote scenarios without pretending mock evidence is production truth.
- Reports clearly explain when remote routing was beneficial, harmful, or inconclusive.
- Tests pass.
```

---

## Prompt 10 - add operator surfaces for remote health, budgets, and placement decisions

```text
Improve runtime inspection and operator controls for the hybrid topology.

Requirements:
- Extend the existing admin/inspection/runtime-diagnostics surfaces so operators can inspect at minimum:
  - currently registered local and remote workers,
  - lifecycle state and last heartbeat,
  - worker locality/device/capability metadata,
  - budget/spillover state,
  - recent remote-routing decisions and reason codes,
  - recent local-versus-remote placement distribution,
  - remote transport errors and cooldown state,
  - which policies are eligible to use remote workers.
- Add practical operator controls for:
  - remote disable/enable,
  - budget resets or budget-window inspection,
  - draining a remote worker,
  - marking a worker quarantined or canary-only,
  - inspecting recent route examples.
- Reuse earlier inspection/reporting patterns instead of building an unrelated operator plane.
- Add tests for the most important admin and inspection paths.

Acceptance criteria:
- A human operator can understand the hybrid topology and why requests are going remote.
- The most important controls and diagnostics are visible and testable.
- Tests pass.

Do not build a giant web console unless the repo already has a light admin surface that can absorb this cleanly.
```

---

## Prompt 11 - package the first cloud-ready worker path and deployment scaffolding

```text
Add deployment/package scaffolding so Switchyard is ready for a later real cloud-GPU worker without requiring one today.

Requirements:
- Audit current worker packaging and deployment assumptions.
- Add or refine a packaging path for a remote Linux worker runtime that can later host CUDA-backed inference engines.
- At minimum, provide:
  - a container image or clear packaging target for a generic remote worker runtime,
  - environment/config contracts for later vLLM-CUDA integration,
  - example deployment manifests or templates for a remote worker service,
  - docs for how the control plane discovers and authenticates remote workers,
  - a mock or stub deployment path that can be exercised in CI or local integration tests without a GPU.
- If the repo structure makes sense for it, add a placeholder or feature-flagged worker backend for a future vLLM-CUDA worker, but do not require CUDA to run tests.
- Keep the packaging/backend split clean so the control plane does not become NVIDIA-specific.

Acceptance criteria:
- The repo contains a credible, reviewable path toward Linux/NVIDIA remote workers.
- Local development and CI remain functional without cloud GPUs.
- Tests and docs pass.

Do not attempt to solve full cloud provisioning in this prompt. This is about packaging and integration boundaries.
```

---

## Prompt 12 - midpoint bootstrap instruction for a later new session
Paste this first in a **new** Codex session if context compaction becomes a problem in the later part of Phase 7.

```text
Read AGENTS.md first and follow it. The repo is in Phase 7 of Switchyard. Earlier phases already delivered: Mac-first local workers, network-addressable worker protocol, benchmark/replay artifacts, explainable routing, and adaptive-policy guardrails. Phase 7 work is expected to already include: remote/cloud-aware topology models, a generic remote-worker transport with a mock implementation, remote-worker registration/lifecycle, hybrid routing modes, spillover budgets/guardrails, remote-aware benchmark/replay/reporting, operator inspection for remote health and budgets, and a cloud-ready worker packaging/deployment path that does not require rented GPUs yet. Continue from there without redoing earlier phases. The remaining major work is: exposing optimization-ready config/knob surfaces for later Forge Stage A, polishing docs/architecture/ADRs/runbooks, and a final exit review with remaining gaps and next-phase recommendations. Keep the design Mac-first, backend-agnostic, testable in CI, and ready for later vLLM-CUDA workers. For every task: inspect the current repo state, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 13 - expose optimization-ready worker and routing config surfaces for later Forge Stage A

```text
Prepare Switchyard for later Forge Stage A autotuning without doing any kernel or code generation yet.

Requirements:
- Audit which worker-launch, serving, routing, and scheduling knobs are currently implicit or scattered.
- Introduce a typed, inspectable config surface for benchmark-relevant knobs that later autotuning can search over safely, such as:
  - concurrency caps,
  - batching-related knobs if the current architecture exposes them,
  - queue thresholds,
  - routing thresholds or policy parameters,
  - remote-spillover thresholds,
  - cooldown durations,
  - canary percentages,
  - worker feature flags or launch presets.
- Ensure these settings can be captured in authoritative benchmark/replay/report artifacts.
- Add a notion of immutable run configuration or config fingerprinting so future optimization loops can compare runs honestly.
- Keep the config surface bounded and explicit; do not build a giant search platform here.
- Add tests for config parsing, fingerprinting, artifact capture, and compatibility.

Acceptance criteria:
- Switchyard has an optimization-ready config surface that later Forge Stage A can consume.
- Benchmark/replay artifacts capture enough configuration truth for future autotuning.
- Tests pass.

Do not implement autotuning or kernel search yet. This prompt is about clean optimization boundaries and experiment truthfulness.
```

---

## Prompt 14 - docs, architecture updates, ADRs, and hybrid-runbooks

```text
Polish the docs and developer ergonomics for Phase 7.

Requirements:
- Update README.md with a clear summary of what Phase 7 adds.
- Add docs/hybrid-workers.md or equivalent that explains:
  - local versus remote worker concepts,
  - topology/capability extensions,
  - remote transport and lifecycle,
  - security/auth assumptions for worker registration,
  - hybrid routing modes and spillover guardrails,
  - remote-aware benchmark/replay/report behavior,
  - operator inspection and budget controls,
  - packaging/deployment boundaries for later Linux/NVIDIA workers,
  - optimization-ready config surfaces and how they prepare for later Forge Stage A.
- Update docs/architecture.md so the hybrid control-plane path, worker lifecycle, and remote placement decision path are explicit.
- Add at least one ADR for a key Phase 7 decision, such as:
  - why remote workers are first-class topology members instead of special cases,
  - why hybrid routing remains explainable instead of opaque,
  - why cloud support is developed against mock workers before real GPUs,
  - or why optimization surfaces are captured explicitly before any Forge work begins.
- Add practical example commands for:
  - running a mock remote worker,
  - registering or discovering a remote worker,
  - enabling/disabling remote spillover,
  - running a hybrid benchmark/replay scenario,
  - inspecting remote health and budget state,
  - packaging the remote worker runtime.

Acceptance criteria:
- A new contributor can understand and operate the Phase 7 hybrid stack from the docs.
- Trust boundaries and limitations are visible.
- All checks still pass.
```

---

## Prompt 15 - Phase 7 exit review

```text
Review the repo against AGENTS.md and the intended Phase 7 definition of done.

Tasks:
- Identify anything missing, weak, too clever, too vendor-specific, or too cloud-hand-wavy in the current Phase 7 implementation.
- Tighten tests, docs, config naming, artifact clarity, remote-lifecycle honesty, and hybrid-routing ergonomics where needed.
- Remove accidental overengineering.
- Verify that:
  - remote/cloud workers are first-class in topology and capability models,
  - the control plane can talk to remote workers through explicit, typed, contract-tested transport paths,
  - remote registration, heartbeats, readiness, and lifecycle are implemented and observable,
  - hybrid routing policies are explainable and compatible with Phase 6 routing intelligence,
  - spillover guardrails and budgets keep remote use bounded and visible,
  - benchmark/replay/report flows can compare local-only, remote-only, and hybrid scenarios honestly,
  - operator surfaces expose remote health, placement, and budget state,
  - deployment/package scaffolding exists for later Linux/NVIDIA workers without breaking Mac-first dev,
  - optimization-ready config surfaces exist for later Forge Stage A,
  - the overall design remains backend-agnostic and ready for later vLLM-CUDA integration.
- Make code changes only where they clearly improve completeness or clarity.

Deliverables:
- a concise Phase 7 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 8 tasks,
- the top 5 recommended real-cloud-rollout tasks,
- the top 5 recommended Forge Stage A tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 7 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add remote/cloud-aware topology models, remote transport, secure worker registration/lifecycle, hybrid routing modes, spillover budgets/guardrails, remote-aware benchmark/replay/reporting, operator surfaces, packaging for later Linux/NVIDIA workers, and optimization-ready config surfaces,
2. the schema and artifact changes you recommend,
3. the simplest practical auth and lifecycle model you recommend for remote workers and why,
4. the implementation order you would use,
5. the test strategy for CI without Apple GPU access or rented cloud GPUs,
6. the main risks in hybrid routing, spillover budgeting, and remote lifecycle management and how you would mitigate them,
7. how you would keep the design backend-agnostic and ready for later vLLM-CUDA plus Forge Stage A work.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
