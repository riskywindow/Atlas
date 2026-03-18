# Phase 7

Phase 7 moves Switchyard from a deployable, explainable, Mac-first control plane into a
hybrid local/remote execution phase. The goal is not to abandon the local Apple-Silicon
path that made earlier phases practical. The goal is to make remote workers first-class
topology members while keeping routing explainable, artifacts authoritative, and the
control plane testable without real rented GPUs.

## Goals

- Preserve the current local-first developer path:
  - host-native Apple-Silicon workers remain the default for real local execution,
  - the control plane still works in local, Compose, and kind-oriented workflows,
  - local-only behavior remains available and testable.
- Make remote workers first-class topology members:
  - worker lifecycle state stays typed,
  - registration and heartbeat posture stay explicit,
  - runtime inspection and deployment diagnostics can summarize remote health honestly.
- Add hybrid execution controls in small slices:
  - local preference and spillover remain explicit,
  - remote budget posture is operator-visible,
  - hybrid routing decisions remain explainable from typed inputs and outputs.
- Extend evidence and reporting:
  - benchmark, replay, and comparison surfaces should distinguish local and remote paths,
  - reports should derive from authoritative artifacts rather than ad hoc reconstruction,
  - offline evaluation should remain useful before live hybrid policy changes.
- Prepare for later cloud-ready workers without locking in a runtime:
  - later Linux/NVIDIA `vllm_cuda` workers should fit the same boundary,
  - later Forge Stage A autotuning should consume typed artifacts and measurements rather
    than hidden router state.

## Definition Of Done

- The repo keeps a clean Python workspace with linting, typing, and tests.
- The control plane still supports at least two real Mac-native backend families:
  - `mlx_lm`
  - `vllm_metal`
- The gateway still serves `GET /healthz`, `GET /readyz`, and
  `POST /v1/chat/completions` with health-aware fallback.
- Routing, overload admission, circuit-breaker protection, session affinity, canaries,
  shadowing, and policy rollout remain outside the HTTP layer.
- Remote workers are first-class topology members:
  - typed instance inventory can describe local and remote workers,
  - registration and heartbeat posture are inspectable,
  - runtime and deployment views expose remote health and topology state.
- Hybrid execution controls are explicit and bounded:
  - local preference is configurable,
  - spillover is guarded,
  - remote budget posture is visible to operators and artifacts.
- Routing decisions remain explainable:
  - local versus remote choices are attributable,
  - hybrid policy behavior is inspectable,
  - logs and artifacts can explain fallback, degradation, and spillover.
- Benchmark, replay, and reporting surfaces remain authoritative:
  - local versus remote execution can be distinguished in machine-readable outputs,
  - operator-facing markdown points back to typed artifacts,
  - offline comparison remains possible without live remote GPU access.
- Apple-specific runtime imports stay lazy and optional so CI and portable control-plane
  packaging do not require Apple GPU dependencies.
- Phase 7 leaves a clean extension path for later Linux/NVIDIA `vllm_cuda` workers and
  later Forge Stage A autotuning without requiring a refactor of the control plane.

## Non-Goals

- No giant refactor of the control plane or routing stack.
- No requirement to rent GPUs or stand up a production cloud platform in this phase.
- No opaque cost or routing logic hidden outside typed schemas and artifacts.
- No direct coupling of router logic to vendor-specific hardware assumptions.
- No production-grade multi-cluster orchestration layer.
- No introduction of CUDA-only runtime dependencies into the default control-plane
  workspace.
- No automatic promotion of remote execution policies based on unreviewed signals.

## Rules

### Local-First Default Rule

Local-first development remains the default. The shortest useful path for a developer
should still be a Mac-first local workflow with host-native Apple-Silicon workers and a
portable control plane.

### No Rented GPU Requirement Rule

Remote and cloud support must be testable without real rented GPUs. Mock workers,
synthetic traces, offline simulation, protocol tests, and typed artifacts should be
enough to validate the control-plane slices added in Phase 7.

### Typed Truth Rule

Topology, cost, and health truth must come from typed contracts and authoritative
artifacts. Runtime inspection, benchmark artifacts, replay outputs, and simulation
artifacts are the source of truth. Logs and markdown reports are supporting surfaces,
not the authoritative model.

### Explainable Hybrid Routing Rule

Hybrid routing must remain explainable. If the system prefers local execution, spills
over to remote capacity, or rejects a remote path for budget, health, or policy
reasons, that reasoning must be representable in typed route decisions, diagnostics, or
artifacts.

### Portability Rule

Phase 7 prepares for later Linux/NVIDIA `vllm_cuda` workers and later Forge Stage A
autotuning. The control plane should expose typed measurements, topology metadata, and
policy evidence that those later phases can consume without rewiring the serving path.

## Audit Notes

The current repo already had most of the Phase 6 foundations that Phase 7 depends on:

- explicit network-addressable worker topology,
- a remote worker adapter and worker protocol,
- deployment-aware diagnostics,
- explainable routing and offline policy evaluation,
- benchmark and replay artifacts with topology context.

The smallest obvious blocker for a clean Phase 7 start was operator visibility. The
repo needed typed runtime and deployment summaries for hybrid local/remote posture and
remote worker lifecycle state before hybrid routing behavior grows further. That is the
kind of small blocker worth patching here. Larger routing or deployment changes should
continue as follow-on vertical slices rather than being bundled into this alignment pass.

The next obvious packaging blocker is similar: Phase 7 needs a reviewable Linux remote
worker packaging path before any real cloud GPU rollout exists. That path should rely on
the existing HTTP worker protocol, static discovery or typed registration, and CI-safe
stub workers rather than rented GPUs. The packaging and documentation slice in this repo
now covers that boundary directly.
