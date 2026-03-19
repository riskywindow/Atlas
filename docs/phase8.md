# Phase 8

Phase 8 moves Switchyard from cloud-ready hybrid execution into the first real
cloud-backed execution phase. The goal is not to replace the Mac-first local path. The
goal is to prove that the existing backend-agnostic control plane can drive the first
real Linux/NVIDIA remote worker path, report that behavior honestly, and keep rollout
controls bounded and reversible.

## Goals

- Add the first real cloud-backed worker path in small vertical slices:
  - a concrete `vllm_cuda`-style remote backend should run behind the generic worker
    contract,
  - the control plane should treat it as another typed deployment rather than a special
    case,
  - cross-backend aliases should remain compatible across local Apple and remote cloud
    paths.
- Preserve the Mac-first developer default:
  - host-native Apple-Silicon workers remain the shortest useful local path,
  - the portable control-plane workspace stays usable without Linux/NVIDIA runtime
    dependencies,
  - local-only and hybrid test paths stay available in CI and on a Mac laptop.
- Keep the control plane backend-agnostic:
  - router and gateway logic should continue to reason about typed capabilities, health,
    topology, budgets, and evidence,
  - hardware-specific logic should remain isolated at adapter, worker, or packaging
    boundaries.
- Make cloud-backed evidence honest:
  - observed runtime/cloud evidence must be distinct from deployment metadata estimates,
    predictor outputs, and mock injections,
  - operator surfaces, benchmark artifacts, and markdown reports should expose that
    distinction directly.
- Keep the first cloud path safe to operate:
  - canaries stay explicit,
  - spillover and remote budget posture stay bounded,
  - kill switches and rollback remain operator-visible and reversible.
- Prepare for later Forge Stage A work:
  - typed artifacts, config surfaces, and replay/report outputs should remain consumable
    by later optimization work,
  - Phase 8 does not implement Forge Stage A itself.

## Definition Of Done

- The repo keeps a clean Python workspace with linting, typing, and tests.
- The existing real Mac-native backend families remain supported:
  - `mlx_lm`
  - `vllm_metal`
- The first real Linux/NVIDIA worker path exists behind the shared worker contract:
  - `vllm_cuda` or an equivalent `vllm_cuda`-style remote backend path is wired through
    the existing backend and worker abstractions.
- The gateway still serves `GET /healthz`, `GET /readyz`, and
  `POST /v1/chat/completions` with health-aware fallback.
- Cross-backend alias compatibility remains intact:
  - a logical alias may map to local Apple paths, the real cloud path, or both,
  - routing, inspection, and benchmark outputs preserve that abstraction.
- The control plane remains backend-agnostic:
  - routing logic does not hardcode hardware assumptions,
  - hardware-specific imports and runtime concerns stay outside the router and gateway.
- Cloud placement, spend, and runtime evidence are operator-visible and typed:
  - runtime/admin views expose cloud placement and health posture,
  - benchmark/replay/report artifacts preserve local versus remote execution truth,
  - observed evidence is kept distinct from estimated or mock evidence.
- The first real cloud path is canaryable, bounded, and reversible:
  - canary and shadow controls remain explicit,
  - remote budget and kill-switch posture are visible,
  - rollback to local-only or non-cloud behavior is straightforward.
- Mac-first local development remains the default:
  - Apple-Silicon local workflows still work without cloud access,
  - the default control-plane workspace does not require CUDA-only dependencies.
- Deployment docs and runbooks cover the first rented-GPU bring-up path without making
  CI depend on real GPU rentals.
- The Phase 8 design prepares for later Forge Stage A by preserving typed evidence,
  config, and replay/report surfaces without implementing Forge Stage A itself.

## Non-Goals

- No giant refactor of the control plane, routing stack, or package layout.
- No production-grade cloud platform buildout, scheduler, or fleet manager.
- No opaque cloud cost or routing logic hidden outside typed schemas, artifacts, and
  runtime inspection.
- No requirement that everyday development or CI use rented GPUs.
- No direct coupling of router policy code to NVIDIA-, Metal-, or vendor-specific
  hardware logic.
- No automatic policy promotion or cloud expansion driven by unreviewed signals.
- No implementation of Forge Stage A autotuning, learned routing, or kernel search in
  this phase.

## Rules

### Mac-First Default Rule

Mac-first local development remains the default. The shortest useful path for a
developer should still be a local Apple-Silicon workflow with host-native workers and a
portable control plane.

### Backend-Agnostic Control Plane Rule

The control plane remains backend-agnostic. Routers, schemas, admin surfaces, and
benchmarks should reason about typed capabilities, health, cost, and topology rather
than hardware brands or runtime internals.

### Evidence Separation Rule

Observed cloud/runtime evidence must be kept distinct from estimates, predictor outputs,
and mock results. Benchmarks, runtime inspection, and reports should never blur those
sources together.

### Canaryable And Reversible Cloud Rule

The first real cloud path should be canaryable, bounded, and reversible. Cloud-backed
execution must remain explicit, operator-visible, and easy to disable or roll back.

### Forge-Preparation Rule

Phase 8 prepares for later Forge Stage A by preserving typed artifacts, config
snapshots, and evidence surfaces that later optimization work can consume. Phase 8 does
not implement Forge Stage A itself.

## Audit Notes

The current repo already carries most of the foundations Phase 8 needs:

- explicit network-addressable worker topology,
- a generic remote worker protocol and adapter boundary,
- registration and lifecycle posture for remote workers,
- explainable routing with admission, circuit-breakers, canaries, shadowing, and policy
  rollout,
- benchmark, replay, comparison, and simulation artifacts with topology context,
- portable packaging boundaries that keep Apple-specific dependencies optional.

The smallest obvious blockers for a clean Phase 8 start were alignment blockers rather
than architecture blockers:

- the main repo guidance still described the project as Phase 7,
- there was no dedicated Phase 8 doc that set the new rollout and evidence rules
  explicitly,
- the repo needed an explicit standard that observed cloud/runtime evidence must stay
  separate from estimated or mock evidence in operator and benchmark surfaces.

Those are the right size of blockers to patch in an audit-and-alignment pass. Larger
runtime bring-up, cloud packaging, or rollout work should continue as follow-on vertical
slices.

## First Concrete Worker Boundary

The first concrete Linux/NVIDIA worker path should be reviewable as a real worker
implementation even when local development and CI do not have CUDA.

- `switchyard.worker.vllm_cuda` and `switchyard.worker.vllm_cuda_cli` are the concrete
  worker-process entrypoints for the first `vllm_cuda` path.
- `switchyard.adapters.vllm_cuda` is responsible for translating the shared worker
  contract into typed request, capability, health, and deployment behavior.
- `switchyard.runtime.vllm_cuda` is the only place that should know about direct
  vLLM-CUDA runtime imports.

The intentionally abstracted or feature-gated parts of this first slice are:

- Direct `vllm` imports stay lazy and optional so Mac-first development and CI do not
  require CUDA.
- The provider boundary in `switchyard.runtime.vllm_cuda` is the seam for future
  host-specific engine flags, tokenizer/runtime tuning, and exact streaming mechanics.
- Tool calling and JSON-schema response formatting remain explicitly disabled until a
  later Phase 8 slice verifies them on real rented-GPU workers.
- The worker contract, topology truth, and operator/benchmark surfaces should already be
  stable even while the concrete runtime internals continue to harden.
