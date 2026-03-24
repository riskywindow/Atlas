# Phase 9

Phase 9 is Forge Stage A: evidence-driven autotuning.

This phase assumes the repo already has the Phase 8 foundation in place:

- a Mac-first control plane,
- real local Apple-Silicon workers behind the generic worker contract,
- hybrid local/remote execution,
- explainable and adaptive routing,
- benchmark, replay, and simulation artifacts,
- operator-visible remote placement, spend, and health posture,
- and a real Linux/NVIDIA cloud-worker path.

Phase 9 adds the first optimization layer on top of that foundation without turning the
project into a code-generation system. The work should stay backend-agnostic, preserve
the Mac-first developer path, and ship in small vertical slices.

## Goals

- Expose a typed optimization surface with explicit knobs, objectives, constraints, and
  promotion guardrails.
- Capture campaign and trial lineage so tuning work is serializable, diffable, and
  reviewable.
- Compare baseline and candidate policies or profiles through offline, replay-backed,
  and simulation-backed evaluation.
- Generate conservative, explainable recommendations rather than opaque automatic
  decisions.
- Support safe promotion through config profiles, bounded canaries, and explicit
  rollback posture.
- Expose operator/admin inspection surfaces for current Forge Stage A campaigns,
  evidence posture, and promotion constraints.
- Prepare the repo for later Forge Stage B without coupling the control plane to
  hardware-specific optimization logic.

## Definition Of Done

Phase 9 is complete when all of the following are true:

1. The repo keeps a clean Python workspace with linting, typing, and tests.
2. The existing Mac-native backend paths remain intact:
   `mlx_lm` and `vllm_metal`.
3. The real Linux/NVIDIA worker path remains intact behind the generic worker contract.
4. The optimization layer exposes explicit knobs, objectives, constraints, and
   promotion limits through typed schemas and export surfaces.
5. Campaigns and trials have typed lineage that can be serialized, diffed, inspected,
   and consumed by offline evaluation workflows.
6. Offline, replay-backed, and simulation-backed evaluation can compare baseline and
   candidate configurations without requiring Apple GPU access in CI.
7. Recommendations are explainable, conservative, and reversible; insufficient evidence
   produces an explicit no-change or recommendation-only outcome.
8. Observed runtime evidence remains distinct from replayed, simulated, estimated, and
   mock evidence in schemas, artifacts, reports, and admin surfaces.
9. Live promotion uses bounded rollout controls and explicit rollback support rather
   than direct uncontrolled policy mutation.
10. Operator/admin inspection surfaces expose the current Forge Stage A campaign
    posture, evidence requirements, candidate lineage, and promotion constraints.
11. The Mac-first local developer workflow and portable control-plane packaging remain
    intact while Phase 9 capabilities are added.
12. Deployment docs and runbooks explain how Forge Stage A fits on top of the existing
    local and hybrid/cloud paths.
13. The repo clearly prepares for Forge Stage B without implementing kernel generation
    or code generation in Phase 9.

## Non-Goals

- No kernel generation.
- No runtime code generation.
- No backend-specific autotuning logic inside the control plane core.
- No uncontrolled automatic live promotion from offline results straight to production
  traffic.
- No large packaging or service-topology refactor.
- No new frontend.

## Phase 9 Rules

### Explicit Optimization Rule

Optimization must be driven by explicit knobs, objectives, and constraints.
It must not infer hidden mutable tuning state from ad hoc runtime behavior alone.

### Evidence Separation Rule

Observed evidence must remain distinct from replayed, simulated, or estimated evidence.
Typed schemas, artifacts, reports, and operator surfaces should preserve that
distinction directly rather than burying it in prose.

### Explainable And Reversible Recommendation Rule

Candidate recommendations must be explainable and reversible.
Every recommendation should make clear:

- what candidate or scope it applies to,
- what evidence posture supports it,
- what guardrails or caveats apply,
- and what safe no-change path exists when evidence is weak.

### Bounded Promotion Rule

Live promotion requires bounded rollout and rollback support.
Promotion should move through explicit config profiles, canaries, and kill switches
rather than mutating the active serving path in an opaque way.

### Forge Stage B Boundary Rule

Phase 9 prepares for later Forge Stage B but does not implement kernel generation,
runtime code generation, or similar synthesis workflows.

## Current Audit Summary

The current repo already carries most of the Phase 9 prerequisites:

- typed optimization profiles and worker launch presets,
- benchmark/replay/simulation and recommendation artifacts,
- hybrid/cloud operator surfaces,
- and a new Forge Stage A campaign inspection surface.

The first executable Phase 9 slice now also includes a bounded offline campaign runner
in `switchyard.bench.campaigns`. It materializes candidate configuration artifacts,
counterfactual trial artifacts, and a campaign artifact from benchmark and/or trace
evidence while keeping unsupported candidate families explicitly partial instead of
pretending they were evaluated.

The repo now also carries a first explicit candidate-generation layer in
`switchyard.bench.candidate_generation`. It supports fixed baselines,
one-factor-at-a-time variants, bounded grid search, bounded random search, and small
heuristic seeds, while pruning hard-constraint violations and unsafe combinations
before the executor sees them.

The main alignment blockers for Phase 9 were small rather than architectural:

- `AGENTS.md` still described the project as Phase 8,
- this Phase 9 doc was only a narrow feature note rather than a phase contract,
- and the default optimization profile naming still reflected the earlier pre-Phase-9
  baseline.

That is the right scope for an audit-and-alignment pass. Phase 9 should now proceed in
small vertical slices on top of the existing Phase 8 foundation rather than through a
large refactor.
