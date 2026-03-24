# ADR 0012: Forge Stage A Optimization and Promotion Model

## Status

Accepted

## Context

Phase 9 adds the first optimization layer (Forge Stage A) on top of the existing
control plane.  The system already has typed routing policies, hybrid local/remote
guardrails, benchmark and replay artifacts, and an exported optimization-ready knob
surface (ADR 0008).

The remaining question is how to move from "here are the tunable knobs" to "here
is a concrete recommendation, and here is a safe way to act on it."

Two concerns are in tension:

1. **Automation pressure**: operators want the system to tell them what to change
   and ideally do it for them.
2. **Trust boundaries**: offline evidence (replay, simulation, estimation) does not
   guarantee production behavior, and the topology or budget may have changed since
   evidence was collected.

## Decision

Forge Stage A uses a three-layer model:

### Layer 1: Offline Campaign Evaluation

Campaigns generate, prune, and evaluate candidates against benchmark and replay
evidence without touching live routing.  The output is a set of typed artifacts:

- `OptimizationCampaignArtifact` — the authoritative campaign result.
- `OptimizationCampaignComparisonArtifact` — multi-objective candidate ranking.
- `OptimizationRecommendationSummary` — per-trial recommendation with disposition,
  confidence, evidence kinds, and reason codes.

Evidence is never collapsed: observed, replayed, simulated, and estimated results
carry distinct labels through every surface.

### Layer 2: Honesty Assessment

Before acting on recommendations, the system validates them against current
environment state:

- **Budget bounds**: does the current remote budget still support what the campaign
  assumed?
- **Topology drift**: are the workers from campaign evidence still present?
- **Staleness**: is the evidence recent enough?
- **Workload coverage**: did the campaign evaluate diverse workload families, or
  might it overfit?
- **Evidence consistency**: do promotion recommendations have observed evidence, or
  only replay/simulation?

Warnings are typed (`CampaignHonestyWarning`) and surfaced in operator inspection
views.  The system recommends `STALE` or `INVALIDATED` status when conditions
warrant it but does not unilaterally overwrite campaign artifacts.

### Layer 3: Bounded Promotion Lifecycle

Promotion follows an explicit multi-step lifecycle with rollback at every stage:

```
PROPOSED -> APPROVED -> CANARY_ACTIVE -> COMPARED -> PROMOTED_DEFAULT
```

Key properties:

- Promotion requires explicit operator review by default.
- Canary percentage is capped by the optimization profile.
- Runtime rollback restores the pre-promotion state atomically.
- Hybrid guardrail knobs (spillover, budget, concurrency) can be part of the
  promoted profile and are also rolled back.
- Every lifecycle transition is recorded as an auditable event.

### What Is Explicitly Out of Scope

- Automatic unreviewed promotion.
- Kernel or code generation.
- Multi-gateway coordination (future work).
- Learned policy parameter tuning beyond routing-policy selection.

## Consequences

Positive:

- Operators get typed, explainable recommendations backed by artifact evidence.
- Trust boundaries are visible: honesty warnings surface when recommendations may
  not be trustworthy.
- Rollback is always available and does not require a control-plane refactor.
- The typed surfaces (profiles, campaigns, comparisons, config profiles) are
  reusable by future Forge Stage B.
- Evidence semantics (observed vs replayed vs simulated vs estimated) are
  preserved end-to-end.

Negative:

- The multi-step promotion lifecycle requires operator engagement.  This is
  intentional but adds workflow friction.
- Honesty checks are heuristic: they catch common staleness and drift scenarios but
  cannot prove that a recommendation is correct.
- The offline campaign executor currently supports routing-policy candidates;
  config-profile and worker-launch-preset candidates are exported for lineage but
  not fully executed.

## Rejected Alternatives

- **Auto-promotion from offline results**: rejected because offline evidence cannot
  guarantee production behavior.  The risk of a bad automatic promotion outweighs
  the convenience.
- **Collapsing evidence kinds into a single quality score**: rejected because the
  trust difference between observed and estimated evidence is material.  Operators
  need to see the mix.
- **Embedding optimization logic in the live router**: rejected because optimization
  evaluation should be reproducible and offline-first, not coupled to request-path
  latency.
- **Skipping the honesty layer**: rejected because recommendations become stale
  quickly in hybrid environments where topology, budget, and workload patterns
  change.
