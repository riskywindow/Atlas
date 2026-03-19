# ADR 0008: Optimization-Ready Knob Surface

## Status

Accepted

## Context

Phase 7 already includes explainable routing, hybrid local/remote controls, remote
worker lifecycle posture, and authoritative benchmark and replay artifacts. The
remaining gap before later Forge Stage A work is a clean typed surface that describes
which control-plane knobs are tunable, what their current values are, and which bounds
or rollout modes remain allowed.

Without that surface, later optimization work would have to infer policy knobs from a
mix of runtime settings, admin endpoints, and artifact conventions. That would make the
Stage A inputs harder to diff, harder to review, and easier to couple to internal
implementation details.

## Decision

Switchyard exports a dedicated optimization-ready profile:

- `OptimizationSettings` in runtime config defines the allowlisted routing policies,
  rollout modes, evidence thresholds, bounded search ranges, and worker launch presets
  for later tuning work.
- `OptimizationProfile` and `OptimizationKnobSurface` are typed schemas that serialize
  the current tunable surface.
- `BenchmarkRunConfig` and `ReplayPlan` carry immutable config snapshots plus a
  canonical fingerprint so future experiment loops can compare runs honestly.
- `switchyard-control-plane export-optimization-profile` emits the resolved profile as
  JSON for offline consumers.

The profile is informational. It does not change live routing behavior by itself.

## Consequences

Positive:

- Later Forge Stage A work can consume a stable, typed, diffable config artifact.
- The exported surface makes local-first, spillover, and operator-review constraints
  explicit instead of implicit.
- Offline tuning can stay separate from the HTTP gateway and router implementation.

Negative:

- Some tuning bounds are duplicated at the export layer rather than inferred purely from
  lower-level settings.
- The profile still needs future extensions if later stages tune richer adaptive-policy
  internals.

## Rejected Alternatives

- Reusing benchmark artifact schemas directly.
  Rejected because artifacts represent observations and results, not the allowed tuning
  surface of the live control plane.

- Reading admin runtime state as the optimization contract.
  Rejected because runtime state is mutable and operational, while Stage A needs a
  reviewable configuration snapshot.
