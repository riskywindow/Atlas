# ADR 0010: Canary-Only Cloud Workers Require Explicit Runtime Rollout

## Status

Accepted

## Context

Phase 8 introduces the first real Linux/NVIDIA cloud worker path. The repo already had:

- typed remote worker registration and lifecycle posture,
- operator-visible quarantine and `canary-only` tags,
- alias overrides,
- hybrid remote enable/disable controls.

That was enough to describe a cloud worker, but it left one unsafe gap: a
`canary-only` worker could become "just another candidate" without a dedicated runtime
gate that operators could inspect and reverse quickly.

## Decision

Switchyard now treats `canary-only` as an enforceable rollout posture, not only an
operator label.

The control plane adds a small runtime `cloud_rollout` controller that:

- blocks `canary-only` cloud backends from primary routing by default,
- allows them only through an explicit deterministic rollout percentage,
- preserves explicit canary-routing selection and explicit backend pins,
- exposes current state and recent decisions through `/admin/hybrid/cloud-rollout`,
- keeps the final primary choice with the normal routing policy once the backend is
  eligible.

This keeps the control plane backend-agnostic: the gate reasons about typed rollout
posture on a remote backend, not CUDA-specific logic.

## Consequences

Positive:

- first rented-GPU traffic is bounded and reversible,
- operators can inspect whether a cloud backend was blocked or merely lost on score,
- `canary-only` becomes meaningful in runtime behavior, not only admin metadata,
- observed versus estimated cloud evidence remains separate from rollout state.

Negative:

- there is another mutable runtime surface to inspect during incidents,
- "eligible" still does not mean "chosen"; policy score remains a separate concern.

## Rejected Alternatives

- Let `canary-only` remain only a label.
  Rejected because that is not a real rollout control.

- Reuse alias pins as the primary rollout mechanism.
  Rejected because pins are coarse overrides, not bounded rollout controls.

- Make rollout selection override the routing score directly.
  Rejected because Phase 8 still wants routing policy, health, and fallback behavior to
  remain explainable and benchmarkable outside the HTTP layer.
