# Phase 8 Exit Review

## Summary

Phase 8 is close to exit on the control-plane side.

The repo now has:

- explicit `vllm_cuda`-style remote worker bring-up and packaging,
- typed remote registration, lifecycle, and operator health posture,
- cross-backend alias compatibility across local Apple and remote cloud paths,
- honest observed-versus-estimated cloud evidence in runtime/admin surfaces and artifacts,
- explicit rollout gating for `canary-only` cloud workers through
  `/admin/hybrid/cloud-rollout`.

The remaining work is not another architecture reset. It is mostly hardening,
operator practice, and proving the real rented-GPU path under a few controlled runs.

## What Looks Ready

- The control plane remains Mac-first and backend-agnostic.
- Remote workers are first-class topology members rather than special cases.
- Cloud placement, spend posture, and health are visible without hiding behind vendor
  branches.
- Rollback posture is explicit:
  `canary-only`,
  `cloud_rollout`,
  remote enable/disable,
  quarantine,
  and budget/cooldown guardrails can all be inspected independently.

## Residual Risks

- The first real cloud rollout still depends on operator judgment for widening
  percentages. That is intentional, but it means the runbook needs to be followed.
- `eligible` is not the same as `chosen`; a cloud backend can clear rollout gating and
  still lose on score. That is correct behavior, but operators need to read
  `x-switchyard-route-decision` and `/admin/hybrid` carefully.
- The current rollout state is process-local. Multi-gateway coordination is Phase 9
  work, not Phase 8 work.

## Phase 9 Recommendations

- Persist rollout and hybrid operator state in a small shared store so multi-process
  gateways do not drift during incidents.
- Add a first-class rollback bundle command that can atomically:
  enable the cloud rollout kill switch,
  disable remote routing,
  and snapshot current hybrid/operator state.
- Add report tooling that summarizes "blocked by rollout" versus "eligible but not
  chosen" counts per serving target and remote backend.
- Harden rented-GPU smoke coverage around transport failures, stale heartbeats, and
  cloud-worker replacement during live traffic.

## Forge Stage A Recommendations

- Consume the explicit observed-versus-estimated evidence split as separate optimizer
  inputs; do not collapse them into one score source.
- Treat `cloud_rollout` state as a guardrail input, never as an optimization target.
- Keep learned or optimized policy promotion behind the existing operator-reviewed
  rollout modes rather than inventing a separate promotion path.
- Preserve the ability to explain why a remote backend was blocked:
  rollout gate,
  spillover guardrail,
  quarantine,
  circuit breaker,
  or policy score.

## Exit Call

The repo does not need a broader refactor before Phase 9.

The pragmatic Phase 8 finish is:

1. run a few documented rented-GPU canaries with the new rollout gate,
2. capture benchmark and operator artifacts from those runs,
3. fix only the concrete issues that show up,
4. then move the remaining state-coordination and automation work into Phase 9.
