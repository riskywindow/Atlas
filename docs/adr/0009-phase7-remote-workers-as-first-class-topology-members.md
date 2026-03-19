# ADR 0009: Remote Workers As First-Class Topology Members

## Status

Accepted

## Context

Phase 7 adds remote workers, hybrid routing, and later cloud-ready worker packaging.
The tempting shortcut would be to treat remote workers as a special case outside the
normal backend and topology contracts.

That shortcut would create several problems:

- routing would need hidden hardware- or environment-specific branches,
- benchmark artifacts would lose honest topology truth,
- operator inspection would split between "normal" backends and "remote" exceptions,
- later Linux/NVIDIA workers would require another contract rewrite.

## Decision

Switchyard models remote workers as first-class topology members.

That means remote workers use the same core concepts as any other serving path:

- backend deployments,
- explicit worker instances,
- capabilities,
- health,
- placement,
- trust,
- lifecycle state,
- serializable artifact references.

Remote-specific lifecycle and transport details are added as typed extensions rather
than hidden side channels.

## Consequences

Positive:

- the control plane stays backend-agnostic and explainable,
- runtime/admin inspection can show local and remote posture in one shared view,
- benchmark and replay artifacts preserve the actual exercised topology,
- later `vllm_cuda` workers can slot into the same control-plane boundary.

Negative:

- the shared schemas become richer and require more careful documentation,
- some local-only workflows now carry remote-aware fields that remain unused.

## Rejected Alternatives

- Treat remote workers as opaque external providers.
  Rejected because it would hide topology truth and make routing behavior harder to
  explain or replay.

- Delay remote topology modeling until real GPUs are available.
  Rejected because the control-plane contracts need to be testable in CI before any
  real cloud rollout exists.
