# ADR 0007: Keep Phase 6 Adaptive Routing Transparent And Conservative

## Status

Accepted

## Context

Phase 6 introduces an adaptive policy, offline simulation, rollout controls, and
recommendation reports. This creates a common failure mode for routing systems:

- simulation output starts being treated as truth instead of evidence,
- adaptive logic becomes hard to explain from artifacts,
- rollout decisions become hard to reverse,
- locality and history signals drift into hidden prompt storage or hardware-specific
  heuristics.

Switchyard needs smarter routing, but the project goal is still a serious and portable
control plane rather than an opaque scheduler.

## Decision

Phase 6 adaptive routing is intentionally transparent and conservative.

Concretely:

- deterministic request features and locality signals are typed and serializable,
- raw prompt text is not stored as a locality signal,
- historical estimates are explicit about scope and sample size,
- adaptive scoring may abstain instead of forcing a decision,
- shadow scoring and report-only modes come before guarded activation,
- rollout controls include kill switch, freeze-learning, reset, and export/import,
- recommendation reports are derived from authoritative artifacts, not ad hoc runtime
  logs,
- simulation distinguishes observed outcomes, predictor estimates, low-confidence
  estimates, and unsupported cases.

## Consequences

Positive:

- operators can inspect why a policy selected, abstained, or fell back,
- CI and local development can evaluate policy behavior without Apple GPU access,
- later cloud or Forge work can reuse the same typed contracts,
- adaptive behavior stays reversible and easier to benchmark.

Tradeoffs:

- the first adaptive policy is less aggressive than a learned online scheduler,
- more recommendation cases end in no-change or shadow-only guidance,
- some rollout actions require explicit operator steps instead of automatic promotion.

That tradeoff is intentional. In Phase 6, trustworthy decision support is more valuable
than squeezing out a few extra routing wins behind an opaque control path.
