# ADR 0005: Deterministic Canary Bucketing

## Status

Accepted

## Context

Phase 4 adds canary routing so a logical alias can gradually shift a bounded percentage
of traffic to a candidate backend. That rollout behavior must be:

- explainable from logs and artifacts,
- stable enough for tests and local debugging,
- compatible with multi-turn chat,
- safe when the candidate backend is unhealthy or breaker-protected.

Non-deterministic sampling would make local debugging and artifact comparison harder.
Two requests with identical context could select different backends for reasons that are
not reconstructable after the fact.

## Decision

Switchyard uses deterministic canary bucketing based on a stable request key:

- prefer the session ID when present,
- otherwise fall back to the request ID.

The resulting bucket is compared against the configured percentage for the rollout
policy. If the candidate backend is selected but is unhealthy or otherwise ineligible,
routing falls back to the baseline path and records that reason in route annotations and
artifacts.

## Consequences

Positive:

- multi-turn chat stays rollout-stable when a session key is present,
- tests can assert canary behavior deterministically,
- artifacts can explain why a request went to canary or baseline,
- rollout behavior stays backend-agnostic.

Tradeoffs:

- distribution is only as stable as the chosen request key,
- local in-process rollout state does not coordinate across multiple gateway processes,
- changing the bucketing key strategy later would affect reproducibility across runs.

## Alternatives Considered

- random per-request sampling
  Rejected because it weakens reproducibility and makes artifact-based explanation worse.
- sticky rollout state in a separate distributed store
  Rejected for Phase 4 because it adds infrastructure complexity before the local control
  plane needs it.
