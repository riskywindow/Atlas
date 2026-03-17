# ADR 0004: Use Benchmark Artifacts As The Source Of Truth For Phase 3 Reporting

## Status

Accepted

## Context

Phase 3 adds reproducible workload generation, replay, A/B comparison, and markdown
reporting. There are now several possible data sources during a run:

- structured logs,
- metrics,
- captured traces,
- benchmark artifacts.

If reports were allowed to derive conclusions directly from logs or opportunistic runtime
state, two problems would appear quickly:

- the same run could produce different summaries depending on what auxiliary systems were
  enabled,
- comparison and replay outputs would be harder to diff, archive, and validate.

Switchyard needs one authoritative representation for benchmark outcomes.

## Decision

Use typed benchmark artifacts as the source of truth for Phase 3 reporting and
comparative analysis.

Concretely:

- benchmark execution writes JSON artifacts with explicit schema versioning,
- replay and comparison reuse the same artifact story instead of inventing a parallel
  report format,
- markdown reports are rendered from artifact contents after the run,
- logs and metrics remain operational aids, not authoritative benchmark records.

## Consequences

Positive:

- reports stay reproducible and easy to diff,
- machine-readable and human-readable outputs stay aligned,
- downstream tooling has one contract to validate,
- comparison runs can remain deterministic without scraping logs.

Tradeoffs:

- artifacts must carry more benchmark metadata up front,
- renderer code must stay disciplined and avoid report-only logic,
- ad hoc operational signals that are not recorded in the artifact cannot be treated as
  benchmark facts later.

That tradeoff is acceptable because Phase 3 is about rigorous local evaluation, not a
free-form analytics platform.
