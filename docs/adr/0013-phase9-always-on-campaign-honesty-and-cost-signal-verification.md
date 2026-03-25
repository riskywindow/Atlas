# ADR 0013: Always-On Campaign Honesty and Cost-Signal Verification

## Status

Accepted

## Context

ADR 0012 established a three-layer Forge Stage A model with an honesty assessment
layer that validates campaign recommendations against current environment state.
The initial implementation gated honesty checks on whether the caller supplied
external environment state (worker inventory, budget posture).  This meant that
staleness, workload coverage, evidence consistency, and cost-signal checks — which
operate entirely on the campaign artifact itself — were silently skipped when no
environment parameters were provided.

Two additional gaps were identified:

1. **Cost signal honesty**: remote budget constraints could be evaluated using
   configured values rather than observed runtime cost signals, making
   recommendations appear budget-safe when actual cloud spend behavior was never
   measured.
2. **Evidence contradictions**: the evidence consistency check only looked at
   evidence-mix shares (estimated vs observed ratio), not at whether observed and
   simulated evidence produced contradictory objective outcomes for the same trial.

## Decision

### Always-on honesty checks

Campaign honesty assessment now always runs during inspection, regardless of
whether external environment state is supplied.  The checks fall into two groups:

- **Artifact-only checks** (always active): staleness, workload coverage,
  evidence consistency, cost signal mismatch.  These require only the campaign
  artifact and produce warnings whenever the data warrants it.
- **Environment-dependent checks** (active when state is provided): budget
  bounds, topology drift.  These degrade gracefully to no warnings when no
  external state is available.

This means `inspect_forge_stage_a_campaigns` always includes honesty warnings
in the inspection output, not only when the caller explicitly opts in.

### Cost-signal mismatch warning

A new `COST_SIGNAL_MISMATCH` warning kind is added.  It fires (severity: `info`)
when a trial's remote budget constraints (budget per minute, remote share, or
concurrency cap) were evaluated without any observed evidence in their
`evidence_kinds`.  This flags cases where configured values stood in for actual
cloud spend observations.

### Strengthened evidence contradiction detection

The evidence consistency check now detects objective-level contradictions: when
a trial's observed evidence and simulated evidence disagree on whether an
objective was satisfied, the check fires with severity `error` (which
recommends `INVALIDATED` status).  This is a stronger signal than the existing
evidence-mix share check, which fires with severity `warning`.

### Concurrency cap in budget bounds

Budget-bound checks now include `REMOTE_CONCURRENCY_CAP` alongside the existing
`REMOTE_REQUEST_BUDGET_PER_MINUTE` and `REMOTE_SHARE_PERCENT` dimensions.

### Trial invalidation API

`mark_trial_invalidated` complements the existing `mark_trial_stale` function,
allowing operators to mark a trial as `INVALIDATED` with a typed reason.

## Consequences

Positive:

- Inspection output is honest by default.  Operators who forget to supply
  environment state still see staleness and coverage warnings.
- Cost-signal warnings make the gap between "configured budget" and "observed
  spend" visible before a promotion decision.
- Objective-level contradiction detection catches cases where the campaign's own
  evidence disagrees with itself, which the share-ratio check would miss.

Negative:

- Inspection results may now include more warnings than before, even for
  campaigns that previously appeared clean.  This is an honesty improvement but
  may require operators to adjust expectations.
- The cost-signal check uses `info` severity by default, which does not affect
  the `trustworthy` flag.  Operators who want stricter treatment would need to
  inspect warnings manually.

## Rejected Alternatives

- **Keeping honesty checks opt-in**: rejected because artifact-only checks
  (staleness, coverage) have no reason to be gated on external state, and
  silently skipping them creates a false sense of trust.
- **Making cost-signal mismatch a `warning`-level severity**: rejected because
  many legitimate local-only campaigns use configured values.  Flagging them as
  non-trustworthy would be too noisy.  `info` surfaces the gap without
  penalizing common workflows.
- **Building a full causal inference system for evidence contradictions**:
  rejected per AGENTS.md scope constraints.  The objective-level contradiction
  check is practical and explicit without requiring statistical modeling.
