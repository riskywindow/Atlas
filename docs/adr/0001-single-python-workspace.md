# ADR 0001: Use A Single Python Workspace For Phase 0

## Status

Accepted

## Context

Phase 0 is about contracts and foundations, not service sprawl. The project needs shared
schemas, a mock backend, routing, a thin gateway, and a benchmark path, all while staying
easy to iterate on from a single Apple Silicon development machine.

## Decision

Use one Python workspace under `src/switchyard` for Phase 0 instead of splitting the repo
into multiple packages or services.

## Consequences

- Shared contracts stay close to the router, gateway, adapters, and benchmark code.
- Tooling is simpler: one `pyproject.toml`, one test suite, and one type-check target.
- Vertical slices are easier to land without spending time on packaging and versioning.
- Future phases can still split components later if real operational boundaries emerge.

For Phase 0, the lower coordination cost is more valuable than early service isolation.
