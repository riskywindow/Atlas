# Phase 2

Phase 2 turns Switchyard from a single-real-backend proof point into a small but honest
multi-backend local control plane. The goal is to keep the control plane portable while
adding the minimum real functionality needed to compare policies and fail over cleanly
across backends.

## Goals

- Support two real Mac-native backend implementations behind the same adapter boundary:
  - `mlx_lm`
  - `vllm_metal`
- Keep the gateway, router, schemas, telemetry, and benchmark paths backend-agnostic.
- Support multi-backend model registration so one logical model alias can map to more
  than one backend implementation.
- Add router v1 policy modes and health-aware fallback without burying routing behavior
  in the HTTP layer.
- Improve route-level observability so route decisions, fallback attempts, latency, and
  backend selection are inspectable and benchmarkable.
- Add comparative benchmark tooling that can evaluate the same workload shape across
  multiple routing policies.

## Definition Of Done

- The repo still has a clean Python workspace with linting, typing, and tests.
- There are at least two real backend paths available behind `BackendAdapter`:
  - `mlx_lm`
  - `vllm_metal`
- Real backend imports remain lazy and optional so CI-friendly tests do not require
  Apple GPU hardware.
- One logical model alias can map to multiple backend implementations through
  configuration and registration.
- `POST /v1/chat/completions` can route to a chosen backend and fall back to another
  healthy candidate when the first backend is unavailable.
- Router policy behavior remains testable outside FastAPI.
- Route-level logging and telemetry capture policy, chosen backend, fallback attempts,
  and latency-oriented execution data.
- Benchmark artifacts can compare routing policies in a reproducible JSON format.
- The control plane remains portable to future `vllm_cuda` and remote backends without
  changing the gateway or router contracts.

## Non-Goals

- No CUDA, Triton, or remote worker runtime integration in this phase.
- No Kubernetes, Ray, or multi-service scheduler buildout in the request path.
- No frontend UI work.
- No adapter hierarchy rewrite or speculative plugin framework.
- No router overbuild beyond simple, typed, testable v1 policy behavior.

## Mac-First Constraint

Switchyard remains Mac-first in Phase 2. Real local inference should run host-native on
macOS Apple Silicon, and Apple-specific runtime details must stay inside the adapter or
runtime/provider boundary rather than leaking into the router, schemas, or HTTP control
plane.

## Adapter Boundary Rule

Future non-Mac backends must fit the same adapter boundary used by the Mac-native
backends. That includes future `vllm_cuda` and remote OpenAI-like workers. The control
plane should reason about capabilities, health, latency, quality, and cost signals, not
about backend-specific runtime mechanics.

## Model Alias Rule

Phase 2 should assume that one logical model alias may map to multiple backend
implementations. Configuration, registry construction, routing, fallback, and benchmark
artifacts should preserve that relationship instead of assuming a one-to-one mapping
between alias and runtime.

## Audit Notes

- The repo already had most of the Phase 1 foundation needed to enter Phase 2: typed
  contracts, a pure router, a FastAPI gateway, observability scaffolding, and real
  backend boundaries.
- The obvious blockers for a Phase 2 start were documentation alignment issues:
  `AGENTS.md` still contained several Phase 0 constraints and `docs/phase2.md` did not
  exist.
- This pass intentionally keeps the change small and focused on repo alignment rather
  than a broad implementation rewrite.
