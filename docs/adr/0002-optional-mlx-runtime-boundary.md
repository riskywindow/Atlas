# ADR 0002: Keep MLX-LM Optional And Behind A Runtime Boundary

## Status

Accepted

## Context

Phase 1 needs one honest local Apple Silicon backend so the gateway, routing, telemetry,
and benchmark path can be validated against real model execution. At the same time, the
repo still needs to work for contributors and CI environments that do not have Apple
Silicon hardware or do not want MLX-LM installed.

## Decision

Use MLX-LM as an optional dependency and isolate it behind a dedicated runtime boundary:

- `MLXLMAdapter` satisfies the shared `BackendAdapter` contract.
- `MLXLMChatRuntime` owns MLX-specific model loading and generation behavior.
- direct MLX imports stay inside the runtime provider path instead of the gateway or
  router.

## Consequences

- The control plane remains testable without MLX-LM installed.
- Apple-specific runtime concerns stay out of routing and HTTP layers.
- Contributors on Apple Silicon can enable the real backend with `uv sync --dev --extra mlx`.
- Future backends can follow the same pattern instead of inheriting MLX-specific design
  assumptions.

The tradeoff is a bit more adapter/runtime plumbing, but that separation is worth keeping
because portability is a project goal, not a future cleanup item.
