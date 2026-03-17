# ADR 0003: Allow One Logical Alias To Map To Multiple Backend Deployments

## Status

Accepted

## Context

Phase 2 needs to support more than one real local backend while keeping the public
serving contract simple. Clients should target one logical model alias, but the control
plane must be able to route between MLX-LM and vLLM-Metal implementations of that alias,
benchmark them, and fall back safely when one deployment becomes unavailable.

If clients were forced to address concrete backend names directly, Switchyard would lose
most of its value as an inference fabric:
- route policy selection would leak into clients,
- health-aware fallback would become much harder to reason about,
- comparative benchmarking would fragment around backend-specific request paths,
- future CUDA and remote workers would require more API churn.

## Decision

Treat the client-facing `model` field as a logical serving target and allow that target
to map to one or more concrete backend deployments.

Concretely:
- the registry indexes deployments by both backend name and serving target,
- the router resolves candidates from the serving target,
- the gateway executes the selected deployment and, when safe, a small ordered fallback
  set,
- explicit backend pinning remains available only through a namespaced internal override
  path for tests and debugging.

## Consequences

Positive:
- the public API stays simple,
- routing and fallback remain centralized and explainable,
- benchmark artifacts can compare alias routing against pinned deployments cleanly,
- the same contract can later host `vllm_cuda` and remote workers.

Tradeoffs:
- config needs to distinguish logical alias from concrete deployment alias,
- telemetry and artifacts must carry both target-level and deployment-level metadata,
- debugging sometimes requires an internal backend pin header.
