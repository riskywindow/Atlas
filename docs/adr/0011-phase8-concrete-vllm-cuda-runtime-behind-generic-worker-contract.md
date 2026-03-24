# ADR 0011: Use A Concrete `vllm_cuda` Runtime Label Behind The Generic Worker Contract

## Status

Accepted

## Context

Phase 8 needs the first real Linux/NVIDIA worker path, not another abstract placeholder.
At the same time, Switchyard still needs to keep the control plane backend-agnostic and
Mac-first:

- the router should not learn CUDA-specific logic,
- the gateway should not depend on `vllm` or NVIDIA runtime imports,
- the first rented-GPU path still needs a concrete, reviewable runtime identity for
  packaging, registration, diagnostics, and artifact truth.

If Phase 8 used only a generic "remote GPU" label, two problems would appear:

- packaging and bring-up guidance would be vague about what is actually running,
- operator and artifact surfaces would lose the ability to distinguish a real
  Linux/NVIDIA `vllm` worker from a stub, mock, or future remote runtime.

## Decision

Switchyard keeps the generic worker contract and adds one concrete Phase 8 runtime label:
`vllm_cuda`.

That means:

- the worker/runtime boundary gets a concrete `vllm_cuda` implementation,
- the control plane still routes to `remote-worker:<worker_name>` deployments rather than
  to CUDA-specific code paths,
- worker registration, inventory, and artifacts preserve the concrete runtime identity,
- optional Linux/NVIDIA dependencies remain isolated to worker extras, images, and CLI
  entrypoints such as `switchyard-vllm-cuda-worker`.

## Consequences

Positive:

- the first real cloud path is explicit and reviewable,
- packaging, preflight, and bring-up docs can be concrete instead of speculative,
- operator surfaces can distinguish real `vllm_cuda` workers from generic remote
  topology,
- future runtimes can join the same contract without rewriting the control plane.

Tradeoffs:

- schemas and docs must carry both the generic deployment identity and the concrete
  runtime identity,
- contributors must understand that `remote-worker:<worker_name>` is the routed backend
  while `vllm_cuda` is the runtime label behind it.

## Rejected Alternatives

- Use only a generic remote GPU runtime label.
  Rejected because it hides what Phase 8 actually proved and weakens packaging and
  artifact clarity.

- Route directly on CUDA-specific backend names from the control plane.
  Rejected because it would couple routing and fallback logic to hardware-specific
  implementation details.
