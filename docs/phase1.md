# Phase 1

Phase 1 is the first real local backend phase for Switchyard. The repo should keep the
Phase 0 control-plane shape, then add one honest Mac-native serving path that proves the
adapter boundary, routing flow, telemetry, and benchmark story hold up against a real
backend.

## Goals

- Integrate one real Apple Silicon backend end to end behind the existing adapter
  contract.
- Keep the gateway, router, schema layer, and benchmark path backend-agnostic.
- Support both standard and streaming chat completions through the same routing path.
- Make the real backend measurable with request IDs, latency, TTFT, output-token, and
  throughput-oriented telemetry where meaningful.
- Preserve a CI-safe mock path so development and tests do not require Apple GPU access.

## Definition Of Done

- Phase 0 functionality still works: mock backend, routing, gateway endpoints,
  observability scaffolding, benchmark artifact writing, and existing test coverage.
- One real Mac-native backend is available behind `BackendAdapter` with honest
  `health`, `capabilities`, `warmup`, and generation behavior.
- The app can boot cleanly when the real backend is not configured or its optional
  dependency is unavailable.
- `POST /v1/chat/completions` works for both non-streaming and streaming requests using
  the same router and adapter boundary.
- Routing remains independent of hardware-specific runtime details and can choose among
  adapters based on capabilities, health, and policy.
- Benchmarks can record real local runs with backend/model metadata and latency-oriented
  summaries in reproducible JSON artifacts.
- The Mac-first local development flow is documented without assuming CI has Apple GPU
  access.

## Non-Goals

- No vLLM-Metal integration yet.
- No CUDA, Triton, or remote GPU worker runtime work yet.
- No Kubernetes, Ray, or cloud scheduler work in the request path.
- No frontend UI buildout.
- No speculative multi-service split or model-registry overbuild.

## Mac-First Constraint

Switchyard is still Mac-first in Phase 1. Real local inference should run host-native on
macOS Apple Silicon, and Apple-specific runtime details should stay at the adapter or
provider boundary rather than leaking into the router or HTTP control plane.

## Adapter Boundary Rule

Future non-Mac backends must fit the same adapter boundary used by the Mac-native path.
That includes future `vllm_metal`, `vllm_cuda`, and remote OpenAI-like workers. The
control plane should learn about capabilities, health, cost, and performance, not about
backend-specific runtime mechanics.

## Audit Summary

- The current repo already satisfies the Phase 0 foundation needed to begin Phase 1:
  typed schemas, a mock adapter, a pure router, a FastAPI gateway, telemetry scaffolding,
  and reproducible benchmark artifacts are present.
- No obvious Phase 0 implementation gap currently blocks Phase 1 kickoff.
- The main mismatch before this pass was documentation: the repo still described itself
  as Phase 0 even though the next intended slice is Phase 1.
