# Phase 0 Architecture

Switchyard Phase 0 is a single Python workspace that establishes the contracts and local
control-plane shape without committing to any real model-serving runtime yet.

## Main Components

- `switchyard.schemas`: shared typed models for chat requests and responses, backend
  capabilities and health, routing decisions, and benchmark artifacts.
- `switchyard.adapters`: the backend adapter contract, a small registry, and a
  deterministic mock adapter used by tests, routing, and benchmark runs.
- `switchyard.router`: a pure Python router service with deterministic policy helpers.
  It evaluates backend capabilities and health and returns a typed `RouteDecision`.
- `switchyard.gateway`: a thin FastAPI layer that creates or propagates request IDs,
  delegates routing to `RouterService`, invokes the chosen adapter, and returns the typed
  chat response.
- `switchyard.logging` and `switchyard.telemetry`: Phase 0 observability scaffolding.
  Logging is structured with `structlog`; telemetry wraps OpenTelemetry setup and a few
  local-friendly counters and histograms.
- `switchyard.bench`: lightweight synthetic benchmarking that exercises the service layer
  and writes reproducible JSON artifacts.

## Request Flow

1. A request enters the FastAPI gateway.
2. Middleware assigns or propagates `x-request-id`.
3. The route handler builds a `RequestContext` and asks `RouterService` for a
   `RouteDecision`.
4. The selected adapter is fetched from the registry and asked to `generate`.
5. The gateway returns the typed chat completion response.
6. Logging and telemetry capture request, routing, and backend metadata along the way.

## Why This Shape

- The router is independent of FastAPI so it can be tested directly and reused by
  benchmarks or future workers.
- Backends live behind a narrow adapter contract, which keeps Apple-specific or
  future CUDA-specific logic at the boundary instead of in the control plane.
- Benchmark artifacts use the same schema layer as the rest of the system, which keeps
  request and result data consistent across gateway and offline evaluation paths.

## Portability Forward

Phase 0 is Mac-first, but not Mac-locked.

- The control plane does not encode runtime details like Metal vs CUDA in the routing
  service.
- Backend-specific concerns belong in adapters and capability declarations.
- Future backends such as MLX-LM, vLLM-Metal, remote OpenAI-like services, or CUDA-backed
  workers can implement the same adapter contract and register capabilities without
  rewriting the gateway or schema layer.

That separation is the main portability bet for the next phases.
