# Phase 0 Checklist

- [x] Clean Python 3.12 workspace with `uv`, linting, typing, and tests.
- [x] Shared domain model for requests, responses, routing, backend health, and benchmarks.
- [x] `BackendAdapter` contract and deterministic `MockBackendAdapter`.
- [x] FastAPI gateway with `GET /healthz`, `GET /readyz`, and `POST /v1/chat/completions`.
- [x] Gateway serves a chat-completions-style response via the mock backend.
- [x] Routing logic lives outside the HTTP layer.
- [x] Structured logging and basic telemetry hooks exist.
- [x] Benchmark artifacts can be written to reproducible JSON.
- [x] Documented local dev boot flow.
