# Switchyard

Switchyard is a Mac-first, backend-agnostic inference fabric. The long-term goal is a
portable control plane that can route inference requests across local and remote backends
without baking hardware-specific assumptions into the core.

Phase 0 is intentionally narrow. The repo currently includes:
- typed shared schemas for chat, backend, routing, and benchmark domains,
- a deterministic `MockBackendAdapter`,
- a pure Python router with a few static policies,
- a thin FastAPI gateway with health, readiness, and chat completions,
- structured logging, telemetry scaffolding, and a lightweight benchmark artifact path.

No real MLX, vLLM, CUDA, or cloud-serving integration is included yet.

## Local Setup

```bash
uv sync --dev
cp .env.example .env
```

## Useful Commands

```bash
make check
make serve
make bench-smoke
```

Optional local infra for later phases:

```bash
docker compose -f infra/compose/compose.yaml --profile optional up -d
```

This is not required for current Phase 0 development.

If you prefer direct commands:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
uv run python -m switchyard.bench.cli --request-count 3
```

## Current Scope

- Phase 0 foundation and contracts only
- Mock backends only
- Routing separated from HTTP
- Local-friendly telemetry and reproducible benchmark artifacts

## Repo Guide

- [`docs/architecture.md`](/Users/rishivinodkumar/Atlas/docs/architecture.md): Phase 0 architecture
- [`docs/infra.md`](/Users/rishivinodkumar/Atlas/docs/infra.md): optional local infra scaffolding
- [`docs/phase0.md`](/Users/rishivinodkumar/Atlas/docs/phase0.md): checklist and implementation status
- [`docs/adr/0001-single-python-workspace.md`](/Users/rishivinodkumar/Atlas/docs/adr/0001-single-python-workspace.md): key Phase 0 decision
