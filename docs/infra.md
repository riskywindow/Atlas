# Optional Local Infra

The files under [`infra/compose`](/Users/rishivinodkumar/Atlas/infra/compose) are optional
local scaffolding for later phases. They are not required for normal Phase 0 development,
tests, routing work, gateway work, or benchmark smoke runs.

## Included Services

- Postgres: placeholder state store for future metadata, experiment, or job persistence
- Redis: placeholder cache/queue service for future coordination needs
- OpenTelemetry Collector: local receiver so traces and metrics can be pointed somewhere
  concrete during later observability work

## What Is Used Now

- Nothing in the compose stack is required by the current code path.
- Phase 0 works with local Python only: mock adapters, in-process routing, FastAPI gateway,
  and JSON benchmark artifacts.

## What This Is For Later

- Postgres and Redis are future-facing infrastructure hooks, not active dependencies.
- The collector config is intentionally minimal and exports to the debug logger only.
- This is development scaffolding, not a production deployment shape.

## Optional Startup

```bash
docker compose -f infra/compose/compose.yaml --profile optional up -d
```

To stop it:

```bash
docker compose -f infra/compose/compose.yaml --profile optional down
```
