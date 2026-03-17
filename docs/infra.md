# Optional Local Infra

The files under [`infra/compose`](/Users/rishivinodkumar/Atlas/infra/compose) and
[`infra/grafana`](/Users/rishivinodkumar/Atlas/infra/grafana), and
[`infra/kind`](/Users/rishivinodkumar/Atlas/infra/kind) are optional local
scaffolding. They are not required for the normal host-native worker path, tests,
routing work, gateway work, or benchmark runs.

Phase 5 adds a new constraint around this folder: the control plane must be deployable
without Apple-specific runtime dependencies, while real Apple-Silicon model workers stay
host-native by default unless a future deployment path explicitly says otherwise.

## Included Services

- Gateway control plane: portable image running the FastAPI gateway and control-plane CLI
- Postgres: placeholder state store for future metadata, experiment, or job persistence
- Redis: placeholder cache/queue service for future coordination needs
- OpenTelemetry Collector: local receiver so traces and metrics can be pointed somewhere
  concrete during later observability work
- Prometheus: optional scrape target for the gateway `/metrics` surface and OTEL collector
- Grafana: optional dashboard viewer for the Prometheus metrics path
- Grafana dashboard JSON: an optional importable dashboard for the local `/metrics` path

## What Is Used Now

- Nothing in the compose stack is required by the current code path.
- The current default path still works with local Python only: adapters, in-process
  routing, the FastAPI gateway, optional `/metrics`, structured logs, and JSON benchmark
  artifacts.

## Optional Metrics Path

Switchyard already exposes a lightweight Prometheus-style metrics endpoint when
`SWITCHYARD_METRICS_ENABLED=true`.

Start the gateway with metrics enabled:

```bash
SWITCHYARD_METRICS_ENABLED=true \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Inspect metrics directly:

```bash
curl -s http://127.0.0.1:8000/metrics
```

Useful local metrics include:
- `switchyard_backend_request_latency_ms`
- `switchyard_backend_ttft_ms`
- `switchyard_backend_tokens_per_second`
- `switchyard_backend_output_tokens`
- `switchyard_backend_warmup_latency_ms`
- `switchyard_requests_total`

This path is enough for local inspection and for the Phase 1 benchmark runner. No
collector, Prometheus server, or Grafana instance is required.

## Optional Grafana Dashboard

If you already have a local Grafana + Prometheus setup, you can import:

- [switchyard-phase1-local-dashboard.json](/Users/rishivinodkumar/Atlas/infra/grafana/switchyard-phase1-local-dashboard.json)

The dashboard visualizes:
- backend request latency,
- TTFT,
- tokens per second,
- request count by backend,
- warmup latency and readiness-related events.

The dashboard is intentionally static and small. It is an optional convenience artifact,
not part of the core developer workflow.

## What This Is For Later

- Postgres and Redis are future-facing infrastructure hooks, not active dependencies.
- The collector config is intentionally minimal and exports to the debug logger only.
- The Grafana dashboard is only a local visualization seed.
- This is development scaffolding, not a production deployment shape.

## Compose Deployment

The Compose stack is now aimed at the Phase 5 Mac-first deployment model:

- the control plane runs in containers,
- Apple-Silicon workers remain host-native by default,
- worker endpoints are configured explicitly through `SWITCHYARD_LOCAL_MODELS`,
- the macOS local-dev default uses `host.docker.internal`, but no code depends on that
  name specifically.
- supporting infra stays on the internal Compose network by default to avoid colliding
  with already-running local Postgres or Redis instances.

Two env files are provided:

- [phase5_compose_m4pro.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_compose_m4pro.env)
  for a realistic M4 Pro setup with host-native MLX and vLLM-Metal workers.
- [phase5_compose_smoke.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_compose_smoke.env)
  for a GPU-free smoke path using a host-native mock worker.

Validate the rendered Compose config:

```bash
docker compose -f infra/compose/compose.yaml config
```

Start the core stack:

```bash
docker compose -f infra/compose/compose.yaml up -d
```

Start the stack plus Prometheus and Grafana:

```bash
docker compose -f infra/compose/compose.yaml --profile observability up -d
```

### Real Worker Workflow

Run host-native Apple workers in separate terminals:

```bash
uv run switchyard-worker serve mlx-lm:mlx-chat --host 127.0.0.1 --port 8101 --warmup-mode eager
uv run switchyard-worker serve vllm-metal:vllm-chat --host 127.0.0.1 --port 8102 --warmup-mode eager
```

Then verify containerized readiness and routing:

```bash
curl -s http://127.0.0.1:8000/readyz | python -m json.tool
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
curl -s http://127.0.0.1:8000/admin/deployment | python -m json.tool
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chat-shared","messages":[{"role":"user","content":"hello from compose"}]}' | python -m json.tool
```

### Smoke Workflow

Run a lightweight host-native mock worker:

```bash
uv run python scripts/host_native_mock_worker.py
```

In another terminal, point Compose at the smoke env:

```bash
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase5_compose_smoke.env \
  docker compose -f infra/compose/compose.yaml up -d
```

Then verify the route:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chat-smoke","messages":[{"role":"user","content":"hello from compose"}]}' | python -m json.tool
```

## Stop The Stack

```bash
docker compose -f infra/compose/compose.yaml down
```

## Portable Control-Plane Image

Phase 5 adds a portable control-plane image definition at
[infra/docker/Dockerfile.control-plane](/Users/rishivinodkumar/Atlas/infra/docker/Dockerfile.control-plane).
It intentionally installs only the base control-plane dependencies from `uv.lock` and
does not include `mlx-lm` or `vllm`.

Build it locally:

```bash
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
```

Smoke-check settings and imports:

```bash
docker run --rm switchyard/control-plane:dev check-config
```

Run the gateway:

```bash
docker run --rm -p 8000:8000 switchyard/control-plane:dev
```

The image exposes a health check against `/healthz`. For deployment readiness, use
`/readyz` after the process starts and the configured adapters are available.

For a local preflight before startup:

```bash
uv run switchyard-control-plane doctor
```

For deployed diagnostics against a running control plane:

```bash
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
```

## kind Deployment

The kind path uses plain Kubernetes YAML with small Kustomize overlays under
[infra/kind](/Users/rishivinodkumar/Atlas/infra/kind). The strategy is intentionally
simple:

- one base deployment and service for the gateway,
- one `smoke` overlay,
- one `m4pro` overlay,
- config injected from env-file-driven ConfigMaps,
- a local registry workflow for reproducible image pushes into kind.

Bootstrap the cluster and local registry:

```bash
./scripts/kind-bootstrap.sh
```

Build and push the control-plane image:

```bash
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
./scripts/kind-push-control-plane.sh switchyard/control-plane:dev
```

Render the manifests locally:

```bash
kubectl kustomize infra/kind/overlays/smoke
kubectl kustomize infra/kind/overlays/m4pro
```

Deploy the smoke overlay:

```bash
./scripts/kind-deploy-control-plane.sh smoke
```

Port-forward the gateway:

```bash
kubectl -n switchyard port-forward service/switchyard-gateway 8000:8000
```

Once port-forwarded, the same deployment diagnostics path works:

```bash
curl -s http://127.0.0.1:8000/admin/deployment | python -m json.tool
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
```

### Worker Endpoints On macOS

The kind env files use `host.docker.internal` as the default worker address:

- [phase5_kind_smoke.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_kind_smoke.env)
- [phase5_kind_m4pro.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_kind_m4pro.env)

That keeps the endpoint abstraction generic because Switchyard still just consumes an
explicit URL. If your local kind pods cannot resolve `host.docker.internal`, replace the
`base_url` entries with an ordinary reachable host address and redeploy. No code changes
should be required.

### kind Smoke Workflow

The repo includes a smoke helper:

```bash
./scripts/kind-smoke.sh
```

It starts the host-native mock worker, bootstraps kind, builds and pushes the
control-plane image, deploys the smoke overlay, port-forwards the gateway, and sends a
request through the cluster-shaped control plane to the host-native worker endpoint.
