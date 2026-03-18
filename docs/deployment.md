# Phase 5 Deployment Guide

Switchyard Phase 5 is built around one deliberate boundary:

- Apple-Silicon model execution stays host-native by default.
- The control plane speaks a typed internal worker protocol over HTTP and can run locally,
  in Docker Compose, or in kind without Apple-specific runtime dependencies.

This guide is the main operator/developer map for that topology.

## Topology Summary

The default Phase 5 shape on a Mac looks like this:

```text
client
  -> Switchyard gateway / control plane
       -> router, admission, affinity, canary, shadow, telemetry
       -> explicit backend-instance inventory
       -> HTTP worker protocol
  -> host-native worker process on macOS
       -> MLX-LM or vLLM-Metal runtime
       -> Apple GPU
```

The same control plane can then move into:

- a local container image,
- a Docker Compose stack,
- a small kind deployment,
- later remote or cloud GPU workers that still speak the same internal worker protocol.

Note on config naming:

- the advanced routing/overload namespace is still `SWITCHYARD_PHASE4`,
- that is retained intentionally for backward compatibility with the existing control-plane
  settings model,
- Phase 5 adds deployment and worker-topology concerns around that control-plane core
  rather than renaming the full settings surface in place.

## 1. Host-Native Apple Workers

Phase 5 keeps Apple workers host-native by default because that is the stable,
Mac-friendly path for MLX-LM and vLLM-Metal.

Install the optional worker dependencies only on the host that will run them:

```bash
uv sync --dev --extra apple-workers
```

Start the current real workers side by side:

```bash
uv run switchyard-worker serve mlx-lm:chat-mlx --host 127.0.0.1 --port 8101 --warmup-mode eager
uv run switchyard-worker serve vllm-metal:chat-metal --host 127.0.0.1 --port 8102 --warmup-mode eager
```

The worker wrapper is intentionally small. It exposes:

- `GET /healthz`
- `GET /internal/worker/ready`
- `GET /internal/worker/capabilities`
- `POST /internal/worker/warmup`
- `POST /internal/worker/generate`
- `POST /internal/worker/generate/stream`
- `POST /v1/chat/completions`

Those are the only network contracts the portable control plane needs to speak.

## 2. Internal Worker Protocol

The internal worker protocol is Switchyard-specific and typed. It exists to keep the
control plane backend-agnostic while making worker reachability explicit.

What it covers:

- health,
- readiness and warm state,
- capabilities,
- warmup,
- generate,
- streaming generate.

Why it matters:

- the control plane can talk to a host-native Apple worker exactly the same way it could
  later talk to a remote CUDA or cloud GPU worker,
- transport failures and malformed responses are explicit,
- runtime inspection and benchmark artifacts can record real worker endpoints and
  instance inventory.

## 3. Containerized Control-Plane Services

The portable control-plane image does not require MLX-LM or vLLM-Metal.

Build it locally:

```bash
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
```

Smoke-check config inside the image:

```bash
docker run --rm switchyard/control-plane:dev check-config
```

Run the gateway directly from the image:

```bash
docker run --rm -p 8000:8000 switchyard/control-plane:dev
```

This image is the base for Compose and kind.

## 4. Compose Deployment

Compose is the easiest Phase 5 deployment path for a Mac-first developer because the
supporting services can be containerized while the Apple workers stay on the host.

Start the real host-native workers first, then start the Compose stack:

```bash
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase5_compose_m4pro.env \
  docker compose -f infra/compose/compose.yaml up -d
```

Send a request through the containerized gateway:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "hello from compose"}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Inspect runtime inventory and deployment diagnostics:

```bash
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
curl -s http://127.0.0.1:8000/admin/deployment | python -m json.tool
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
```

Compose defaults use `host.docker.internal` for worker addresses on macOS, but the
typed endpoint model stays generic. Later environments can use ordinary hostnames or
load-balancer addresses without code changes.

## 5. kind Deployment

kind gives Switchyard a local cluster-shaped deployment path without pretending Apple GPU
workers are in-cluster.

Bootstrap the cluster and local registry:

```bash
./scripts/kind-bootstrap.sh
```

Build and push the control-plane image into the local kind workflow:

```bash
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
./scripts/kind-push-control-plane.sh switchyard/control-plane:dev
```

Deploy the M4 Pro overlay:

```bash
./scripts/kind-deploy-control-plane.sh m4pro
kubectl -n switchyard port-forward service/switchyard-gateway 8000:8000
```

Send a request through the kind-hosted control plane:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "hello from kind"}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Inspect inventory and diagnostics:

```bash
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
curl -s http://127.0.0.1:8000/admin/deployment | python -m json.tool
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
```

The kind env examples also default to `host.docker.internal` on macOS:

- [phase5_kind_m4pro.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_kind_m4pro.env)
- [phase5_kind_smoke.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_kind_smoke.env)

## 6. Backend-Instance Inventory And Registration

Phase 5 gives Switchyard an explicit concept of backend instances instead of only
abstract backends.

Each configured or discovered instance can carry:

- a stable `instance_id`,
- a concrete endpoint URL,
- backend type and transport,
- source of truth such as static config or registration,
- health and last-seen metadata,
- optional tags such as `local`, `canary`, or `experimental`,
- image/build metadata where relevant.

That matters because:

- route decisions can be explained in terms of real worker endpoints,
- runtime inspection can show which instances are healthy right now,
- future cloud workers can register into the same inventory model,
- benchmark artifacts can preserve the exact worker topology that was exercised.

Inspect current runtime inventory:

```bash
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
```

Inspect deployment-aware config plus runtime alignment:

```bash
curl -s http://127.0.0.1:8000/admin/deployment | python -m json.tool
```

## 7. Deployment-Aware Benchmarking And Replay

Phase 5 benchmark artifacts are the source of truth for deployed topology, not ad hoc
logs.

Compose benchmark example:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_compose_benchmark_workload.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target compose \
  --deployment-profile compose \
  --config-profile-name phase5-compose-m4pro \
  --control-plane-image-tag switchyard/control-plane:dev \
  --warmup-request-count 1 \
  --markdown-report
```

kind smoke benchmark example:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_kind_smoke_workload.json \

## 8. Phase 7 Remote Worker Packaging

Phase 7 keeps the control plane portable while adding a credible path to later Linux and
NVIDIA-backed workers.

The repo now includes:

- [Dockerfile.remote-worker](/Users/rishivinodkumar/Atlas/infra/docker/Dockerfile.remote-worker)
  for a generic remote worker image,
- [compose.remote-worker.yaml](/Users/rishivinodkumar/Atlas/infra/compose/compose.remote-worker.yaml)
  as a Compose overlay for a stub remote worker,
- [phase7_remote_worker_stub_control_plane.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_control_plane.env)
  for control-plane static discovery,
- [phase7_remote_worker_stub_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_worker.env)
  for the CI-safe worker stub,
- [remote-workers.md](/Users/rishivinodkumar/Atlas/docs/remote-workers.md)
  for the discovery and authentication contract.

This path is intentionally split cleanly:

- the control plane only knows typed remote worker metadata and the shared HTTP protocol,
- the remote worker image can stay generic today,
- later `vllm_cuda` runtime dependencies can be added to a derived Linux worker image
  without changing the control-plane contract or forcing CUDA into CI.
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target kind \
  --deployment-profile kind \
  --config-profile-name phase5-kind-smoke \
  --control-plane-image-tag localhost:5001/switchyard/control-plane:dev \
  --markdown-report
```

Replay against a deployed topology:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target compose \
  --deployment-profile compose \
  --config-profile-name phase5-compose-m4pro \
  --control-plane-image-tag switchyard/control-plane:dev \
  --policy balanced \
  --replay-mode fixed_concurrency \
  --concurrency 2 \
  --markdown-report
```

Artifacts should now preserve:

- deployment target,
- deployment profile,
- config/profile name,
- control-plane image metadata,
- worker-instance inventory snapshot when available.

## 8. Diagnostics And Preflight Tooling

Phase 5 ships two lightweight diagnostic paths:

- `switchyard-control-plane doctor`
  Local preflight from config, or remote diagnostics when pointed at a deployed gateway.
- `GET /admin/deployment`
  Runtime-truth deployment diagnostics from inside the running control plane.

Local preflight:

```bash
uv run switchyard-control-plane doctor
```

Fail the command when the diagnostic report contains unreachable or error states:

```bash
uv run switchyard-control-plane doctor --fail-on-issues
```

Remote deployment diagnostics:

```bash
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
```

These diagnostics report at minimum:

- effective deployment profile,
- configured worker endpoints,
- reachability of worker health endpoints,
- current backend-instance inventory,
- image/build metadata where available,
- supporting-service reachability when the repo defines a clear endpoint target.

If a check cannot be verified, the output should say `not_verifiable` or
`not_configured` rather than silently marking it healthy.

## 9. Why This Prepares For Later Cloud GPU Workers

Phase 5 does not add cloud GPU workers yet, but it builds the pieces needed for them:

- the control plane no longer assumes model execution is in-process,
- worker addressing is explicit and typed,
- instance inventory can track many concrete workers behind one logical alias,
- containerized control-plane paths already exist,
- benchmark artifacts can preserve deployed topology,
- diagnostics and runtime inspection already reason about remote endpoints.

That means a later `vllm_cuda` or cloud GPU worker can plug into an existing boundary
instead of forcing a control-plane redesign.

## Experimental Note

Future local variants such as minikube, krunkit, or in-cluster Apple GPU experiments are
explicitly non-core for Phase 5. They may be interesting later, but the supported Phase 5
path remains:

- host-native Apple workers on macOS,
- portable control-plane services in local Python, Compose, or kind,
- explicit network endpoints between the two.
