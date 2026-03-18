# Remote Worker Packaging And Discovery

This document defines the Phase 7 packaging and integration boundary for later
Linux/NVIDIA workers without requiring CUDA or rented GPUs today.

## Goals

- keep the control plane backend-agnostic,
- provide a reviewable Linux/container packaging path for remote workers,
- preserve a CI-safe stub path that exercises the same HTTP worker protocol,
- document the discovery and authentication contract the control plane already expects.

## Packaging Targets

### Generic Remote Worker Image

The repo now includes [Dockerfile.remote-worker](/Users/rishivinodkumar/Atlas/infra/docker/Dockerfile.remote-worker).
It packages the shared worker protocol runtime without Apple-specific dependencies.

Default behavior:

- starts a deterministic fake remote worker,
- serves the same internal HTTP protocol as real workers,
- is suitable for local integration tests, Compose smoke runs, and CI.

Startup modes:

- `SWITCHYARD_REMOTE_WORKER_START_MODE=fake`
  Uses `switchyard-fake-remote-worker`.
- `SWITCHYARD_REMOTE_WORKER_START_MODE=configured`
  Uses `switchyard-worker serve $SWITCHYARD_REMOTE_WORKER_TARGET`.
  This exists as the clean extension point for a later real Linux worker adapter.

The image intentionally does not install CUDA-only dependencies. A later
`vllm_cuda`-capable image should derive from this packaging target or mirror its
entrypoint contract while adding the Linux/NVIDIA runtime pieces separately.

## Worker Runtime Environment Contract

The container/runtime contract is modeled by
[config.py](/Users/rishivinodkumar/Atlas/src/switchyard/worker/config.py) as
`RemoteWorkerRuntimeSettings`.

Current fields cover:

- bind settings: host, port, log level,
- worker identity: worker name and worker ID,
- advertised backend metadata: backend type, engine type, device class,
- served model and serving target,
- fake-runtime behavior for CI-safe stubs: latency, queue depth, concurrency, streaming,
- future control-plane bootstrap metadata: control-plane URL, auth mode, registration token,
  enrollment token, and heartbeat interval.

The stub runtime uses the typed backend/device/model fields today. The control-plane
bootstrap fields are documented now so later registration sidecars or bootstrap scripts
can adopt a stable contract without reworking the packaging surface.

Example worker env file:

- [phase7_remote_worker_stub_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_worker.env)

## Control-Plane Discovery Contract

Switchyard supports two honest discovery paths for remote workers.

### 1. Static Discovery

The control plane can be configured with an explicit remote worker inventory through
`SWITCHYARD_LOCAL_MODELS` using:

- `worker_transport=http`,
- `execution_mode=remote_worker`,
- explicit remote `instances`,
- remote placement, cost, trust, and network metadata.

This is the easiest CI-safe path because it requires no registration bootstrap.

Example control-plane env file:

- [phase7_remote_worker_stub_control_plane.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_control_plane.env)

The example is intentionally shaped like a future `vllm_cuda` worker while still being
backed by the fake worker image.

### 2. Dynamic Registration

The control plane also exposes typed lifecycle endpoints:

- `POST /internal/control-plane/remote-workers/register`
- `POST /internal/control-plane/remote-workers/heartbeat`
- `POST /internal/control-plane/remote-workers/deregister`

Authentication modes are:

- `none`
- `static_token`
- `signed_enrollment`

Operators can inspect the resulting lifecycle and health state through:

- `GET /admin/remote-workers`
- `GET /admin/runtime`
- `GET /admin/hybrid`

The stub worker image does not auto-register yet. That boundary is intentional: the
control plane contract is stable and testable today, while later Linux worker bootstrap
logic can be added without changing the protocol or packaging split.

## Example Deployment Templates

Compose overlay for a stub remote worker:

- [compose.remote-worker.yaml](/Users/rishivinodkumar/Atlas/infra/compose/compose.remote-worker.yaml)

kind/Kubernetes template for a stub remote worker:

- [remote-worker-stub.yaml](/Users/rishivinodkumar/Atlas/infra/kind/remote-worker-stub.yaml)

These templates keep the worker image separate from the control-plane image and preserve
the typed remote-worker HTTP boundary.

## Local Or CI Stub Path

Build the remote worker image:

```bash
docker build -f infra/docker/Dockerfile.remote-worker -t switchyard/remote-worker:dev .
```

Run the base Compose stack with the remote-worker overlay:

```bash
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase7_remote_worker_stub_control_plane.env \
SWITCHYARD_REMOTE_WORKER_ENV_FILE=../../docs/examples/phase7_remote_worker_stub_worker.env \
docker compose -f infra/compose/compose.yaml -f infra/compose/compose.remote-worker.yaml up -d
```

Inspect the topology:

```bash
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
curl -s http://127.0.0.1:8000/admin/remote-workers | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid | python -m json.tool
```

This path is intentionally honest:

- the remote worker is a stub, not production evidence,
- the control plane still sees a remote `vllm_cuda`-shaped worker over the real protocol,
- later Linux/NVIDIA images can replace the stub worker without changing the control-plane
  transport contract.
