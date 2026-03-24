# Remote Worker Packaging And Discovery

For the broader Phase 7 hybrid operator/developer guide, start with
[hybrid-workers.md](/Users/rishivinodkumar/Atlas/docs/hybrid-workers.md).

For the Phase 8 real-cloud operator/developer guide, start with
[real-cloud-workers.md](/Users/rishivinodkumar/Atlas/docs/real-cloud-workers.md).

This document now covers both:

- the Phase 7 CI-safe stub path, and
- the Phase 8 first practical bring-up path for a real rented Linux/NVIDIA
  `vllm_cuda` worker.

Treat this file as the packaging and discovery appendix. The new real-cloud guide is the
better entry point when you need rollout, evidence, alias, and operator-context
explained together.

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
- `SWITCHYARD_REMOTE_WORKER_START_MODE=vllm_cuda`
  Runs the concrete dependency-gated `switchyard-vllm-cuda-worker` path.
  The entrypoint now runs a preflight check first so missing `vllm`,
  missing `nvidia-smi`, or invalid rented-GPU settings fail fast with a concrete error.

The image intentionally does not install CUDA-only dependencies. A later
`vllm_cuda`-capable image should derive from this packaging target or mirror its
entrypoint contract while adding the Linux/NVIDIA runtime pieces separately.

### Concrete `vllm_cuda` Image Target

The same Dockerfile now exposes a concrete `runtime-vllm-cuda` target for the
first rented-GPU worker.

Build targets:

- generic stub image:

```bash
docker build -f infra/docker/Dockerfile.remote-worker -t switchyard/remote-worker:dev .
```

- concrete rented-GPU image:

```bash
docker build -f infra/docker/Dockerfile.remote-worker \
  --target runtime-vllm-cuda \
  -t switchyard/remote-worker-vllm-cuda:dev .
```

The concrete target installs the optional `vllm-cuda` dependency group while
keeping the default control-plane workspace and CI path GPU-free.

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
- first rented-GPU engine knobs:
  tensor parallel size,
  GPU memory utilization,
  maximum model length,
  and `trust_remote_code`,
- future control-plane bootstrap metadata: control-plane URL, auth mode, registration token,
  enrollment token, and heartbeat interval.

The concrete `vllm_cuda` path now validates the most important startup invariants:

- `backend_type` must be `vllm_cuda`,
- `device_class` must be `nvidia_gpu`,
- `engine_type` must be `vllm` or `vllm_cuda`,
- `gpu_count` must be at least `1`,
- `tensor_parallel_size` must not exceed `gpu_count`.

Those checks run both when building the concrete worker app and in the new CLI
preflight command, so obviously broken rented-GPU settings fail before the worker
starts serving.

Example worker env file:

- [phase7_remote_worker_stub_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_worker.env)
- [phase8_vllm_cuda_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase8_vllm_cuda_worker.env)
- [phase8_vllm_cuda_worker.secrets.env.example](/Users/rishivinodkumar/Atlas/docs/examples/phase8_vllm_cuda_worker.secrets.env.example)

The Phase 8 split is intentional:

- the main env file carries non-secret runtime identity, placement, and engine settings,
- the secrets env file is where control-plane registration tokens and model-access
  credentials belong.

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

Compose overlay for a first rented-GPU `vllm_cuda` worker:

- [compose.vllm-cuda-worker.yaml](/Users/rishivinodkumar/Atlas/infra/compose/compose.vllm-cuda-worker.yaml)

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

## Phase 8 First Rented-GPU Bring-Up

This is the smallest realistic path for a human operator bringing up the first real
cloud-backed worker on a rented Linux/NVIDIA host.

1. Build the concrete worker image:

```bash
docker build -f infra/docker/Dockerfile.remote-worker \
  --target runtime-vllm-cuda \
  -t switchyard/remote-worker-vllm-cuda:dev .
```

2. Prepare env files:

- config:
  [phase8_vllm_cuda_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase8_vllm_cuda_worker.env)
- secrets:
  [phase8_vllm_cuda_worker.secrets.env.example](/Users/rishivinodkumar/Atlas/docs/examples/phase8_vllm_cuda_worker.secrets.env.example)

3. Run worker preflight before serving:

```bash
uv run switchyard-vllm-cuda-worker check-config \
  --verify-runtime-import \
  --require-nvidia-smi \
  --fail-on-issues
```

4. Start the worker with the Compose reference overlay:

```bash
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase7_remote_worker_stub_control_plane.env \
SWITCHYARD_REMOTE_WORKER_ENV_FILE=../../docs/examples/phase8_vllm_cuda_worker.env \
SWITCHYARD_REMOTE_WORKER_SECRET_ENV_FILE=/absolute/path/to/phase8_vllm_cuda_worker.secrets.env \
docker compose -f infra/compose/compose.yaml -f infra/compose/compose.vllm-cuda-worker.yaml up -d
```

5. Check worker health and readiness:

```bash
curl -s http://127.0.0.1:8090/healthz | python -m json.tool
curl -s http://127.0.0.1:8090/internal/worker/ready | python -m json.tool
```

6. Check the control plane sees the worker path honestly:

```bash
curl -s http://127.0.0.1:8000/admin/remote-workers | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid | python -m json.tool
```

## Safe Real-Cloud Rollout Runbook

The first rented GPU should not go straight from "registered" to "fully serving."
Switchyard now exposes an explicit runtime gate for `canary-only` cloud workers so the
primary path can stay bounded and reversible.

1. Mark the new worker `canary-only` before widening traffic:

```bash
curl -sS http://127.0.0.1:8000/admin/remote-workers/worker-1/canary-only \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"reason":"phase8 first rented GPU canary"}' \
  | python -m json.tool
```

2. Inspect current rollout posture:

```bash
curl -s http://127.0.0.1:8000/admin/hybrid/cloud-rollout | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid | python -m json.tool
```

3. Open a small deterministic rollout window:

```bash
curl -sS http://127.0.0.1:8000/admin/hybrid/cloud-rollout \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"canary_percentage":5.0,"kill_switch_enabled":false}' \
  | python -m json.tool
```

4. Observe whether the worker is merely eligible or actually wins routing under the
current policy:

```bash
curl -i -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: phase8-rollout-check-001' \
  -H 'x-switchyard-routing-policy: burst_to_remote' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Summarize the cloud rollout posture."}],
    "max_output_tokens": 64
  }'
```

Inspect `x-switchyard-route-decision`:

- `protected_backends` still listing the cloud worker means the rollout gate blocked it,
- `considered_backends` including the cloud worker means it was eligible for primary
  routing,
- the final `backend_name` still depends on the active policy and observed scores.

5. Roll back immediately if transport, spend, or correctness looks wrong:

```bash
curl -sS http://127.0.0.1:8000/admin/hybrid/cloud-rollout \
  -H 'content-type: application/json' \
  -d '{"kill_switch_enabled":true}' \
  | python -m json.tool

curl -sS http://127.0.0.1:8000/admin/hybrid/remote-enabled \
  -H 'content-type: application/json' \
  -d '{"enabled":false,"reason":"phase8 rollback"}' \
  | python -m json.tool
```

6. Once the worker is proven healthy, either widen the deterministic percentage in small
steps or clear the `canary-only` tag after an operator review:

```bash
curl -sS http://127.0.0.1:8000/admin/remote-workers/worker-1/canary-only \
  -H 'content-type: application/json' \
  -d '{"enabled":false,"reason":"canary complete"}' \
  | python -m json.tool
```

### Failure Modes

Common first-bring-up failures now fail with clearer messages:

- `vllm is not installed; install it to enable the vLLM-CUDA worker`
  The wrong image target was built or the optional CUDA worker dependency set is missing.
- `nvidia-smi was not found`
  The host or container runtime is not exposing the NVIDIA driver stack into the worker.
- `tensor_parallel_size must not exceed gpu_count`
  The worker env contract is internally inconsistent before model load even starts.
- `vLLM engine initialization failed; verify NVIDIA drivers, GPU visibility, and the container GPU runtime on the rented host`
  `vllm` imported, but actual CUDA-backed engine startup failed.
- `vLLM engine initialization failed while resolving model assets`
  Model identifier, Hugging Face token, or remote-code policy needs attention.

### Operator Notes

- `/healthz` is a liveness signal.
- `/internal/worker/ready` is the better readiness gate for deployments because it
  reflects warmup and draining state.
- The Compose reference uses separate config and secrets env files on purpose.
- The reference overlay uses `gpus: all` because it targets the simplest single-host
  Docker bring-up, not a production fleet manager.
