# Real Cloud Workers

This is the Phase 8 developer/operator guide for the first real cloud-backed worker
path. For lower-level packaging details, keep
[remote-workers.md](/Users/rishivinodkumar/Atlas/docs/remote-workers.md) nearby. For
the system model, read
[architecture.md](/Users/rishivinodkumar/Atlas/docs/architecture.md).

## What Phase 8 Adds

Phase 8 keeps the Mac-first local path as the default, but it proves that the same
control plane can also drive a real Linux/NVIDIA worker path.

Concretely, Phase 8 adds:

- the first concrete Linux/NVIDIA worker runtime behind the generic worker contract,
- explicit real-cloud worker registration and inventory,
- alias routing across local Apple and remote cloud workers,
- route and artifact semantics that keep observed cloud evidence separate from
  estimates, predictors, and mocks,
- bounded rollout controls for the first rented GPU: canary-only posture, rollout
  percentage, budget, quarantine, drain, and local-only failback.

## The First Real Linux/NVIDIA Worker Path

The first real remote runtime is the concrete `vllm_cuda` worker path:

- runtime CLI: `switchyard-vllm-cuda-worker`
- worker app/runtime modules:
  - `switchyard.worker.vllm_cuda`
  - `switchyard.worker.vllm_cuda_cli`
  - `switchyard.runtime.vllm_cuda`
- image target:
  [infra/docker/Dockerfile.remote-worker](/Users/rishivinodkumar/Atlas/infra/docker/Dockerfile.remote-worker)
  with `--target runtime-vllm-cuda`

Why it exists:

- Phase 8 needed one honest, concrete Linux/NVIDIA path instead of another stub.
- The control plane still needs a stable runtime/backend label for inventory,
  registration, artifacts, and diagnostics.
- The router and gateway still stay backend-agnostic because the concrete runtime is
  isolated at the worker/runtime boundary.

In practice that means:

- the control plane sees typed capability, health, placement, and cost metadata,
- the worker runtime knows about `vllm`, CUDA, GPU count, and engine bring-up,
- the control plane never needs a direct CUDA-specific branch in route logic.

## Runtime And Backend Labels

Phase 8 uses concrete labels intentionally:

- `backend_type=vllm_cuda`
- `device_class=nvidia_gpu`
- `engine_type=vllm` or `vllm_cuda`
- control-plane backend name for a registered worker:
  `remote-worker:<worker_name>`

This split matters:

- `vllm_cuda` is the runtime identity and packaging boundary,
- `remote-worker:<worker_name>` is the routable deployment identity inside the control
  plane,
- the logical alias remains the client-facing contract.

Example:

- logical alias: `chat-shared`
- local deployments:
  - `mlx-lm:chat-mlx`
  - `vllm-metal:chat-metal`
- remote deployment:
  - `remote-worker:remote-chat`
- concrete runtime behind that remote deployment:
  - `vllm_cuda`

## Registration And Inventory

Real cloud workers behave like first-class topology members, not opaque external
providers.

Two discovery paths exist:

- static inventory in control-plane config,
- dynamic registration through:
  - `POST /internal/control-plane/remote-workers/register`
  - `POST /internal/control-plane/remote-workers/heartbeat`
  - `POST /internal/control-plane/remote-workers/deregister`

Operator/runtime views:

- `GET /admin/remote-workers`
- `GET /admin/runtime`
- `GET /admin/hybrid`

The inventory keeps these facts explicit:

- worker ID and worker name,
- backend/runtime identity,
- serving targets,
- endpoint and transport,
- health, readiness, queue depth, and active requests,
- placement and cost metadata,
- quarantine, canary-only, draining, and lifecycle state.

That is what lets Phase 8 treat a real cloud worker as normal topology rather than a
transport exception.

## Aliases Across Local And Remote Workers

Logical aliases deliberately span multiple backends.

Example:

- client sends `model="chat-shared"`
- the registry may resolve:
  - `mlx-lm:chat-mlx`
  - `vllm-metal:chat-metal`
  - `remote-worker:remote-chat`

This is why alias compatibility is explicit instead of heuristic:

- clients stay pinned to the logical contract,
- operators can add or remove real cloud workers without changing client payloads,
- benchmarks and replays can compare alias routing against pinned backends honestly.

## How Routing Uses Real Observed Cloud Signals

Routing still reasons over typed signals, not hardware brands.

For a real cloud worker, the control plane can consume:

- observed health and readiness,
- observed queue depth and active request count,
- observed runtime identity and GPU metadata,
- observed placement metadata such as provider, region, and zone,
- observed cost-profile metadata when the worker reports it,
- remote spillover guardrails such as budget, concurrency, cooldown, and remote enable
  state.

Routing does not do this:

- branch on CUDA internals inside the router,
- guess that a worker is safe because it "looks like" a cloud backend,
- blur real runtime observations into predicted or mocked evidence.

## Evidence And Artifact Semantics

Phase 8 keeps evidence sources separate on purpose.

Observed cloud evidence means:

- the request really executed on a cloud/remote worker,
- runtime placement or cost metadata came from the exercised worker or observed instance.

Estimated or non-observed evidence means:

- configured placement/cost assumptions,
- predictor output,
- simulation estimates,
- mock/stub cloud paths.

That separation is visible in:

- `/admin/hybrid` recent route examples,
- runtime summaries and recent cloud evidence counts,
- benchmark artifacts,
- replay outputs,
- markdown reports derived from those artifacts.

Practical consequence:

- a report can say "remote worker was eligible" without pretending that a real cloud run
  happened,
- a mock `vllm_cuda`-shaped run does not count as direct observed cloud evidence,
- a real cloud benchmark stays distinguishable from local-only or estimated comparisons.

## Rollout Safety Model

The first real cloud rollout is intentionally bounded and reversible.

### Controls

- `canary-only`
  Keeps a new cloud worker out of normal primary traffic by default.
- `/admin/hybrid/cloud-rollout`
  Adds the explicit outer gate for canary-only workers.
- remote spillover budget and concurrency limits
  Bound how much cloud traffic can happen even when remote is enabled.
- quarantine
  Removes a failing worker from eligibility without deleting inventory truth.
- drain
  Stops new placement while preserving operator-visible lifecycle state.
- `/admin/hybrid/remote-enabled`
  Immediate failback to local-only behavior.

### Current Behavior

- canary-only workers stay blocked unless cloud rollout allows them,
- kill switch overrides canary selection,
- repeated remote transport failures can auto-quarantine a matching registered worker,
- drained or quarantined workers are blocked before route selection,
- shadow traffic never changes the primary user-visible response,
- session affinity is preserved where practical, but the control plane avoids rebinding
  affinity on canary, spillover, or operator-override routes that are meant to stay
  temporary or exceptional.

### Route Diagnostics

Route headers and `/admin/hybrid` examples now make the reason explicit when remote
routing happened because of:

- `canary`
- `spillover`
- `operator_override`

They also show rollout disposition and blocked/protected reasons when the cloud worker
was eligible in principle but intentionally kept out of primary traffic.

## Build And Bring Up The First Rented GPU

### 1. Build The Concrete Worker Runtime

```bash
docker build -f infra/docker/Dockerfile.remote-worker \
  --target runtime-vllm-cuda \
  -t switchyard/remote-worker-vllm-cuda:dev .
```

### 2. Validate The Worker Contract Before Serving

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run switchyard-vllm-cuda-worker check-config \
  --verify-runtime-import \
  --require-nvidia-smi \
  --fail-on-issues
```

### 3. Start The Worker

Reference env files:

- [phase8_vllm_cuda_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase8_vllm_cuda_worker.env)
- [phase8_vllm_cuda_worker.secrets.env.example](/Users/rishivinodkumar/Atlas/docs/examples/phase8_vllm_cuda_worker.secrets.env.example)

Compose example:

```bash
SWITCHYARD_REMOTE_WORKER_ENV_FILE=../../docs/examples/phase8_vllm_cuda_worker.env \
SWITCHYARD_REMOTE_WORKER_SECRET_ENV_FILE=/absolute/path/to/phase8_vllm_cuda_worker.secrets.env \
docker compose -f infra/compose/compose.yaml -f infra/compose/compose.vllm-cuda-worker.yaml up -d
```

### 4. Register The First Cloud Worker

If the worker is not auto-registering yet, register it explicitly:

```bash
curl -sS http://127.0.0.1:8000/internal/control-plane/remote-workers/register \
  -H 'content-type: application/json' \
  -d '{
    "worker_id": "gpu-canary-001",
    "worker_name": "remote-chat",
    "backend_type": "vllm_cuda",
    "model_identifier": "meta-llama/Llama-3.1-8B-Instruct",
    "serving_targets": ["chat-shared"],
    "endpoint": {"base_url": "https://remote-chat.internal", "transport": "https"},
    "capabilities": {
      "backend_type": "vllm_cuda",
      "engine_type": "vllm",
      "device_class": "nvidia_gpu",
      "model_ids": ["meta-llama/Llama-3.1-8B-Instruct"],
      "serving_targets": ["chat-shared"],
      "max_context_tokens": 8192,
      "supports_streaming": true,
      "concurrency_limit": 8
    },
    "device_class": "nvidia_gpu",
    "runtime": {
      "runtime_family": "vllm_cuda",
      "runtime_label": "vllm_cuda",
      "runtime_version": "0.6.5",
      "engine_type": "vllm_cuda",
      "backend_type": "vllm_cuda"
    },
    "gpu": {
      "accelerator_type": "cuda",
      "vendor": "nvidia",
      "model": "L4",
      "count": 1,
      "memory_per_device_gib": 24.0,
      "cuda_version": "12.4"
    },
    "environment": "staging",
    "placement": {"provider": "aws", "region": "us-east-1"},
    "cost_profile": {"profile": "premium", "budget_bucket": "gpu-canary"},
    "lifecycle_state": "registering",
    "ready": false
  }' | python -m json.tool
```

Then mark it ready:

```bash
curl -sS http://127.0.0.1:8000/internal/control-plane/remote-workers/heartbeat \
  -H 'content-type: application/json' \
  -H 'x-switchyard-lease-token: <lease-token>' \
  -d '{
    "worker_id": "gpu-canary-001",
    "lifecycle_state": "ready",
    "ready": true,
    "active_requests": 0,
    "queue_depth": 0,
    "health": {"state": "healthy", "load_state": "ready", "latency_ms": 12.0}
  }' | python -m json.tool
```

## First Rollout Commands

### Mark The Worker Canary-Only

```bash
curl -sS http://127.0.0.1:8000/admin/remote-workers/gpu-canary-001/canary-only \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"reason":"phase8 first rented GPU"}' \
  | python -m json.tool
```

### Open A Small Cloud Rollout Window

```bash
curl -sS http://127.0.0.1:8000/admin/hybrid/cloud-rollout \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"canary_percentage":5.0,"kill_switch_enabled":false}' \
  | python -m json.tool
```

### Inspect Health, Budget, And Hybrid State

```bash
curl -s http://127.0.0.1:8000/admin/remote-workers | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid/budget | python -m json.tool
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
```

### Drain Or Quarantine A Cloud Worker

```bash
curl -sS http://127.0.0.1:8000/admin/remote-workers/gpu-canary-001/drain \
  -H 'content-type: application/json' \
  -d '{"reason":"rotate rented node"}' \
  | python -m json.tool

curl -sS http://127.0.0.1:8000/admin/remote-workers/gpu-canary-001/quarantine \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"reason":"transport instability"}' \
  | python -m json.tool
```

### Fail Back To Local-Only

```bash
curl -sS http://127.0.0.1:8000/admin/hybrid/remote-enabled \
  -H 'content-type: application/json' \
  -d '{"enabled":false,"reason":"phase8 rollback"}' \
  | python -m json.tool
```

## First Real Cloud Benchmark And Replay

Benchmark the first real cloud path through the normal gateway:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_compose_benchmark_workload.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target compose \
  --deployment-profile compose \
  --config-profile-name phase8-vllm-cuda-canary \
  --warmup-request-count 1 \
  --markdown-report
```

Replay captured traces against the same gateway once you have a trace file:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --policy balanced \
  --replay-mode fixed_concurrency \
  --concurrency 3 \
  --warmup-request-count 1 \
  --markdown-report
```

Use the resulting artifacts to answer two separate questions:

- did a real cloud execution happen,
- if so, was the evidence direct observation or a mix of observation and non-observed
  assumptions.

## Trust Boundaries And Limitations

Trust boundaries that matter in Phase 8:

- the worker runtime is trusted to report its own typed runtime, GPU, placement, and
  cost metadata honestly,
- the control plane is trusted to preserve that evidence source instead of silently
  rewriting it into estimates,
- operator overrides, canaries, and rollout gates are runtime controls, not benchmark
  evidence.

Intentionally out of scope:

- cloud autoscaling or fleet management,
- production incident-management tooling,
- multi-region scheduling,
- automatic promotion from shadow/canary to full rollout,
- heuristic alias inference,
- direct Forge Stage A autotuning or learned cloud routing.

## How Phase 8 Prepares For Forge Stage A

Phase 8 is not Forge Stage A, but it establishes the inputs Stage A will need:

- concrete runtime labels and explicit worker topology,
- typed observed-versus-estimated evidence in artifacts,
- reproducible benchmark and replay runs,
- explicit rollout and budget controls that later optimization work must respect,
- config/profile surfaces that can be exported, reviewed, and diffed.

That is the right preparation layer: Forge Stage A can later consume the typed evidence
and bounded knobs without having to refactor the request path first.
