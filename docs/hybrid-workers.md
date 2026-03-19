# Hybrid Workers

This guide is the Phase 7 entrypoint for understanding and operating Switchyard's
hybrid local/remote stack.

Phase 7 keeps the project Mac-first while making remote workers first-class topology
members. The control plane still prefers host-native Apple-Silicon workers for the
shortest local path, but it can now reason about remote workers through the same typed
inventory, worker protocol, lifecycle metadata, benchmark artifacts, and operator
surfaces.

## What Phase 7 Adds

- explicit local-versus-remote worker placement in shared backend inventory,
- remote-worker registration, heartbeat, and retirement posture,
- hybrid routing modes with bounded spillover and budget guardrails,
- operator-visible hybrid runtime state and remote transport errors,
- remote-aware benchmark, replay, and report artifacts,
- portable Linux/container packaging boundaries for later `vllm_cuda` workers,
- optimization-ready config surfaces and config fingerprints for later Forge Stage A.

## Mental Model

### Local Workers

Local workers are the Mac-first path:

- they usually run host-native on macOS,
- they wrap MLX-LM or vLLM-Metal behind the worker HTTP protocol,
- they remain the default path for real local execution.

### Remote Workers

Remote workers are not special cases bolted onto the side. They are typed worker
instances with:

- explicit network endpoints,
- capability and health metadata,
- placement, cost, trust, and network characteristics,
- lifecycle state when they are dynamically registered,
- remote execution mode and locality.

In Phase 7 they can be:

- statically configured in `SWITCHYARD_LOCAL_MODELS`, or
- dynamically registered through the control-plane lifecycle endpoints.

### Why This Matters

The router and gateway do not need vendor-specific logic such as "Metal versus CUDA".
They consume typed capability, health, locality, and budget signals. That keeps the
control plane portable to later Linux/NVIDIA workers without making local development
depend on real rented GPUs.

## Topology And Capability Extensions

Phase 7 extends the existing deployment and backend schemas rather than creating a
parallel remote-worker system.

Relevant additions include:

- remote `execution_mode` and `locality_class`,
- explicit remote `instances` with network endpoints,
- placement metadata such as provider, region, and zone,
- cost and trust posture,
- registration and heartbeat metadata for discovered workers,
- remote worker lifecycle state such as registering, ready, draining, lost, or retired.

Trust boundary:

- the control plane trusts only typed metadata and live protocol responses,
- remote workers may advertise cloud-like metadata today even when the runtime is still a
  fake worker,
- benchmark artifacts preserve that distinction instead of pretending the stub is real
  GPU evidence.

## Remote Transport And Lifecycle

The shared internal worker protocol remains small:

- `GET /healthz`
- `GET /internal/worker/ready`
- `GET /internal/worker/capabilities`
- `POST /internal/worker/warmup`
- `POST /internal/worker/generate`
- `POST /internal/worker/generate/stream`

Phase 7 adds lifecycle endpoints on the control plane:

- `POST /internal/control-plane/remote-workers/register`
- `POST /internal/control-plane/remote-workers/heartbeat`
- `POST /internal/control-plane/remote-workers/deregister`

Operator endpoints for inspection and posture changes:

- `GET /admin/runtime`
- `GET /admin/deployment`
- `GET /admin/remote-workers`
- `GET /admin/hybrid`
- `POST /admin/hybrid/remote-enabled`
- `POST /admin/hybrid/budget/reset`
- `POST /admin/remote-workers/{worker_id}/drain`
- `POST /admin/remote-workers/{worker_id}/quarantine`
- `POST /admin/remote-workers/{worker_id}/canary-only`
- `POST /admin/remote-workers/cleanup`

Lifecycle assumptions:

- static discovery is the simplest local and CI-safe path,
- dynamic registration is typed and testable before any real GPU rollout,
- lease tokens gate heartbeat and deregistration after successful registration,
- stale remote workers can be retained briefly for diagnosis and then evicted.

## Security And Auth Assumptions

Phase 7 supports three registration modes:

- `none`
- `static_token`
- `signed_enrollment`

The trust model is intentionally conservative:

- `none` is suitable only for local development or tightly controlled test setups,
- `static_token` is the default simple contract for local Compose and CI-safe stubs,
- `signed_enrollment` is the typed extension point for later stronger bootstrap flows.

Current limitation:

- the fake remote worker image does not auto-register itself yet,
- this is intentional so the control-plane contract stays reviewable before bootstrap
  automation is added.

## Hybrid Routing Modes And Guardrails

Hybrid routing remains explicit and explainable. The control plane can prefer local,
burst to remote capacity, or block remote execution due to policy or guardrails.

Current remote-aware policies:

- `local_preferred`
- `burst_to_remote`
- `latency_slo`
- `quality_on_demand`
- `remote_preferred_if_local_unhealthy`

Current guardrails include:

- `prefer_local`
- `spillover_enabled`
- `require_healthy_local_backends`
- `max_remote_share_percent`
- `remote_request_budget_per_minute`
- `remote_concurrency_cap`
- `remote_kill_switch_enabled`
- `remote_cooldown_seconds`
- `allow_high_priority_remote_escalation`
- `allowed_remote_environments`
- per-tenant remote spillover rules

Trust boundary:

- remote spillover is bounded and inspectable,
- shadow traffic never changes the primary response path,
- hybrid routing does not hide cost or health logic in opaque model-specific code.

## Remote-Aware Benchmarks, Replay, And Reports

Phase 7 benchmark and replay artifacts now preserve:

- deployment target and deployment profile,
- runtime topology snapshots,
- remote health and lifecycle summaries,
- hybrid execution context per request,
- immutable benchmark config snapshots,
- deterministic config fingerprints for honest run comparison.

This matters for later Forge Stage A work because optimization loops need to compare
runs using the same configuration truth, not just the same prompt set.

Current limitation:

- batching knobs are not captured because the current runtime boundary does not expose
  explicit batching controls yet.

## Operator Inspection And Budget Controls

Use the runtime and hybrid admin endpoints together:

- `/admin/runtime` shows the broad control-plane view,
- `/admin/remote-workers` shows the lifecycle registry,
- `/admin/hybrid` shows recent route examples, transport errors, and the effective
  remote-enabled posture.

Budget posture remains operator-visible rather than inferred from logs:

- per-minute remote budget usage,
- remaining remote budget,
- active cooldown state,
- effective remote-enabled override,
- recent remote-blocked route examples.

## Packaging And Deployment Boundaries

Phase 7 keeps packaging split cleanly:

- the control-plane image stays portable and Apple-runtime-free,
- the generic remote-worker image speaks the same protocol without CUDA-only runtime
  dependencies,
- later Linux/NVIDIA worker images should derive from or mirror that remote-worker
  contract rather than changing the control plane.

Relevant files:

- [Dockerfile.remote-worker](/Users/rishivinodkumar/Atlas/infra/docker/Dockerfile.remote-worker)
- [compose.remote-worker.yaml](/Users/rishivinodkumar/Atlas/infra/compose/compose.remote-worker.yaml)
- [remote-worker-stub.yaml](/Users/rishivinodkumar/Atlas/infra/kind/remote-worker-stub.yaml)
- [phase7_remote_worker_stub_control_plane.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_control_plane.env)
- [phase7_remote_worker_stub_worker.env](/Users/rishivinodkumar/Atlas/docs/examples/phase7_remote_worker_stub_worker.env)
- [remote-workers.md](/Users/rishivinodkumar/Atlas/docs/remote-workers.md)

## Optimization-Ready Config Surfaces

Phase 7 now exposes bounded, typed optimization surfaces before any actual autotuning:

- `SWITCHYARD_OPTIMIZATION` defines the allowed search surface,
- worker launch presets capture bounded host-native and remote startup options,
- `switchyard-control-plane export-optimization-profile` exports the current tuning
  surface,
- benchmark and replay configs carry immutable snapshots and a deterministic fingerprint.

This is the preparation for later Forge Stage A:

- the search space is explicit rather than inferred,
- run comparisons can reject mismatched config fingerprints,
- operators can review optimization constraints before any loop starts changing policy.

## Practical Commands

### 1. Run A Mock Remote Worker

Local fake worker process:

```bash
env $(grep -v '^#' docs/examples/phase7_remote_worker_stub_worker.env | xargs) \
  uv run switchyard-fake-remote-worker
```

Container image:

```bash
docker build -f infra/docker/Dockerfile.remote-worker -t switchyard/remote-worker:dev .
docker run --rm -p 8090:8090 --env-file docs/examples/phase7_remote_worker_stub_worker.env \
  switchyard/remote-worker:dev
```

### 2. Configure Or Discover A Remote Worker

Static discovery through the control-plane env file:

```bash
uv run switchyard-control-plane check-config
```

Dynamic registration example:

```bash
curl -sS http://127.0.0.1:8000/internal/control-plane/remote-workers/register \
  -H 'content-type: application/json' \
  -H 'x-switchyard-registration-token: phase7-demo-registration-token' \
  -d '{
    "worker_id": "stub-vllm-cuda-worker-1",
    "worker_name": "stub-vllm-cuda-worker",
    "backend_type": "vllm_cuda",
    "model_identifier": "meta-llama/Llama-3.1-8B-Instruct",
    "serving_targets": ["chat-shared"],
    "endpoint": {"base_url": "http://127.0.0.1:8090", "transport": "http"},
    "capabilities": {
      "backend_type": "vllm_cuda",
      "device_class": "nvidia_gpu",
      "model_ids": ["meta-llama/Llama-3.1-8B-Instruct"],
      "model_aliases": {"chat-shared": "meta-llama/Llama-3.1-8B-Instruct"},
      "default_model": "meta-llama/Llama-3.1-8B-Instruct",
      "max_context_tokens": 8192,
      "supports_streaming": true,
      "concurrency_limit": 8
    },
    "device_class": "nvidia_gpu",
    "environment": "compose-remote-stub",
    "locality": "remote",
    "locality_class": "remote_cloud",
    "placement": {"provider": "aws", "region": "us-east-1", "zone": "us-east-1a"},
    "ready": true,
    "health": {"state": "healthy"},
    "tags": ["registered", "remote"]
  }' | python -m json.tool
```

Inspect the discovered worker:

```bash
curl -s http://127.0.0.1:8000/admin/remote-workers | python -m json.tool
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
```

### 3. Enable Or Disable Remote Spillover

Disable remote spillover immediately:

```bash
curl -sS http://127.0.0.1:8000/admin/hybrid/remote-enabled \
  -H 'content-type: application/json' \
  -d '{"enabled":false,"reason":"pause remote spend while debugging"}' \
  | python -m json.tool
```

Re-enable it:

```bash
curl -sS http://127.0.0.1:8000/admin/hybrid/remote-enabled \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"reason":"resume bounded spillover"}' \
  | python -m json.tool
```

Reset the remote budget window:

```bash
curl -sS -X POST http://127.0.0.1:8000/admin/hybrid/budget/reset | python -m json.tool
```

### 4. Run A Hybrid Benchmark Or Replay

Hybrid workload benchmark against a Compose stack with the remote-worker overlay:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_compose_benchmark_workload.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target compose \
  --deployment-profile compose \
  --config-profile-name phase7-remote-stub \
  --control-plane-image-tag switchyard/control-plane:dev \
  --warmup-request-count 1 \
  --markdown-report
```

Hybrid replay:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target compose \
  --deployment-profile compose \
  --config-profile-name phase7-remote-stub \
  --control-plane-image-tag switchyard/control-plane:dev \
  --policy burst_to_remote \
  --replay-mode fixed_concurrency \
  --concurrency 2 \
  --markdown-report
```

### 5. Inspect Remote Health And Budget State

```bash
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid | python -m json.tool
curl -s http://127.0.0.1:8000/admin/hybrid/routes | python -m json.tool
curl -s http://127.0.0.1:8000/admin/remote-workers | python -m json.tool
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
uv run switchyard-control-plane export-optimization-profile | python -m json.tool
```

### 6. Package The Remote Worker Runtime

```bash
docker build -f infra/docker/Dockerfile.remote-worker -t switchyard/remote-worker:dev .
```

Compose overlay:

```bash
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase7_remote_worker_stub_control_plane.env \
SWITCHYARD_REMOTE_WORKER_ENV_FILE=../../docs/examples/phase7_remote_worker_stub_worker.env \
docker compose -f infra/compose/compose.yaml -f infra/compose/compose.remote-worker.yaml up -d
```

kind stub worker template:

```bash
kubectl apply -f infra/kind/remote-worker-stub.yaml
```

## Limitations

- The fake remote worker is suitable for protocol, topology, and operator workflow
  validation, not for real performance claims about NVIDIA workers.
- Dynamic registration is typed, but bootstrap automation is still intentionally small.
- Config fingerprints improve experiment truthfulness, but they do not yet enforce
  hard cross-run compatibility checks in report rendering.
