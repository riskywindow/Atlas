# Phase 5 Control Plane

Switchyard Phase 5 keeps the Phase 4 control-plane core and makes it deployment-aware.
This doc is the operator and developer guide for the routing, overload, rollout, and
artifact behaviors that still define the control plane regardless of whether workers are
in-process or network-addressable.

## Goals

- Keep the request path OpenAI-like while adding explicit Switchyard control-plane
  context.
- Make overload, degradation, failover, and rollout behavior visible in artifacts and
  runtime inspection, not only in logs.
- Keep the implementation local-first, Mac-first, and CI-friendly.
- Preserve backend portability: no control-plane schema or workflow should assume Apple
  GPU specifics.

## Local M4 Pro Starting Point

For a 24 GB M4 Pro, start small and explicit:

- concurrency: `2` to `4`
- bounded queue size: `2` to `8`
- tenant cap: `1` to `2`
- session-affinity TTL: `60` to `180` seconds
- canary percentage: `5` to `10`
- shadow sampling: `0.05` to `0.10`

Example config files:

- [phase5_local_m4pro.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_local_m4pro.env)
- [phase5_local_m4pro_workload.json](/Users/rishivinodkumar/Atlas/docs/examples/phase5_local_m4pro_workload.json)

## Tenant And Request Classification

Switchyard keeps the public chat payload unchanged. Phase 5 control-plane context is
carried in explicit headers:

- `x-switchyard-tenant-id`
  Stable tenant identifier. Defaults to `default` for local development.
- `x-switchyard-request-class`
  One of `standard`, `latency_sensitive`, or `bulk`.
- `x-switchyard-session-id`
  Optional conversation/session key for sticky multi-turn routing.
- `x-switchyard-internal-backend-pin`
  Explicit backend pin for internal debugging only.

Example request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: cp-context-001' \
  -H 'x-switchyard-tenant-id: tenant-priority' \
  -H 'x-switchyard-request-class: latency_sensitive' \
  -H 'x-switchyard-session-id: convo-001' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Keep the route stable for this conversation."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

## Admission Control And Per-Tenant Limits

Admission control is a local in-process subsystem that decides whether a request is:

- admitted immediately,
- queued briefly,
- or rejected before execution begins.

It currently supports:

- global concurrency caps,
- per-tenant concurrency caps,
- bounded queue depth,
- queue timeout / stale-request expiration,
- explicit reason codes.

The current local-first overload response is `429 Too Many Requests`.

Run the gateway with bounded admission control:

```bash
SWITCHYARD_METRICS_ENABLED=true \
SWITCHYARD_DEFAULT_MODEL_ALIAS=chat-shared \
SWITCHYARD_LOCAL_MODELS='[{"alias":"chat-mlx","serving_target":"chat-shared","model_identifier":"mlx-community/Qwen2.5-3B-Instruct-4bit","backend_type":"mlx_lm","configured_priority":80,"configured_weight":1.1},{"alias":"chat-metal","serving_target":"chat-shared","model_identifier":"NousResearch/Meta-Llama-3-8B-Instruct","backend_type":"vllm_metal","configured_priority":100,"configured_weight":1.0}]' \
SWITCHYARD_PHASE4='{"admission_control":{"enabled":true,"global_concurrency_cap":4,"global_queue_size":4,"default_concurrency_cap":2,"default_queue_size":2,"request_timeout_seconds":20.0,"queue_timeout_seconds":2.0,"per_tenant_limits":[{"tenant_id":"tenant-priority","concurrency_cap":1,"queue_size":1},{"tenant_id":"tenant-standard","concurrency_cap":1,"queue_size":0}]},"circuit_breakers":{"enabled":true,"failure_threshold":2,"recovery_success_threshold":1,"open_cooldown_seconds":15.0,"request_timeout_seconds":20.0},"session_affinity":{"enabled":true,"ttl_seconds":120.0,"max_sessions":2000},"canary_routing":{"enabled":false,"default_percentage":0.0,"policies":[]},"shadow_routing":{"enabled":false,"default_sampling_rate":0.0,"policies":[]}}' \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Exercise a tenant cap from another terminal:

```bash
for id in 1 2 3; do
  curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H "content-type: application/json" \
    -H "x-request-id: tenant-cap-$id" \
    -H "x-switchyard-tenant-id: tenant-priority" \
    -H "x-switchyard-request-class: latency_sensitive" \
    -d '{
      "model": "chat-shared",
      "messages": [{"role": "user", "content": "Hold the slot briefly and answer in one sentence."}],
      "max_output_tokens": 64
    }' &
done
wait
```

Inspect the response headers or the benchmark artifact for:

- `x-switchyard-admission-decision`
- queue wait timing
- queue/full or timeout reason codes

## Circuit Breakers

The local circuit breaker protects a backend after repeated invocation or timeout-like
failures. It transitions through:

- `closed`
- `open`
- `half_open`

The router avoids `open` backends and allows controlled recovery probes through
`half_open`.

Current local-friendly way to observe breaker transitions with a flaky backend:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/control/test_circuit.py tests/router/test_service.py -k breaker -q
```

That path is intentionally documented because the repo does not yet expose a public
runtime flakiness injector for a live gateway. The targeted tests are deterministic and
exercise the same control-plane transitions the gateway records in artifacts.

## Session Affinity

Session affinity is local, bounded, and TTL-based. Related requests with the same
session key prefer the same backend while that backend is still healthy and eligible.

Send repeated requests with a session key:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: session-001' \
  -H 'x-switchyard-session-id: session-demo-1' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Turn one of a multi-turn exchange."}],
    "max_output_tokens": 64
  }' | python -m json.tool

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: session-002' \
  -H 'x-switchyard-session-id: session-demo-1' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Turn two of the same exchange."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Check the route header or artifact annotations for:

- `affinity_disposition`
- sticky-route creation or reuse
- failover notes when the sticky backend becomes ineligible

## Shadow Traffic

Shadow traffic is off by default and must be explicitly enabled. It never changes the
primary user-visible response.

Use concrete backend deployment names in `target_backend`, such as
`vllm-metal:chat-metal`, not just the local alias.

Run the gateway with safe shadow traffic:

```bash
SWITCHYARD_PHASE4='{"admission_control":{"enabled":true,"global_concurrency_cap":4,"global_queue_size":4,"default_concurrency_cap":2,"default_queue_size":2,"request_timeout_seconds":20.0,"queue_timeout_seconds":2.0,"per_tenant_limits":[]},"circuit_breakers":{"enabled":true,"failure_threshold":2,"recovery_success_threshold":1,"open_cooldown_seconds":15.0,"request_timeout_seconds":20.0},"session_affinity":{"enabled":true,"ttl_seconds":120.0,"max_sessions":2000},"canary_routing":{"enabled":false,"default_percentage":0.0,"policies":[]},"shadow_routing":{"enabled":true,"default_sampling_rate":0.05,"policies":[{"policy_name":"tenant-shadow","enabled":true,"serving_target":"chat-shared","tenant_id":"tenant-shadow","target_backend":"vllm-metal:chat-metal","sampling_rate":0.1}]}}' \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Send a shadow-eligible request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: shadow-001' \
  -H 'x-switchyard-tenant-id: tenant-shadow' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Primary response should stay authoritative."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Inspect telemetry or artifacts for:

- the matched shadow policy,
- `shadow_disposition`,
- the configured shadow target,
- a link back to the primary request ID.

## Canary Routing

Canary routing is also off by default and opt-in. A logical alias can shift a bounded
percentage of eligible traffic to a candidate backend while the remainder stays on the
baseline path.

Use concrete backend deployment names for `baseline_backend` and allocation
`backend_name` values, such as `mlx-lm:chat-mlx` and `vllm-metal:chat-metal`.

Enable a small rollout:

```bash
SWITCHYARD_PHASE4='{"admission_control":{"enabled":true,"global_concurrency_cap":4,"global_queue_size":4,"default_concurrency_cap":2,"default_queue_size":2,"request_timeout_seconds":20.0,"queue_timeout_seconds":2.0,"per_tenant_limits":[]},"circuit_breakers":{"enabled":true,"failure_threshold":2,"recovery_success_threshold":1,"open_cooldown_seconds":15.0,"request_timeout_seconds":20.0},"session_affinity":{"enabled":true,"ttl_seconds":120.0,"max_sessions":2000},"canary_routing":{"enabled":true,"default_percentage":5.0,"policies":[{"policy_name":"chat-rollout","serving_target":"chat-shared","enabled":true,"baseline_backend":"mlx-lm:chat-mlx","allocations":[{"backend_name":"vllm-metal:chat-metal","percentage":10.0}]}]},"shadow_routing":{"enabled":false,"default_sampling_rate":0.0,"policies":[]}}' \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Send a canary-stable request with a session key:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: canary-001' \
  -H 'x-switchyard-session-id: canary-session-1' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Explain why deterministic rollout bucketing matters."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Artifacts and route headers should show:

- the matched canary policy,
- `rollout_disposition`,
- canary-vs-baseline reasons,
- fallback to baseline when the canary candidate is unhealthy or breaker-protected.

## Phase 6 Policy Rollout Controls

Phase 6 keeps intelligent-policy rollout local-first and explicit. The same
`SWITCHYARD_PHASE4` block now carries a `policy_rollout` section for candidate-policy
controls.

Important:

- the rollout controller and admin endpoints are live in the stock gateway,
- but the default `create_app()` path does not yet instantiate adaptive candidate
  scorers from settings alone,
- `candidate_policy_id` and `shadow_policy_id` therefore only take effect when the
  gateway is constructed with registered candidate scorers.

Use the config block below as the rollout-state shape, not as a full policy-registration
mechanism yet.

```bash
SWITCHYARD_PHASE4='{"admission_control":{"enabled":true,"global_concurrency_cap":4,"global_queue_size":4,"default_concurrency_cap":2,"default_queue_size":2,"request_timeout_seconds":20.0,"queue_timeout_seconds":2.0,"per_tenant_limits":[]},"circuit_breakers":{"enabled":true,"failure_threshold":2,"recovery_success_threshold":1,"open_cooldown_seconds":15.0,"request_timeout_seconds":20.0},"session_affinity":{"enabled":true,"ttl_seconds":120.0,"max_sessions":2000},"canary_routing":{"enabled":false,"default_percentage":0.0,"policies":[]},"shadow_routing":{"enabled":false,"default_sampling_rate":0.0,"policies":[]},"policy_rollout":{"mode":"shadow_only","candidate_policy_id":"adaptive-balanced-v1","shadow_policy_id":"adaptive-balanced-v1","canary_percentage":0.0,"kill_switch_enabled":false,"learning_frozen":false,"max_recent_decisions":25}}' \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

At runtime, inspect and mutate rollout state through the admin endpoints:

```bash
curl -s http://127.0.0.1:8000/admin/policy-rollout | python -m json.tool

curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"canary","canary_percentage":10.0}' \
  | python -m json.tool

curl -sS -X POST http://127.0.0.1:8000/admin/policy-rollout/reset | python -m json.tool
curl -s http://127.0.0.1:8000/admin/policy-rollout/export | python -m json.tool
```

Use these controls to keep candidate policies in shadow, recommendation, canary, or
guarded-active modes without changing the baseline compatibility policy configuration.

## Runtime Inspection

The current admin surface is read-only and local-dev friendly.

Inspect runtime state:

```bash
curl -sS http://127.0.0.1:8000/admin/runtime | python -m json.tool
```

That view exposes:

- backend health snapshots,
- breaker state,
- admission queue depth and in-flight totals,
- tenant limiter summaries,
- active canary config,
- active shadow config,
- session-affinity cache summary.

## Integration With Benchmarking And Replay

Phase 5 builds on the Phase 3 artifact-first benchmark and replay system.

- Workload generation now includes `queue_saturation`, `tenant_contention`,
  `backend_flakiness`, `session_stickiness`, `canary_rollout`, and `shadow_traffic`.
- Benchmark records carry `route_decision` and `control_plane_metadata`.
- Markdown reports derive control-plane summaries from the authoritative JSON artifact.
- Trace replay preserves tenant, request-class, and session context where available.

Generate a Phase 5 local workload:

```bash
uv run python -m switchyard.bench.cli generate-workload \
  --family tenant_contention \
  --model-alias chat-shared \
  --request-count 8 \
  --seed 21
```

Capture an overload-oriented artifact:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_local_m4pro_workload.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --warmup-request-count 1 \
  --markdown-report
```

Capture a rollout-oriented artifact:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/canary_rollout_21_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --markdown-report
```

Replay captured traces back through the Phase 5 control-plane path:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --replay-mode fixed_concurrency \
  --concurrency 2 \
  --markdown-report
```

See also:

- [docs/benchmarking.md](/Users/rishivinodkumar/Atlas/docs/benchmarking.md)
- [docs/phase4.md](/Users/rishivinodkumar/Atlas/docs/phase4.md)
