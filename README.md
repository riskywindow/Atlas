# Switchyard

Switchyard is a Mac-first, backend-agnostic inference fabric. In Phase 9 it can expose
one logical model alias through either MLX-LM, vLLM-Metal, or both, route requests
across explicit network-addressable worker instances (local and remote), and emit typed
route decisions that are benchmarkable, replayable, and explainable. The current routing
slice carries tenant and session context, rejects saturated backends with bounded
admission rules, uses circuit-breaker protection to stop hammering failing backends,
supports bounded session-affinity reuse, records canary/shadow decisions in runtime
inspection and benchmark artifacts, and preserves deterministic request-feature
extraction for cache/locality-aware scorers and offline policy simulation.

Phase 9 adds Forge Stage A — evidence-driven autotuning — on top of the Phase 8
hybrid local/remote baseline:

- typed optimization campaigns generate, prune, and evaluate candidate configurations
  offline against benchmark and replay evidence,
- multi-objective candidate ranking surfaces explicit tradeoffs across latency, quality,
  remote spend, and workload diversity,
- campaign honesty assessment validates recommendations against current environment state
  (budget bounds, topology drift, staleness, workload coverage, evidence consistency),
- typed recommendations carry dispositions, confidence levels, evidence kinds, and
  reason codes,
- a bounded promotion lifecycle (propose, approve, canary, compare, promote/rollback)
  keeps changes reversible and operator-reviewed,
- evidence semantics (observed vs replayed vs simulated vs estimated) are preserved
  end-to-end and never collapsed.

Earlier phases delivered:

- remote workers as first-class topology members with hybrid local/remote guardrails
  (Phase 7-8),
- the first real Linux/NVIDIA worker path behind the same generic worker contract
  (Phase 8),
- optimization-ready config surfaces and config fingerprints (Phase 7-8).

More specifically, the current repo includes:
- deterministic request and workload feature extraction,
- repeated-prefix and locality-aware signals without storing raw prompt text,
- historical performance summaries and transparent predictor inputs,
- richer scorer and policy interfaces with per-candidate reasoning and shadow scoring,
- offline simulation that distinguishes direct observations from estimates,
- a conservative adaptive policy with abstention, confidence thresholds, and guardrails,
- safe rollout controls with shadow, report-only, canary, and guarded-active modes,
- evidence-based recommendation reports that do not auto-apply changes,
- typed hybrid local/remote execution settings and runtime inspection,
- operator-visible remote spillover budgets and remote-health summaries,
- remote worker lifecycle posture for later secure registration and cloud-ready workers,
- optimization-ready config and export surfaces for Forge Stage A autotuning.
- explicit separation between observed cloud/runtime evidence and estimated or mock
  evidence in typed operator and benchmark surfaces.
- offline optimization campaigns with candidate generation, pruning, and ranking.
- campaign honesty assessment for budget, topology, staleness, and evidence validation.
- bounded promotion lifecycle with canary rollout and atomic rollback.
- typed recommendation summaries with disposition, confidence, and evidence posture.

## Phase 9 At A Glance

Phase 9 is the first release with an optimization layer (Forge Stage A) on top of the
existing hybrid control plane.

- typed optimization profiles declare what is safely tunable (routing policy, rollout
  mode, hybrid budget posture, shadow sampling) and what is excluded (kernels,
  quantization, model loading),
- offline campaigns generate candidates, evaluate them against benchmark and replay
  evidence, and rank them with multi-objective comparison,
- campaign honesty assessment warns when recommendations may not be trustworthy due to
  budget changes, topology drift, stale evidence, narrow workload coverage, or
  inconsistent evidence,
- recommendations carry explicit dispositions (PROMOTE_CANDIDATE, KEEP_BASELINE,
  NEED_MORE_EVIDENCE, INVALIDATE_TRIAL) and evidence kinds,
- promotion follows a bounded lifecycle: PROPOSED -> APPROVED -> CANARY_ACTIVE ->
  COMPARED -> PROMOTED_DEFAULT with rollback at every stage,
- evidence semantics (observed, replayed, simulated, estimated) are never collapsed
  and are visible in every operator surface.

Start with these docs:

- [docs/forge-stage-a.md](/Users/rishivinodkumar/Atlas/docs/forge-stage-a.md)
- [docs/architecture.md](/Users/rishivinodkumar/Atlas/docs/architecture.md)
- [docs/remote-workers.md](/Users/rishivinodkumar/Atlas/docs/remote-workers.md)
- [docs/phase8.md](/Users/rishivinodkumar/Atlas/docs/phase8.md)
- [docs/phase9.md](/Users/rishivinodkumar/Atlas/docs/phase9.md)

Phase 9 is Mac-first, not Mac-locked:
- real local backends are currently Apple Silicon focused,
- the gateway, router, schemas, and artifacts stay backend-agnostic,
- Apple-specific imports stay behind adapter and runtime boundaries,
- the control plane should remain deployable without Apple-specific runtime dependencies,
- the same control plane is intended to grow into future `vllm_cuda` and remote worker
  backends.

## Setup

### Prerequisites

- Apple Silicon Mac
- Python 3.12
- `uv`

### Base Workspace

```bash
uv sync --dev
cp .env.example .env
```

This base environment is enough for:
- the mock backend,
- unit and integration tests,
- synthetic benchmarks,
- gateway and control-plane work that does not require a real Apple worker runtime.

It is also the intended portable control-plane environment:
- no `mlx-lm` dependency is required,
- no `vllm` dependency is required,
- the control plane should remain importable and testable on CI hosts without Apple GPU
  access.

### Optional Backend Installs

Use the repo extras so the optional runtime dependencies stay explicit.

MLX-LM:

```bash
uv sync --dev --extra mlx
```

Equivalent Make target:

```bash
make setup-mlx
```

vLLM-Metal:

```bash
uv sync --dev --extra vllm-metal
```

Equivalent Make target:

```bash
make setup-vllm-metal
```

Both backends in one local environment:

```bash
uv sync --dev --extra mlx --extra vllm-metal
```

Combined Apple-worker extra:

```bash
uv sync --dev --extra apple-workers
```

Notes:
- MLX-LM is imported lazily and is only required when an `mlx_lm` backend is configured.
- vLLM-Metal uses the `vllm` module behind a dedicated runtime/provider boundary and is
  only required when a `vllm_metal` backend is configured.
- CI-friendly tests do not require either optional dependency or Apple GPU hardware.
- Portable control-plane packaging should continue to work without either Apple-worker
  extra installed.

## Mac-First Workflow

The current workflow is built around one explicit split:

- Apple-Silicon model workers stay host-native on macOS by default.
- The control plane can run locally, in Docker Compose, or in kind without Apple-specific runtime dependencies.

The shortest useful path on an M4 Pro is:

1. Install the portable control-plane workspace:

```bash
uv sync --dev
```

2. Install Apple-worker extras only on the host that will run MLX-LM or vLLM-Metal:

```bash
uv sync --dev --extra apple-workers
```

3. Start host-native workers in separate terminals:

```bash
uv run switchyard-worker serve mlx-lm:chat-mlx --host 127.0.0.1 --port 8101 --warmup-mode eager
uv run switchyard-worker serve vllm-metal:chat-metal --host 127.0.0.1 --port 8102 --warmup-mode eager
```

4. Start one of the control-plane paths:

Local gateway with host-native workers:

```bash
cp docs/examples/phase5_local_m4pro.env .env
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Compose control plane with host-native workers:

```bash
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase5_compose_m4pro.env \
  docker compose -f infra/compose/compose.yaml up -d
```

kind control plane with host-native workers:

```bash
./scripts/kind-bootstrap.sh
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
./scripts/kind-push-control-plane.sh switchyard/control-plane:dev
./scripts/kind-deploy-control-plane.sh m4pro
kubectl -n switchyard port-forward service/switchyard-gateway 8000:8000
```

5. Inspect the deployment boundary and runtime truth:

```bash
curl -s http://127.0.0.1:8000/readyz | python -m json.tool
curl -s http://127.0.0.1:8000/admin/runtime | python -m json.tool
curl -s http://127.0.0.1:8000/admin/deployment | python -m json.tool
uv run switchyard-control-plane doctor --gateway-base-url http://127.0.0.1:8000
uv run switchyard-control-plane export-optimization-profile | python -m json.tool
```

6. Send one request through the gateway:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Explain the Phase 8 topology in one sentence."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

7. Run a deployment-aware benchmark or replay:

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

8. Run offline policy simulation against captured benchmark artifacts:

```bash
uv run python -m switchyard.bench.cli simulate-policy \
  benchmarks/20260317T000000Z_balanced.json \
  --policy-id adaptive-balanced-v1 \
  --objective balanced \
  --mode recommend \
  --require-sufficient-data \
  --max-predicted-error-rate 0.10 \
  --max-predicted-latency-regression-ms 25 \
  --markdown-report
```

This produces a typed simulation artifact plus a markdown report with counterfactual
route recommendations, guardrail blocks, and a compact rollout recommendation.

To compare several offline candidate policies against the same benchmark artifacts or
captured traces:

```bash
uv run python -m switchyard.bench.cli compare-offline-policies \
  --trace-path traces/captured.jsonl \
  --routing-policy balanced \
  --routing-policy latency_first \
  --candidate-policy adaptive-safe:balanced \
  --markdown-report
```

The comparison artifact keeps per-policy evidence quality explicit and distinguishes
direct observations, predictor estimates, low-confidence estimates, and unsupported
cases.

To turn recent benchmark and simulation artifacts into operator guidance:

```bash
uv run python -m switchyard.bench.cli recommend-policies \
  benchmarks/20260317T000000Z_balanced.json \
  benchmarks/20260317T001500Z_policy-comparison.json \
  --markdown-report
```

This writes a typed recommendation artifact plus markdown guidance with evidence
windows, sample sizes, workload buckets, no-change cases, and caveats.

If a gateway process has a candidate adaptive policy registered, inspect rollout state:

```bash
curl -s http://127.0.0.1:8000/admin/policy-rollout | python -m json.tool
```

Move that candidate into shadow-only mode:

```bash
curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"shadow_only","kill_switch_enabled":false,"learning_frozen":false}' \
  | python -m json.tool
```

Enable or disable adaptive-policy canary mode:

```bash
curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"canary","canary_percentage":10.0}' \
  | python -m json.tool

curl -sS http://127.0.0.1:8000/admin/policy-rollout \
  -H 'content-type: application/json' \
  -d '{"mode":"disabled"}' \
  | python -m json.tool
```

Reset or export rollout state:

```bash
curl -sS -X POST http://127.0.0.1:8000/admin/policy-rollout/reset | python -m json.tool
curl -s http://127.0.0.1:8000/admin/policy-rollout/export | python -m json.tool
```

Inspect and mutate the real-cloud rollout gate for `canary-only` remote workers:

```bash
curl -s http://127.0.0.1:8000/admin/hybrid/cloud-rollout | python -m json.tool

curl -sS http://127.0.0.1:8000/admin/hybrid/cloud-rollout \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"canary_percentage":5.0,"kill_switch_enabled":false}' \
  | python -m json.tool

curl -sS http://127.0.0.1:8000/admin/hybrid/cloud-rollout \
  -H 'content-type: application/json' \
  -d '{"kill_switch_enabled":true}' \
  | python -m json.tool
```

Use the longer guides together:

- [deployment.md](/Users/rishivinodkumar/Atlas/docs/deployment.md)
- [hybrid-workers.md](/Users/rishivinodkumar/Atlas/docs/hybrid-workers.md)
- [remote-workers.md](/Users/rishivinodkumar/Atlas/docs/remote-workers.md)
- [control-plane.md](/Users/rishivinodkumar/Atlas/docs/control-plane.md)
- [benchmarking.md](/Users/rishivinodkumar/Atlas/docs/benchmarking.md)
- [phase6.md](/Users/rishivinodkumar/Atlas/docs/phase6.md)
- [phase7.md](/Users/rishivinodkumar/Atlas/docs/phase7.md)
- [phase8.md](/Users/rishivinodkumar/Atlas/docs/phase8.md)
- [phase8-exit-review.md](/Users/rishivinodkumar/Atlas/docs/phase8-exit-review.md)
- [intelligent-routing.md](/Users/rishivinodkumar/Atlas/docs/intelligent-routing.md)
- [architecture.md](/Users/rishivinodkumar/Atlas/docs/architecture.md)

### Host-Native Worker Mode

Each worker exposes `GET /healthz`, `GET /internal/worker/ready`,
`GET /internal/worker/capabilities`, `POST /internal/worker/warmup`,
`POST /internal/worker/generate`, `POST /internal/worker/generate/stream`, and
`POST /v1/chat/completions`. Point the control plane at those workers by setting the
corresponding `local_models[*].worker_transport` to `http` and
`local_models[*].instances[*].base_url` to the worker address.

Then drive a few current control-plane behaviors:

Bounded admission and tenant classification:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: quickstart-tenant-001' \
  -H 'x-switchyard-tenant-id: tenant-priority' \
  -H 'x-switchyard-request-class: latency_sensitive' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Answer briefly and keep latency low."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Session affinity:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: quickstart-session-001' \
  -H 'x-switchyard-session-id: session-quickstart-1' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "This is turn one."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Phase 5 benchmark artifact capture:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_local_m4pro_workload.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --warmup-request-count 1 \
  --markdown-report
```

Compose deployment-aware benchmark artifact capture:

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

## Config Examples

Switchyard clients send a logical alias in `model`. A configured backend deployment may
either serve its own alias directly or share a `serving_target` with other deployments.

Configuration lives in `SWITCHYARD_LOCAL_MODELS` as JSON.

### MLX-only Alias

```dotenv
SWITCHYARD_DEFAULT_MODEL_ALIAS=chat-mlx
SWITCHYARD_LOCAL_MODELS=[{"alias":"chat-mlx","model_identifier":"mlx-community/Qwen2.5-3B-Instruct-4bit","backend_type":"mlx_lm","generation_defaults":{"max_output_tokens":256,"temperature":0.2,"top_p":1.0},"warmup":{"enabled":true,"eager":false,"timeout_seconds":30}}]
```

### vLLM-Metal-only Alias

```dotenv
SWITCHYARD_DEFAULT_MODEL_ALIAS=chat-metal
SWITCHYARD_LOCAL_MODELS=[{"alias":"chat-metal","model_identifier":"NousResearch/Meta-Llama-3-8B-Instruct","backend_type":"vllm_metal","generation_defaults":{"max_output_tokens":256,"temperature":0.2,"top_p":1.0},"warmup":{"enabled":true,"eager":false,"timeout_seconds":30}}]
```

### Dual-backend Alias

Both deployments serve the same logical alias, `chat-shared`. Clients still send only
`"model": "chat-shared"`.

```dotenv
SWITCHYARD_DEFAULT_MODEL_ALIAS=chat-shared
SWITCHYARD_LOCAL_MODELS=[{"alias":"chat-mlx","serving_target":"chat-shared","model_identifier":"mlx-community/Qwen2.5-3B-Instruct-4bit","backend_type":"mlx_lm","configured_priority":80,"configured_weight":1.2,"generation_defaults":{"max_output_tokens":256,"temperature":0.2,"top_p":1.0},"warmup":{"enabled":true,"eager":false,"timeout_seconds":30}},{"alias":"chat-metal","serving_target":"chat-shared","model_identifier":"NousResearch/Meta-Llama-3-8B-Instruct","backend_type":"vllm_metal","configured_priority":100,"configured_weight":1.0,"generation_defaults":{"max_output_tokens":256,"temperature":0.2,"top_p":1.0},"warmup":{"enabled":true,"eager":false,"timeout_seconds":30}}]
```

What the fields mean:
- `alias`: the concrete deployment name in config.
- `serving_target`: the logical model alias exposed to clients. If omitted, the alias
  serves itself.
- `backend_type`: the adapter/runtime family, such as `mlx_lm` or `vllm_metal`.
- `configured_priority` and `configured_weight`: deterministic router hints used by
  policy routing.
- `deployment_profile`: where that deployment is expected to run, such as
  `host_native`, `compose`, or `kind`.
- `worker_transport`: how the control plane reaches workers, such as `in_process` or
  `http`.
- `instances`: explicit static worker inventory for deployment-aware topologies.

## Start The Gateway

Metrics enabled:

```bash
make serve-metrics
```

Direct command:

```bash
SWITCHYARD_METRICS_ENABLED=true \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Check readiness:

```bash
curl -s http://127.0.0.1:8000/readyz | python -m json.tool
```

## Example Requests

### Basic Chat Completion

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: readme-basic-001' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Explain Switchyard in two sentences."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

### Streaming Response

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-request-id: readme-stream-001' \
  -d '{
    "model": "chat-shared",
    "stream": true,
    "messages": [{"role": "user", "content": "List three reasons typed contracts help routing."}],
    "max_output_tokens": 64
  }'
```

### Select A Routing Policy

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-switchyard-routing-policy: quality_first' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Answer briefly."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Supported policy modes:
- `latency_first`
- `balanced`
- `quality_first`
- `local_only`

### Attach Switchyard Request Context

Tenant and request classification use Switchyard-specific headers so the OpenAI-like chat
payload stays unchanged.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-switchyard-tenant-id: local-dev' \
  -H 'x-switchyard-request-class: latency_sensitive' \
  -H 'x-switchyard-session-id: convo-001' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Use the typed request context."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

Supported request classes:
- `standard`
- `latency_sensitive`
- `bulk`

When `SWITCHYARD_PHASE4.session_affinity.enabled` is on, repeated requests that carry the
same `x-switchyard-session-id` will prefer the same eligible backend until the sticky
binding expires. Affinity is bounded by TTL and `max_sessions`, bypassed by
`x-switchyard-internal-backend-pin`, and will fail over cleanly when the sticky backend
is unhealthy, overloaded, or breaker-protected.

Shadow traffic is separate from the primary response path. It stays off by default, is
only launched when a configured shadow policy matches, and may target either a concrete
backend or another alias. Shadow failures are best-effort observations only and never
change the user-visible primary result.

Canary routing is also explicit and off by default. When enabled for a logical alias, a
named rollout policy can deterministically bucket requests by session ID or request ID
and shift only the configured fraction of eligible traffic to a candidate backend while
the rest stays on the baseline path. If the candidate backend is unhealthy or otherwise
ineligible, routing falls back to the normal baseline decision and records why.

For local inspection, the gateway now exposes a read-only `GET /admin/runtime` endpoint.
It returns the current backend status plus live summaries for admission control, circuit
breakers, canary config, shadow config, and the session-affinity cache.

When admission control is enabled and the local gateway is saturated, Switchyard rejects
new work before execution with `429 Too Many Requests` and includes an
`x-switchyard-admission-decision` header describing the typed overload outcome.

### Pin A Backend For Debugging

This is intentionally namespaced as an internal override path, not part of the base chat
schema.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-switchyard-internal-backend-pin: vllm-metal:chat-metal' \
  -d '{
    "model": "chat-shared",
    "messages": [{"role": "user", "content": "Use the pinned backend."}],
    "max_output_tokens": 64
  }' | python -m json.tool
```

### Warmup

```bash
export SWITCHYARD_DEFAULT_MODEL_ALIAS=chat-shared
make warmup
```

## Phase 5 Workflow

Phase 5 is centered on a small Mac-first loop:
1. generate a deterministic workload or enable opt-in trace capture,
2. execute that workload or replay those traces through the normal gateway path,
3. compare policies or pinned backends against the same inputs,
4. layer in overload, protection, and progressive-delivery decisions in small slices,
5. generate markdown from the authoritative JSON artifacts.

On an M4 Pro, keep first runs intentionally small:
- `request-count`: `4` to `12`
- workload burst size: `2` to `4`
- replay concurrency: `1` to `4`
- warmup requests: `1` to `2`
- timeout: `30` to `60` seconds

For the full current operator/developer workflow, including admission control, tenant
limits, breakers, session affinity, shadow traffic, canaries, runtime inspection, and
artifact-oriented experiments, see
[docs/control-plane.md](/Users/rishivinodkumar/Atlas/docs/control-plane.md).

### Generate A Workload

Generate a deterministic mixed workload manifest:

```bash
uv run python -m switchyard.bench.cli generate-workload \
  --family mixed \
  --model-alias chat-shared \
  --request-count 8 \
  --seed 17
```

Generate a repeated-prefix workload for cache-sensitive local testing:

```bash
uv run python -m switchyard.bench.cli generate-workload \
  --family repeated_prefix \
  --model-alias chat-shared \
  --request-count 6 \
  --seed 11
```

### Run Benchmarks

Run a workload manifest against the logical alias:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --markdown-report
```

Run the same workload with an explicit routing policy:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --policy latency_first \
  --warmup-request-count 1 \
  --markdown-report
```

Run the same workload pinned to one backend deployment:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --pinned-backend mlx-lm:chat-mlx \
  --warmup-request-count 1 \
  --markdown-report
```

For quick local iteration without a workload manifest, the synthetic runner remains useful:

```bash
uv run python -m switchyard.bench.cli run-synthetic \
  --request-count 6 \
  --workload-pattern repeated_prefix \
  --shared-prefix "Shared retrieval context: customer tier gold" \
  --markdown-report
```

Artifacts are written to `artifacts/benchmarks/` by default. JSON remains the primary
machine-readable artifact; markdown is a derived view.

### Capture Traces Safely

Trace capture is opt-in and off by default.

Safe metadata-only capture:

```bash
SWITCHYARD_TRACE_CAPTURE_MODE=metadata_only \
SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH=artifacts/traces/gateway-traces.jsonl \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Replayable redacted-content capture:

```bash
SWITCHYARD_TRACE_CAPTURE_MODE=redacted_content \
SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH=artifacts/traces/gateway-traces.jsonl \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Use `full_content` only when you explicitly want raw prompts and outputs on disk. Avoid it
for normal local iteration.

### Replay Captured Traces

Replay captured traces against the same logical alias:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --replay-mode sequential \
  --markdown-report
```

Replay the same traces against a routing policy:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --policy balanced \
  --replay-mode fixed_concurrency \
  --concurrency 3 \
  --warmup-request-count 1 \
  --markdown-report
```

Replay against a pinned backend:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --pinned-backend vllm-metal:chat-metal \
  --replay-mode preserve_order_without_original_timing \
  --markdown-report
```

### Run A/B Comparisons

Compare policy A vs policy B on the same workload:

```bash
uv run python -m switchyard.bench.cli compare-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --left-policy balanced \
  --right-policy latency_first \
  --warmup-request-count 1 \
  --markdown-report
```

Compare normal alias routing vs a pinned backend:

```bash
uv run python -m switchyard.bench.cli compare-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --right-pinned-backend mlx-lm:chat-mlx \
  --markdown-report
```

Compare the same captured trace set across two targets:

```bash
uv run python -m switchyard.bench.cli compare-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --left-policy balanced \
  --right-pinned-backend vllm-metal:chat-metal \
  --replay-mode fixed_concurrency \
  --concurrency 2 \
  --markdown-report
```

### Generate Markdown Reports

Render a markdown report from one artifact:

```bash
uv run python -m switchyard.bench.cli generate-report \
  artifacts/benchmarks/20260316T120000Z_balanced.json
```

Render one markdown file from multiple artifacts:

```bash
uv run python -m switchyard.bench.cli generate-report \
  artifacts/benchmarks/20260316T120000Z_balanced.json \
  artifacts/benchmarks/20260316T121500Z_compare_abc123.json \
  --output-path artifacts/benchmarks/phase3-review.md
```

See [docs/benchmarking.md](/Users/rishivinodkumar/Atlas/docs/benchmarking.md) for the
benchmarking and replay guide that the current phase builds on.

## Observability

The current Phase 5 slice is useful without standing up Prometheus or Grafana.

## Control-Plane Container

The control plane can be packaged into a portable image without the Apple-worker extras.
This image is for the gateway/control-plane path and bench/admin-style entrypoints. It
is not for MLX-LM or vLLM-Metal worker execution.

Build the image:

```bash
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
```

Run a config smoke check in the image:

```bash
docker run --rm switchyard/control-plane:dev check-config
```

Run the gateway container:

```bash
docker run --rm -p 8000:8000 \
  -e SWITCHYARD_METRICS_ENABLED=true \
  switchyard/control-plane:dev
```

The image entrypoint also supports:

- `gateway`
- `check-config`
- `bench ...`

Container health uses `GET /healthz`, and readiness should still be checked with
`GET /readyz` once the container is running.

## Mac-First Docker Compose

Switchyard now includes a Mac-first Compose stack for the portable control plane and
supporting infra. Real Apple-Silicon workers still run host-native on macOS and are
reached from containers through explicit worker endpoint config.

Real-worker M4 Pro example:

```bash
docker compose -f infra/compose/compose.yaml up -d
uv run switchyard-worker serve mlx-lm:mlx-chat --host 127.0.0.1 --port 8101 --warmup-mode eager
uv run switchyard-worker serve vllm-metal:vllm-chat --host 127.0.0.1 --port 8102 --warmup-mode eager
curl -s http://127.0.0.1:8000/readyz | python -m json.tool
```

The default Compose env file is
[phase5_compose_m4pro.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_compose_m4pro.env),
which points the gateway at:

- `http://host.docker.internal:8101`
- `http://host.docker.internal:8102`

GPU-free smoke workflow:

```bash
uv run python scripts/host_native_mock_worker.py
SWITCHYARD_COMPOSE_ENV_FILE=../../docs/examples/phase5_compose_smoke.env \
  docker compose -f infra/compose/compose.yaml up -d
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chat-smoke","messages":[{"role":"user","content":"hello from compose"}]}' | python -m json.tool
```

That smoke flow proves the containerized control plane can route to a host-native worker
through the explicit HTTP worker boundary.

## Mac-First kind

Switchyard also includes a small kind deployment path for the same portable control
plane image. Apple-Silicon workers stay outside the cluster by default and are still
configured through explicit worker endpoint inventory.

Bootstrap the local registry and cluster:

```bash
./scripts/kind-bootstrap.sh
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
./scripts/kind-push-control-plane.sh switchyard/control-plane:dev
```

Deploy the smoke overlay:

```bash
./scripts/kind-deploy-control-plane.sh smoke
kubectl -n switchyard port-forward service/switchyard-gateway 8000:8000
```

The kind env examples are:

- [phase5_kind_smoke.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_kind_smoke.env)
- [phase5_kind_m4pro.env](/Users/rishivinodkumar/Atlas/docs/examples/phase5_kind_m4pro.env)

On macOS, those examples use `host.docker.internal` as the default external worker
address. If your local kind setup does not resolve that name inside pods, replace the
`base_url` values in the env file with an ordinary reachable host address and redeploy.

Local-friendly default:

```bash
curl -s http://127.0.0.1:8000/metrics
```

The metrics and structured logs now include route-facing fields such as:
- logical model alias,
- chosen backend,
- routing policy,
- candidate backend count,
- compact route reason,
- fallback occurrence,
- latency, TTFT, output tokens, and tokens/sec when available.

Optional trace capture is available for benchmarking and replay preparation.

- Default: `SWITCHYARD_TRACE_CAPTURE_MODE=off`
- Safe non-replay mode: `metadata_only`
- Replayable safer mode: `redacted_content`
- Explicit raw-content mode: `full_content`

Trace records are written to `SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH` as JSONL when enabled.
The safe default avoids raw prompt and response content. Only `redacted_content` and
`full_content` should be treated as replay-oriented capture modes.

## Troubleshooting

Missing MLX dependency:
- Symptom: logs or health output mention `mlx-lm is not installed`.
- Fix: run `uv sync --dev --extra mlx` or `make setup-mlx`, then restart the gateway.

Missing vLLM-Metal dependency:
- Symptom: logs or health output mention `vllm is not installed`.
- Fix: run `uv sync --dev --extra vllm-metal` or `make setup-vllm-metal`, then restart
  the gateway.

Backend type mismatch:
- Symptom: startup fails because the adapter/runtime expects `backend_type='mlx_lm'` or
  `backend_type='vllm_metal'`.
- Fix: verify each `SWITCHYARD_LOCAL_MODELS` entry uses the correct `backend_type` for
  the configured `model_identifier`.

Alias or serving-target mismatch:
- Symptom: requests to `model="chat-shared"` return no route available, or backend pin
  errors mention the backend does not belong to the target.
- Fix: ensure each deployment that should participate in the same route shares the same
  `serving_target`, and pin using the concrete deployment name such as
  `mlx-lm:chat-mlx`.

Dependency installed but backend still unhealthy:
- Symptom: `/readyz` lists the adapter but shows it unavailable.
- Fix: read the structured logs. Missing optional dependencies, invalid model identifiers,
  or backend-specific runtime configuration errors are surfaced at the adapter/runtime
  boundary instead of being swallowed by the router.

Streaming fallback confusion:
- Symptom: a streaming request fails after some chunks rather than switching backends.
- Fix: this is intentional. The current failover policy only allows failover before the first visible token
  is emitted. Mid-stream migration is not attempted.

Benchmark artifacts miss richer route metadata:
- Symptom: client-side artifact fields look sparse.
- Fix: start the gateway with `SWITCHYARD_METRICS_ENABLED=true` so the benchmark runner
  can merge route and backend execution metrics into each request record.

## Useful Commands

```bash
make setup
make setup-mlx
make setup-vllm-metal
make serve
make serve-metrics
make warmup
make bench-smoke
make bench-gateway
make check
```

Direct checks:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Repo Guide

- [README.md](/Users/rishivinodkumar/Atlas/README.md)
- [docs/architecture.md](/Users/rishivinodkumar/Atlas/docs/architecture.md)
- [docs/forge-stage-a.md](/Users/rishivinodkumar/Atlas/docs/forge-stage-a.md)
- [docs/hybrid-workers.md](/Users/rishivinodkumar/Atlas/docs/hybrid-workers.md)
- [docs/control-plane.md](/Users/rishivinodkumar/Atlas/docs/control-plane.md)
- [docs/benchmarking.md](/Users/rishivinodkumar/Atlas/docs/benchmarking.md)
- [docs/phase0.md](/Users/rishivinodkumar/Atlas/docs/phase0.md)
- [docs/phase1.md](/Users/rishivinodkumar/Atlas/docs/phase1.md)
- [docs/phase2.md](/Users/rishivinodkumar/Atlas/docs/phase2.md)
- [docs/phase3.md](/Users/rishivinodkumar/Atlas/docs/phase3.md)
- [docs/phase4.md](/Users/rishivinodkumar/Atlas/docs/phase4.md)
- [docs/phase7.md](/Users/rishivinodkumar/Atlas/docs/phase7.md)
- [docs/phase8.md](/Users/rishivinodkumar/Atlas/docs/phase8.md)
- [docs/phase8-exit-review.md](/Users/rishivinodkumar/Atlas/docs/phase8-exit-review.md)
- [docs/phase9.md](/Users/rishivinodkumar/Atlas/docs/phase9.md)
- [docs/infra.md](/Users/rishivinodkumar/Atlas/docs/infra.md)
- [docs/adr/0001-single-python-workspace.md](/Users/rishivinodkumar/Atlas/docs/adr/0001-single-python-workspace.md)
- [docs/adr/0002-optional-mlx-runtime-boundary.md](/Users/rishivinodkumar/Atlas/docs/adr/0002-optional-mlx-runtime-boundary.md)
- [docs/adr/0003-logical-alias-to-multiple-backend-deployments.md](/Users/rishivinodkumar/Atlas/docs/adr/0003-logical-alias-to-multiple-backend-deployments.md)
- [docs/adr/0004-artifact-source-of-truth-for-phase3-reporting.md](/Users/rishivinodkumar/Atlas/docs/adr/0004-artifact-source-of-truth-for-phase3-reporting.md)
- [docs/adr/0005-deterministic-canary-bucketing.md](/Users/rishivinodkumar/Atlas/docs/adr/0005-deterministic-canary-bucketing.md)
- [docs/adr/0008-phase7-optimization-ready-knob-surface.md](/Users/rishivinodkumar/Atlas/docs/adr/0008-phase7-optimization-ready-knob-surface.md)
- [docs/adr/0009-phase7-remote-workers-as-first-class-topology-members.md](/Users/rishivinodkumar/Atlas/docs/adr/0009-phase7-remote-workers-as-first-class-topology-members.md)
- [docs/adr/0010-phase8-canary-only-cloud-rollout-gating.md](/Users/rishivinodkumar/Atlas/docs/adr/0010-phase8-canary-only-cloud-rollout-gating.md)
- [docs/adr/0011-phase8-concrete-vllm-cuda-runtime-behind-generic-worker-contract.md](/Users/rishivinodkumar/Atlas/docs/adr/0011-phase8-concrete-vllm-cuda-runtime-behind-generic-worker-contract.md)
- [docs/adr/0012-phase9-forge-stage-a-optimization-and-promotion-model.md](/Users/rishivinodkumar/Atlas/docs/adr/0012-phase9-forge-stage-a-optimization-and-promotion-model.md)
