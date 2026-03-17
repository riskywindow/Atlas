# Phase 5 Benchmarking

Switchyard Phase 5 treats benchmarking, replay, comparison, and reporting as first-class
developer workflows built on typed artifacts.

The intended loop is:
1. generate a deterministic workload or opt into trace capture,
2. execute through the normal gateway path,
3. compare the same inputs across two execution targets,
4. generate markdown from the authoritative JSON artifact.

## Principles

- Mac-first, not Mac-locked: examples assume an Apple Silicon Mac, but workload,
  benchmark, replay, and report logic stay backend-agnostic.
- Artifact-first: JSON artifacts are the source of truth. Markdown is derived.
- Deployment-aware: when a run depends on a deployed topology, the artifact should carry
  enough endpoint context to explain what was exercised.
- Privacy-conscious: trace capture is off by default and must be enabled deliberately.
- Local-friendly: example sizes are realistic for an M4 Pro and should not require Apple
  GPU access in CI.

## Recommended Local Sizes

These settings are intentionally modest for a 24 GB M4 Pro:

- workload request count: `4` to `12`
- warmup requests: `1` to `2`
- burst size: `2` to `4`
- replay concurrency: `1` to `4`
- timeout: `30` to `60` seconds

Use larger values only after the base path is healthy and the artifact contents look
correct.

## Synthetic Workload Generation

Generate deterministic workload manifests from a seed. The built-in families are:

- `short_chat`
- `long_prompt`
- `repeated_prefix`
- `concurrency_burst`
- `queue_saturation`
- `tenant_contention`
- `backend_flakiness`
- `session_stickiness`
- `canary_rollout`
- `shadow_traffic`
- `mixed`

Example:

```bash
uv run python -m switchyard.bench.cli generate-workload \
  --family mixed \
  --model-alias chat-shared \
  --request-count 8 \
  --seed 17
```

Repeated-prefix example:

```bash
uv run python -m switchyard.bench.cli generate-workload \
  --family repeated_prefix \
  --model-alias chat-shared \
  --request-count 6 \
  --seed 11
```

Phase 5 local-control-plane example:

```bash
uv run python -m switchyard.bench.cli generate-workload \
  --family tenant_contention \
  --model-alias chat-shared \
  --request-count 8 \
  --seed 21
```

The repo also includes a local M4 Pro-sized example manifest:

- `docs/examples/phase5_local_m4pro_workload.json`
- `docs/examples/phase5_compose_benchmark_workload.json`
- `docs/examples/phase5_kind_smoke_workload.json`

The generated manifest is reusable across:
- normal alias routing,
- routing-policy overrides,
- pinned-backend runs,
- future backend targets.

The current families are intentionally small and artifact-oriented:

- `queue_saturation`
  Useful for exercising bounded admission, short queueing, and explicit rejection.
- `tenant_contention`
  Mixes a hot tenant with a colder tenant so per-tenant caps and fairness decisions are visible.
- `backend_flakiness`
  Provides a stable request shape for mock or flaky backends where breaker behavior should show up.
- `session_stickiness`
  Reuses session IDs across turns so sticky-hit, miss, expiry, and failover behavior is reconstructable.
- `canary_rollout`
  Uses stable session bucket keys so rollout selection is deterministic enough to test.
- `shadow_traffic`
  Generates explicit shadow-eligible requests. Primary responses remain authoritative.

## Benchmark Execution

Benchmark execution should reuse the normal gateway/control-plane path where possible.
The benchmark artifact remains the source of truth for admission, breaker, affinity,
canary, shadow, and deployed-topology decisions. Markdown is derived from those recorded
request records.

Run a workload against the logical alias:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --markdown-report
```

Run the same path against a Compose deployment and capture deployed topology metadata into
the artifact:

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

Run a smoke-scale kind benchmark against a local cluster-shaped deployment:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path docs/examples/phase5_kind_smoke_workload.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --deployment-target kind \
  --deployment-profile kind \
  --config-profile-name phase5-kind-smoke \
  --control-plane-image-tag localhost:5001/switchyard/control-plane:dev \
  --markdown-report
```

By default the benchmark runner will try to snapshot `/admin/runtime` from the target
gateway and fold that worker-instance inventory into the JSON artifact. Use
`--runtime-inspection-path none` only when the deployed gateway intentionally does not
expose runtime inspection.

Run with a routing policy override:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --policy balanced \
  --warmup-request-count 1 \
  --markdown-report
```

Run pinned to a backend:

```bash
uv run python -m switchyard.bench.cli run-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --pinned-backend mlx-lm:chat-mlx \
  --warmup-request-count 1 \
  --markdown-report
```

For quick smoke tests without a manifest:

```bash
uv run python -m switchyard.bench.cli run-synthetic \
  --request-count 6 \
  --workload-pattern repeated_prefix \
  --shared-prefix "Shared retrieval context: customer tier gold" \
  --markdown-report
```

## Trace Capture Modes

Trace capture must be enabled explicitly with environment variables.

Configuration:

```bash
export SWITCHYARD_TRACE_CAPTURE_MODE=metadata_only
export SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH=artifacts/traces/gateway-traces.jsonl
```

Modes:

- `off`
  Default. No traces are written.
- `metadata_only`
  Safest mode. Stores routing and request/response metadata without prompt or output
  bodies. Useful for operational inspection, not replay.
- `redacted_content`
  Stores normalized request and response shapes with redacted content markers. This is
  the recommended replay-oriented mode when you want safer local artifacts.
- `full_content`
  Stores normalized request and response payloads with raw content. Use this only when
  you intentionally need exact local replay inputs and have accepted the privacy risk.

Start the gateway with safe trace capture:

```bash
SWITCHYARD_TRACE_CAPTURE_MODE=metadata_only \
SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH=artifacts/traces/gateway-traces.jsonl \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

Start the gateway with replayable redacted capture:

```bash
SWITCHYARD_TRACE_CAPTURE_MODE=redacted_content \
SWITCHYARD_TRACE_CAPTURE_OUTPUT_PATH=artifacts/traces/gateway-traces.jsonl \
uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000
```

## Privacy And Safety

Trace capture is for benchmarking, debugging, and replay preparation. It is not a general
analytics pipeline.

Rules:

- leave capture off unless you actively need it,
- prefer `metadata_only` for routine local development,
- use `redacted_content` when you need replay structure without raw prompt retention,
- use `full_content` only on intentionally controlled data,
- do not treat auth headers or unrelated HTTP metadata as benchmark inputs,
- rotate or delete local trace files when they are no longer needed.

`metadata_only` traces are intentionally not replayable because they do not contain a
request body.

## Trace Replay

Replay captured traces through Switchyard rather than talking directly to one backend.

Replay against the same logical alias:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --replay-mode sequential \
  --markdown-report
```

Replay against a deployed Compose or kind control plane in the same way. The replay
artifact should carry the deployment target, deployment profile, config profile name,
control-plane image tag, and captured worker-instance inventory snapshot so the artifact
remains the source of truth for what topology was exercised.

Replay against a routing policy:

```bash
uv run python -m switchyard.bench.cli replay-traces \
  --trace-path artifacts/traces/gateway-traces.jsonl \
  --gateway-base-url http://127.0.0.1:8000 \
  --policy latency_first \
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

Replay modes currently supported:

- `sequential`
- `fixed_concurrency`
- `preserve_order_without_original_timing`

The code is intentionally structured so time-scaled replay can be added later without
rewriting artifact or planning contracts.

## Phase 5 Report Signals

Phase 5 reports should make control-plane behavior explainable from the artifact:

- admission outcomes and queue wait summaries,
- breaker phases and reasons,
- session-affinity hit/miss summaries,
- canary-selection summaries and policy names,
- shadow-routing dispositions and target summaries,
- notable control-plane notes captured on route annotations.

## A/B Comparison

Phase 5 comparisons must use the same workload items or captured trace set on both sides.

Policy A vs policy B:

```bash
uv run python -m switchyard.bench.cli compare-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --left-policy balanced \
  --right-policy latency_first \
  --warmup-request-count 1 \
  --markdown-report
```

Pinned backend A vs pinned backend B:

```bash
uv run python -m switchyard.bench.cli compare-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --left-pinned-backend mlx-lm:chat-mlx \
  --right-pinned-backend vllm-metal:chat-metal \
  --warmup-request-count 1 \
  --markdown-report
```

Alias routed normally vs alias pinned to one backend:

```bash
uv run python -m switchyard.bench.cli compare-workload \
  --manifest-path artifacts/benchmarks/mixed_17_8.json \
  --gateway-base-url http://127.0.0.1:8000 \
  --right-pinned-backend mlx-lm:chat-mlx \
  --markdown-report
```

Trace-set comparison:

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

## Markdown Report Generation

Reports are derived from typed artifacts rather than ad hoc logs.

Generate a report for one artifact:

```bash
uv run python -m switchyard.bench.cli generate-report \
  artifacts/benchmarks/20260316T120000Z_balanced.json
```

Generate one markdown report from multiple artifacts:

```bash
uv run python -m switchyard.bench.cli generate-report \
  artifacts/benchmarks/20260316T120000Z_balanced.json \
  artifacts/benchmarks/20260316T121500Z_compare_abc123.json \
  --output-path artifacts/benchmarks/phase3-review.md
```

The report layer should summarize:

- run metadata,
- environment summary,
- benchmark configuration,
- scenario mix,
- aggregate metrics,
- per-scenario tables,
- route and backend distributions,
- fallback and error summaries,
- notable deltas for comparison runs.

If something appears in markdown but not in the JSON artifact, that is a design bug.

## Artifact Expectations

Benchmark and replay artifacts should remain easy to diff and archive.

At minimum they carry:

- schema version,
- timestamps,
- git revision when available,
- environment snapshot,
- execution target,
- scenario seed and workload metadata,
- route and fallback outcomes,
- per-request timings and token counts,
- error details when a request fails.

This is the contract that downstream comparison and reporting should trust.
