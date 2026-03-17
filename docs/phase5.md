# Phase 5

Phase 5 prepares Switchyard to move from a Mac-first local control plane into a
deployment-aware inference fabric. The goal is not a rebuild. The goal is to preserve
the existing routing and benchmarking core while making worker addressing, deployment
shape, and control-plane packaging explicit.

## Goals

- Preserve the existing typed control-plane foundation:
  - two real Mac-native backend families,
  - backend-agnostic routing,
  - health-aware fallback,
  - replayable benchmarking and trace capture with safe defaults.
- Make worker addressing explicit and explainable:
  - backend deployments may expose multiple worker instances,
  - network endpoints should be represented in typed config and artifacts,
  - operators should be able to explain which worker addresses were in play.
- Keep the Mac-first serving rule intact:
  - real Apple-Silicon model workers stay host-native by default,
  - the control plane can still run in portable deployment shapes around them.
- Keep the control plane deployable without Apple-specific runtime dependencies:
  - Apple imports remain optional and lazy,
  - containerized control-plane paths must not require MLX-LM or vLLM-Metal packages
    unless a host-native worker process actually needs them.
- Add deployment-aware local paths:
  - a containerized control plane,
  - a Docker Compose stack,
  - a small `kind` path for future cluster-oriented smoke coverage.
- Preserve benchmark artifacts as the source of truth:
  - deployed-topology benchmark artifacts remain authoritative,
  - markdown and runbooks derive from those artifacts rather than ad hoc log scraping.

## Definition Of Done

- The repo keeps a clean Python workspace with linting, typing, and tests.
- The shared contracts still support at least two real backend adapter paths:
  - `mlx_lm`
  - `vllm_metal`
- The FastAPI gateway still serves `GET /healthz`, `GET /readyz`, and
  `POST /v1/chat/completions` with health-aware fallback.
- Routing, admission, backend protection, affinity, and rollout logic remain outside the
  HTTP layer and benchmarkable without spinning up the API.
- Worker addressing is explicit in typed config or typed artifacts rather than being
  inferred from adapter names or hidden environment state.
- Real Apple-Silicon model workers remain host-native by default for local development.
- The control plane is deployable without Apple-specific runtime dependencies installed.
- The repo includes a documented Docker Compose path and a documented `kind` path.
- Benchmark and replay artifacts can record deployed topology and remain the source of
  truth for comparative analysis and reporting.
- CI-friendly tests do not require Apple GPU hardware.

## Non-Goals

- No giant package split or service mesh buildout.
- No production-grade Kubernetes platform.
- No request-path Ray integration.
- No default move of Apple-Silicon model workers into containers.
- No hidden worker discovery that cannot be explained from config or artifacts.
- No reporting layer that supersedes benchmark artifacts as the authoritative record.

## Rules

### Mac-First Worker Rule

Real Apple-Silicon model workers stay host-native by default. Switchyard can add
containerized or clustered control-plane paths around them, but the default local path
should continue to respect host-native Apple runtime expectations.

### Portable Control Plane Rule

The control plane must be deployable without Apple-specific runtime dependencies. Apple
backend packages belong at the adapter/runtime boundary and should remain optional for
CI, local control-plane containers, and future remote-worker control-plane deployments.

### Explicit Worker Addressing Rule

Worker addressing must be explicit and explainable. If the system can talk to a worker
over the network, the relevant endpoint should be representable in typed config, typed
inventory, or benchmark artifacts.

### Artifact Source Of Truth Rule

Deployed-topology benchmark artifacts remain the source of truth. Markdown reports,
runbooks, and operator explanations should derive from those artifacts rather than
reconstructing topology from log fragments after the fact.

## Packaging And Topology Contracts

Phase 5 packaging should stay coherent and explicit:

- base install: portable control plane plus built-in observability surface,
- dev/test tooling: linting, typing, and tests,
- Apple-worker extras: `mlx`, `vllm-metal`, and the combined `apple-workers` extra.

Phase 5 topology contracts should stay typed:

- deployment profile or mode,
- worker transport type,
- worker endpoint and instance descriptor,
- static instance inventory,
- registration and heartbeat metadata,
- image or build metadata where relevant,
- environment-specific topology layers.

Instance inventory should stay explicit and explainable:

- one logical alias may map to several concrete worker instances,
- each instance should carry a stable instance ID and endpoint,
- source of truth, health, last-seen state, and tags should stay serializable,
- routing and runtime inspection may still select at the deployment level, but they should
  be able to surface instance-level inventory and health where it matters.

## Host-Native Worker Serving

Phase 5 adds an explicit host-native worker-serving mode for the current Mac-native
backends. The worker wrapper stays intentionally small: it serves one backend adapter
over the Switchyard internal worker protocol plus a minimal public
`POST /v1/chat/completions` path.

On an M4 Pro Mac, a minimal side-by-side setup looks like:

```bash
uv run switchyard-worker serve mlx-lm:mlx-chat --host 127.0.0.1 --port 8101 --warmup-mode eager
uv run switchyard-worker serve vllm-metal:vllm-chat --host 127.0.0.1 --port 8102 --warmup-mode eager
```

The control plane should then target those processes through explicit instance
inventory and `worker_transport=http`. This keeps Apple worker runtimes host-native
while preserving a portable control-plane boundary.

## Status

Phase 5 now has:

- explicit network-addressable worker endpoints,
- typed backend-instance inventory,
- host-native worker serving for the Mac-native backends,
- a portable control-plane image,
- a Docker Compose workflow,
- a kind workflow,
- deployment-aware benchmark and replay artifacts,
- deployment diagnostics and preflight tooling,
- docs that make the host-native-worker plus portable-control-plane split explicit.

Remaining work should now be Phase 6 work rather than Phase 5 alignment cleanup.
