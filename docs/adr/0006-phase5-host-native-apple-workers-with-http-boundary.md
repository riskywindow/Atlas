# ADR 0006: Keep Apple Workers Host-Native And Use An Explicit HTTP Worker Boundary

## Status

Accepted

## Context

Phase 5 needs Switchyard to become deployment-aware without losing its Mac-first
development path.

Two constraints matter at the same time:

- real Apple-Silicon runtimes such as MLX-LM and vLLM-Metal are most stable as
  host-native macOS processes,
- the control plane needs a portable boundary so it can later talk to remote or cloud GPU
  workers without being coupled to Apple-specific imports or process layout.

If the control plane continues to assume in-process execution only, later remote-worker
 support becomes a redesign. If Apple workers are forced into containers too early, the
main local path becomes less reliable and less explainable.

## Decision

Keep Apple worker processes host-native by default and make them explicit
network-addressable workers behind a small Switchyard-internal HTTP protocol.

That protocol covers:

- health,
- readiness,
- capabilities,
- warmup,
- generate,
- streaming generate.

The portable control plane may then run:

- locally in Python,
- in Docker Compose,
- in kind,
- later against remote or cloud GPU workers,

while still talking to the same typed worker boundary.

## Consequences

- Apple-specific runtime dependencies remain optional and isolated from the portable
  control plane.
- Host-native worker endpoints become explicit in config, diagnostics, runtime inspection,
  and benchmark artifacts.
- Deployment-aware benchmarking and replay can preserve real topology metadata.
- Local Compose and kind workflows stay honest: the control plane is containerized or
  clustered, but Apple workers are still external host processes by default.
- Future remote CUDA or cloud GPU workers can reuse the same worker protocol and
  instance-inventory model.
