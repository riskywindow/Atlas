# Phase 5 Codex Prompts for Switchyard

Use these prompts **one at a time** in Codex. Each prompt assumes Codex can read the repo and should follow `AGENTS.md`.

Keep the same discipline as earlier phases: do not ask Codex to build all of Phase 5 in one shot. Push it through small, reviewable vertical slices.

Phase 5 is the **deployment-topology and packaging** phase. The goal is to make Switchyard feel like a real local control plane that can run in a cluster-shaped environment on your Mac, while keeping real Apple-Silicon model backends host-native today and making future cloud GPU workers easy to add later.

Core Phase 5 outcomes:
- the control plane is explicitly separated from model execution,
- the control plane can talk to **network-addressable worker endpoints** instead of only in-process adapters,
- Apple-specific backend dependencies are isolated so the control-plane image is portable,
- a Mac-first local deployment path exists using **containerized control-plane services + host-native engine workers**,
- a **Docker Compose** stack exists for local deployment,
- a **kind** deployment path exists for a cluster-shaped local environment,
- local image build / registry flow is documented and reproducible,
- backend instance inventory and registration are explicit enough for later cloud workers,
- benchmarking and replay can target deployed topologies and preserve authoritative artifacts,
- docs and runbooks make the topology understandable instead of magical.

Recommended scenario families to test in Phase 5:
- a containerized gateway calling host-native Apple-Silicon workers,
- a kind-deployed control plane calling external worker endpoints,
- worker endpoint failure and recovery,
- multiple worker instances registered for the same logical alias,
- replay and benchmark capture against Compose and kind deployments,
- image rebuild + redeploy loops,
- profile switching between local dev, Compose, and kind.

Non-goals for Phase 5:
- no real cloud GPU rental yet,
- no vLLM-CUDA worker yet,
- no autoscaler or HPA yet,
- no service mesh,
- no production secrets manager,
- no multi-region or HA claims,
- no full dynamic service-discovery platform unless a tiny one is clearly justified,
- no full Kubernetes operator,
- no Forge or kernel optimization work yet,
- no requirement that Apple GPU workers run inside containers.

A good theme for this phase: **separate control plane from execution, package the control plane cleanly, and make deployment topologies explicit**.

---

## Prompt 0 - bootstrap instruction
Paste this first in a fresh Codex session.

```text
Read AGENTS.md first and follow it, but treat the repo as now entering Phase 5. The old current-phase text in AGENTS.md can be updated as part of this work. Switchyard is still a Mac-first, backend-agnostic inference fabric. For Phase 5, the major additions are: explicit network-addressable worker endpoints, isolation of Apple-specific backend dependencies from the portable control plane, containerization of the control plane, a Docker Compose deployment, a kind deployment path, explicit backend-instance inventory for future remote workers, and deployment-aware benchmarking and runbooks. Keep the design portable to later vLLM-CUDA and cloud GPU workers, keep tests CI-friendly without Apple GPU access, avoid overengineering, and ship in small vertical slices. For every task: inspect the repo, make a short plan, implement the smallest coherent change, run relevant checks, and summarize files changed plus commands run.
```

---

## Prompt 1 - Phase 5 kickoff and repo audit

```text
Inspect the current repo and prepare it for Phase 5.

Requirements:
- Review the codebase against the intended Phase 5 outcomes.
- Update AGENTS.md so the project phase is now Phase 5 instead of Phase 4.
- Add or update docs/phase5.md with:
  - Phase 5 goals,
  - definition of done,
  - non-goals,
  - the Mac-first rule that real Apple-Silicon model workers stay host-native by default,
  - the rule that the control plane must be deployable without Apple-specific runtime dependencies,
  - the rule that worker addressing must be explicit and explainable,
  - the rule that deployed-topology benchmark artifacts remain the source of truth.
- Identify any tiny Phase 4 gaps that obviously block Phase 5 and patch only the smallest necessary blockers.
- Do not do a giant refactor.

Acceptance criteria:
- AGENTS.md and docs reflect Phase 5 accurately.
- The repo has a crisp Phase 5 definition of done.
- Existing tests still pass.

Keep this focused. This is a repo-audit-and-alignment pass, not a rebuild.
```

---

## Prompt 2 - isolate dependency groups and harden topology/config contracts

```text
Prepare the repo so the control plane is portable while Apple-specific workers remain optional.

Requirements:
- Audit the current dependency layout and separate it into coherent groups or extras, for example:
  - core/control-plane,
  - observability/dev/test,
  - Apple worker extras such as MLX-LM and vLLM-Metal,
  - future remote-worker or CUDA placeholders only if clearly useful.
- Ensure control-plane code paths do not import Apple-specific worker dependencies at module import time. Use lazy imports or clean boundaries where appropriate.
- Extend or refine typed schemas/config for deployment topology, including at minimum:
  - deployment profile or mode,
  - worker transport type,
  - worker endpoint / instance descriptor,
  - static instance inventory,
  - health/registration metadata,
  - image tag / build metadata where relevant,
  - environment-specific config layering.
- Preserve existing Phase 4 behavior where practical. If a schema/config version bump is needed, make it explicit and documented.
- Add tests that prove the portable control-plane modules can load without Apple-specific extras installed.

Acceptance criteria:
- The control-plane environment can be installed and imported without MLX-LM or vLLM-Metal being present.
- Topology/config contracts for external workers are typed and documented.
- Existing tests still pass and new import-boundary tests exist.

Do not implement the worker server or deployment manifests yet unless a tiny placeholder is required to keep the code coherent.
```

---

## Prompt 3 - define a Switchyard worker protocol and implement a remote worker adapter

```text
Add an explicit network boundary between the control plane and model execution.

Requirements:
- Design and implement a small Switchyard-internal worker API/protocol for network-addressable workers.
- Keep the protocol focused and typed. It should cover at minimum:
  - health,
  - readiness or warm status,
  - capabilities,
  - warmup,
  - generate,
  - streaming generate if the current public API already supports streaming.
- Implement a RemoteWorkerAdapter (or similarly named adapter) that speaks this internal protocol over HTTP.
- Keep existing in-process adapters working for tests and local direct execution where useful.
- Make timeout, transport, and malformed-response handling explicit and observable.
- Ensure route decisions and telemetry can distinguish between local in-process execution and remote worker execution.
- Add tests using fake or mock HTTP workers so CI does not require Apple GPU access.

Acceptance criteria:
- The control plane can route to a worker over the network through a clean adapter boundary.
- Transport failures are explicit and testable.
- The design remains backend-agnostic and ready for later cloud workers.
- Tests pass.

Do not yet build the full worker-serving CLI for real Apple backends unless a tiny stub is necessary.
```

---

## Prompt 4 - wrap existing MLX-LM and vLLM-Metal backends as host-native worker services

```text
Make the existing Mac-native backends runnable as explicit worker processes.

Requirements:
- Add a lightweight worker-serving CLI or service entrypoint that can wrap an existing backend adapter and expose the Switchyard worker protocol over HTTP.
- Support the current real local backends, which should include the Mac-native MLX-LM path and the Phase 2 vLLM-Metal path if present.
- Keep the implementation host-native and Mac-friendly. Do not containerize the Apple GPU workers by default.
- Support configurable bind address, port, model alias or backend identity, and warmup behavior.
- Ensure health/readiness status reflects actual worker state rather than fake green lights.
- Preserve or restore public-path parity where practical, including streaming if the backend supports it.
- Add tests using mock or fake adapters so CI remains GPU-free.
- Document the minimal command set to run two local workers side-by-side on an M4 Pro Mac.

Acceptance criteria:
- A developer can run one or more Switchyard worker processes on the host and point the control plane at them.
- The worker service exposes typed health/capability/generation behavior.
- CI does not require Apple hardware to test the worker server wrapper.
- Tests pass.

Keep this practical. This is a worker boundary and host-native serving mode, not a large distributed worker framework.
```

---

## Prompt 5 - containerize the control plane cleanly

```text
Package the control plane into portable container images.

Requirements:
- Add Dockerfile(s) and supporting scripts/config needed to containerize the control-plane services.
- Prefer a small number of coherent images over a proliferation of tiny images. It is acceptable to use one main application image with multiple entrypoints if that matches the repo well.
- Ensure the control-plane image does not require Apple-specific runtime dependencies.
- Support at minimum the main gateway/control-plane path and any small admin/bench entrypoints that are clearly useful in deployment.
- Add health/readiness behavior that works in containerized environments.
- Keep image builds deterministic and documented enough for local iteration.
- If helpful, add a small build helper such as a Makefile target, task runner entry, or simple script, but keep it aligned with the repo style.
- Add smoke-level validation where practical, such as config validation or import checks.

Acceptance criteria:
- The control plane can be built into a portable container image on the Mac without dragging in MLX-LM or vLLM-Metal runtime dependencies.
- The image can start the gateway/control-plane successfully with documented commands.
- Tests and any relevant container smoke checks pass.

Do not try to containerize Apple GPU worker execution here.
```

---

## Prompt 6 - add a Docker Compose deployment for Mac-first local operation

```text
Create a Mac-first Docker Compose deployment for Switchyard.

Requirements:
- Add a Compose-based local deployment that includes the control-plane services and supporting infra that already exist or are clearly justified, such as Postgres, Redis, OpenTelemetry Collector, Prometheus, and Grafana.
- Real Apple-Silicon model workers should remain host-native and be addressed explicitly from the containerized control plane.
- Use explicit configuration for host-native worker endpoints. On macOS, it is acceptable for local-dev defaults to use host.docker.internal, but keep the endpoint abstraction generic so later environments can use ordinary addresses without code changes.
- Keep the first Compose setup developer-friendly and explainable. Avoid clever networking tricks unless clearly necessary.
- Ensure logs, health checks, and config wiring are coherent.
- Add at least one realistic local profile/config example for an M4 Pro Mac.
- Add a smoke test or documented smoke workflow that proves the containerized control plane can reach a host-native worker.

Acceptance criteria:
- A developer can run the control plane and supporting infra with Docker Compose while running the real model workers natively on macOS.
- The control plane can successfully route a request to a host-native worker endpoint.
- The workflow is documented and repeatable.

Do not add Kubernetes yet in this prompt.
```

---

## Prompt 7 - make backend instance inventory and registration first-class

```text
Add explicit backend-instance inventory so Switchyard is ready for future remote and cloud workers.

Requirements:
- Extend the backend/registry model so a logical alias can map to one or more concrete worker instances.
- Support at minimum:
  - stable instance IDs,
  - endpoint URL/address,
  - backend type,
  - transport type,
  - source of truth such as static config vs discovered/registered,
  - health/last-seen metadata,
  - optional tags such as local, canary, or experimental.
- Start with a practical local-first design. A static declarative inventory is enough if it is clean, but a tiny registration heartbeat can be added if it is clearly useful.
- Ensure routing, circuit breakers, canaries, and runtime inspection can operate at the instance level where appropriate.
- Keep the design explainable in artifacts, logs, and docs.
- Add tests for multiple instances behind one alias, instance selection, health-aware filtering, and inventory serialization.

Acceptance criteria:
- Switchyard has an explicit concept of backend instances rather than only abstract backends.
- The design is useful for host-native workers today and cloud workers later.
- Tests pass.

Do not build a heavyweight service-discovery system here.
```

---

## Prompt 8 - add a kind deployment path and local registry flow

```text
Create a cluster-shaped local deployment path using kind.

Requirements:
- Add a kind-based deployment path for the Switchyard control plane.
- Choose a simple manifest strategy that fits the repo, such as Kustomize overlays or clean plain YAML. Do not introduce Helm unless there is a compelling reason.
- Add the minimal Kubernetes resources needed to run the control plane and supporting config cleanly.
- Add a local registry workflow so locally built images can be loaded or pushed into kind in a reproducible way.
- Keep Apple-Silicon model workers external to the cluster by default. The kind deployment should call explicit worker endpoints rather than pretending Apple GPU workers are in-cluster.
- Document how worker endpoints are configured for kind on macOS.
- Add a smoke-level workflow that validates the kind deployment path.

Acceptance criteria:
- A developer can build the control-plane image, deploy it to kind, and configure it to talk to host-native worker endpoints.
- The manifest strategy is understandable and not overbuilt.
- Docs and scripts are sufficient for local iteration.

Do not add autoscaling, ingress controllers, or a service mesh here unless one is absolutely required for a tiny smoke path.
```

---

## Prompt 9 - make benchmarking and replay deployment-aware

```text
Extend the benchmarking and replay system so it works against deployed topologies and preserves topology metadata.

Requirements:
- Allow benchmark/replay tooling to target a deployed gateway URL rather than only an in-process or local-dev path.
- Extend authoritative artifacts to record deployment/topology metadata such as:
  - deployment profile,
  - control-plane version or image tag,
  - worker instance inventory snapshot,
  - relevant config/profile name,
  - whether the run targeted local dev, Compose, or kind.
- Preserve the rule that artifacts, not ad hoc logs, are the source of truth.
- Add at least one realistic example scenario for Compose and one for kind, even if the kind example is a smoke-scale scenario.
- Ensure Phase 3 and Phase 4 control-plane signals still appear coherently in the artifacts.
- Add tests for topology metadata serialization and any new targeting logic.

Acceptance criteria:
- Benchmark and replay can exercise deployed Switchyard topologies.
- Artifacts make deployment context explicit and reproducible.
- Tests pass.

Do not turn this into a giant deployment platform. Keep the benchmarking changes tightly aligned with Switchyard.
```

---

## Prompt 10 - add deployment diagnostics and preflight tooling

```text
Add lightweight deployment diagnostics so local and cluster workflows are debuggable.

Requirements:
- Provide a small CLI, admin endpoint, or tightly scoped diagnostic tool that can check at minimum:
  - effective deployment profile,
  - configured worker endpoints,
  - reachability of worker health endpoints,
  - current backend-instance inventory,
  - image/build metadata where available,
  - health of key supporting services that already exist in the repo.
- Prefer a local-dev-friendly and automation-friendly design.
- Keep output honest. If something cannot be verified, say so instead of inventing status.
- If useful, add a doctor/preflight mode that validates configuration before startup.
- Add tests for any new diagnostic helpers.

Acceptance criteria:
- A developer can quickly diagnose why a Compose or kind deployment is not reaching the expected workers.
- The diagnostics align with runtime truth and current config.
- Tests pass.

Do not build a large web dashboard here. A CLI and/or small admin endpoint is enough.
```

---

## Prompt 11 - docs, runbooks, and optional experimental notes for advanced local deployment

```text
Polish the docs and developer ergonomics for Phase 5 deployment workflows.

Requirements:
- Update README.md with a clear Mac-first Phase 5 workflow.
- Add docs/deployment.md or equivalent that explains:
  - host-native Apple worker processes,
  - the internal worker protocol,
  - containerized control-plane services,
  - Compose deployment,
  - kind deployment,
  - backend-instance inventory and registration,
  - deployment-aware benchmarking/replay,
  - diagnostic/preflight tooling,
  - how this phase prepares for later cloud GPU workers.
- Add example commands for:
  - starting host-native workers,
  - starting the Compose stack,
  - sending a request through the containerized gateway,
  - building and deploying the image to kind,
  - inspecting backend-instance inventory,
  - running deployment-aware replay/benchmark scenarios,
  - running deployment diagnostics.
- Update docs/architecture.md to show the control-plane / worker boundary explicitly.
- Add at least one short ADR for a key Phase 5 decision, such as the worker protocol boundary, external worker endpoints for Mac-first deployment, or the choice to keep Apple workers host-native.
- Optionally add a clearly labeled experimental note for future local variants such as minikube/krunkit or in-cluster Apple GPU experiments, but keep those explicitly non-core and unsupported for the main Phase 5 definition of done.

Acceptance criteria:
- A new contributor can understand and run the Phase 5 topology from the docs.
- The docs make the host-native-worker + containerized-control-plane split obvious.
- All checks still pass.
```

---

## Prompt 12 - Phase 5 exit review

```text
Review the repo against AGENTS.md and the intended Phase 5 definition of done.

Tasks:
- Identify anything missing, weak, or too clever in the current Phase 5 implementation.
- Tighten tests, docs, config naming, artifact clarity, deployment ergonomics, and topology observability where needed.
- Remove accidental overengineering.
- Verify that:
  - the control plane can run without Apple-specific runtime dependencies,
  - real Apple-Silicon workers can run host-native and be addressed explicitly,
  - a network-addressable worker boundary exists and is tested,
  - backend instances are first-class enough for future cloud expansion,
  - Docker Compose deployment works for the Mac-first local workflow,
  - kind deployment works as a cluster-shaped local path,
  - deployed-topology benchmarks and replay produce authoritative artifacts,
  - deployment diagnostics are useful and honest,
  - the design stays clean for later remote CUDA/cloud workers and later Forge work.
- Make code changes only where they clearly improve completeness or clarity.

Deliverables:
- a concise Phase 5 status summary,
- remaining gaps if any,
- the top 5 recommended Phase 6 tasks,
- code changes only where they clearly improve completeness or clarity.
```

---

## Optional planning prompt if you want Codex to reason before coding

```text
Read AGENTS.md and inspect the current repo. I want a Phase 5 implementation plan before any major coding. Produce:
1. the smallest set of code changes needed to add a network-addressable worker boundary, isolate Apple-specific dependencies from the portable control plane, package the control plane in containers, add Compose and kind deployment paths, and make backend instances first-class,
2. any schema/config changes needed,
3. the simplest local-first worker protocol and instance-inventory model you recommend,
4. the implementation order you would use,
5. the test strategy for CI without Apple GPU access,
6. the main correctness or ergonomics risks in a host-native-worker + containerized-control-plane deployment and how you would mitigate them,
7. how you would extend artifacts and runbooks so deployed-topology benchmarks remain explainable and reproducible.

Do not make big code changes yet unless you spot a tiny blocker worth fixing immediately.
```
