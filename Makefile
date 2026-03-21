.PHONY: setup setup-mlx setup-vllm-metal lint typecheck test check serve serve-metrics warmup bench-smoke bench-gateway container-build-control-plane container-build-remote-worker container-build-vllm-cuda-worker container-check-control-plane container-check-vllm-cuda-worker compose-config compose-up compose-up-observability compose-down compose-up-vllm-cuda-worker kind-render-smoke kind-render-m4pro kind-bootstrap kind-push-control-plane kind-deploy-smoke kind-deploy-m4pro kind-smoke

setup:
	uv sync --dev

setup-mlx:
	uv sync --dev --extra mlx

setup-vllm-metal:
	uv sync --dev --extra vllm-metal

lint:
	uv run ruff check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check: lint typecheck test

serve:
	uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000

serve-metrics:
	SWITCHYARD_METRICS_ENABLED=true uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000

warmup:
	curl -sS http://127.0.0.1:8000/v1/chat/completions \
		-H 'content-type: application/json' \
		-d '{"model":"'"$$SWITCHYARD_DEFAULT_MODEL_ALIAS"'","messages":[{"role":"user","content":"ping"}],"max_output_tokens":1}' | python -m json.tool

bench-smoke:
	uv run python -m switchyard.bench.cli run-synthetic --request-count 3

bench-gateway:
	uv run python -m switchyard.bench.cli run-gateway --model "$$SWITCHYARD_DEFAULT_MODEL_ALIAS"

container-build-control-plane:
	docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .

container-build-remote-worker:
	docker build -f infra/docker/Dockerfile.remote-worker -t switchyard/remote-worker:dev .

container-build-vllm-cuda-worker:
	docker build -f infra/docker/Dockerfile.remote-worker --target runtime-vllm-cuda -t switchyard/remote-worker-vllm-cuda:dev .

container-check-control-plane:
	uv run switchyard-control-plane check-config

container-check-vllm-cuda-worker:
	uv run switchyard-vllm-cuda-worker check-config --verify-runtime-import --no-require-nvidia-smi

compose-config:
	docker compose -f infra/compose/compose.yaml config

compose-up:
	docker compose -f infra/compose/compose.yaml up -d

compose-up-vllm-cuda-worker:
	docker compose -f infra/compose/compose.yaml -f infra/compose/compose.vllm-cuda-worker.yaml up -d

compose-up-observability:
	docker compose -f infra/compose/compose.yaml --profile observability up -d

compose-down:
	docker compose -f infra/compose/compose.yaml down

kind-render-smoke:
	kubectl kustomize infra/kind/overlays/smoke

kind-render-m4pro:
	kubectl kustomize infra/kind/overlays/m4pro

kind-bootstrap:
	./scripts/kind-bootstrap.sh

kind-push-control-plane:
	./scripts/kind-push-control-plane.sh switchyard/control-plane:dev

kind-deploy-smoke:
	./scripts/kind-deploy-control-plane.sh smoke

kind-deploy-m4pro:
	./scripts/kind-deploy-control-plane.sh m4pro

kind-smoke:
	./scripts/kind-smoke.sh
