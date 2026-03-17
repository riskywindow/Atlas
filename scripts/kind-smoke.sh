#!/bin/sh
set -eu

WORKER_LOG="${TMPDIR:-/tmp}/switchyard-kind-mock-worker.log"
PORT_FORWARD_LOG="${TMPDIR:-/tmp}/switchyard-kind-port-forward.log"

cleanup() {
  if [ -n "${PORT_FORWARD_PID:-}" ]; then
    kill "${PORT_FORWARD_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${WORKER_PID:-}" ]; then
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

uv run python scripts/host_native_mock_worker.py >"${WORKER_LOG}" 2>&1 &
WORKER_PID=$!
sleep 2

./scripts/kind-bootstrap.sh
docker build -f infra/docker/Dockerfile.control-plane -t switchyard/control-plane:dev .
./scripts/kind-push-control-plane.sh switchyard/control-plane:dev
./scripts/kind-deploy-control-plane.sh smoke

kubectl -n switchyard port-forward service/switchyard-gateway 18000:8000 >"${PORT_FORWARD_LOG}" 2>&1 &
PORT_FORWARD_PID=$!
sleep 3

curl -sS http://127.0.0.1:18000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"chat-smoke","messages":[{"role":"user","content":"hello from kind"}]}'
