#!/usr/bin/env sh
set -eu

MODE="${SWITCHYARD_REMOTE_WORKER_START_MODE:-fake}"
HOST="${SWITCHYARD_REMOTE_WORKER_HOST:-0.0.0.0}"
PORT="${SWITCHYARD_REMOTE_WORKER_PORT:-8090}"
VERIFY_RUNTIME_IMPORT="${SWITCHYARD_REMOTE_WORKER_VERIFY_RUNTIME_IMPORT:-true}"
REQUIRE_NVIDIA_SMI="${SWITCHYARD_REMOTE_WORKER_REQUIRE_NVIDIA_SMI:-false}"

runtime_import_flag="--verify-runtime-import"
if [ "$VERIFY_RUNTIME_IMPORT" = "false" ]; then
  runtime_import_flag="--no-verify-runtime-import"
fi

nvidia_smi_flag="--no-require-nvidia-smi"
if [ "$REQUIRE_NVIDIA_SMI" = "true" ]; then
  nvidia_smi_flag="--require-nvidia-smi"
fi

case "$MODE" in
  fake)
    exec switchyard-fake-remote-worker --host "$HOST" --port "$PORT"
    ;;
  vllm_cuda)
    switchyard-vllm-cuda-worker check-config \
      "$runtime_import_flag" \
      "$nvidia_smi_flag" \
      --fail-on-issues
    exec switchyard-vllm-cuda-worker serve --host "$HOST" --port "$PORT"
    ;;
  configured)
    TARGET="${SWITCHYARD_REMOTE_WORKER_TARGET:-}"
    if [ -z "$TARGET" ]; then
      echo "SWITCHYARD_REMOTE_WORKER_TARGET is required when START_MODE=configured" >&2
      exit 2
    fi
    exec switchyard-worker serve "$TARGET" --host "$HOST" --port "$PORT"
    ;;
  *)
    echo "unsupported SWITCHYARD_REMOTE_WORKER_START_MODE: $MODE" >&2
    exit 2
    ;;
esac
