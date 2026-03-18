#!/usr/bin/env sh
set -eu

MODE="${SWITCHYARD_REMOTE_WORKER_START_MODE:-fake}"
HOST="${SWITCHYARD_REMOTE_WORKER_HOST:-0.0.0.0}"
PORT="${SWITCHYARD_REMOTE_WORKER_PORT:-8090}"

case "$MODE" in
  fake)
    exec switchyard-fake-remote-worker --host "$HOST" --port "$PORT"
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
