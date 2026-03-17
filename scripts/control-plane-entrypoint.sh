#!/bin/sh
set -eu

command_name="${1:-gateway}"

case "$command_name" in
  gateway)
    shift
    exec switchyard-control-plane gateway \
      --host "${SWITCHYARD_GATEWAY_HOST:-0.0.0.0}" \
      --port "${SWITCHYARD_GATEWAY_PORT:-8000}" \
      --log-level "${SWITCHYARD_LOG_LEVEL:-info}" \
      "$@"
    ;;
  bench)
    shift
    exec python -m switchyard.bench.cli "$@"
    ;;
  check-config)
    shift
    exec switchyard-control-plane check-config "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
