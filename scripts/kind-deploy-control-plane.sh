#!/bin/sh
set -eu

OVERLAY="${1:-smoke}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi

kubectl apply -k "infra/kind/overlays/${OVERLAY}"
kubectl -n switchyard rollout status deployment/switchyard-gateway --timeout=180s
