#!/bin/sh
set -eu

CLUSTER_NAME="${SWITCHYARD_KIND_CLUSTER_NAME:-switchyard}"
REGISTRY_NAME="${SWITCHYARD_KIND_REGISTRY_NAME:-switchyard-kind-registry}"
REGISTRY_PORT="${SWITCHYARD_KIND_REGISTRY_PORT:-5001}"
CLUSTER_CONFIG="${SWITCHYARD_KIND_CLUSTER_CONFIG:-infra/kind/cluster.yaml}"

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is required" >&2
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required" >&2
  exit 1
fi

if ! docker inspect -f '{{.State.Running}}' "${REGISTRY_NAME}" >/dev/null 2>&1; then
  docker run -d --restart=always -p "127.0.0.1:${REGISTRY_PORT}:5000" --name "${REGISTRY_NAME}" registry:2
fi

if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${CLUSTER_CONFIG}"
fi

if [ "$(docker inspect -f='{{json .NetworkSettings.Networks.kind}}' "${REGISTRY_NAME}")" = "null" ]; then
  docker network connect kind "${REGISTRY_NAME}"
fi

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REGISTRY_PORT}"
    help: "Build or tag images as localhost:${REGISTRY_PORT}/switchyard-control-plane:dev and push them before deploying to kind."
EOF
