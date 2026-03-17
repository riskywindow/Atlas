#!/bin/sh
set -eu

SOURCE_IMAGE="${1:-switchyard/control-plane:dev}"
REGISTRY_PORT="${SWITCHYARD_KIND_REGISTRY_PORT:-5001}"
TARGET_IMAGE="localhost:${REGISTRY_PORT}/switchyard-control-plane:dev"

docker image inspect "${SOURCE_IMAGE}" >/dev/null
docker tag "${SOURCE_IMAGE}" "${TARGET_IMAGE}"
docker push "${TARGET_IMAGE}"

printf '%s\n' "${TARGET_IMAGE}"
