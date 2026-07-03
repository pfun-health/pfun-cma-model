#!/usr/bin/env bash

# scripts/push-docker.sh : Build and push Docker images for the current version.

set -e

# Load common functions
SCRIPT_DIRNAME="$(dirname "$0")"
source "${SCRIPT_DIRNAME}/_funcs.def.sh"

# Extract the current version
VERSION="$(uv version | grep -o '[0-9]*\.[0-9]*\.[0-9]*')"

if [ -z "$VERSION" ]; then
	echo "ERROR: Could not extract version from 'uv version'."
	exit 1
fi

IMAGE_BASE="ghcr.io/pfun-health/pfun-cma-model"
VERSION_TAG="${IMAGE_BASE}:${VERSION}"
LATEST_TAG="${IMAGE_BASE}:latest"

echo "Building Docker image..."
docker build \
	-t "${VERSION_TAG}" \
	-t "${LATEST_TAG}" \
	.

echo "Pushing version tag (${VERSION_TAG})..."
docker push "${VERSION_TAG}"

echo "Pushing latest tag (${LATEST_TAG})..."
docker push "${LATEST_TAG}"

echo "Done. Successfully built and pushed ${VERSION_TAG} and ${LATEST_TAG}."
