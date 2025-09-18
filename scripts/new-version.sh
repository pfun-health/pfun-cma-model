#!/usr/bin/env bash

# new-version.sh
# Bump pfun-cma-model to a new patch version, record it.

set -e

# bump pfun-cma-model package version
uv version --bump patch && \
    uv sync && \
    uv build

# build and start the services in the background
docker compose up -d --build

create_new_tag() {
    # create tags for the latest version.
    # tags: VERSION, prod-VERSION
    local VERSION=$(uv version | grep -o '[0-9]*\.[0-9]*\.[0-9]*')
    echo "$VERSION" | xargs -I {} git tag {} && \
        echo "$VERSION" | xargs -I {} git tag prod-{}
}

# create a new commit
git add -A && \
    git commit -m "($(uv version)) bump to new version." && \
    git push && \
    git push github && \
    git push --tags && \
    git push --tags github