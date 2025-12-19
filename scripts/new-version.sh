#!/usr/bin/env bash

# new-version.sh
# Bump pfun-cma-model to a new patch version, record it.

set -e

# Load common functions
source "$(dirname "$0")/_funcs.def.sh"

# bump pfun-cma-model package version
uv version --bump patch --project pfun-cma-model &&
/usr/bin/env bash -c 'cd packages/pfun_gradio && uv version --bump patch --project pfun-gradio && cd -' &&
/usr/bin/env bash -c 'cd packages/pfun_common && uv version --bump patch --project pfun-common && cd -'

# sync uv.lock and build the package
full_uv_sync && uv build

# # build and start the services in the background
# docker compose up -d --build --quiet || echo -e "Skipping docker rebuild..."
# sleep 1s

create_new_tag() {
	# create tags for the latest version.
	# tags: VERSION, prod-VERSION
	local VERSION=$(uv version | grep -o '[0-9]*\.[0-9]*\.[0-9]*')
	echo "$VERSION" | xargs -I {} git tag {}
	echo "$VERSION" | xargs -I {} git tag "prod-{}"
}

# regenerate the openapi.json and updated client
# ! runs in the background
nohup ./scripts/openapi-generate-pfun.sh &

# create a new commit
git add -A &&
	git commit -m "($(uv version)) bump to new version."

# create new tags
create_new_tag &&
	git push &&
	git push github &&
	git push --tags &&
	git push --tags github

# # watch the cloud build (update every n=5 seconds)
# sleep 1s
# bash -c 'scripts/monitor-cloud-build.sh'

echo -e "\n🎉 Successfully bumped to new version: $(uv version) 🎉\n"
