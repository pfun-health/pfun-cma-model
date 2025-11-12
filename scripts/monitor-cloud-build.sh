#!/usr/bin/env bash

# watch the cloud build (update every n=5 seconds)
monitor_cloud_build() {
    watch -n5 gcloud builds list --project=pfun-cma-model --sort-by=CREATE_TIME --limit=3
}

sleep 1s;
monitor_cloud_build
