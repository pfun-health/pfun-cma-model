#!/usr/bin/env sh

# test-opentelemetry-collector.sh

set -e

. ./.envrc

generate_sample_traces() {
  # #$GOBIN/telemetrygen traces --otlp-insecure --traces 3
  $GOBIN/telemetrygen traces --otlp-insecure \
    --traces 3 2>&1 | grep -E 'start|traces|stop'
}

generate_sample_traces