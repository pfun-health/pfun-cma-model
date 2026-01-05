#!/usr/bin/env sh

set -e

# setup-opentelemetry-collector.sh
# + ref: https://opentelemetry.io/docs/collector/quick-start/
# + ref: https://opentelemetry.io/docs/collector/install/docker/

# --- Prerequisites ---

# set the collector output logs filepath
export collector_output_log_fn='./logs/opentelemetry-collector-output.log'

# set Golang bin directory
export GOBIN=${GOBIN:-$(go env GOPATH)/bin}

setup_core_env() {
  # pull opentelemetry collector core docker image
  docker pull otel/opentelemetry-collector:0.142.0
  # install telemetrygen utility
  go install github.com/open-telemetry/opentelemetry-collector-contrib/cmd/telemetrygen@latest
}

setup_core_env

# --- initialize collector: collect, monitor telemetry

initialize_collector_collect_telemetry() {
  tee "${collector_output_log_fn}"
  docker run \
    -p 127.0.0.1:4317:4317 \
    -p 127.0.0.1:4318:4318 \
    -p 127.0.0.1:55679:55679 \
    otel/opentelemetry-collector:0.142.0 \
    2>&1 | tee -a "${collector_output_log_fn}" # Optionally tee output for easier search later
    echo "Generating, collecting telemetry collector output..."
}

initialize_collector_collect_telemetry &

check_collector_logs() {
  echo "# --- Collector Logs: ---"
  grep -E '^Span|(ID|Name|Kind|time|Status \w+)\s+:' "${collector_output_log_fn}"
  printf "\n"
  echo "Open: http://localhost:55679/debug/tracez"
  echo "Select one of the samples in the table to see any generated traces."
}

