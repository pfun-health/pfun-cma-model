#!/usr/bin/env bash

set -e
set -x

# Run pytest with options to show the 5 slowest tests
# and use line-based tracebacks for easier readability in CI logs.

run_tests() {
    uv run python -m pytest \
        --durations=5 \
        --tb=line \
        tests \
        "${@}"
}

export -f run_tests

run_tests

# run the tests for sub-packages too

#(cd packages/pfun_qt_gui && run_tests)

(cd packages/pfun_common && run_tests)
