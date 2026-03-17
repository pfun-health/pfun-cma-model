#!/usr/bin/env bash

set -e
set -x

# Run pytest with options to show the 5 slowest tests and use line-based tracebacks for easier readability in CI logs.

pytest \
	--durations=5 \
	--tb=line \
	tests \
	"${@}"
