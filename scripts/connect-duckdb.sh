#!/usr/bin/env bash

set -e

# scripts/connect-duckdb.sh

duckdb results/duckdb.db 'select * from cma_pgrid'
