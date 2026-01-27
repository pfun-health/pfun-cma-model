#!/usr/bin/env bash

set -e

# scripts/connect-duckdb.sh

duckdb results/duckdb.db 'SELECT * FROM cma_pgrid LIMIT 10'
duckdb results/duckdb.db 'SELECT COUNT(*) FROM cma_pgrid'

echo -e "\n===============================================\n"

duckdb results/duckdb.db 'SELECT * FROM cma_scenes LIMIT 10'
duckdb results/duckdb.db 'SELECT COUNT(*) FROM cma_scenes'