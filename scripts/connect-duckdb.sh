#!/usr/bin/env sh

set +e

# scripts/connect-duckdb.sh

echo "Connecting to DuckDB and running test queries..."
echo "  ===========================================  "


echo -e "\nTesting cma_pgrid table:\n"
echo "  ------------------------------------------------  "
duckdb results/duckdb.db 'SELECT * FROM cma_pgrid LIMIT 10'
duckdb results/duckdb.db 'SELECT COUNT(*) FROM cma_pgrid'

echo -e "\n===============================================\n"

echo -e "\nTesting cma_scenes table:\n"
echo "  ------------------------------------------------  "
duckdb results/duckdb.db 'SELECT * FROM cma_scenes LIMIT 10'
duckdb results/duckdb.db 'SELECT COUNT(*) FROM cma_scenes'

echo -e "\n===============================================\n"

echo -e "\nTesting cma_recs table:\n"
echo "  ------------------------------------------------  "
duckdb results/duckdb.db 'SELECT * FROM cma_recs LIMIT 10'
duckdb results/duckdb.db 'SELECT COUNT(*) FROM cma_recs'
