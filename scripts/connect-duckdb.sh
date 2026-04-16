#!/usr/bin/env sh

set +e

# scripts/connect-duckdb.sh

echo "Connecting to DuckDB and running test queries..."
echo "  ===========================================  "


# echo -e "\nTesting cma_pgrid table:\n"
# echo "  ------------------------------------------------  "
# duckdb results/duckdb-local.db 'SELECT * FROM cma_pgrid LIMIT 10'
# duckdb results/duckdb-local.db 'SELECT COUNT(*) FROM cma_pgrid'
# echo -e "\n===============================================\n"

# echo -e "\nTesting cma_scenes table:\n"
# echo "  ------------------------------------------------  "
# duckdb results/duckdb-local.db 'SELECT * FROM cma_scenes LIMIT 10'
# duckdb results/duckdb-local.db 'SELECT COUNT(*) FROM cma_scenes'
# echo -e "\n===============================================\n"

test_cma_recs() {
    echo -e "\n# Testing cma_recs table:\n"
    echo "  ------------------------------------------------  "

    echo -e "\n# Head (first 10 results):"
    echo      "........................."
    duckdb results/duckdb-local.db 'SELECT * FROM cma_recs LIMIT 10'
    echo ""

    echo -e "\n# Tail (last 10 results):"
    echo      "........................."
    duckdb results/duckdb-local.db "$(cat ./snippets/sql/select_last_10_rows.sql)"
    echo ""

    echo -e "\n# Count (total number of rows):"
    echo      "........................."
    duckdb results/duckdb-local.db 'SELECT COUNT(*) FROM cma_recs'
}


# Execute duckdb tests:

test_cma_recs

echo ""
echo "...done."
