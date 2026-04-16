#!/usr/bin/env sh

set +e

# scripts/connect-duckdb.sh

echo -e "Connecting to DuckDB'results/duckdb-local.db', running test queries..."
echo -e "=============================================\n"

test_cma_recs() {
    echo -e "\n# Testing cma_recs table:"
    echo      "........................."
    echo -e   ".........................\n"

    echo -e "\n# Head (first 10 results):"
    echo      "........................."
    duckdb results/duckdb-local.db 'SELECT * FROM cma_recs LIMIT 10'
    echo ""

    echo -e "\n# Tail (last 10 results):"
    echo      "........................."
    duckdb \
	"results/duckdb-local.db" \
	"$(cat snippets/sql/macro_select_last_n_rows.sql)"
    echo ""

    echo -e "\n# Count (total number of rows):"
    echo      "..............................."
    duckdb results/duckdb-local.db 'SELECT COUNT(*) FROM cma_recs'
}

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


# Execute duckdb tests:

test_cma_recs

echo ""
echo "...done."
