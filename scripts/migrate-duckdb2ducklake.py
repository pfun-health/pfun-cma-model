import duckdb
import argparse
import re
import os
from collections import deque

TYPE_MAPPING = {
    "VARINT": "::VARCHAR::INT",
    "UNION/ENUM": "::VARCHAR",
    "BIT": "::VARCHAR",
}


def get_postgres_secret():
    return f"""
        CREATE SECRET postgres_secret(
            TYPE postgres,
            HOST '{os.getenv("POSTGRES_HOST", "localhost")}',
            PORT {os.getenv("POSTGRES_PORT", "5432")},
            DATABASE {os.getenv("POSTGRES_DB", "migration_test")},
            USER '{os.getenv("POSTGRES_USER", "user")}',
            PASSWORD '{os.getenv("POSTGRES_PASSWORD", "simple")}'
        );"""


def _resolve_data_types(
    table: str, schema: str, catalog: str, conn: duckdb.DuckDBPyConnection
):
    excepts = []
    casts = []
    for col in conn.execute(
        f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}' AND table_schema = '{schema}' AND table_catalog = '{catalog}'"
    ).fetchall():
        col_name, col_type = col[0], col[1]
        # Handle mapped types
        if col_type in TYPE_MAPPING or re.match(r"(ENUM|UNION)\b", col_type):
            cast = TYPE_MAPPING.get(col_type) or TYPE_MAPPING["UNION/ENUM"]
            casts.append(f"{col_name}{cast} AS {col_name}")
            excepts.append(col_name)
        # Handle array types
        elif re.fullmatch(r"(INTEGER|VARCHAR|FLOAT)\[\d+\]", col_type):
            base_type = re.match(r"(INTEGER|VARCHAR|FLOAT)", col_type).group(1)
            cast = f"::{base_type}[]"
            casts.append(f"{col_name}{cast} AS {col_name}")
            excepts.append(col_name)
    return excepts, casts


def migrate_tables_and_views(duckdb_catalog: str, con: duckdb.DuckDBPyConnection):
    """
    Migrate tables and views from the DuckDB catalog to DuckLake using a queue system.
    If migration of a table or view fails, it will be re-added to the back of the queue.
    """
    rows = con.execute(
        f"SELECT table_catalog, table_schema, table_name, table_type "
        f"FROM information_schema.tables WHERE table_catalog = '{duckdb_catalog}'"
    ).fetchall()

    # The idea behind this queue is to retry failed migration of views due to missing dependencies.
    # The failed item is re-added to the back of the queue and waits for the rest of the dependencies to be migrated.
    # This avoids the need to generate a full dependency graph, which would make this script very complex.
    queue = deque(rows)
    failed_last_round = set()

    while queue:
        catalog, schema, table, table_type = queue.popleft()
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        try:
            if table_type == "VIEW":
                view_definition = con.execute(
                    f"SELECT view_definition FROM information_schema.views "
                    f"WHERE table_name = '{table}' AND table_schema = '{schema}' AND table_catalog = '{catalog}'"
                ).fetchone()[0]
                con.execute(
                    f"CREATE VIEW IF NOT EXISTS {view_definition.removeprefix('CREATE VIEW ')}"
                )
                print(f"Migrating Catalog: {catalog}, Schema: {schema}, View: {table}")
            else:
                excepts, casts = _resolve_data_types(table, schema, catalog, con)
                if casts:
                    select_clause = (
                        "* EXCLUDE(" + ", ".join(excepts) + "),\n" + ",\n".join(casts)
                    )
                    con.execute(
                        f"CREATE TABLE IF NOT EXISTS {schema}.{table} AS "
                        f"SELECT {select_clause} FROM {catalog}.{schema}.{table}"
                    )
                else:
                    con.execute(
                        f"CREATE TABLE IF NOT EXISTS {schema}.{table} AS "
                        f"SELECT * FROM {catalog}.{schema}.{table}"
                    )
                print(f"Migrating Catalog: {catalog}, Schema: {schema}, Table: {table}")
        except Exception as e:
            print(f"WARNING: Requeuing {table_type} {table}")
            # Prevent infinite loop if no progress is possible
            if (catalog, schema, table, table_type) in failed_last_round:
                print(
                    f"Skipping {table_type} {table} permanently due to repeated failure. {e}"
                )
                continue
            else:
                queue.append((catalog, schema, table, table_type))
                failed_last_round.add((catalog, schema, table, table_type))
        else:
            # Success — ensure we clear from failure set
            failed_last_round.discard((catalog, schema, table, table_type))


def migrate_macros(con: duckdb.DuckDBPyConnection, duckdb_catalog: str):
    """
    Migrate macros from the DuckDB catalog to DuckLake metadata database.
    """
    for row in con.execute(
        f"SELECT function_name, parameters, macro_definition FROM duckdb_functions() "
        f"WHERE database_name='{duckdb_catalog}'"
    ).fetchall():
        name, parameters, definition = row[0], row[1], row[2]
        print(f"Migrating Macro: {name}")
        con.execute(
            f"CREATE OR REPLACE MACRO {name}({','.join(parameters)}) AS {definition}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate DuckDB catalog to DuckLake.")
    parser.add_argument("--duckdb-catalog", required=True, help="DuckDB catalog name")
    parser.add_argument("--duckdb-file", required=True, help="Path to DuckDB file")
    parser.add_argument(
        "--ducklake-catalog", required=True, help="DuckLake catalog name"
    )
    parser.add_argument(
        "--catalog-type",
        choices=["duckdb", "postgresql", "sqlite"],
        required=True,
        help="Choose one of: duckdb, postgresql, sqlite",
    )
    parser.add_argument("--ducklake-file", required=False, help="Path to DuckLake file")
    parser.add_argument(
        "--ducklake-data-path", required=True, help="Data path for DuckLake"
    )

    args = parser.parse_args()

    con = duckdb.connect(database=args.duckdb_file)

    if args.catalog_type == "postgresql":
        con.execute(get_postgres_secret())

    secret = (
        "CREATE SECRET ducklake_secret (TYPE ducklake"
        + (
            f"\n,METADATA_PATH '{args.ducklake_file if args.catalog_type == 'duckdb' else f'sqlite:{args.ducklake_file}'}'"
            if args.catalog_type in ("duckdb", "sqlite")
            else "\n,METADATA_PATH ''"
        )
        + f"\n,DATA_PATH '{args.ducklake_data_path}'"
        + (
            "\n,METADATA_PARAMETERS MAP {'TYPE': 'postgres', 'SECRET': 'postgres_secret'});"
            if args.catalog_type == "postgresql"
            else ");"
        )
    )
    con.execute(secret)

    con.execute(
        f"ATTACH '{args.duckdb_file}' AS {args.duckdb_catalog};"
        f"ATTACH 'ducklake:ducklake_secret' AS {args.ducklake_catalog}; USE {args.ducklake_catalog};"
    )

    migrate_tables_and_views(
        duckdb_catalog=args.duckdb_catalog,
        con=con,
    )

    if args.catalog_type == "duckdb":
        # DETACH DuckLake to be able to attach to the metadata database in migrate_macros
        con.execute(f"USE {args.duckdb_catalog}; DETACH {args.ducklake_catalog};")
        con.execute(
            f"ATTACH '{args.ducklake_file}' AS ducklake_metadata; USE ducklake_metadata;"
        )
        migrate_macros(
            con=con,
            duckdb_catalog=args.duckdb_catalog,
        )
    con.close()
