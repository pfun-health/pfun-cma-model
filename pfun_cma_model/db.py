import logging
import pandas as pd
import duckdb


def save2duckdb(df: pd.DataFrame, db_path: str, table_id: str) -> None:
    with duckdb.connect(db_path) as con:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {table_id} AS SELECT * FROM df WHERE FALSE"
        )
        con.execute(f"INSERT INTO {table_id} SELECT * FROM df")
    logging.debug(
        "...saved df_result to table '%s', located in database '%s'", table_id, db_path
    )


def query_duckdb(query: str, db_path: str) -> pd.DataFrame:
    with duckdb.connect(db_path) as con:
        result = con.sql(query).df()
    logging.debug("Executed query against database '%s': %s", db_path, query)
    return result
