import logging
import duckdb
from pandas import DataFrame


def save2duckdb(
    df_result: DataFrame, db_path: str = "results/duckdb.db", table_id: str = "cma_recs"
):
    """
    Save the contents of 'df_result' to the specified duckdb database 'db_path',
    creating 'table_id' if it doesn't already exist.
    """

    with duckdb.connect(database=db_path) as connection:
        # create the table if it doesn't yet exist
        connection.sql(
            f"CREATE TABLE IF NOT EXISTS {table_id} AS SELECT * FROM df_result"
        )
        # update the table otherwise
        connection.sql(f"INSERT INTO {table_id} SELECT * FROM df_result")
        connection.commit()

    logging.debug(
        "...saved df_result to table '%s', located in database '%s'", table_id, db_path
    )


def query_duckdb(query: str, db_path: str = "results/duckdb.db") -> DataFrame:
    """
    Execute the provided SQL query against the specified duckdb database and return the results as a DataFrame.
    """
    with duckdb.connect(database=db_path) as connection:
        result_df = connection.sql(query).df()
    logging.debug("Executed query against database '%s': %s", db_path, query)
    return result_df
