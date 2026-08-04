import pytest
from pathlib import Path
import pandas as pd
import duckdb
from unittest.mock import patch
from pfun_cma_model.db import save2duckdb, query_duckdb

@pytest.fixture
def temp_db_path(tmp_path: Path):
    return str(tmp_path / "test.db")

@pytest.fixture
def sample_df():
    # Keep it simple, but do check_dtype=False in assertion
    return pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})

def test_save2duckdb_creates_and_inserts(temp_db_path, sample_df):
    save2duckdb(sample_df, db_path=temp_db_path, table_id="test_table")
    with duckdb.connect(temp_db_path) as con:
        result = con.sql("SELECT * FROM test_table").df()

    expected_df = pd.concat([sample_df, sample_df], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected_df, check_dtype=False)

def test_save2duckdb_appends_to_table(temp_db_path, sample_df):
    # First save creates AS SELECT + INSERT = 2 copies
    save2duckdb(sample_df, db_path=temp_db_path, table_id="test_table")

    # Second save does CREATE IF NOT EXISTS (noop) + INSERT = 1 copy
    # Total = 3 copies
    save2duckdb(sample_df, db_path=temp_db_path, table_id="test_table")

    # Verify
    with duckdb.connect(temp_db_path) as con:
        result = con.sql("SELECT * FROM test_table").df()

    expected_df = pd.concat([sample_df, sample_df, sample_df], ignore_index=True)
    pd.testing.assert_frame_equal(result, expected_df, check_dtype=False)

def test_query_duckdb_valid_query(temp_db_path, sample_df):
    save2duckdb(sample_df, db_path=temp_db_path, table_id="test_table")

    query = "SELECT id FROM test_table WHERE value = 'a'"
    result = query_duckdb(query, db_path=temp_db_path)

    # Since it inserts twice
    expected_df = pd.DataFrame({"id": [1, 1]})
    pd.testing.assert_frame_equal(result, expected_df, check_dtype=False)

def test_query_duckdb_invalid_query(temp_db_path):
    with pytest.raises(duckdb.CatalogException):
        query_duckdb("SELECT * FROM non_existent_table", db_path=temp_db_path)

def test_save2duckdb_empty_dataframe(temp_db_path):
    empty_df = pd.DataFrame(columns=["id", "value"])
    save2duckdb(empty_df, db_path=temp_db_path, table_id="test_table")

    with duckdb.connect(temp_db_path) as con:
        result = con.sql("SELECT * FROM test_table").df()

    pd.testing.assert_frame_equal(result, empty_df, check_dtype=False)

@patch("pfun_cma_model.db.logging.debug")
def test_save2duckdb_logging(mock_debug, temp_db_path, sample_df):
    save2duckdb(sample_df, db_path=temp_db_path, table_id="test_table")
    mock_debug.assert_called_once_with(
        "...saved df_result to table '%s', located in database '%s'", "test_table", temp_db_path
    )

@patch("pfun_cma_model.db.logging.debug")
def test_query_duckdb_logging(mock_debug, temp_db_path, sample_df):
    save2duckdb(sample_df, db_path=temp_db_path, table_id="test_table")
    query = "SELECT id FROM test_table WHERE value = 'a'"
    query_duckdb(query, db_path=temp_db_path)
    mock_debug.assert_called_with(
        "Executed query against database '%s': %s", temp_db_path, query
    )
