import pfun_path_helper as pph

pph.append_path(path=pph.get_lib_path("pfun_cma_model"))  # noqa: E402
from . import test_base

test_base.setup_test_environment()
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import Request, status
from fastapi.responses import Response
from fastapi.testclient import TestClient

from pfun_cma_model.app import app
from pfun_cma_model.data import read_sample_data

client = TestClient(app)


@pytest.fixture
def fake_request():
    # Minimal mock for FastAPI Request
    return MagicMock(spec=Request)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]
    )


@pytest.fixture()
def get_sample_dataset(fake_request, nrows=None):
    response = client.get("/data/sample/download", params={"nrows": nrows} if nrows is not None else {})
    return response


@patch("pfun_cma_model.data.read_sample_data")
def test_get_sample_dataset_route_integration(mock_read_sample_data, sample_df):
    # Integration test using TestClient
    mock_read_sample_data.return_value = sample_df
    response = client.get("/data/sample/download?nrows=2")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_get_sample_dataset_route_invalid_nrows():
    response = client.get("/data/sample/download?nrows=-5")
    assert response.status_code == 400
    assert "nrows" in response.text
