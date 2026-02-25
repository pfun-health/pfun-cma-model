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

client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


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


@patch("pfun_cma_model.routes.data.read_sample_data")
def test_get_sample_dataset_route_integration(mock_read_sample_data, sample_df):
    # Integration test using TestClient
    mock_read_sample_data.return_value = sample_df
    response = client.get("/data/sample/download?nrows=2&media_type=json")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@patch("pfun_cma_model.routes.data.read_sample_data")
def test_get_sample_dataset_route_invalid_nrows(mock_read_sample_data, sample_df):
    mock_read_sample_data.return_value = sample_df
    response = client.get("/data/sample/download?nrows=-5")
    assert response.status_code == 400
    assert "nrows" in response.text


def test_run_at_time_double_encoding():
    """Verify that /model/run-at-time returns a JSON object, not a JSON string."""
    response = client.post(
        "/model/run-at-time",
        params={"t0": 0, "t1": 24, "n": 10},
        json={}
    )
    assert response.status_code == 200
    content = response.content.decode('utf-8')
    assert content.startswith('{')
    assert not (content.startswith('"') and content.endswith('"'))


def test_fit_model_data_body():
    """Verify that /model/fit accepts data as JSON body."""
    data = [{"t": 0, "G": 5.0}, {"t": 1, "G": 5.0}]
    response = client.post("/model/fit", json={"data": data})
    # Accept 400 or 500 as long as it's not 422 (validation error) or the specific UnboundLocalError
    assert response.status_code in [400, 500]
    assert "error" in response.json()
    error_msg = response.json().get("error", "") + response.json().get("exception", "")
    assert "failed to fit data" in error_msg.lower() or "no raw time column" in error_msg.lower()
