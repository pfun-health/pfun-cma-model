import test_base
import pytest
from fastapi import status, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pfun_cma_model.app import app
import pandas as pd
from pfun_cma_model.app import get_sample_dataset
import json

client = TestClient(app)


@pytest.fixture
def fake_request():
    # Minimal mock for FastAPI Request
    return MagicMock(spec=Request)


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ])


def test_get_sample_dataset_invalid_nrows(fake_request):
    # nrows < -1 should raise HTTPException 400
    with pytest.raises(Exception) as excinfo:
        get_sample_dataset(fake_request, nrows=-2)
    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST


@patch("pfun_cma_model.app.read_sample_data")
def test_get_sample_dataset_full_dataset(mock_read_sample_data, fake_request, sample_df):
    # nrows = -1 should return the full dataset as JSON
    mock_read_sample_data.return_value = sample_df
    resp = get_sample_dataset(fake_request, nrows=-1)
    assert isinstance(resp, Response)
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "application/json"
    data = json.loads(resp.body)
    assert isinstance(data, list)
    assert len(data) == 3


@patch("pfun_cma_model.app.read_sample_data")
def test_get_sample_dataset_nrows_given(mock_read_sample_data, fake_request, sample_df):
    # nrows = 2 should return only first 2 rows
    mock_read_sample_data.return_value = sample_df
    resp = get_sample_dataset(fake_request, nrows=2)
    assert isinstance(resp, Response)
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["a"] == 1
    assert data[1]["a"] == 3


@patch("pfun_cma_model.app.read_sample_data")
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
