import pytest
from fastapi.testclient import TestClient
import os

os.environ["GUARD_PASSIVE_MODE"] = "true"  # Set guard_passive_mode to True for testing
import logging

logging.info("Starting test_demo_routes.py with GUARD_PASSIVE_MODE=%s", os.environ["GUARD_PASSIVE_MODE"])
from pfun_cma_model.app import app


# Fixture to ensure app is initialized and served for each test
@pytest.fixture(scope="module")
def test_client():
    with TestClient(
        app,
        base_url="https://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as c:
        yield c


DemoRoutes = pytest.mark.parametrize(
    "route",
    [
        "/demo/llm",
        "/demo/data-stream",
        "/demo/run-at-time",
        "/demo/canvas-wave",
        "/demo/webgl-demo",
        "/demo/full-model-run",
    ],
)


@DemoRoutes
def test_demo_route_exists(test_client, route):
    """Test that the demo route exists and returns a successful response."""
    response = test_client.get(route)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html" in response.text or "<html" in response.text


@DemoRoutes
def test_demo_html_structure(test_client, route):
    """Test that the demo route returns HTML content with basic structure."""
    response = test_client.get(route)
    assert "<html" in response.text
    assert "</html>" in response.text


def test_llm_demo_contains_bootstrap(test_client):
    """Test that the LLM demo route includes Bootstrap CSS and JS."""
    response = test_client.get("/demo/llm")
    assert response.status_code == 200
    assert "bootstrap.min.css" in response.text
    assert "cdn.jsdelivr.net" in response.text
    assert "year" in response.text or str(response.text).find(str(2026)) != -1


def test_canvas_wave_demo_contains_socketio(test_client):
    response = test_client.get("/demo/canvas-wave")
    assert response.status_code == 200
    assert "socket.io.min.js" in response.text
