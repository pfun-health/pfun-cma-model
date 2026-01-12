"""Unit tests for pfun_gradio.gradio_ui module."""

import pfun_path_helper as pph

from . import test_base

test_base.setup_test_environment()
pph.append_path(path=pph.get_lib_path("pfun_cma_model"))
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Adjust path to import the gradio_ui module
sys.path.insert(0, str(Path(__file__).parent.parent / "pfun_gradio" / "pfun_gradio"))


@pytest.mark.asyncio
async def test_async_generate_parameters_success():
    """Test successful response with valid parameters."""
    from pfun_gradio.pfun_gradio.gradio_ui import async_generate_parameters

    description = "Test patient with diabetes"
    endpoint = "https://cloud.tail38611b.ts.net/llm/generate-scenario"

    mock_response = {
        "qualitative_description": "A 45-year-old with type 2 diabetes",
        "parameters": {
            "param1": 0.5,
            "param2": 1.2,
        },
    }

    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 200
    mock_response_obj.json.return_value = mock_response

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response_obj
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "## Description:" in result
        assert "A 45-year-old with type 2 diabetes" in result
        assert "## Generated Parameters:" in result
        assert "param1" in result
        assert "param2" in result
        mock_client.post.assert_called_once_with(
            endpoint, json={"description": description}, timeout=27
        )


@pytest.mark.asyncio
async def test_async_generate_parameters_no_parameters():
    """Test response with no parameters generated."""
    from pfun_gradio.pfun_gradio.gradio_ui import async_generate_parameters

    description = "Test description"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    mock_response = {"qualitative_description": "Some description", "parameters": {}}

    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 200
    mock_response_obj.json.return_value = mock_response

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response_obj
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "## Description:" in result
        assert "😞 No parameters generated." in result


@pytest.mark.asyncio
async def test_async_generate_parameters_http_error():
    """Test error response from endpoint."""
    from gradio_ui import async_generate_parameters

    description = "Test description"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 500
    mock_response_obj.text = "Internal Server Error"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response_obj
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "Error: 500 - Internal Server Error" in result


@pytest.mark.asyncio
async def test_async_generate_parameters_http_404():
    """Test 404 error response from endpoint."""
    from gradio_ui import async_generate_parameters

    description = "Test description"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 404
    mock_response_obj.text = "Not Found"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response_obj
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "Error: 404 - Not Found" in result


@pytest.mark.asyncio
async def test_async_generate_parameters_request_exception():
    """Test exception during HTTP request."""
    from gradio_ui import async_generate_parameters

    description = "Test description"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.side_effect = httpx.ConnectError("Connection failed")
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "Request failed:" in result
        assert "Connection failed" in result


@pytest.mark.asyncio
async def test_async_generate_parameters_timeout():
    """Test timeout exception during HTTP request."""
    from gradio_ui import async_generate_parameters

    description = "Test description"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "Request failed:" in result
        assert "Request timeout" in result


@pytest.mark.asyncio
async def test_async_generate_parameters_json_decode_error():
    """Test exception when JSON decoding fails."""
    from gradio_ui import async_generate_parameters

    description = "Test description"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 200
    mock_response_obj.json.side_effect = ValueError("Invalid JSON")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response_obj
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "Request failed:" in result
        assert "Invalid JSON" in result


@pytest.mark.asyncio
async def test_async_generate_parameters_with_multiple_params():
    """Test response with multiple parameters."""
    from gradio_ui import async_generate_parameters

    description = "Complex patient scenario"
    endpoint = "http://localhost:8000/llm/generate-scenario"

    mock_response = {
        "qualitative_description": "A complex case",
        "parameters": {
            "glucose_sensitivity": 2.5,
            "cortisol_amplitude": 1.8,
            "sleep_phase": 0.3,
            "metabolic_rate": 1.1,
        },
    }

    mock_response_obj = MagicMock()
    mock_response_obj.status_code = 200
    mock_response_obj.json.return_value = mock_response

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response_obj
        mock_client_cls.return_value = mock_client

        result = await async_generate_parameters(description, endpoint)

        assert "## Description:" in result
        assert "A complex case" in result
        assert "glucose_sensitivity" in result
        assert "cortisol_amplitude" in result
        assert "sleep_phase" in result
        assert "metabolic_rate" in result
        assert "## Generated Parameters:" in result
