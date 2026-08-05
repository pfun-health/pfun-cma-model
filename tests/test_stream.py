import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pfun_cma_model.stream import async_generate_parameters

@pytest.mark.asyncio
async def test_async_generate_parameters_exception():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.stream.side_effect = Exception("Connection Timeout")

        description = "Test scenario"
        endpoint = "http://test-endpoint"

        result = await async_generate_parameters(description, endpoint)

        assert result == "Request failed: Connection Timeout"

@pytest.mark.asyncio
async def test_async_generate_parameters_success_with_params():
    with patch("httpx.AsyncClient") as mock_client_class, \
         patch("pandas.DataFrame.to_markdown") as mock_to_markdown:

        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.__aenter__.return_value = mock_client_instance

        mock_stream_context = MagicMock()
        mock_client_instance.stream.return_value = mock_stream_context

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "qualitative_description": "A successful test description.",
            "parameters": {
                "param1": {"value": 1},
                "param2": {"value": 2}
            }
        }
        mock_stream_context.__aenter__.return_value = mock_response

        # Mock to_markdown since pandas will convert the dict to df
        mock_to_markdown.return_value = "| Parameter | value |\n| --- | --- |\n| param1 | 1 |\n| param2 | 2 |"

        description = "Test scenario"
        endpoint = "http://test-endpoint"

        result = await async_generate_parameters(description, endpoint)

        assert "## Description:\n" in result
        assert "A successful test description." in result
        assert "## Generated Parameters:\n" in result
        assert "| param1 | 1 |" in result

@pytest.mark.asyncio
async def test_async_generate_parameters_success_no_params():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.__aenter__.return_value = mock_client_instance

        mock_stream_context = MagicMock()
        mock_client_instance.stream.return_value = mock_stream_context

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "qualitative_description": "A successful test description."
        }
        mock_stream_context.__aenter__.return_value = mock_response

        description = "Test scenario"
        endpoint = "http://test-endpoint"

        result = await async_generate_parameters(description, endpoint)

        assert result == "😞 No parameters generated.\n"

@pytest.mark.asyncio
async def test_async_generate_parameters_non_200():
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.__aenter__.return_value = mock_client_instance

        mock_stream_context = MagicMock()
        mock_client_instance.stream.return_value = mock_stream_context

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_stream_context.__aenter__.return_value = mock_response

        description = "Test scenario"
        endpoint = "http://test-endpoint"

        result = await async_generate_parameters(description, endpoint)

        assert result == "Error: 500 - Internal Server Error"
