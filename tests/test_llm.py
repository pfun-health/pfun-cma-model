import pfun_path_helper as pph

pph.append_path(path=pph.get_lib_path("pfun_cma_model"))  # noqa: E402
from . import test_base

test_base.setup_test_environment()

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from pfun_cma_model.llm import (
    generate_scenario,
    _call_llm_for_json,
    _parse_generated_response,
    PFunLLMGeneratedScenario
)


@pytest.mark.asyncio
class TestGenerateScenario:
    @patch("pfun_cma_model.llm._call_llm_for_json")
    async def test_generate_scenario_success(self, mock_call_llm):
        # Setup mock return value
        mock_response = {
            "forecasted_events": "Test event",
            "qualitative_description": "Test description",
            "parameters": {
                "Cm": {"value": 1.5, "stderr": 0.1, "description": "Test Cm"}
            },
            "recommendations": {
                "diet": "Test diet recommendation"
            }
        }
        mock_call_llm.return_value = mock_response

        # Call function
        result = await generate_scenario(
            query="Test query",
            include_recommendations=True
        )

        # Assertions
        assert isinstance(result, PFunLLMGeneratedScenario)
        assert result.forecasted_events == "Test event"
        assert result.qualitative_description == "Test description"
        assert "Cm" in result.parameters
        assert "diet" in result.recommendations
        # The mocked LLM response omits health_info, so the fixed sample
        # profile is substituted and the flag must reflect that.
        assert result.used_fallback_health_info is True
        # The internal flag must never leak into model_dump() (persistence protection).
        assert "used_fallback_health_info" not in result.model_dump()

        # Verify call arguments
        mock_call_llm.assert_called_once()
        args, kwargs = mock_call_llm.call_args
        prompt = args[0]
        assert "User: \"Test query\"" in prompt
        assert "recommendations" in prompt

    @patch("pfun_cma_model.llm._call_llm_for_json")
    async def test_generate_scenario_with_health_info_no_fallback(self, mock_call_llm):
        mock_response = {
            "forecasted_events": "Test event",
            "qualitative_description": "Test description",
            "parameters": {
                "Cm": {"value": 1.5, "stderr": 0.1, "description": "Test Cm"}
            },
            "health_info": {
                "age": 50,
                "sex": "m",
                "stress_level": "low",
                "sleep_quality": "low",
                "circadian_misalignment": "low",
            },
            "recommendations": {
                "diet": "Test diet recommendation"
            },
        }
        mock_call_llm.return_value = mock_response

        result = await generate_scenario(query="Test query")

        # The LLM provided health_info, so no fallback substitution happens
        # and the flag must be False.
        assert isinstance(result, PFunLLMGeneratedScenario)
        assert result.used_fallback_health_info is False
        assert result.health_info.age == 50
        assert result.health_info.sex == "m"
        # The internal flag must never leak into model_dump() (persistence protection).
        assert "used_fallback_health_info" not in result.model_dump()
        # The internal flag must not be part of the LLM-facing schema.
        assert (
            "used_fallback_health_info"
            not in PFunLLMGeneratedScenario.model_json_schema()["properties"]
        )

    @patch("pfun_cma_model.llm._call_llm_for_json")
    async def test_generate_scenario_no_recommendations(self, mock_call_llm):
        mock_response = {
            "forecasted_events": "Test event",
            "qualitative_description": "Test description",
            "parameters": {
                "Cm": {"value": 1.5, "stderr": 0.1, "description": "Test Cm"}
            },
            "recommendations": {}
        }
        mock_call_llm.return_value = mock_response

        # Call function
        result = await generate_scenario(
            query="Another query",
            include_recommendations=False
        )

        # Assertions
        assert isinstance(result, PFunLLMGeneratedScenario)

        mock_call_llm.assert_called_once()
        args, kwargs = mock_call_llm.call_args
        prompt = args[0]
        assert "User: \"Another query\"" in prompt
        assert "\"recommendations\":" not in prompt.split("Assistant:")[0] # Make sure recommendations isn't in the prompt schema

    @patch("pfun_cma_model.llm._call_llm_for_json")
    async def test_generate_scenario_no_query(self, mock_call_llm):
        mock_response = {
            "forecasted_events": "Test event",
            "qualitative_description": "Test description",
            "parameters": {
                "Cm": {"value": 1.5, "stderr": 0.1, "description": "Test Cm"}
            },
            "recommendations": {}
        }
        mock_call_llm.return_value = mock_response

        result = await generate_scenario()

        args, kwargs = mock_call_llm.call_args
        prompt = args[0]
        assert "User: \"No query provided.\"" in prompt

    @patch("pfun_cma_model.llm._call_llm_for_json")
    async def test_generate_scenario_exception_bubbling(self, mock_call_llm):
        mock_call_llm.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await generate_scenario()

@pytest.mark.asyncio
class TestCallLLMForJson:
    @patch("pfun_cma_model.llm.GenerativeModel")
    @patch("pfun_cma_model.llm._parse_generated_response")
    async def test_call_llm_for_json_markdown_block(self, mock_parse, mock_gen_model):
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        # Markdown JSON response
        json_data = {"key": "value"}
        mock_parse.return_value = f"Some text here\n```json\n{json.dumps(json_data)}\n```\nMore text"

        result = await _call_llm_for_json("test prompt")
        assert result == json_data

    @patch("pfun_cma_model.llm.GenerativeModel")
    @patch("pfun_cma_model.llm._parse_generated_response")
    async def test_call_llm_for_json_no_markdown(self, mock_parse, mock_gen_model):
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        # Raw JSON response
        json_data = {"key": "value", "nested": {"test": 123}}
        mock_parse.return_value = f"   {json.dumps(json_data)}   "

        result = await _call_llm_for_json("test prompt")
        assert result == json_data

    @patch("pfun_cma_model.llm.GenerativeModel")
    @patch("pfun_cma_model.llm._parse_generated_response")
    async def test_call_llm_for_json_invalid_json(self, mock_parse, mock_gen_model):
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance

        mock_parse.return_value = "```json\n{invalid json}\n```"

        with pytest.raises(Exception, match="Failed to parse LLM API response"):
            await _call_llm_for_json("test prompt")


@pytest.mark.asyncio
class TestParseGeneratedResponse:
    async def test_parse_generated_response_sync(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "message": {"content": "Hello World"}
        }
        # Deliberately remove __await__ to simulate sync object
        del mock_response.__await__

        result = await _parse_generated_response(mock_response)
        assert result == "Hello World"

    async def test_parse_generated_response_async(self):
        mock_sync_response = MagicMock()
        mock_sync_response.model_dump.return_value = {
            "message": {"content": "Async Hello"}
        }
        del mock_sync_response.__await__

        async def mock_async_call():
            return mock_sync_response

        # mock_async_call() is an awaitable (coroutine)
        result = await _parse_generated_response(mock_async_call())
        assert result == "Async Hello"

    async def test_parse_generated_response_unicode_handling(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "message": {"content": "Unicode text '"}
        }
        del mock_response.__await__

        result = await _parse_generated_response(mock_response)
        assert result == 'Unicode text "'
