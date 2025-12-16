"""Demo UI endpoint for LLM-based CMA parameter suggestions.

Uses Gradio for the interface. Hits the /llm/generate-scenario endpoint
to generate a scenario based on user input.
"""

import logging
import gradio as gr
import httpx
import asyncio
from pfun_common.utils import load_environment_variables, setup_logging
import os


# Initially, Get the logger (globally accessible)
# Will be overridden by setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info("Logger initialized for pfun_cma_model (logger name: %s)", logger.name)

# Ensure the .env file is loaded
load_environment_variables(logger=logger)

# Global variables and constants
debug_mode: bool = os.getenv("DEBUG", "0") in ["1", "true"]
# Perform logging setup...
setup_logging(logger, debug_mode=debug_mode)


def get_default_description():
    return "The patient is a 45-year-old male with type 2 diabetes and a history of hypoglycemia."


async def async_generate_parameters(description, llm_gen_scenario_endpoint):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                llm_gen_scenario_endpoint, json={"description": description}
            )
            if response.status_code == 200:
                return response.json().get(
                    "suggested_parameters", "No parameters returned."
                )
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request failed: {e}"


def setup_gradio_ui(
    llm_gen_scenario_endpoint: str,
):
    """Set up the Gradio demo interface using gr.Interface."""

    async def interface_fn(description):
        return await async_generate_parameters(description, llm_gen_scenario_endpoint)

    placeholder_text = "E.g., 'The patient has type 1 diabetes and struggle with high blood sugar after meals.'"
    default_value = get_default_description()

    iface = gr.Interface(
        fn=interface_fn,
        inputs=gr.Textbox(
            value=default_value,
            label="Scenario Description (third-person)",
            placeholder=placeholder_text,
            lines=4,
        ),
        outputs=gr.Textbox(
            label="Likely CMA Parameters",
            placeholder="CMA parameters will appear here...",
            lines=10,
        ),
        title="PFun CMA Model - Generate Condition-Based Parameters",
        description=(
            "This demo uses a Large Language Model (LLM) to suggest CMA model parameters "
            "based on a brief description of the user's condition. "
            "Enter a description below and click 'Submit' to see the suggestions."
        ),
        allow_flagging="never",
        concurrency_limit=1,
        examples=[
            [
                "The patient has type 1 diabetes and struggles with high blood sugar after meals."
            ],
            [
                "A 60-year-old woman with well-controlled type 2 diabetes and mild hypertension."
            ],
        ],
        cache_examples=False,
    )
    return iface


def launch_demo(
    server_scheme: str = "http",
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    **kwargs,
):
    endpoint = f"{server_scheme}://{server_name}:{server_port}/llm/generate-scenario"
    demo = setup_gradio_ui(llm_gen_scenario_endpoint=endpoint)
    return demo.launch(server_name=server_name, server_port=server_port, **kwargs)


if __name__ == "__main__":
    launch_demo()
