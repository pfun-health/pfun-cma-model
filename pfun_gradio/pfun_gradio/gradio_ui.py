"""Demo UI endpoint for LLM-based CMA parameter suggestions.

Uses Gradio for the interface. Hits the /llm/generate-scenario endpoint
to generate a scenario based on user input.
"""

import logging
import gradio as gr
import httpx
import asyncio
from pathlib import Path
import pfun_path_helper as pph  # type: ignore
pph.append_path(Path(__file__).parent.parent)
import os


# Initially, Get the logger (globally accessible)
# Will be overridden by setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.debug("Logger initialized for pfun-gradio (logger name: %s)", logger.name)

def get_default_description():
    return "The patient is a 45-year-old male with type 2 diabetes and a history of hypoglycemia."


async def async_generate_parameters(description, llm_gen_scenario_endpoint):
    logger.debug("Hitting llm generation endpoint: %s", str(llm_gen_scenario_endpoint))
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                llm_gen_scenario_endpoint,
                json={"description": description},
                timeout=27
            )
            if response.status_code == 200:
                # Successful response (JSON object)
                response_jdict = response.json()
                description_text = response_jdict.get("qualitative_description", "")
                import pandas as pd
                parameters = response_jdict.get("parameters", {})
                if parameters:
                    param_df = pd.DataFrame.from_dict(parameters, orient="index")
                    param_df.index.name = "Parameter"
                    param_df.reset_index(inplace=True)
                    formatted_params_table = param_df.to_markdown(index=False)
                else:
                    formatted_params_table = "😞 No parameters generated.\n"
                formatted_response = (
                    "## Description:\n"
                    f"{description_text}\n\n"
                    "## Generated Parameters:\n"
                    f"{formatted_params_table}\n"
                )
                return formatted_response
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request failed: {e}"


def setup_gradio_ui(
    llm_gen_scenario_endpoint: str,
):
    """Set up the Gradio demo interface using gr.Interface."""

    async def interface_fn(description, llm_gen_scenario_endpoint=llm_gen_scenario_endpoint):
        return await async_generate_parameters(description, llm_gen_scenario_endpoint)

    placeholder_text = "E.g., 'The patient has type 1 diabetes and struggle with high blood sugar after meals.'"
    default_value = get_default_description()

    iface = gr.Interface(
        fn=interface_fn,
        inputs=gr.Textbox(
            value=default_value,
            label="Input Scenario Description ~ *(for best results, use the third-person tense)*",
            placeholder=placeholder_text,
            lines=4,
        ),
        outputs=gr.Markdown(
            label="Generated Scenario, Scenario-driven PFun Model Parameters",
            elem_id="output-markdown",
            container=True,
            show_label=True,
            height="100%"
        ),
        title="PFun CMA Model - Generate Condition-Based Parameters",
        description=(
            "This demo uses a Large Language Model (LLM) to suggest CMA model parameters "
            "based on a brief description of the user's condition. "
            "Enter a description below and click 'Submit' to see the suggestions."
        ),
        flagging_mode="never",
        examples=[
            [
                "The patient has type 1 diabetes and struggles with high blood sugar after meals."
            ],
            [
                "A 60-year-old woman with well-controlled type 2 diabetes and mild hypertension."
            ],
        ],
        cache_examples=False,
        concurrency_limit=1,
        time_limit=30
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
