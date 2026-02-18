"""Demo UI endpoint for LLM-based CMA parameter suggestions.

Uses Gradio for the interface. Hits the /llm/generate-scenario endpoint
to generate a scenario based on user input.
"""

import logging
import importlib
from pathlib import Path
import asyncio

import gradio as gr
import pfun_path_helper as pph  # type: ignore

pph.append_path(Path(__file__).parent.parent)

try:
    from pfun_common.settings import Settings, get_settings
except (ImportError, ModuleNotFoundError):
    from pfun_common.pfun_common.settings import Settings, get_settings
gen_scene = importlib.import_module('.llm', package='pfun_cma_model').generate_scenario

# Initially, Get the logger (globally accessible)
# Will be overridden by setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.debug("Logger initialized for pfun-gradio (logger name: %s)", logger.name)


def get_default_description():
    return "The patient is a 45-year-old male with type 2 diabetes and a history of hypoglycemia."


async def generate_fn(description: str):
    return await gen_scene(description, include_recommendations=True)


async def wait_gen_fn(description: str):
    try:
        return await asyncio.wait_for(generate_fn(description), timeout=25)
    except asyncio.TimeoutError as exc:
        return "Timed out waiting for response. Please try again."

    
def _run_asyncio(coro):
    loop = asyncio.get_event_loop()
    result = asyncio.get_event_loop().run_until_complete(coro)
    return result


def interface_fn(description: str):
    return _run_asyncio(wait_gen_fn(description))


def setup_gradio_ui() -> gr.Interface:
    """Set up the Gradio demo interface using gr.Interface."""

    placeholder_text = "E.g., 'The patient has type 1 diabetes and struggle with high blood sugar after meals.'"
    default_value = get_default_description()

    iface = gr.Interface(
        fn=interface_fn,
        inputs=gr.Textbox(
            value=default_value,
            label="Input Scenario Description <h4>(for best results, use the third-person tense)</h4>",
            placeholder=placeholder_text,
            lines=4,
        ),
        outputs=gr.Markdown(
            label="<h3>Generated Scenario</h3><h4>Scenario-driven PFun Model Parameters</h4>",
            elem_id="output-markdown",
            container=True,
            show_label=True,
            height="20vh",
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
        concurrency_limit='default',
    )
    return iface


def launch_demo(
    **kwargs,
):
    settings = get_settings()
    server_scheme = settings.server_scheme
    server_name = settings.server_host
    server_port = settings.server_port
    demo = setup_gradio_ui()
    return demo.launch(
        server_name=server_name, server_port=server_port, mcp_server=True, **kwargs
    )


if __name__ == "__main__":
    launch_demo()
