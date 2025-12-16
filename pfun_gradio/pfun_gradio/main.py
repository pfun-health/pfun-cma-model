import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import gradio as gr
from pfun_gradio.pfun_gradio.ui.gradio_ui import setup_gradio_ui


async def _mount_gradio_app(app: FastAPI) -> FastAPI:
    """Mount the gradio demo instance to the FastAPI app."""
    # Dynamically determine the endpoint for the LLM scenario generator
    scheme = os.getenv("GRADIO_SERVER_SCHEME", "http")
    host = os.getenv("GRADIO_SERVER_HOST", "localhost")
    port = os.getenv("GRADIO_SERVER_PORT", "7860")
    llm_gen_scenario_endpoint = f"{scheme}://{host}:{port}/llm/generate-scenario"
    demo_blocks_iface = setup_gradio_ui(
        llm_gen_scenario_endpoint=llm_gen_scenario_endpoint
    )
    app = gr.mount_gradio_app(app, demo_blocks_iface, path="/gradio")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to set up Gradio app on startup."""
    app = await _mount_gradio_app(app)
    yield
    # Any shutdown code can go here if needed


app = FastAPI(app_name="PFun Gradio Demo", lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the PFun Gradio Demo API. Visit /gradio for the demo interface."
    }
