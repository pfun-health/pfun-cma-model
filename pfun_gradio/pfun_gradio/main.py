import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
from pfun_gradio.pfun_gradio.ui.gradio_ui import setup_gradio_ui
from dataclasses import dataclass


@dataclass
class Settings:
    scheme = os.getenv("GRADIO_SERVER_SCHEME", "http")
    host = os.getenv("GRADIO_SERVER_HOST", "localhost")
    port = os.getenv("GRADIO_SERVER_PORT", "7860")

    @property
    def llm_gen_scenario_endpoint() -> str:
        """Dynamically determine the llm-generate-scenario endpoint."""
        return f"{self.scheme}://{self.host}:{self.port}/llm/generate-scenario"

    @property
    def gradio_demo_endpoint() -> str:
        return f"{self.scheme}://{self.host}:{self.port}/gradio/gradio/"


def get_settings() -> Settings:
    """Initialize the settings object (dependency injection helper method)."""
    return Settings()


# pre-init for global settings instance
settings = None


async def _mount_gradio_app(app: FastAPI) -> FastAPI:
    """Mount the gradio demo instance to the FastAPI app."""
    # Dynamically determine the endpoint for the LLM scenario generator
    demo_blocks_iface = setup_gradio_ui(
        llm_gen_scenario_endpoint=settings.llm_gen_scenario_endpoint
    )
    app = gr.mount_gradio_app(app, demo_blocks_iface, path="/gradio")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to set up Gradio app on startup."""
    global settings
    # setup global settings (load environment variables)
    settings = Settings()
    # mount the gradio UI
    app = await _mount_gradio_app(app)
    yield
    # Any shutdown code can go here if needed
    settings = None


app = FastAPI(app_name="PFun Gradio Demo", lifespan=lifespan)


@app.get("/")
async def root():
    return RedirectResponse(
        settings.gradio_demo_endpoint,
        status_code=307
    )
