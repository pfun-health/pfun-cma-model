import logging
from pathlib import Path
import sys
import pfun_path_helper as pph  # type: ignore
import os


# Initially, Get the logger (globally accessible)
# Will be overridden by setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pfun_cma_model")
logger.info("Logger initialized for pfun_cma_model (logger name: %s)", logger.name)

# Global variables and constants
debug_mode: bool = os.getenv("DEBUG", "0") in ["1", "true"]

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
import importlib
setup_gradio_ui = \
    importlib.import_module("gradio_ui", package="pfun_gradio.pfun_gradio").setup_gradio_ui
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


def _mount_gradio_app(app: FastAPI) -> FastAPI:
    """Mount the gradio demo instance to the FastAPI app."""
    # Dynamically determine the endpoint for the LLM scenario generator
    scheme = os.getenv("GRADIO_SERVER_SCHEME", "http")
    host = os.getenv("SERVER_HOST", "localhost")
    if debug_mode is True:
        logging.debug("DEBUG MODE, so server port is being set...")
        port = ":" + os.getenv("SERVER_PORT", "8001")
    else:
        port = ""  # no explicit port needed for prod
    llm_gen_scenario_endpoint = f"{scheme}://{host}{port}/llm/generate-scenario"
    logging.info("llm_gen_scenario_endpoint: %s", str(llm_gen_scenario_endpoint))
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
    logger.debug("...mounted gradio app.")
    yield
    # Any shutdown code can go here if needed
    settings = None


app = FastAPI(app_name="PFun Gradio Demo", lifespan=lifespan)

# mount the Gradio demo instance to the app
app = _mount_gradio_app(app)


@app.get("/")
async def root():
    return RedirectResponse(
        settings.gradio_demo_endpoint,
        status_code=307
    )
