import logging
from pathlib import Path
import sys
from typing import Annotated
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
from fastapi import Depends, FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr
import importlib

try:
    from pfun_common.settings import Settings, get_settings
except (ImportError, ModuleNotFoundError):
    from pfun_common.pfun_common.settings import Settings, get_settings
setup_gradio_ui = importlib.import_module(
    "gradio_ui", package="pfun_gradio.pfun_gradio"
).setup_gradio_ui


#: settings dependency injection type
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _mount_gradio_app(app: FastAPI, settings: Settings) -> FastAPI:
    """Mount the gradio demo instance to the FastAPI app."""
    logger.info(
        "llm_gen_scenario_endpoint: %s", str(settings.llm_gen_scenario_endpoint)
    )
    demo_blocks_iface = setup_gradio_ui(
        llm_gen_scenario_endpoint=settings.llm_gen_scenario_endpoint
    )
    app = gr.mount_gradio_app(app, demo_blocks_iface, path="/gradio")
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to set up Gradio app on startup."""
    # mount the Gradio demo instance to the app
    app = _mount_gradio_app(app, get_settings())
    logger.debug("...mounted gradio app.")
    yield
    # Any shutdown code can go here if needed


app = FastAPI(
    app_name="PFun Gradio Demo App",
    lifespan=lifespan,
    title="PFun Gradio Demo App",
    description="A FastAPI app that serves a Gradio UI for generating pfun scenarios."
)


@app.get("/")
async def root(settings: SettingsDep):
    return RedirectResponse(settings.gradio_demo_endpoint, status_code=307)
