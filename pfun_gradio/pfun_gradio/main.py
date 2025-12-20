from fastapi.middleware.cors import CORSMiddleware
import importlib
import gradio as gr
from fastapi.responses import RedirectResponse
from fastapi import Depends, FastAPI
from contextlib import asynccontextmanager
import logging
from typing import Annotated
import pfun_path_helper as pph  # type: ignore



# Initially, Get the logger (globally accessible)
# Will be overridden by setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pfun_cma_model")
logger.info(
    "Logger initialized for pfun_cma_model (logger name: %s)", logger.name)


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
        "llm_gen_scenario_endpoint: %s", str(
            settings.llm_gen_scenario_endpoint)
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


def setup_middleware(app: FastAPI):
    """Set up CORS middleware for the FastAPI app."""
    logger.info("Setting up CORS middleware.")
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:7860",
            "http://localhost:8001",
            f"{settings.server_scheme}://{settings.server_host}:{settings.server_port}",
            f"{settings.gradio_server_scheme}://{settings.gradio_server_host}:{settings.gradio_server_port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.debug("...CORS middleware set up.")
    return app


# Apply CORS middleware to the FastAPI app
app = setup_middleware(app)


@app.get("/")
async def root(settings: SettingsDep):
    return RedirectResponse(settings.gradio_demo_endpoint, status_code=307)
