"""
Pfun CMA Model API Backend Routes.
"""

# import necessary modules and packages
from pfun_cma_model.routes import llm as llm_routes
from pfun_cma_model.routes import demo as demo_routes
from pfun_cma_model.routes import params as params_routes
from pfun_cma_model.routes import data as data_routes
from pfun_cma_model.engine.cma_model_params import (
    _BOUNDED_PARAM_KEYS_DEFAULTS,
)
import pfun_cma_model
import importlib
from pandas import DataFrame
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.engine.cma import CMASleepWakeModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import FastAPI, Request, Response, Body, Depends, Header
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Mapping, Optional

import pfun_path_helper as pph  # type: ignore
from fastapi import Body, Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pandas import DataFrame
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

import pfun_cma_model
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.engine.cma_model_params import (
    _BOUNDED_PARAM_KEYS_DEFAULTS,
    CMAModelParams,
)

# import necessary modules and packages
from pfun_cma_model.routes import data as data_routes
from pfun_cma_model.routes import demo as demo_routes
from pfun_cma_model.routes import llm as llm_routes
from pfun_cma_model.routes import params as params_routes

pph.append_path(Path(__file__).parent.parent)
from pfun_common import setup_logging  # type: ignore
from pfun_common.settings import get_settings

from pfun_cma_model.misc.templating import get_templates
from pfun_cma_model.routes import dexcom as dexcom_routes

# Ensure the .env file is loaded
settings = get_settings()

# Global variables and constants
debug_mode: bool = settings.debug

# --- Setup app Lifespan events ---

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG if debug_mode is True else logging.INFO)
#: globally accessible logger (with appropriate logging level)

redis_client: Redis | None = None
#: Global Redis client instance

templates: Jinja2Templates | None = None
#: Global Jinja2 templates instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""

    # --- Startup task: initialize templates ---
    global templates
    templates = get_templates()

    # --- Startup task: connect to Redis ---
    global redis_client
    try:
        redis_client = Redis(
            host=settings.redis_host,
            port=int(settings.redis_port),
            db=int(settings.redis_db),
            password=settings.redis_password,
            decode_responses=True,
        )
        await redis_client.ping()  # type: ignore
    except Exception as exc:
        logging.warning("Failed to setup redis client: %s", str(exc))
        redis_client = None

    # --- Startup task: download sample data if not present ---
    from pfun_cma_model.misc.pathdefs import PFunDataPaths

    pfun_data_paths = PFunDataPaths()
    pfun_data_paths.download_sample_data()

    # ---
    # --- Shutdown tasks will be handled after this point ---
    # ---
    yield
    # ---
    # NOTE: Yes, these steps are technically unnecessary.
    # ...It's an extra guarantee when you're using hot reload.
    # --- Delete the templates instance ---
    templates = None
    # TODO: sample data needs to be loaded via db connection.
    # --- Delete the sample data ---
    pfun_data_paths.remove_sample_data()
    # --- Shutdown task: disconnect from Redis ---
    if redis_client is not None:
        await redis_client.close()
        logging.info("Redis client connection closed.")


# --- Instantiate FastAPI app ---

app = FastAPI(
    app_name="PFun CMA Model Backend",
    lifespan=lifespan,
    servers=[
        (
            {
                "url": "https://cloud.tail38611b.ts.net",
                "description": "tailscale-funnel for pfun demos.",
            }
            if not debug_mode
            else {
                "url": "http://localhost:8001",
                "description": "Local development server.",
            }
        ),
    ],
)

# --- Application Configuration ---


# Set the application title and description
app.title = "PFun CMA Model Routing API"
app.description = (
    "Server-side operations for operating the PFun CMA model; "
    + "schema definitions, data IO, model execution."
)


def set_app_version(app: FastAPI = app) -> FastAPI:
    """Set the application version based on the package version and `app.py` file modification time."""
    fmod_time = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime(
        "%Y%m%d%H%M%S"
    )
    app.version = str(pfun_cma_model.__version__) + f"-dev.{fmod_time}"
    logging.debug("pfun-cma-model version: %s", pfun_cma_model.__version__)
    logging.debug("FastAPI app version set to: %s", app.version)
    return app


app = set_app_version(app=app)

# Configure debug mode based on environment variable
if debug_mode:
    app.debug = True
    logging.info("Running in DEBUG mode.")
    logging.debug("Debug mode is enabled.")
else:
    app.debug = False
    logging.info("Running in PRODUCTION mode.")
    logging.debug("Debug mode is disabled.")

# Mount the static directory to serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Setup middleware ---

# Add client request tracking middleware (added first, executes last)
from pfun_cma_model.misc.middleware import track_client_request_middleware

app.add_middleware(BaseHTTPMiddleware, dispatch=track_client_request_middleware)

# Add CORS middleware to allow cross-origin requests
allow_all_origins = {
    True: ["*", "localhost", "127.0.0.1", "::1"],  # for debug mode, allow all
    False: {
        "localhost",
        "127.0.0.1",
        "pfun.run",
        "pfun.one",
        "pfun.me",
        "pfun.app",
        "cloud.tail38611b.ts.net",
    },
}
allowed_hosts = allow_all_origins[debug_mode]  # type: ignore  # pyright: ignore[reportArgumentType]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_hosts,
    allow_headers=[
        "Authorization",
        "Access-Control-Allow-Origin",
        "Content-Security-Policy",
        "Content-Type",
    ],
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_credentials=True,
    max_age=300,
)

# Add TrustedHost middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts,
)

# Add Session middleware (added last, executes first)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
)

# --- Include Routers ---

app.include_router(dexcom_routes.router, prefix="/dexcom", tags=["dexcom"])

app.include_router(data_routes.router, prefix="/data", tags=["data"])

app.include_router(params_routes.router, prefix="/params", tags=["params"])

app.include_router(demo_routes.router, prefix="/demo", tags=["demo"])

app.include_router(llm_routes.router, prefix="/llm", tags=["llm"])


@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.debug("Health check endpoint accessed.")
    return {"status": "ok", "message": "PFun CMA Model API is running."}


@app.get("/pitch")
def pitch_document(request: Request):
    """PFun pitch document."""
    return templates.TemplateResponse(
        "pitch-doc.html",
        context={"request": request}
    )


@app.get("/")
def root(request: Request, real_ip: str = Header(None, alias="X-Real-IP")):
    """Root endpoint to display the homepage."""
    ts_msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug("Root endpoint accessed at %s", ts_msg)
    # Render the index.html template
    return templates.TemplateResponse(  # type: ignore
        "index.html",
        {
            "request": request,
            "year": datetime.now().year,
            "message": f"Accessed at: {ts_msg}; from: {real_ip}",
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    img = None
    with open(STATIC_DIR / "icons" / "pfun-cutielogo-icon.ico", "rb") as f:
        img = f.read()
    return Response(content=img, media_type="image/x-icon")


# -- CMA Model endpoints --


def get_model_instance():
    """FastAPI dependency to get a singleton CMA model instance."""
    # This function body runs only once, and the result is cached for subsequent calls.
    return CMASleepWakeModel()


@app.post("/model/run")
async def run_model(
    config: Annotated[CMAModelParams, Body()] | None = None,
    model: CMASleepWakeModel = Depends(get_model_instance),
):
    """Runs the CMA model."""
    if config is not None:
        model.update(config)
    df = model.run()
    output = df.to_json()
    response = Response(
        content=output,
        status_code=200,
        headers={
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    )
    if hasattr(response.body, "decode"):
        # maintain backward compatibility
        logger.debug("Response: %s", bytes(response.body).decode("utf-8"))
    return response


async def run_at_time_func(
    model: CMASleepWakeModel, t0: float | int, t1: float | int, n: int, **config
) -> str:
    """calculate the glucose signal for the given timeframe"""
    logger.debug(
        "(run_at_time_func) Running model at time: t0=%s, t1=%s, n=%s, config=%s",
        t0,
        t1,
        n,
        config,
    )
    bounded_params = {
        k: v for k, v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS
    }
    model.update(bounded_params)
    logger.debug("(run_at_time_func) Model parameters updated: %s", model.params)
    logger.debug(f"(run_at_time_func) Generating time vector<{t0}, {t1}, {n}>...")
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    output = df.to_json()
    return output


@app.post("/model/run-at-time")
async def run_at_time_route(
    t0: float | int,
    t1: float | int,
    n: int,
    # type: ignore
    config: Optional[CMAModelParams] = None,
    model: CMASleepWakeModel = Depends(get_model_instance),
):
    """Run the CMA model at a specific time.

    Parameters:
    - t0 (float | int): The start time (in decimal hours).
    - t1 (float | int): The end time (in decimal hours).
    - n (int): The number of samples.
    - config (CMAModelParams): The model configuration parameters.
    """
    try:
        if config is None:
            config_obj = CMAModelParams()  # type: ignore
        else:
            config_obj = config
        config_dict: Mapping = config_obj.model_dump()  # type: ignore
        output = await run_at_time_func(model, t0, t1, n, **config_dict)
        return output
    except Exception as err:
        logger.error("failed to run at time.", exc_info=True)
        error_response = Response(
            content=json.dumps(
                {
                    "error": "failed to run at time. See error message on server log.",
                    "exception": str(err),
                }
            ),
            status_code=500,
        )
        return error_response


@app.post("/model/run-at-time/stream")
async def run_at_time_stream_route(
    t0: float | int,
    t1: float | int,
    n: int,
    # type: ignore
    config: Optional[CMAModelParams] = None,
    model: CMASleepWakeModel = Depends(get_model_instance),
):
    """Streaming version of the run-at-time route."""
    from pfun_cma_model.stream import stream_run_at_time_func

    try:
        config_obj = config
        if config_obj is None:
            config_obj = CMAModelParams()  # type: ignore
        config_dict: Mapping = config_obj.model_dump()  # type: ignore
        async for row in stream_run_at_time_func(model, t0, t1, n, **config_dict):
            yield row
    except Exception as err:
        logger.error("failed to run at time.", exc_info=True)
        error_content = json.dumps(
            {
                "error": "failed to run at time. See error message on server log.",
                "exception": str(err),
                "status_code": 500,
            }
        )
        for err_row in [
            error_content,
        ]:
            yield err_row


# -- WebSocket Routes --

# Import websockets module to register events
PFunSocketIOSession = importlib.import_module(
    "pfun_cma_model.misc.sessions"
).PFunSocketIOSession
PFunWebsocketNamespace = importlib.import_module(
    "pfun_cma_model.routes.ws"
).PFunWebsocketNamespace
pfun_sio_session = PFunSocketIOSession(app=app, ns=PFunWebsocketNamespace())


@app.get("/health/ws/run-at-time")
async def health_check_run_at_time():
    """Health check endpoint for the 'run-at-time' WebSocket functionality."""
    logger.debug("Health check for 'run-at-time' WebSocket endpoint accessed.")
    # @todo: implement further health check logic as needed
    return {"status": "ok", "message": "'run-at-time' WebSocket is running."}


# -- Model Fitting Endpoints --


@app.post("/model/fit")
async def fit_model_to_data(
    data: dict | str, config: Optional[CMAModelParams | str] = None  # type: ignore
):
    from pandas import DataFrame

    from pfun_cma_model.data import read_sample_data
    from pfun_cma_model.engine.fit import fit_model as cma_fit_model

    if len(data) == 0:
        logger.debug("Sample data will be loaded as no data was provided in query.")
        data = read_sample_data(convert2json=False)  # type: ignore
        logger.debug("...Sample data retrieved:\n'%s'\n\n", data[:100])
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(config, str):
        logger.debug("Config received as string, parsing JSON.")
        # @note: config expected as JSON string
        config_dict = json.loads(config)
        # @note: config -> CMAModelParams object
        config: CMAModelParams = CMAModelParams(**config_dict)  # type: ignore
    try:
        df = DataFrame(data)
        fit_result = cma_fit_model(df, **config.model_dump())  # type: ignore
        logger.debug("Model fitted successfully.")
        logger.debug("Fit result: %s", fit_result)
        if fit_result is None:
            raise ValueError("Fit result is None. Model fitting failed.")
        output = fit_result.model_dump_json()
    except Exception as exc:
        logger.error(
            "Exception encountered. Failed to fit to data. Exception:\n%s",
            str(exc),
            exc_info=False,
        )
        error_response = Response(
            content={
                "error": "failed to fit data. See error message on server log.",
                "exception": str(exc),
            },
            status_code=500,
            headers={"Content-Type": "application/json"},
        )
        return error_response
    response = Response(
        content=output,
        status_code=200,
        headers={"Content-Type": "application/json"},
    )
    return response


# Setup the Socket.IO session
socketio_session = PFunSocketIOSession(app)
