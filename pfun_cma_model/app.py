"""
PFun CMA Model API Backend Routes.
"""
from jinja2 import pass_context
from fastapi.responses import RedirectResponse
from starlette.responses import StreamingResponse
from redis.asyncio import Redis
from contextlib import asynccontextmanager
from dataclasses import dataclass, InitVar
from dataclasses import dataclass
from typing import Optional
from pfun_cma_model.engine.cma_model_params import _BOUNDED_PARAM_KEYS_DEFAULTS, CMAModelParams
from typing import Dict, Any
import pfun_cma_model
from pfun_cma_model.data import read_sample_data
import importlib
from pandas import DataFrame
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.engine.cma import CMASleepWakeModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, HTTPException, Request, Response, status, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Annotated, Mapping, AsyncGenerator
from pfun_common.utils import load_environment_variables, setup_logging
from pfun_cma_model.routes import dexcom as dexcom_routes

# Initially, Get the logger (globally accessible)
# Will be overridden by setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(
    "Logger initialized for pfun_cma_model (logger name: %s)", logger.name)

# Ensure the .env file is loaded
load_environment_variables(logger=logger)

# Global variables and constants
debug_mode: bool = os.getenv("DEBUG", "0") in ["1", "true"]
# Perform logging setup...
setup_logging(logger, debug_mode=debug_mode)

# --- Setup app Lifespan events ---

redis_client: Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global redis_client
    # --- Startup task: connect to Redis ---
    redis_client = Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True,
    )
    try:
        await redis_client.ping()
        logging.info("Connected to Redis server successfully.")
    except Exception as e:
        logging.error("Failed to connect to Redis server: %s", str(e))
        redis_client = None
    yield
    # --- Shutdown task: disconnect from Redis ---
    if redis_client is not None:
        await redis_client.close()
        logging.info("Redis client connection closed.")


# --- Instantiate FastAPI app ---

app = FastAPI(
    app_name="PFun CMA Model Backend",
    lifespan=lifespan
)

# --- Application Configuration ---

# Set the application title and description
app.title = "PFun CMA Model Backend"
app.description = "Backend API for the PFun CMA Model, providing endpoints for model parameters, data handling, and model execution."

# Set the app version based on package version and file modification time


def set_app_version(app: FastAPI = app) -> FastAPI:
    """Set the application version based on the package version and `app.py` file modification time."""
    fmod_time = datetime.fromtimestamp(
        Path(__file__).stat().st_mtime
    ).strftime("%Y%m%d%H%M%S")
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

# Setup Jinja2 templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@pass_context
def https_url_for(context: dict, name: str, **path_params: Any) -> str:
    """Convert http to https.

    ref: https://waylonwalker.com/thoughts-223
    """
    request = context["request"]
    http_url = request.url_for(name, **path_params)
    return str(http_url).replace("http", "https", 1)


def get_templates() -> Jinja2Templates:
    """Get the Jinja2 templates object, include https_url_for filter.

    Returns:
        Jinja2Templates: The Jinja2 templates object.
    """
    global templates
    templates.env.globals["https_url_for"] = https_url_for
    # only use the default url_for for local development, for dev, qa, and prod use https
    if not debug_mode:
        templates.env.globals["url_for"] = https_url_for
        logger.debug("Using HTTPS")
    else:
        logger.debug("Using HTTP")
    return templates

# -- Setup middleware

# Add Session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "a-secure-secret-key-for-development")
)

# Add CORS middleware to allow cross-origin requests
allow_all_origins = {
    True: ["*", "localhost", "127.0.0.1", "::1"],  # for debug mode, allow all
    False: set([
        "localhost",
        "127.0.0.1",
        "*.robcapps.com",
        "*.run.app",
        "pfun-cma-model.local.pfun.run",
        "*.pfun.run",
        "*.pfun.one",
        "*.pfun.me",
        "*.pfun.app",
        "*.robcapps.com"
    ])
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_all_origins[debug_mode],
    allow_headers=[
        "Authorization",
        "Access-Control-Allow-Origin",
        'Content-Security-Policy',
        'Content-Type',
    ],
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_credentials=True,
    max_age=300,
)


app.include_router(dexcom_routes.router, prefix="/dexcom", tags=["dexcom"])

@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "PFun CMA Model API is running."}


@app.get("/")
def root(request: Request):
    """Root endpoint to display the homepage."""
    ts_msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug("Root endpoint accessed at %s", ts_msg)
    # Render the index.html template
    return get_templates().TemplateResponse("index.html", {
        "request": request,
        "year": datetime.now().year,
        "message": f"Accessed at: {ts_msg}"
    })


@app.get("/demo/dexcom")
def demo_dexcom(request: Request):
    return get_templates().TemplateResponse("dexcom-demo.html", {
        "request": request,
        "year": datetime.now().year
    })




# -- Model Parameters Endpoints --


@app.get("/params/schema")
def params_schema():
    from pfun_cma_model.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return Response(
        content=json.dumps(params.model_json_schema()),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@app.get("/params/default")
def default_params():
    from pfun_cma_model.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return Response(
        content=params.model_dump_json(),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@app.post("/params/describe")
def describe_params(
    params: CMAModelParams | Mapping[str, Any]
):
    """
    Describe a given (single) or set of parameters using CMAModelParams.describe and generate_qualitative_descriptor.
    Args:
        config (Optional[BoundedCMAModelParams | Mapping]): The configuration parameters to describe.
    Returns:
        dict: Dictionary of parameter descriptions and qualitative descriptors.
    """
    if params is not None:
        params = CMAModelParams(**params)  # type: ignore
    else:
        params = CMAModelParams()
    bounded_keys = list(params.bounded_param_keys)
    result = {}
    for key in bounded_keys:
        try:
            desc = params.describe(key)
            qual = params.generate_qualitative_descriptor(key)
            result[key] = {
                "description": desc,
                "qualitative": qual,
                "value": getattr(params, key, None)
            }
        except Exception as e:
            result[key] = {"error": str(e)}
    return Response(
        content=json.dumps(result),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@app.post("/params/tabulate")
def tabulate_params(
    params: CMAModelParams | Mapping[str, Any]
):
    """
    Generate a markdown table of a given (single) or set of parameters using CMAModelParams.generate_markdown_table.
    Args:
        config (Optional[BoundedCMAModelParams | Mapping]): The configuration parameters to describe.
    Returns:
        dict: Dictionary of parameter descriptions and qualitative descriptors.
    """
    if params is not None:
        params = CMAModelParams(**params)  # type: ignore
    else:
        params = CMAModelParams()
    result = {}
    try:
        result = params.generate_markdown_table()
    except Exception as e:
        result = json.dumps({"error": str(e)})  # type: ignore
    finally:
        return Response(
            content=result,
            status_code=200,
            headers={"Content-Type": "application/json"},
        )


@dataclass
class PFunDatasetResponse:
    data: DataFrame | None = None
    nrows: InitVar[int] = 23
    nrows_given: bool | None = None

    def __post_init__(self, nrows: int):
        """Post-initialization to parse nrows and data."""
        _, self.nrows_given = self._parse_nrows(nrows)
        self.data = self._parse_data(self.data, nrows, self.nrows_given)

    @property
    def streaming_response(self) -> StreamingResponse:
        """Generate a streaming Response object with the dataset as JSON."""
        return StreamingResponse(
            content=self._stream,
            media_type="application/json"
        )

    @property
    def response(self) -> Response:
        """Generate a Response object with the dataset as JSON."""
        output = self.data.to_json(orient='records')  # type: ignore
        return Response(
            content=output,
            status_code=200,
            headers={"Content-Type": "application/json"}
        )

    @classmethod
    def _parse_data(cls, data: DataFrame | None, nrows: int, nrows_given: bool):
        """Parse and limit the dataset based on nrows and nrows_given."""
        # If no data provided, read the default sample dataset
        if data is None:
            data = read_sample_data(convert2json=False)  # type: ignore
        # ensure DataFrame
        dataset = DataFrame(data)
        logging.debug("Sample dataset loaded with %d rows.", len(dataset))
        if nrows_given:
            return dataset.iloc[:nrows, :]  # type: ignore
        return dataset

    @property
    def _stream(self) -> Any:
        """Yield the dataset as streamable chunks."""
        for record in self.data.to_dict(orient='records'):  # type: ignore
            yield json.dumps(record) + '\n'

    @classmethod
    def _parse_nrows(cls, nrows: int) -> tuple[int, bool]:
        """Parse and validate the nrows parameter for dataset retrieval.
        Args:
            nrows (int): The number of rows to return. If -1, return the full dataset.
        Returns:
            tuple: A tuple containing the validated nrows and a boolean indicating if nrows was given.
        """
        # Check if nrows is valid
        if nrows < -1:
            logging.error(
                "Invalid nrows value: %s. Must be -1 or greater.", nrows)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="nrows must be -1 (for full dataset) or a non-negative integer.",
            )
        if nrows == -1:
            nrows_given = False  # -1 means no limit, return full dataset
        else:
            nrows_given = True  # nrows is given, return only the first nrows
        logging.debug(
            "Received request for sample dataset with nrows=%s", nrows)
        logging.debug("(nrows_given) Was nrows_given? %s",
                      "'Yes.'" if nrows_given else "'No.'")
        return nrows, nrows_given


@app.get("/data/sample/download")
def get_sample_dataset(request: Request, nrows: int = 23):
    """(slow) Download the sample dataset with optional row limit.

    Args:
        request (Request): The FastAPI request object.
        nrows (int): The number of rows to return. If -1, return the full dataset.
    """
    # Read the sample dataset (data=None means use default sample data)
    dataset_response = PFunDatasetResponse(data=None, nrows=nrows)
    return dataset_response.response


@app.get("/data/sample/stream")
async def stream_sample_dataset(request: Request, nrows: int = -1):
    """(fast) Stream the sample dataset with optional row limit.
    Args:
        request (Request): The FastAPI request object.
        nrows (int): The number of rows to include in the stream. If -1, stream the full dataset.
    """
    dataset_response = PFunDatasetResponse(data=None, nrows=nrows)
    # return the iterable (generating) streaming response
    return dataset_response.streaming_response


CMA_MODEL_INSTANCE = None


async def initialize_model():
    """Initialize the CMA model instance if not already done."""
    global CMA_MODEL_INSTANCE
    if CMA_MODEL_INSTANCE is not None:
        return CMA_MODEL_INSTANCE
    model = CMASleepWakeModel()
    CMA_MODEL_INSTANCE = model
    return CMA_MODEL_INSTANCE


@app.post("/translate-results")
async def translate_model_results_by_language(results: Dict, from_lang: Literal["python", "javascript"]):
    """Translate model results between Python and JavaScript formats."""
    to_lang = "python" if from_lang == "javascript" else "javascript"
    from pandas import DataFrame

    translation_dict = {
        "python": {
            "javascript": lambda x: DataFrame(x).to_json(orient="records"),
        },
        "javascript": {
            "python": lambda x: DataFrame.from_records(x).to_json(orient="columns"),
        },
    }
    return Response(
        content=translation_dict[from_lang][to_lang](results), status_code=200
    )


@app.post("/model/run")
async def run_model(config: Annotated[CMAModelParams, Body()] | None = None):
    """Runs the CMA model."""
    model = await initialize_model()
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
    if hasattr(response.body, 'decode'):
        # maintain backward compatibility
        logger.debug("Response: %s", bytes(response.body).decode('utf-8'))
    return response


async def run_at_time_func(t0: float | int, t1: float | int, n: int, **config) -> str:
    """calculate the glucose signal for the given timeframe"""
    model = await initialize_model()
    logger.debug(
        "(run_at_time_func) Running model at time: t0=%s, t1=%s, n=%s, config=%s", t0, t1, n, config)
    bounded_params = {k: v for k,
                      v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS}
    model.update(bounded_params)
    logger.debug(
        "(run_at_time_func) Model parameters updated: %s", model.params)
    logger.debug(
        f"(run_at_time_func) Generating time vector<{t0}, {t1}, {n}>...")
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    output = df.to_json()
    return output


async def stream_run_at_time_func(t0: float | int, t1: float | int, n: int, **config) -> AsyncGenerator[str, None]:
    """calculate the glucose signal for the given timeframe and stream the results."""
    model = await initialize_model()
    logger.debug(
        "(stream_run_at_time_func) Running model at time: t0=%s, t1=%s, n=%s, config=%s", t0, t1, n, config)
    bounded_params = {k: v for k,
                      v in config.items() if k in _BOUNDED_PARAM_KEYS_DEFAULTS}
    model.update(bounded_params)
    logger.debug(
        "(stream_run_at_time_func) Model parameters updated: %s", model.params)
    logger.debug(
        f"(stream_run_at_time_func) Generating time vector<{t0}, {t1}, {n}>...")
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    for index, row in df.iterrows():
        point = {'x': index, 'y': row.iloc[0]}
        yield json.dumps(point)


@app.post("/model/run-at-time")
async def run_at_time_route(t0: float | int,
                            t1: float | int,
                            n: int,
                            # type: ignore
                            config: Optional[CMAModelParams] = None
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
        output = await run_at_time_func(t0, t1, n, **config_dict)
        return output
    except Exception as err:
        logger.error("failed to run at time.", exc_info=True)
        error_response = Response(
            content=json.dumps({
                "error": "failed to run at time. See error message on server log.",
                "exception": str(err),
            }),
            status_code=500,
        )
        return error_response

# -- WebSocket Routes --

# Import websockets module to register events
PFunSocketIOSession = importlib.import_module(
    "pfun_cma_model.misc.sessions").PFunSocketIOSession
PFunWebsocketNamespace = importlib.import_module(
    "pfun_cma_model.routes.ws").PFunWebsocketNamespace
pfun_sio_session = PFunSocketIOSession(app=app, ns=PFunWebsocketNamespace())


@app.get("/health/ws/run-at-time")
async def health_check_run_at_time():
    """Health check endpoint for the 'run-at-time' WebSocket functionality."""
    logger.info("Health check for 'run-at-time' WebSocket endpoint accessed.")
    # @todo: implement actual health check logic if needed
    return {"status": "ok", "message": "'run-at-time' WebSocket is running."}


# -- Demo routes --

@app.get("/demo/run-at-time")
async def demo_run_at_time(request: Request):
    """Demo UI endpoint to run the model at a specific time (using websockets)."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS, _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS, _MID_DEFAULTS, _UB_DEFAULTS
    )
    default_config = dict(cma_params.bounded_params_dict)
    # formatted parameters to appear in the rendered template
    params = {}
    for ix, pk in enumerate(default_config):
        if pk in default_config:
            params[pk] = {
                "name": _BOUNDED_PARAM_KEYS_DEFAULTS[ix],
                "value": default_config[pk],
                "description": _BOUNDED_PARAM_DESCRIPTIONS[ix],
                "min": _LB_DEFAULTS[ix],
                "max": _UB_DEFAULTS[ix],
                "default": _MID_DEFAULTS[ix]
            }
    # formulate the render context
    rand0, rand1 = os.urandom(16).hex(), os.urandom(16).hex()
    context_dict = {
        "request": request,
        "params": params,
        "cdn": {
            "chartjs": {
                "url": f"https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js?dummy={rand0}"
            },
            "socketio": {
                "url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"
            }
        }
    }
    logger.debug("Demo context: %s", context_dict)
    return get_templates().TemplateResponse(
        "run-at-time-demo.html", context=context_dict, headers={"Content-Type": "text/html"})


# -- Model Fitting Endpoints --


@app.post("/model/fit")
async def fit_model_to_data(
    data: dict | str,
    config: Optional[CMAModelParams | str] = None  # type: ignore

):
    from pandas import DataFrame
    from pfun_cma_model.engine.fit import fit_model as cma_fit_model
    if len(data) == 0:
        data = read_sample_data(convert2json=False)  # type: ignore
        logger.info("...Sample data loaded as no data provided.")
        logger.debug("...Sample data retrieved:\n'%s'\n\n", data[:100])
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(config, str):
        logger.info("Config received as string, parsing JSON.")
        # @note: config expected as JSON string
        config_dict = json.loads(config)
        # @note: config -> CMAModelParams object
        config: CMAModelParams = CMAModelParams(**config_dict)  # type: ignore
    try:
        df = DataFrame(data)
        fit_result = cma_fit_model(df, **config.model_dump())  # type: ignore
        logger.info("Model fitted successfully.")
        logger.debug("Fit result: %s", fit_result)
        if fit_result is None:
            raise ValueError("Fit result is None. Model fitting failed.")
        output = fit_result.model_dump_json()
    except Exception as exc:
        logger.error(
            "Exception encountered. Failed to fit to data. Exception:\n%s",
            str(exc),
            exc_info=False
        )
        error_response = Response(
            content={
                "error": "failed to fit data. See error message on server log.",
                "exception": str(exc)
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
