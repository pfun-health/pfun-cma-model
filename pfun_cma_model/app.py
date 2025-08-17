"""
PFun CMA Model API Backend Routes.
"""
from pfun_cma_model.engine.cma_model_params import CMAModelParams, BoundedCMAModelParams
from typing import Dict, Any
import pfun_cma_model
from pfun_cma_model.data import read_sample_data
from fastapi import Query
import importlib
from pandas import DataFrame
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.engine.cma import CMASleepWakeModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response, status, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Annotated, Mapping
from pfun_common.utils import load_environment_variables, setup_logging

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

# Initialize FastAPI app
app = FastAPI(app_name="PFun CMA Model Backend")
# Set the application title and description
app.title = "PFun CMA Model Backend"
app.description = "Backend API for the PFun CMA Model, providing endpoints for model parameters, data handling, and model execution."
# set the version of the API
app.version = pfun_cma_model.__version__
if debug_mode:
    app.debug = True
    logging.info("Running in DEBUG mode.")
    logging.debug("Debug mode is enabled.")
else:
    app.debug = False
    logging.info("Running in PRODUCTION mode.")
    logging.debug("Debug mode is disabled.")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent /
          "static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=Path(__file__).parent / "static")

# Add CORS middleware to allow cross-origin requests
allow_all_origins = {
    True: ["*", "localhost", "127.0.0.1", "::1"],
    False: [
        "pfun-cma-model-446025415469.*.run.app",
        "*.pfun.run",
        "*.pfun.one",
        "*.pfun.me",
        "*.pfun.app",
        "*.robcapps.com"
    ]
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_all_origins[debug_mode],
    allow_headers=[
        "X-API-Key",
        "Authorization",
        "Access-Control-Allow-Origin",
    ],
    allow_methods=["*"],
    allow_credentials=True,
    max_age=300,
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed.")
    return {"status": "ok", "message": "PFun CMA Model API is running."}


@app.get("/")
def root():
    """Root endpoint to display the homepage."""
    ts_msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug("Root endpoint accessed at %s", ts_msg)
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": {}, "message": f"Accessed at: {ts_msg}"})


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
    params: BoundedCMAModelParams | Mapping[str, Any] = BoundedCMAModelParams(
    ).bounded_params_dict
):
    """
    Describe a given (single) or set of parameters using CMAModelParams.describe and generate_qualitative_descriptor.
    Args:
        config (Optional[BoundedCMAModelParams | Mapping]): The configuration parameters to describe.
    Returns:
        dict: Dictionary of parameter descriptions and qualitative descriptors.
    """
    if params is not None:
        params = BoundedCMAModelParams(**params)  # type: ignore
    else:
        params = BoundedCMAModelParams()
    result = {}
    for key in params.bounded_param_keys:  # type: ignore
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


@app.get("/data/sample")
def get_sample_dataset(request: Request, nrows: int = -1):
    """Get the sample dataset with optional row limit.

    Args:
        request (Request): The FastAPI request object.
        nrows (int): The number of rows to return. If -1, return the full dataset.
    """
    # Check if nrows is valid
    if nrows < -1:
        logging.error("Invalid nrows value: %s. Must be -1 or greater.", nrows)
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
    logging.debug("Was nrows_given? %s", "'Yes.'" if nrows_given else "'No.'")
    # Read sample dataset (keep as DataFrame)
    dataset = DataFrame(read_sample_data(convert2json=False))
    if nrows_given is False:
        logging.debug("Returning full dataset as JSON.")
        # if nrows is not given, return the full dataset as JSON
        return Response(
            content=dataset.to_json(orient='records'),
            status_code=200,
            headers={"Content-Type": "application/json"},
        )
    # if nrows is given, return only the first nrows of the dataset
    logging.debug("Returning first %d rows of the dataset as JSON.", nrows)
    dataset = dataset.iloc[:nrows, :]
    output = dataset.to_json(orient='records')
    return Response(
        content=output,
        status_code=200,
        headers={"Content-Type": "application/json"})


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


@app.post("/run")
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
    logging.debug(
        "Running model at time: t0=%s, t1=%s, n=%s, config=%s", t0, t1, n, config)
    model.update(config)
    logging.debug("Model parameters updated: %s", model.params)
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    output = df.to_json()
    return output


@app.post("/run-at-time")
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
            config = CMAModelParams()  # type: ignore
        config: Mapping = config.model_dump()  # type: ignore
        output = await run_at_time_func(t0, t1, n, **config)  # type: ignore
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


@app.get("/demo/run-at-time")
async def demo_run_at_time(request: Request, t0: float | int = 0, t1: float | int = 100, n: int = 100, config: CMAModelParams | None = None):
    """Demo UI endpoint to run the model at a specific time (using websockets)."""
    default_config = {
        "eps": 0.00,  # set noise to zero for this demo
    }
    # load default bounded parameters
    default_config.update(CMAModelParams().bounded.bounded_params_dict)
    ws_port = os.getenv("WS_PORT", 443)
    context_dict = {
        "request": request,
        "params": default_config,
        "ws_prefix": 'wss' if ws_port == 443 else 'ws',
        "host": os.getenv("WS_HOST", request.base_url.hostname),
        "port": ws_port,
    }
    return templates.TemplateResponse(
        "run-at-time-demo.html", context=context_dict)


@app.post("/fit")
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
