"""
PFun CMA Model API Backend Routes.
"""
import requests
from fastapi import WebSocket
from pandas import DataFrame
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.routes.websockets import ConnectionManager, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response, status, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Annotated, Mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SDK_CLIENT = None
BASE_URL: Optional[str] = None
STATIC_BASE_URL: str | None = None
BODY: Optional[str] = None

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# trunk-ignore(bandit/B108)
file_handler = logging.FileHandler("/tmp/FastAPI-logs-backend.log")
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

app = FastAPI(app_name="PFun CMA Model Backend")
if os.getenv("DEBUG", "0") in ["1", "true"]:
    app.debug = True
    logging.info("Running in DEBUG mode.")
    logging.debug("Debug mode is enabled.")
else:
    app.debug = False
    logging.info("Running in PRODUCTION mode.")
    logging.debug("Debug mode is disabled.")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/static/run-at-time-plot", StaticFiles(directory=Path(__file__).parent / "clients/run-at-time-client/public"), name="run-at-time-plot")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=Path(__file__).parent / "static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*.pfun.run"],
    allow_headers=[
        "X-RapidAPI-Key",
        "X-RapidAPI-Proxy-Secret",
        "X-RapidAPI-Host",
        "X-API-Key",
        "Authorization",
        "Access-Control-Allow-Origin",
    ],
    allow_methods=["*"],
    allow_credentials=True,
    max_age=300,
)


@app.get("/")
def root():
    return Response(
        content='''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>pfun_cma_model API Demo</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                a {
                    color: #2980b9;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
        <h1>Welcome to the pfun_cma_model API Demo</h1>
        <p>This API provides endpoints to run the CMA model for sleep-wake prediction.</p>
        <p>To get started, you can:</p>
        <ul>
            <li>View the interactive websocket demo at <a href="/demo/run-at-time">/demo/run-at-time</a>.</li>
            <li>Get the sample dataset using the <a href="/data/sample">/data/sample</a> endpoint.</li>
            <li>View the model parameters schema using the <a href="/params/schema">/params/schema</a> endpoint.</li>
            <li>Get the default model parameters using the <a href="/params/default">/params/default</a> endpoint.</li>
            <li>Use the WebSocket endpoint <a href="/ws/run-at-time">/ws/run-at-time</a> to run the model at specific times.</li>
            <li>Fit the model to your own data using the <a href="/fit">/fit</a> endpoint.</li>
        </ul>
        <p>For more information on how to use the API, please refer to the documentation available at:</p>
        <ul>
            <li><a href="/docs">(SwaggerUI) Interactive API Documentation</a></li>
            <li><a href="/redoc">(ReDoc) Readable API Documentation</a></li>
            <li><a href="/openapi.json">(JSON) OpenAPI Schema</a></li>
        </ul>
        <h2>pfun_cma_model - API Demo Homepage</h2>
        <h3><a href="/docs">View the API docs</a> to learn more about the available endpoints.</h3>
        </body>
        </html>
        ''', status_code=200,
        headers={"Content-Type": "text/html"},
        media_type="text/html"
    )


@app.get("/params/schema")
def params_schema():
    from pfun_cma_model.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return params.model_json_schema()


@app.get("/params/default")
def default_params():
    from pfun_cma_model.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return params.model_dump_json()


def read_sample_data(convert2json: bool = True):
    from pfun_cma_model.engine.data_utils import format_data
    from pfun_cma_model.misc.pathdefs import PFunDataPaths
    df = PFunDataPaths().read_sample_data()
    if convert2json is False:
        return df
    return df.to_json(orient='records')


@app.get("/data/sample")
def get_sample_dataset(request: Request):
    return read_sample_data(convert2json=True)


CMA_MODEL_INSTANCE = None


async def initialize_model():
    global CMA_MODEL_INSTANCE
    if CMA_MODEL_INSTANCE is not None:
        return CMA_MODEL_INSTANCE
    model = CMASleepWakeModel()
    CMA_MODEL_INSTANCE = model
    return CMA_MODEL_INSTANCE


@app.post("/translate-results")
async def translate_model_results_by_language(results: Dict, from_lang: Literal["python", "javascript"]):
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
        logger.debug("Response: %s", response.body.decode('utf-8'))
    return response


async def run_at_time_func(t0: float | int, t1: float | int, n: int, **config) -> str:
    """calculate the glucose signal for the given timeframe"""
    model = await initialize_model()
    model.update(config)
    t = model.new_tvector(t0, t1, n)
    df: DataFrame = model.calc_Gt(t=t)
    output = df.to_json()
    return output


manager = ConnectionManager()


@app.websocket("/ws/run-at-time")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to run the model at specific times."""
    await manager.connect(websocket)
    try:
        while True:
            input_command = await websocket.receive_text()
            logging.info("Received command: %s", input_command)
            run_at_time_func_args = json.loads(input_command)
            t0 = run_at_time_func_args.get("t0", 0)
            t1 = run_at_time_func_args.get("t1", 100)
            n = run_at_time_func_args.get("n", 100)
            config = run_at_time_func_args.get("config", {})
            output = await run_at_time_func(t0, t1, n, **config)
            await manager.send_personal_message(output, websocket)
            logging.info("Sent output: %s", output)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/run-at-time")
async def run_at_time_route(t0: float | int, t1: float | int, n: int, config: CMAModelParams | None = None):
    try:
        if config is None:
            config = CMAModelParams()
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


@app.get("/demo/run-at-time")
async def demo_run_at_time(request: Request, t0: float | int = 0, t1: float | int = 100, n: int = 100, config: CMAModelParams | None = None):
    """Demo UI endpoint to run the model at a specific time (using websockets)."""
    default_config = {
        "eps": 0.00,  # set noise to zero for this demo
    }
    # load default bounded parameters
    default_config.update(CMAModelParams().bounded_params_dict)
    return templates.TemplateResponse("run-at-time-demo.html", {"request": request, "params": default_config})


@app.post("/fit")
async def fit_model_to_data(data: dict | str, config: CMAModelParams | str | None = None):
    from pandas import DataFrame
    from pfun_cma_model.engine.fit import fit_model as cma_fit_model
    if len(data) == 0:
        data = read_sample_data()
        logger.info("\n...Sample data retrieved:\n'%s'\n\n", data[:100])
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


def run_app(host: str = "0.0.0.0", port: int = 8001, **kwargs: Any):
    """Run the FastAPI application."""
    import uvicorn
    # remove unwanted kwargs
    valid_kwargs = uvicorn.run.__kwdefaults__
    for key in list(kwargs.keys()):
        if key not in valid_kwargs:
            logger.warning(f"Unrecognized keyword argument '{key}' for uvicorn.run(). Ignoring it.")
            del kwargs[key]
    logger.info(f"Running FastAPI app on {host}:{port} with kwargs: {kwargs}")
    # must pass the app parameter as a module path to enable hot-reloading
    kwargs.pop("host", None)  # avoid duplicate host/port arguments
    kwargs.pop("port", None)
    if kwargs.get("reload", False):
        # with hot-reloading
        logging.info("Running with hot-reloading enabled.")
        # remove reload from kwargs to avoid passing it twice
        reload = kwargs.pop("reload", False)
        uvicorn.run("pfun_cma_model.app:app", host=host, port=port, reload=reload, **kwargs)
    else:
        # without hot-reloading
        logging.info("Running without hot-reloading.")
        uvicorn.run(app, host=host, port=port, **kwargs)

if __name__ == "__main__":
    run_app()
