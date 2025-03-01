"""
PFun CMA Model API Backend Routes.
"""
from fastapi import WebSocket
from pfun_cma_model.misc.errors import BadRequestError
from pfun_cma_model.misc.pathdefs import PFunAPIRoutes
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from fastapi import FastAPI, HTTPException, Request, Response, status
import pfun_path_helper
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pfun_path_helper.append_path(Path(__file__).parent.parent)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return Response(content='<h1>Welcome to the pfun_cma_model API</h1><h3>Check out <a href="/docs">/docs</a> to get started...</h3>')


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


@app.get("/log")
def logging_route(request: Request, level: Literal["info", "warning", "error"] = "info"):
    current_request: Request = request
    if current_request is None:
        raise RuntimeError("Logging error! No request was provided!")
    if current_request.query_params is None:
        raise RuntimeError("Logging error! No query parameters were provided!")
    msg = current_request.query_params.get("msg") or current_request.query_params.get(
        "message"
    )
    level = current_request.query_params.get("level", level)
    if msg is None:
        return Response(
            content="No message provided.", status_code=BadRequestError.STATUS_CODE
        )
    loggers = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
    }
    loggers[level](msg)
    return Response(content={"message": msg, "level": level}, status_code=200)


def get_params(
        request: Request, key: str, default: Any = None, load_json: bool = False
) -> Dict:
    current_request: Request = request
    if current_request is None:
        raise RuntimeError("No request was provided!")
    params = {} if current_request.json_body is None else current_request.json_body
    if isinstance(params, (str, bytes)):
        params = json.loads(params)
    if key in params:
        params = params[key]
    if current_request.query_params is not None:
        params.update(current_request.query_params)
    if key in params:
        params = params[key]
    if params is None:
        params = default
    if load_json and isinstance(params, (str, bytes)):
        params = json.loads(params)
    return params


def get_model_config(request: Request, key: str = "model_config") -> Dict:
    return get_params(request, key=key)


CMA_MODEL_INSTANCE = None


def initialize_model():
    global CMA_MODEL_INSTANCE
    if CMA_MODEL_INSTANCE is not None:
        return CMA_MODEL_INSTANCE
    from pfun_cma_model.engine.cma_model_params import CMAModelParams
    from pfun_cma_model.engine.cma import CMASleepWakeModel

    model_config = get_model_config(Request({}))
    if model_config is None:
        model_config = {}
    model_config = CMAModelParams(**model_config)
    model = CMASleepWakeModel(model_config)
    CMA_MODEL_INSTANCE = model
    return CMA_MODEL_INSTANCE


@app.post("/translate-results")
def translate_model_results_by_language(request: Request):
    results = get_params(request, "results")
    from_lang = get_params(request, "from", "python", load_json=True)
    if from_lang not in ["python", "javascript"]:
        return Response(
            content="Invalid from language.", status_code=BadRequestError.STATUS_CODE
        )
    to_lang = get_params(request, "to", "javascript", load_json=True)
    if to_lang not in ["python", "javascript"]:
        return Response(
            content="Invalid to language.", status_code=BadRequestError.STATUS_CODE
        )
    if from_lang == to_lang:
        return Response(content=json.dumps(results), status_code=200)
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
@app.get("/run")
def run_model(request: Request):
    """Runs the CMA model."""
    if request is None:
        raise RuntimeError("No request was provided!")
    model_config = get_model_config(request)
    model = initialize_model()
    model.update(**model_config)
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
    logger.info("Response: %s", json.dumps(response.to_dict()))
    return response


def run_at_time_func(request: Request) -> str:
    model_config = get_model_config(request)
    calc_params = get_params(request, "calc_params")
    # pylint-disable=import-outside-toplevel
    from pandas import DataFrame

    from pfun_cma_model.engine.cma_model_params import CMAModelParams

    logger.info("model_config: %s", json.dumps(model_config))
    logger.info("calc_params: %s", json.dumps(calc_params))
    if model_config is None:
        model_config = {}
    model = initialize_model()
    model_config = CMAModelParams(**model_config)
    model.update(model_config)  # ! this occurs inplace !
    if calc_params is None:
        calc_params = {}
    df: DataFrame = model.calc_Gt(**calc_params)
    output = df.to_json()
    return output


@app.websocket("/ws/run_at_time")
async def run_at_time_ws(websocket: WebSocket):
    await websocket.accept()
    msg = await websocket.receive_text()
    logger.info("Received message: %s", msg[:20], "...")
    data = json.loads(msg)
    interim_response = await requests.post("/run-at-time", data=data)
    output: str = interim_response.text
    await websocket.send_text(output)


@app.post("/run-at-time")
@app.get("/run-at-time")
def run_at_time_route(request: Request):
    try:
        output = run_at_time_func(request)
        return Response(
            content=output,
            status_code=200,
            headers={"Content-Type": "application/json"},
        )
    except Exception as err:
        logger.error("failed to run at time.", exc_info=True)
        error_response = Response(
            content={
                "error": "failed to run at time. See error message on server log.",
                "exception": str(err),
            },
            status_code=500,
        )
        return error_response


@app.post("/fit")
def fit_model_to_data(request: Request):
    from pandas import DataFrame
    from pfun_cma_model.engine.fit import fit_model as cma_fit_model
    data = get_params(request, "data")
    if data is None:
        raise RuntimeError("no data was provided!")
    if isinstance(data, str):
        data = json.loads(data)
    model_config = get_model_config(request)
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    try:
        df = DataFrame(data)
        fit_result = cma_fit_model(df, **model_config)
        output = fit_result.model_dump_json()
    except Exception:
        logger.error("failed to fit to data.", exc_info=True)
        error_response = Response(
            content={
                "error": "failed to fit data. See error message on server log."},
            status_code=500,
            headers={"Content-Type": "application/json"},
        )
        return json.dumps(error_response.to_dict())
    response = Response(
        content={"output": output},
        status_code=200,
        headers={"Content-Type": "application/json"},
    )
    return response


def run_app():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_app()
