"""
PFun CMA Model API Backend Routes.
"""
import requests
from fastapi import WebSocket
from pandas import DataFrame
from pfun_cma_model.misc.errors import BadRequestError
from pfun_cma_model.misc.pathdefs import PFunAPIRoutes
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from fastapi import FastAPI, HTTPException, Request, Response, status, Body
import pfun_path_helper
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Annotated

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


CMA_MODEL_INSTANCE = None


async def initialize_model():
    global CMA_MODEL_INSTANCE
    if CMA_MODEL_INSTANCE is not None:
        return CMA_MODEL_INSTANCE
    from pfun_cma_model.engine.cma_model_params import CMAModelParams
    from pfun_cma_model.engine.cma import CMASleepWakeModel

    model_config = {}
    model_config = CMAModelParams(**model_config)
    model = CMASleepWakeModel(model_config)
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
async def run_model(config: Annotated[CMAModelParams, Body()] = None):
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


@app.websocket("/ws/run_at_time")
async def run_at_time_ws(websocket: WebSocket):
    """websocket endpoint for run_at_time"""
    await websocket.accept()
    msg = await websocket.receive_text()
    logger.info("Received message: %s", msg[:20], "...")
    data = json.loads(msg)
    interim_response = await requests.post("/run-at-time", data=data)
    output: str = interim_response.text
    await websocket.send_text(output)


@app.post("/run-at-time")
async def run_at_time_route(t0: float | int, t1: float | int, n: int, config: Annotated[CMAModelParams, Body()] = None):
    try:
        config = config.model_dump()
        output = await run_at_time_func(t0, t1, n, **config)
        return Response(
            content=json.dumps(output, indent=3),
            status_code=200,
            headers={"Content-Type": "application/json"},
        )
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


@app.post("/fit")
async def fit_model_to_data(data: dict, config: CMAModelParams | str = None):
    from pandas import DataFrame
    from pfun_cma_model.engine.fit import fit_model as cma_fit_model
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(config, str):
        config = json.loads(config)
    try:
        df = DataFrame(data)
        fit_result = cma_fit_model(df, **config)
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
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    run_app()
