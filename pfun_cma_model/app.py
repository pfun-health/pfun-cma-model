"""
PFun CMA Model API Backend Routes.
"""
import requests
from fastapi import WebSocket
from pandas import DataFrame
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.websockets import ConnectionManager, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Response, status, Body
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
        config: Mapping = config.model_dump()
        output = await run_at_time_func(t0, t1, n, **config) # type: ignore
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
        config: Mapping = json.loads(config)
    try:
        df = DataFrame(data)
        fit_result = cma_fit_model(df, **config.model_dump()) # type: ignore
        logger.info("Model fitted successfully.")
        logger.debug("Fit result: %s", fit_result)
        if fit_result is None:
            raise ValueError("Fit result is None. Model fitting failed.")
        output = fit_result.model_dump_json()
    except Exception as exc:
        logger.error("Exception encountered. Failed to fit to data. Exception:\n%s",
                     str(exc),
                     exc_info=0)
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


def run_app():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    run_app()
