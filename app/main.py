import os
import json
import sys
import shlex
import subprocess
import ssl
import asyncio
import logging
from typing import (
    Annotated,
    AnyStr,
    Any,
    Dict
)
from pathlib import Path
from importlib.metadata import version
import uvicorn
from fastapi.logger import logger
from fastapi import FastAPI, Request, Depends, status, Body, Cookie, APIRouter
from fastapi.responses import Response, JSONResponse, ORJSONResponse
import pandas as pd

logging.getLogger("fastapi")


#: pfun imports (relative)
top_path = Path(__file__).parents[2]
root_path = Path(__file__).parents[1]
for pth in [top_path, root_path]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)


from app.engine.cma_sleepwake import normalize_glucose, CMASleepWakeModel, CMAFitResult, fit_model as cma_fit_model


certspath = Path(__file__).parents[1].resolve()
key_fn = certspath.joinpath("localhost+2-key.pem")
cert_fn = certspath.joinpath("localhost+2.pem")


async def generate_selfsigned():
    #: ref: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/security.html#infrastructure-security
    #: ref: https://web.dev/how-to-use-local-https/#caution
    print(f"certspath={certspath}")
    # cmd = f"openssl req -subj '/CN=localhost' -x509 -newkey rsa:4096 -nodes -keyout {str(key_fn)} -out {str(cert_fn)} -days 1"
    cmd = f"mkcert localhost 127.0.0.1 ::1"
    out = subprocess.check_call(shlex.split(cmd), stdout=sys.stdout, stderr=sys.stdout)
    print()
    return out


async def delete_selfsigned():
    out = subprocess.check_call(shlex.split(f"rm {key_fn} {cert_fn}"))
    return out


app = FastAPI(title='PFun CMA Model Microservice')


@app.on_event("startup")
async def startup():
    pass

@app.on_event("shutdown")
async def shutdown():
    pass


@app.get("/")
def index_message():
    return {"message": app.title}


@app.get("/log")
async def logging_route(request: Request, response: Response, msg: Any | AnyStr = ''):
    return msg


cma_router = APIRouter(prefix="/cma")


@cma_router.api_route("/run", methods=["POST"], response_class=JSONResponse)
def run_model_with_config(request: Request, response: Response, model_config: Any | Dict = {}):
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    return output


@cma_router.api_route("/fit", methods=["POST"], response_class=JSONResponse)
def fit_model_to_data(request: Request, response: Response, model_config: AnyStr | Any | Dict = {}, data: AnyStr | Any | Dict = {}):
    if data is None:
        raise RuntimeError("no data was provided!")
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    try:
        df = pd.DataFrame(data)
        fit_result = cma_fit_model(df, **model_config)
        output = fit_result.json()
    except:
        logger.error('failed to fit to data.', exc_info=1)
        return {"error": "failed to fit data. See error message on server log."}, 500
    return JSONResponse({"output": output})


app.include_router(cma_router)


if __name__=='__main__':
    asyncio.run(generate_selfsigned())
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.handlers.RotatingFileHandler(str(certspath.joinpath("app.log")), mode='a', backupCount=1)
    logger.addHandler(handler)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.info("...setup logger.")

    # log_config=str(certspath.joinpath("log.ini"))
    uvicorn.run('main:app', host="0.0.0.0", port=5000, ssl_certfile=str(cert_fn), ssl_keyfile=str(key_fn), reload=True)
    asyncio.run(delete_selfsigned())
