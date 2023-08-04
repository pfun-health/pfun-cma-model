from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from chalicelib.engine.cma_sleepwake import fit_model as cma_fit_model
from chalice import (
    Chalice, Response,
    JSONResponse,
    CORSConfig,
)
from chalice.app import Request
import json
import sys
from pathlib import Path
from typing import Any, AnyStr, Dict, Literal
import pandas as pd

#: pfun imports (relative)
root_path = Path(__file__).parents[1]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)

#: init app, set cors
cors_config = CORSConfig(allow_origin='*', allow_credentials=True)
app = Chalice(app_name='PFun CMA Model Backend')
app.api.cors = cors_config


@app.on_event("startup")
async def startup():
    pass


@app.on_event("shutdown")
async def shutdown():
    pass


@app.route("/")
def index_message():
    return {"message": "Welcome to the PFun CMA Model API!"}


@app.route("/log")
async def logging_route(msg: Any | AnyStr = '',
                        level: Literal['info', 'warning', 'error'] = 'info'):
    loggers = {
        'info': app.log.info,
        'warning': app.log.warning,
        'error': app.log.error
    }
    loggers[level](msg)
    return {'message': msg, 'level': level}


@app.route("/run", methods=["POST"], response_class=JSONResponse)
def run_model_with_config(request: Request, response: Response,
                          model_config: Any | Dict = {}):
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    return output


@app.route("/fit", methods=["POST"], response_class=JSONResponse)
def fit_model_to_data(request: Request, response: Response,
                      model_config: AnyStr | Any | Dict = {},
                      data: AnyStr | Any | Dict = {}):
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
    except Exception:
        app.log.error('failed to fit to data.', exc_info=1)
        return {"error":
                "failed to fit data. See error message on server log."}, 500
    return JSONResponse({"output": output})
