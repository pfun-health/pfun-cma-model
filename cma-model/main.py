from chalice import (
    Chalice, Response,
    Request, JSONResponse,
    CORSConfig, CORSView,
)
from cma_model.engine.cma_sleepwake import fit_model as cma_fit_model
from cma_model.engine.cma_sleepwake import CMASleepWakeModel
import json
import logging
import sys
from pathlib import Path
from typing import Any, AnyStr, Dict
import pandas as pd

logging.getLogger()


#: pfun imports (relative)
root_path = Path(__file__).parents[1]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)

app = Chalice(app_name='PFun CMA Model Backend')


@app.on_event("startup")
async def startup():
    pass


@app.on_event("shutdown")
async def shutdown():
    pass


@app.get("/")
def index_message(request: Request):
    return {"message": "Welcome to the PFun CMA Model API!"}


@app.get("/log")
async def logging_route(request: Request, response: Response,
                        msg: Any | AnyStr = ''):
    return msg


@app.api_route("/run", methods=["POST"], response_class=JSONResponse)
def run_model_with_config(request: Request, response: Response,
                          model_config: Any | Dict = {}):
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    return output


@app.api_route("/fit", methods=["POST"], response_class=JSONResponse)
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
        logging.error('failed to fit to data.', exc_info=1)
        return {"error":
                "failed to fit data. See error message on server log."}, 500
    return JSONResponse({"output": output})


app.include_router(app)
