from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from chalicelib.engine.cma_sleepwake import fit_model as cma_fit_model
from chalice import (
    Chalice,
    CORSConfig,
)
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


@app.route("/")
def index_message():
    return {"message": "Welcome to the PFun CMA Model API!"}


@app.route("/log")
async def logging_route(msg: str,
                        level: Literal['info', 'warning', 'error'] = 'info'):
    msg = app.current_request.query_params.get('msg')
    if msg is None:
        raise RuntimeError("Logging error! No message was provided!")
    loggers = {
        'info': app.log.info,
        'warning': app.log.warning,
        'error': app.log.error
    }
    loggers[level](msg)
    return {'message': msg, 'level': level}


def get_params(app: Chalice, key: str) -> Dict:
    params = {} if app.current_request.json_body is None else \
        app.current_request.json_body
    if key in params:
        params = params[key]
    params.update(app.current_request.query_params)
    return params


def get_model_config(app: Chalice, key: str = 'model_config') -> Dict:
    return get_params(app, key=key)


@app.route("/run", methods=["POST"])
def run_model_with_config():
    model_config = get_model_config(app)
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    return output


@app.route("/fit", methods=["POST"])
def fit_model_to_data():
    model_config = get_model_config(app)
    data = get_params(app, key='data')
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
    return {"output": output}
