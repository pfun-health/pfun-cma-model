from chalice import (
    Chalice,
    CORSConfig,
    AuthResponse
)
import json
import sys
from pathlib import Path
from typing import Dict, Literal
import boto3

#: pfun imports (relative)
root_path = Path(__file__).parents[1]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)

#: init app, set cors
cors_config = CORSConfig(allow_origin='*')
app = Chalice(app_name='PFun CMA Model Backend')
app.api.cors = cors_config


@app.authorizer()
def fake_auth(auth_request):
    """
    TODO: continue with this guide: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-use-lambda-authorizer.html
    ...create an authorization token.
    """
    token = auth_request.token
    if token == 'allow':
        return AuthResponse(routes=['/', '/log', '/fit', '/run'],
                            principal_id='user')
    else:
        return AuthResponse(routes=[], principal_id='user')


@app.route("/")
def index_message():
    routes = json.dumps({k: str(v) for k, v in app.routes.items()}, indent=4)
    return {
        "message": "Welcome to the PFun CMA Model API!\nRoutes:\n{}"
        .format(routes)
    }


@app.route("/log", methods=['GET', 'POST'], authorizer=fake_auth)
def logging_route(level: Literal['info', 'warning', 'error'] = 'info'):
    msg = app.current_request.query_params.get('msg')
    level = app.current_request.query_params.get('level', level)
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
    if app.current_request.query_params is not None:
        params.update(app.current_request.query_params)
    return params


def get_model_config(app: Chalice, key: str = 'model_config') -> Dict:
    return get_params(app, key=key)


def get_lambda_params(event, key: str) -> Dict:
    http_method = event.get('httpMethod')
    query_params = event.get('queryStringParams', {})
    app.log.info('(lambda) http_method: {}'.format(http_method))
    body = event.get('body')
    params = {}
    if body is not None:
        params = json.loads(body)
    if key in params:
        params = params[key]
    params.update(query_params)
    return params


@app.lambda_function("run_model")
def run_model_with_config(event, context):
    from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
    model_config = get_lambda_params(event, 'model_config')
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    return output


@app.route('/run', methods=["GET", "POST"], authorizer=fake_auth)
def run_model_route():
    model_config = get_model_config(app)
    response = boto3.client('lambda') \
        .invoke(FunctionName='run_model', Payload=json.dumps(model_config))
    return json.loads(response.get('body', '[]'))


@app.lambda_function("fit_model")
def fit_model_to_data(event, context):
    from chalicelib.engine.fit import fit_model as cma_fit_model
    import pandas as pd
    data = get_lambda_params(event, 'data')
    if data is None:
        raise RuntimeError("no data was provided!")
    if isinstance(data, str):
        data = json.loads(data)
    model_config = get_lambda_params(event, 'model_config')
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


@app.route('/fit', methods=['POST'], authorizer=fake_auth)
def fit_model_route():
    model_config = get_model_config(app)
    response = boto3.client('lambda').invoke(
        FunctionName='fit_model', Payload=json.dumps(model_config))
    return json.loads(response.get('body', '[]'))