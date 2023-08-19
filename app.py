from apispec import APISpec
from chalice import (
    Chalice,
    CORSConfig,
    AuthResponse,
    CustomAuthorizer,
    Response
)
try:
    import apispec_chalice
except (ModuleNotFoundError, ImportError):
    apispec_chalice = None
import json
import sys
from pathlib import Path
from typing import Dict, Literal
import boto3

CLIENT = boto3.client('lambda')

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

PUBLIC_ROUTES = [
    '/',
    '/run',
    '/fit',
    '/run-at-time'
]

spec = None
if apispec_chalice is not None:
    try:
        spec = APISpec(
            title='PFun CMA Model API',
            version='0.1.0',
            openapi_version='3.0',
            plugins=['apispec_chalice'],
        )
    except Exception as e:
        spec = None


def aws_get_rapidapi_key():
    key = str(boto3.client('secretsmanager').get_secret_value(
        SecretId='pfun-cma-model-rapidapi-key')['SecretString'])
    return key


def aws_get_rapidapi_proxy_secret():
    key = str(boto3.client('secretsmanager').get_secret_value(
        SecretId='pfun-cma-model-rapid-api-proxy-secret')['SecretString'])
    return key

@app.authorizer()
def fake_auth(auth_request):
    """
    ... ref (original): https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-use-lambda-authorizer.html
    ...
    ... TODO: implement with Cognito User Pool (plus oauth2): 
    ...
    ... https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-configuring-app-integration.html
    ...
    ... https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-add-custom-domain.html
    ...
    ... https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-assign-domain-prefix.html

    """
    authorized = auth_request.token in ['Bearer allow', 'allow']
    if not authorized:
        return Response(status_code=401) 
    if authorized:
        return AuthResponse(routes=['/', '/log', '/fit', '/run'],
                            principal_id='user')
    else:
        return AuthResponse(routes=[], principal_id='user')


@app.route("/")
def index_message():
    """
    A function that returns a message containing the welcome message and 
    the routes of the PFun CMA Model API.

    Returns:
        dict: A dictionary containing the welcome message and the routes of
        the API.
    """
    routes = json.dumps({k: str(v) for k, v in app.routes.items()
                         if k in PUBLIC_ROUTES}, indent=4)
    return Response(
        body='''
        <html>
        <body>
        <h3>Welcome to the PFun CMA Model API!</h3>
        <hr />
        <br />
        {}
        </body>
        </html>
        '''.format(f"<pre>{routes}</pre>"),
        status_code=200,
        headers={'Content-Type': 'text/html'})


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


@app.lambda_function(name="run_model")
def run_model_with_config(event, context):
    from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
    model_config = get_lambda_params(event, 'model_config')
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    return output


@app.route('/run', methods=["GET", "POST"], authorizer=fake_auth)
def run_model_route():
    global CLIENT
    request = app.current_request
    authorized = all([
        request.headers['Authorization'] == 'Bearer allow',
        request.headers.get('X-RapidAPI-Host') == 'pfun-cma-model-api.p.rapidapi.com',
        request.headers.get('X-RapidAPI-Key') == aws_get_rapidapi_key(),
        request.headers.get('X-RapidAPI-Proxy-Secret') == aws_get_rapidapi_proxy_secret()
    ]) or request.headers['Host'] == '127.0.0.1'
    if not authorized:
        return Response(status_code=401)
    if CLIENT is None:
        CLIENT = boto3.client('lambda')
    model_config = get_model_config(app)
    response = CLIENT \
        .invoke(FunctionName='run_model', Payload=json.dumps(model_config))
    return json.loads(response.get('body', '[]'))


@app.route("/run-at-time", methods=['GET', 'POST'], authorizer=fake_auth)
def run_at_time_route():
    request = app.current_request
    global CLIENT
    if CLIENT is None:
        CLIENT = boto3.client('lambda')
    model_config = get_model_config(app)
    calc_params = get_params(app, 'calc_params')
    params = {
        "model_config": model_config,
        "calc_params": calc_params
    }
    response = CLIENT \
        .invoke(FunctionName='run_at_time', Payload=json.dumps(params))
    return json.loads(response.get('body', '[]'))


@app.lambda_function(name='run_at_time')
def run_at_time(event, context):
    import numpy as np
    import pandas as pd
    from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
    model_config = get_lambda_params(event, 'model_config')
    calc_params = get_lambda_params(event, 'calc_params')
    model = CMASleepWakeModel(**model_config)
    output: np.ndarray | tuple = model.calc_Gt(**calc_params)
    output = pd.json_normalize(output).to_json()  # type: ignore
    return output


@app.lambda_function(name="fit_model")
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
        output = fit_result.json(exclude=['cma'])
    except Exception:
        app.log.error('failed to fit to data.', exc_info=True)
        return {"error":
                "failed to fit data. See error message on server log."}, 500
    return {"output": output}


@app.route('/fit', methods=['POST'], authorizer=fake_auth)
def fit_model_route():
    global CLIENT
    if CLIENT is None:
        CLIENT = boto3.client('lambda')
    model_config = get_model_config(app)
    response = CLIENT.invoke(
        FunctionName='fit_model', Payload=json.dumps(model_config))
    return json.loads(response.get('body', '[]'))

try:
    spec.add_path(path='/', operations={'get': {'summary': 'Welcome'}})
    spec.add_path(path='/fit', operations={'post': {'summary': 'Fit the model to timstamped blood glucose data.'}})
    spec.add_path(path='/run', operations={'get': {'summary': 'Run the model to simulate a specified time period.'}})
    spec.add_path(path='/run-at-time', operations={'post': {'summary': 'Generate model output for the specifed time points.'}})
except Exception:
    pass


@app.route('/apidocs', methods=['GET'])
def openapi():
    """
    A function that serves the OpenAPI JSON file.

    Parameters:
    - None

    Returns:
    - A dictionary object representing the OpenAPI JSON file.
    """
    if spec is not None:
        return spec.to_dict()
    schema = json.loads(Path(__file__).parent.
                        joinpath('openapi.json').
                        read_text(encoding='utf-8'))
    return Response(body=json.dumps(schema, indent=5),
                    status_code=200,
                    headers={'Content-Type': 'application/json'})
