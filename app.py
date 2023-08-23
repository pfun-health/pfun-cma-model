"""
PFun CMA Model API routes.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Literal
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from botocore.session import Session as SessionCore
from botocore.config import Config as ConfigCore
import boto3
import threading
from chalice import (
    AuthRoute,
    CORSConfig,
    AuthResponse,
    Response,
    Chalice,
)
from chalice.app import Request, AuthRequest


def new_boto3_session():
    session_ = boto3.Session()
    session_.client('iam').attach_role_policy(
        RoleName='pfun-cma-model-dev',
        PolicyArn='arn:aws:iam::860311922912:policy/pfun-cma-model-dev'
    )
    return session_


def new_boto3_client(service_name, session=None, *args, **kwds):
    config = ConfigCore(
        region_name='us-east-1',
    )
    if session is None:
        session = new_boto3_session()
    client_ = session.client(service_name, *args, config=config, **kwds)
    return client_


LAMBDA_CLIENT = new_boto3_client('lambda')

#: pfun imports (relative)
root_path = Path(__file__).parents[1]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)

#: init app, set cors
cors_config = CORSConfig(
    allow_origin='pfun-cma-model-api.p.rapidapi.com',
    allow_headers=['X-RapidAPI-Key',
                   'X-RapidAPI-Proxy-Secret',
                   'X-RapidAPI-Host',
                   'X-API-Key',
                   'Authorization'],
    allow_credentials=True,
    max_age=300,
    expose_headers=['X-RapidAPI-Key',
                    'X-RapidAPI-Proxy-Secret', 'X-RapidAPI-Host']
)
app = Chalice(app_name='PFun CMA Model Backend')

app.api.cors = cors_config
app.websocket_api.session = new_boto3_session()
app.experimental_feature_flags.update([
    'WEBSOCKETS'
])

PUBLIC_ROUTES: list[str | AuthRoute] = [
    '/',
    '/run',
    '/fit',
    '/run-at-time',
    '/routes'
]


class SecretsWrapper(threading.Thread):
    """
    Wrapper class for boto3 client 'secretsmanager'
    """

    def __init__(self, secrets_lock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._secrets_lock = secrets_lock
        self._secrets_manager: BaseClient | None = None
        self._session: boto3.Session | None = None

    def run(self):
        self.session = new_boto3_session()

    @property
    def session(self):
        if not self._session:
            with self._secrets_lock:
                if not self._session:
                    self._session = new_boto3_session()
        return self._session

    @session.setter
    def session(self, session):
        with self._secrets_lock:
            self._session = session

    @property
    def secrets_manager(self):
        """
        Retrieves the secrets manager client.

        Returns:
            BaseClient: _description_
        """
        def test_client(client_):
            try:
                client_.get_secret_value()
                return True
            except ClientError:
                return False
        with self._secrets_lock:
            if self._secrets_manager is None:
                self._secrets_manager = \
                    new_boto3_client('secretsmanager', session=self.session)
            return self._secrets_manager

    def authorize(self, request: Request):
        authorized = all([
            request.headers['Authorization'] == 'Bearer allow',
            request.headers.get(
                'X-RapidAPI-Host') == 'pfun-cma-model-api.p.rapidapi.com',
            request.headers.get('X-RapidAPI-Key') ==
            self.aws_get_rapidapi_key(),
            request.headers.get('X-RapidAPI-Proxy-Secret') ==
            self.aws_get_rapidapi_proxy_secret()
        ]) or request.headers['Host'] == '127.0.0.1'
        return authorized

    def aws_get_rapidapi_key(self):
        """
        Retrieves the RapidAPI proxy secret from the AWS Secrets Manager.

        :param self: The reference to the current object.
        :return: The RapidAPI proxy secret as a string.
        """
        key = str(self.secrets_manager.get_secret_value(
            SecretId='pfun-cma-model-rapidapi-key')['SecretString'])
        return key

    def aws_get_rapidapi_proxy_secret(self):
        """
        Retrieves the RapidAPI proxy secret from the AWS Secrets Manager.

        :param self: The reference to the current object.
        :return: The RapidAPI proxy secret as a string.
        """
        key = str(self.secrets_manager.get_secret_value(
            SecretId='pfun-cma-model-rapid-api-proxy-secret')['SecretString'])
        return key


class SecretsManagerContainer:
    """
    Container class for boto3 client 'secretsmanager'
    """

    secrets_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        self._secrets = SecretsWrapper(self.__class__.secrets_lock, *args,
                                       **kwargs)
        self._secrets.start()

    def __del__(self):
        self._secrets.join(timeout=1.0)

    def authorize(self, *args, **kwargs):
        return self._secrets.authorize(*args, **kwargs)


secman = SecretsManagerContainer()


@app.authorizer()
def fake_auth(auth_request: AuthRequest):
    """Temporary authorizer for testing purposes.
    """
    authorized = auth_request.token in ['Bearer allow', 'allow']
    if not authorized:
        return Response(body='Unauthorized', status_code=401)
    if authorized:
        return AuthResponse(routes=PUBLIC_ROUTES,
                            principal_id='user')
    else:
        return AuthResponse(routes=[], principal_id='user')


@app.route("/")
def index():
    SCRIPTS = '''
    <script type="text/javascript" src="lib/axios/dist/axios.standalone.js"></script>
    <script type="text/javascript" src="lib/CryptoJS/rollups/hmac-sha256.js"></script>
    <script type="text/javascript" src="lib/CryptoJS/rollups/sha256.js"></script>
    <script type="text/javascript" src="lib/CryptoJS/components/hmac.js"></script>
    <script type="text/javascript" src="lib/CryptoJS/components/enc-base64.js"></script>
    <script type="text/javascript" src="lib/url-template/url-template.js"></script>
    <script type="text/javascript" src="lib/apiGatewayCore/sigV4Client.js"></script>
    <script type="text/javascript" src="lib/apiGatewayCore/apiGatewayClient.js"></script>
    <script type="text/javascript" src="lib/apiGatewayCore/simpleHttpClient.js"></script>
    <script type="text/javascript" src="lib/apiGatewayCore/utils.js"></script>
    <script type="text/javascript" src="apigClient.js"></script>
    '''
    pypath = '/opt/python/lib/python%s.%s/site-packages/chalicelib' % sys.version_info[:2]
    if not Path(pypath).exists():
        pypath = Path(__file__).parent.joinpath("chalicelib")
    body = Path(pypath).joinpath('www', 'index.html') \
        .read_text(encoding='utf-8')
    ROUTES = '\n'.join([
        f'<li><a class="dropdown-item" href="{name}">{name}</a></li>'
        for name, _ in app.routes.items() if name in PUBLIC_ROUTES
    ])
    body = body.format(
        SCRIPTS=SCRIPTS,
        ROUTES=ROUTES
    )
    return Response(
        body=body,
        status_code=200,
        headers={'Content-Type': 'text/html'}
    )


@app.route("/routes")
def get_routes():
    routes = json.dumps({k: list(v.keys()) for k, v in app.routes.items()
                         if k in PUBLIC_ROUTES}, indent=4)
    return Response(body=routes, status_code=200)


@app.route("/log", methods=['GET', 'POST'], authorizer=fake_auth)
def logging_route(level: Literal['info', 'warning', 'error'] = 'info'):
    if app.current_request is None:
        raise RuntimeError("Logging error! No request was provided!")
    if app.current_request.query_params is None:
        raise RuntimeError("Logging error! No query parameters were provided!")
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
    if app.current_request is None:
        raise RuntimeError("No request was provided!")
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
    app.log.info('(lambda) http_method: %s', http_method)
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
    """
    A function that returns a message containing the welcome message and the
    routes of the PFun CMA Model API.
    """
    global LAMBDA_CLIENT  # type: ignore
    request: Request | None = app.current_request
    if request is None:
        raise RuntimeError("No request was provided!")
    authorized = secman.authorize(request)
    if not authorized:
        return Response(body='Unauthorized', status_code=401)
    model_config = get_model_config(app)
    payload = json.dumps(model_config).encode('utf-8')
    response = LAMBDA_CLIENT \
        .invoke(FunctionName='run_model', Payload=payload)
    return json.loads(response.get('body', b'[]'))


@app.on_ws_message(name="run_at_time")
def run_at_time_route(event):
    global LAMBDA_CLIENT
    model_config = get_model_config(app)
    calc_params = get_params(app, 'calc_params')
    params = {
        "model_config": model_config,
        "calc_params": calc_params
    }
    payload = json.dumps(params).encode('utf-8')
    response = LAMBDA_CLIENT \
        .invoke(FunctionName='run_at_time', Payload=payload)
    lambda_response = response.get('body', b'[]').decode('utf-8')
    app.websocket_api.send(event.connection_id, lambda_response)


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
        output = fit_result.model_dump_json()
    except Exception:
        app.log.error('failed to fit to data.', exc_info=True)
        return {"error":
                "failed to fit data. See error message on server log."}, 500
    return {"output": output}


@app.route('/fit', methods=['POST'], authorizer=fake_auth)
def fit_model_route():
    global LAMBDA_CLIENT
    authorized = secman.authorize(app.current_request)
    if not authorized:
        return Response(body='Unauthorized', status_code=401)
    model_config = get_model_config(app)
    response = LAMBDA_CLIENT.invoke(
        FunctionName='fit_model', Payload=json.dumps(model_config).encode('utf-8'))
    return json.loads(response.get('body', b'[]'))
