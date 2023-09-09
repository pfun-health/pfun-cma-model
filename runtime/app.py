"""
PFun CMA Model API routes.
"""
import base64
import os
import json
import sys
import uuid
from chalice.app import LambdaFunction, ConvertToMiddleware, Request, AuthRequest
import requests
from pathlib import Path
import urllib.parse as urlparse
from typing import (
    Any, Dict, Literal
)
from botocore.client import BaseClient
from botocore.config import Config as ConfigCore
import boto3
import threading
import importlib
from chalice import (
    AuthRoute,
    CORSConfig,
    AuthResponse,
    Response,
    Chalice,
)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def new_boto3_session():
    session_ = boto3.Session()
    return session_


def new_boto3_client(service_name, session=None, *args, **kwds):
    config = ConfigCore(
        region_name='us-east-1',
    )
    if session is None:
        session = new_boto3_session()
    client_ = session.client(service_name, *args, config=config, **kwds)
    return client_


#: pfun imports (relative)
root_path = Path(__file__).parents[2]
for pth in [root_path, ]:
    pth = str(pth)
    if pth not in sys.path:
        sys.path.insert(0, pth)

get_secret_func = importlib.import_module(
    '.secrets', package='chalicelib').get_secret_func

#: init app, set cors
cors_config = CORSConfig(
    allow_origin='*',
    allow_headers=['X-RapidAPI-Key',
                   'X-RapidAPI-Proxy-Secret',
                   'X-RapidAPI-Host',
                   'X-API-Key',
                   'Authorization',
                   'Access-Control-Allow-Origin'],
    allow_credentials=True,
    max_age=300,
    expose_headers=['X-RapidAPI-Key',
                    'X-RapidAPI-Proxy-Secret',
                    'X-RapidAPI-Host',
                    'X-API-Key',
                    'Authorization',
                    'Access-Control-Allow-Origin']
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('/tmp/chalice-logs.log')
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

app = Chalice(app_name='PFun CMA Model Backend')
if os.getenv('DEBUG_CHALICE', False) in ['1', 'true']:
    app.debug = True
app.log = logger
app.log.setLevel(logging.INFO)
local_environ = {k: v for k, v in dict(os.environ).items() if 'AWS' in k}
logger.info('app environment: %s', json.dumps(local_environ, indent=2))

app.api.cors = cors_config
app.websocket_api.session = new_boto3_session()
app.experimental_feature_flags.update([
    'WEBSOCKETS'
])


def fix_headers(func):
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        logger.debug('Original headers: %s', json.dumps(response.headers))
        if isinstance(func, LambdaFunction) or len(args) > 0:
            return response
        request = app.current_request
        response.headers['Access-Control-Allow-Origin'] = '*'
        if not hasattr(request, 'headers'):
            return response
        if 'Host' not in request.headers and 'Origin' not in request.headers:
            return response
        header = 'Origin' if 'Origin' in request.headers else 'Host'
        if request.headers[header].split(':')[0] == '127.0.0.1':
            response.headers['Access-Control-Allow-Origin'] = request.headers[header]
        return response
    return wrapper


app.register_middleware(ConvertToMiddleware(fix_headers), event_type='all')


PUBLIC_ROUTES: list[str | AuthRoute] = [
    '/',
    '/run',
    '/fit',
    '/run-at-time',
    '/routes',
    '/sdk'
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
        with self._secrets_lock:
            if self._secrets_manager is None:
                self._secrets_manager = \
                    new_boto3_client('secretsmanager', session=self.session)
            return self._secrets_manager

    def authorize(self, request: Request):
        """
        Authorizes a request based on the provided headers.

        Args:
            request (Request): The request object containing the headers.

        Returns:
            bool: True if the request is authorized, False otherwise.
        """
        try:
            authorized = all([
                request.headers.get('X-RapidAPI-Key') ==
                self.aws_get_rapidapi_key(),
                request.headers.get('X-RapidAPI-Proxy-Secret') ==
                self.aws_get_rapidapi_proxy_secret()
            ])
        except KeyError:
            authorized = False
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

    def aws_get_aws_api_key(self):
        """
        Retrieves the AWS API key from the AWS Secrets Manager.
        """
        key = str(self.secrets_manager.get_secret_value(
            SecretId='pfun-cma-model-aws-api-key')['SecretString'])
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

    def authorize(self, request, *args, **kwargs):
        """
        Authorizes the request by checking if the secret is authorized or if the API key is valid.

        Args:
            request (object): The request object containing the headers.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bool: True if the request is authorized, False otherwise.
        """
        secret_authorized = self._secrets.authorize(request, *args, **kwargs)
        apikey_authorized = request.headers['X-API-Key'] == \
            self._secrets.aws_get_aws_api_key()
        return secret_authorized or apikey_authorized


secman = SecretsManagerContainer()


@app.authorizer()
def fake_auth(auth_request: AuthRequest):
    """Temporary authorizer for testing purposes.
    """
    authorized = False
    try:
        current_request = app.current_request
    except AttributeError:
        logger.error("Can't authenticate because this is a local instance.")
        logger.info('Current app type: %s', type(app))
        if hasattr(app, '_THREAD_LOCAL'):
            #: ! authorize automatically for local requests
            authorized = True
    else:
        logger.info(
            '(Authorizer) Request method: %s', current_request.method.lower())
        authorized = all([
            secman.authorize(current_request)
        ])
    if authorized:
        logger.info('Authorized request: %s', str(vars(auth_request)))
        return AuthResponse(routes=PUBLIC_ROUTES,
                            principal_id='user')
    #: ! Unauthorized
    app.log.warning('Unauthorized request: %s', str(vars(auth_request)))
    logger.warning('Unauthorized request: %s', str(vars(auth_request)))
    return AuthResponse(routes=[], principal_id='user')


@app.route('/sdk', methods=['GET'], authorizer=fake_auth)
def generate_sdk():
    logger.info('Generating SDK')
    client = boto3.client('apigateway')
    rest_api_id = next(item for item in client.get_rest_apis()['items']
                       if item.get('name') == 'PFun CMA Model Backend')['id']
    response = client.get_sdk(
        restApiId=rest_api_id,
        stageName='api',
        sdkType='javascript',
    )
    sdk_stream = response['body']
    sdk_bytes = sdk_stream.read()

    # Return the zipped SDK as binary response
    return Response(
        body=sdk_bytes,
        headers={'Content-Type': 'application/zip'},
        status_code=200,
    )

@app.route('/static', methods=['GET'])
def static_files():
    """
    Serves the static files for the web application.
    """
    # pylint: disable=consider-using-f-string
    pypath = '/opt/python/lib/python%s.%s/site-packages/chalicelib' % \
        sys.version_info[:2]
    if not Path(pypath).exists():
        pypath = Path(__file__).parent.joinpath("chalicelib")
    else:
        pypath = Path(pypath)
    if app.current_request.query_params is None:
        app.log.warning('No query params provided (requested static resource).')
        return Response(body='No query prameters provided when requesting static resource.', status_code=400)
    filename = app.current_request.query_params.get('filename', 'index.template.html')
    filepath = Path(str(pypath) + '/www/')
    available_files = [f for f in filepath.rglob('*') if f.is_file()]
    filepath = Path(str(filepath) + filename)
    logger.info('Requested static resource: %s', str(filepath))
    if not filepath.exists():
        app.log.warning('Requested static resource does not exist: %s', str(filepath))
        return Response(
            body='Requested static resource does not exist (requested static resource: %s).' % str(filepath), status_code=404)
    if filepath not in available_files:
        app.log.warning('Requested static resource is not available: %s', str(filepath))
        return Response(
            body='Requested static resource is not available (requested static resource: %s).' % str(filepath), status_code=404)
    content_type = app.current_request.query_params.get('ContentType', 'text/*')
    return Response(body=filepath.read_text(encoding='utf-8'),
                    status_code=200, headers={'Content-Type': content_type, 'Access-Control-Allow-Origin': '*'})


@app.route("/")
def index():
    """
    Generates the index page for the web application.

    Returns:
        Response: The HTTP response object containing the index page.
    """
    # pylint: disable=consider-using-f-string
    pypath = '/opt/python/lib/python%s.%s/site-packages/chalicelib' % \
        sys.version_info[:2]
    if not Path(pypath).exists():
        pypath = Path(__file__).parent.joinpath("chalicelib")
    body = Path(pypath).joinpath('www', 'index.template.html') \
        .read_text(encoding='utf-8')
    ROUTES = '\n'.join([
        f'<li><a class="dropdown-item" href="/api{name}">{name}</a></li>'
        for name in PUBLIC_ROUTES])
    PFUN_ICON_BLOB = base64.b64encode(Path(pypath).joinpath(
        'www', 'icons', 'mattepfunlogolighter.png').read_bytes()) \
        .decode('utf-8')
    PFUN_ICON_BLOB = f'data:image/png;base64,{PFUN_ICON_BLOB}'
    STATIC_BASE_URL = app.current_request.headers['host']
    if '127.0.0.1' in STATIC_BASE_URL:
        STATIC_BASE_URL = f'http://{STATIC_BASE_URL}'
    else:
        STATIC_BASE_URL = f'https://{STATIC_BASE_URL}/api'
    STATIC_BASE_URL += '/static?filename='
    body = body.format(
        STATIC_BASE_URL=STATIC_BASE_URL,
        PFUN_ICON_BLOB=PFUN_ICON_BLOB,
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


@app.route('/run', methods=["GET", "POST"], authorizer=fake_auth)
def run_model_route():
    """
    A function that returns a message containing the welcome message and the
    routes of the PFun CMA Model API.
    """
    request: Request | None = app.current_request
    if request is None:
        raise RuntimeError("No request was provided!")
    model_config = get_model_config(app)
    from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
    if model_config is None:
        model_config = {}
    model = CMASleepWakeModel(**model_config)
    df = model.run()
    output = df.to_json()
    response = Response(body=output, status_code=200,
                        headers={'Content-Type': 'application/json',
                                 'Access-Control-Allow-Origin': '*'})
    return response


@app.on_ws_connect(name="run_at_time")
def run_at_time_connect(event):
    logger.info('New connection: %s' % event.connection_id)
    app.websocket_api.send(event.connection_id, "Connected")


def run_at_time_func(app: Chalice) -> str:
    model_config = get_model_config(app)
    calc_params = get_params(app, 'calc_params')
    # pylint-disable=import-outside-toplevel
    import numpy as np
    # pylint-disable=import-outside-toplevel
    import pandas as pd
    # pylint-disable=import-outside-toplevel
    from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
    logger.info('model_config: %s', json.dumps(model_config))
    logger.info('calc_params: %s', json.dumps(calc_params))
    if model_config is None:
        model_config = {}
    model = CMASleepWakeModel(**model_config)
    if calc_params is None:
        calc_params = {}
    output: np.ndarray | tuple = model.calc_Gt(**calc_params)
    output_json: str = pd.json_normalize(output).to_json()  # type: ignore
    return output_json


@app.on_ws_message(name="run_at_time")
def run_at_time_ws(event):
    logger.info('Received message: %s', event.body)
    output: str = run_at_time_func(app)
    app.websocket_api.send(event.connection_id, output)


@app.route('/run-at-time', methods=["GET", "POST"], authorizer=fake_auth)
def run_at_time_route():
    return Response(body=run_at_time_func(app), status_code=200,
                    headers={'Content-Type': 'application/json'})


@app.route('/fit', methods=['POST'], authorizer=fake_auth)
def fit_model_to_data():
    from chalicelib.engine.fit import fit_model as cma_fit_model
    import pandas as pd
    data = get_params(app, 'data')
    if data is None:
        raise RuntimeError("no data was provided!")
    if isinstance(data, str):
        data = json.loads(data)
    model_config = get_model_config(app)
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    try:
        df = pd.DataFrame(data)
        fit_result = cma_fit_model(df, **model_config)
        output = fit_result.model_dump_json()
    except Exception:
        app.log.error('failed to fit to data.', exc_info=True)
        error_response = Response(body={"error": "failed to fit data. See error message on server log."},
                                  status_code=500, headers={'Content-Type': 'application/json'})
        return json.dumps(error_response.to_dict())
    response = Response(body={"output": output}, status_code=200,
                        headers={'Content-Type': 'application/json'})
    return response


DexcomEndpoint = Literal["dataRange", "egvs", "alerts", "calibrations",
                         "devices", "events"]


def get_oauth_info(event):
    oauth_info = {
        "creds": {},
        "host": "",
        "login_url": "",
        "redirect_uri": "",
        "state": str(uuid.uuid4()),
        "oauth2_tokens": None
    }
    secret = get_secret_func("dexcom_pfun-app_glucose")
    oauth_info["creds"] = json.loads(secret)
    os.environ['DEXCOM_CLIENT_ID'] = oauth_info['creds']['client_id']
    os.environ['DEXCOM_CLIENT_SECRET'] = oauth_info['creds']['client_secret']
    oauth_info["host"] = os.getenv(
        "DEXCOM_HOST", get_params(app, 'DEXCOM_HOST')) or \
        'https://api.dexcom.com'
    oauth_info["login_url"] = urlparse.urljoin(
        oauth_info["host"], "/v2/oauth2/login")
    oauth_info["redirect_uri"] = os.getenv("DEXCOM_REDIRECT_URI") or \
        '/login-success'
    oauth_info["token_url"] = urlparse.urljoin(
        oauth_info["host"], "v2/oauth2/token")
    oauth_info['refresh_url'] = urlparse.urljoin(
        oauth_info["host"], "v2/oauth2/token")
    oauth_info['endpoint_urls'] = {}
    for endpoint in DexcomEndpoint.__args__:
        oauth_info["endpoint_urls"][endpoint] = urlparse.urljoin(
            oauth_info['host'], f"v3/users/{{}}/{endpoint}")
    return oauth_info


oauth_info = None


@app.route('/login-dexcom', methods=['GET', 'POST'])
def oauth2_dexcom(event, context):
    """Handles the Dexcom login request."""

    global oauth_info
    if oauth_info is None:
        oauth_info = get_oauth_info(event)

    # Get the authorization code from the event.
    authorization_code = get_params(app, 'authorization_code')
    if authorization_code is not None and oauth_info['oauth2_tokens'] is None:
        # Exchange the authorization code for an access token.
        url = oauth_info['token_url']
        payload = {
            'client_id': os.environ['DEXCOM_CLIENT_ID'],
            'client_secret': os.environ['DEXCOM_CLIENT_SECRET'],
            'code': authorization_code,
            'grant_type': 'authorization_code',
            'redirect_uri': oauth_info['redirect_uri']
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(
            url, data=payload, timeout=10, headers=headers)
        data = response.json()
        oauth_info['oauth2_tokens'] = data

        # Return the refresh, access tokens.
        response = Response(body={
            'refresh_token': data['refresh_token'],
            'access_token': data['access_token'],
            'expires_in': data['expires_in'],
            'token_type': data['token_type'],
            'message': 'Successfully authorized.'
        }, status_code=200, headers={'Content-Type': 'application/json'})
    elif authorization_code is None and oauth_info['oauth2_tokens'] is None:
        #: Redirect to dexcom, get the authorization code.
        url = oauth_info['login_url']
        payload = {
            'client_id': os.environ['DEXCOM_CLIENT_ID'],
            'redirect_uri': oauth_info['redirect_uri'],
            'response_type': 'code',
            'scope': 'offline_access',
            'state': oauth_info['state']
        }
        response = Response(
            status_code=301,
            headers={'Location': url + '?' + urlparse.urlencode(payload)},
            body=''
        )
    elif oauth_info['oauth2_tokens'] is not None:
        #: Refresh the token.
        url = oauth_info['refresh_url']
        payload = {
            'client_id': os.environ['DEXCOM_CLIENT_ID'],
            'client_secret': os.environ['DEXCOM_CLIENT_SECRET'],
            'refresh_token': oauth_info['oauth2_tokens']['refresh_token'],
            'grant_type': 'refresh_token',
            'redirect_uri': oauth_info['redirect_uri']
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(
            url, data=payload, timeout=10, headers=headers)
        data = response.json()
        oauth_info['oauth2_tokens'] = data
        response = Response(body={
            'refresh_token': data['refresh_token'],
            'access_token': data['access_token'],
            'expires_in': data['expires_in'],
            'token_type': data['token_type'],
            'message': 'Successfully refreshed token.'
        }, status_code=200, headers={'Content-Type': 'application/json'})
    else:
        logger.warning(
            "(oauth2_dexcom) Not sure how this would occur, but thought you should know...")
        response = Response(body='Unauthorized', status_code=401)
    return response


@app.route('/login-success', methods=['GET'])
def login_success():
    return Response(status_code=200, body='<h1>Dexcom Login Success!</h1>')
