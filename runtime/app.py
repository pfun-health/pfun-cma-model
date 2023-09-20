"""
PFun CMA Model API routes.
"""
import os
import json
import sys
import uuid
from chalice.app import (
    UnauthorizedError, ConvertToMiddleware,
    Request, BadRequestError, Chalice,
    Response, AuthRoute, CORSConfig,
    CaseInsensitiveMapping
)
from requests import post
from requests.sessions import Session
from pathlib import Path
import urllib.parse as urlparse
from typing import (
    Any, Dict, Literal
)
from botocore.config import Config as ConfigCore
import boto3
from botocore.exceptions import ClientError
import importlib

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: pfun imports (relative)
root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)
utils = importlib.import_module('.utils', package='chalicelib')


BOTO3_SESSION = None


def new_boto3_session():
    global BOTO3_SESSION
    if BOTO3_SESSION is not None:
        return BOTO3_SESSION
    BOTO3_SESSION = boto3.Session()
    return BOTO3_SESSION


def new_boto3_client(service_name: str, session=None, *args, **kwds):
    """
    Creates a new Boto3 client for a specified AWS service.

    Args:
        service_name (str): The name of the AWS service for which the client is being created.
        session (boto3.Session, optional): An existing Boto3 session to use. If not provided, a new session will be created.
        *args: Additional arguments that will be passed to the Boto3 client constructor.
        **kwds: Additional keyword arguments that will be passed to the Boto3 client constructor.

    Returns:
        boto3.client: The newly created Boto3 client for the specified AWS service.
    """
    config = ConfigCore(region_name='us-east-1')
    session = session or new_boto3_session()
    client = session.client(service_name, *args, config=config, **kwds)
    return client


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
app.log.setLevel(logging.INFO)

app.api.cors = cors_config
app.websocket_api.session = new_boto3_session()
app.experimental_feature_flags.update([
    'WEBSOCKETS'
])

API_ROUTES = [
    '/run',
    '/run-at-time',
    '/params/schema',
    '/params/default'
]

PUBLIC_ROUTES: list[str | AuthRoute] = [
    '/',
    '/run',
    '/fit',
    '/run-at-time',
    '/routes',
    '/sdk',
    '/params/schema',
    '/params/default',
]

PRIVATE_ROUTES: list[str] = [
    '/run',
    '/fit',
    '/run-at-time',
    '/sdk'
]

SECRETS_CLIENT = None


def authorization_required(func):
    """
    A wrapper function that handles authentication for the API.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of the wrapped function.

    Raises:
        UnauthorizedError: If the authentication parameters are invalid.

    """
    def wrapper(*args, **kwargs):
        global SECRETS_CLIENT
        if not hasattr(app, 'current_request'):
            #: skip authorization for lambda functions
            return func(*args, **kwargs)
        if app.current_request.path not in PRIVATE_ROUTES:
            #: skip authorization for public routes
            return func(*args, **kwargs)
        if SECRETS_CLIENT is None:
            # lazy load secrets client
            SECRETS_CLIENT = new_boto3_client('secretsmanager')
        api_key = SECRETS_CLIENT.get_secret_value(
            SecretId='pfun-cma-model-aws-api-key')['SecretString']
        rapidapi_key = SECRETS_CLIENT.get_secret_value(
            SecretId='pfun-cma-model-rapidapi-key')['SecretString']
        logger.info('RapidAPI key: %s', rapidapi_key)
        logger.info('API key: %s', api_key)
        try:
            current_request = app.current_request
            api_key_given = current_request.headers.get('X-API-Key')
            apikey_authorized = api_key_given == api_key
            rapidapi_key_given = current_request.headers.get('X-RapidAPI-Key')
            rapidapi_authorized = rapidapi_key_given == rapidapi_key
            if any([apikey_authorized, rapidapi_authorized]):
                logger.info('Authorized request: %s', str(vars(current_request)))
                return func(*args, **kwargs)
            else:
                raise UnauthorizedError('Unauthorized request: %s' % str(vars(current_request)))
        except UnauthorizedError:
            logger.error(
                'authorization parameters given:\n\tapi_key: %s (%s),\n\trapidapi_key: %s (%s)',
                api_key_given, str(apikey_authorized), rapidapi_key_given, str(rapidapi_authorized))
            return Response(
                body='Unauthorized request.\nAuth params:\n\tapi_key: %s,\n\trapidapi_key: %s' % (api_key_given, rapidapi_key_given), status_code=401
            )
    return wrapper


app.register_middleware(ConvertToMiddleware(authorization_required), event_type='all')


SDK_CLIENT = None


@app.route('/sdk', methods=['GET'])
def generate_sdk():
    
    global SDK_CLIENT
    logger.info('Generating SDK')
    if SDK_CLIENT is None:
        SDK_CLIENT = new_boto3_client('apigateway')
    rest_api_id = next(item for item in SDK_CLIENT.get_rest_apis()['items']
                       if item.get('name') == 'PFun CMA Model Backend')['id']
    response = SDK_CLIENT.get_sdk(
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


S3_CLIENT = None

@app.route('/params/schema', methods=['GET'])
def params_schema():
    from chalicelib.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return params.model_json_schema()

@app.route('/params/default', methods=['GET'])
def default_params():
    from chalicelib.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return params.model_dump_json()


@app.route('/static', methods=['GET'])
def static_files():
    """
    Serves the static files for the web application.
    """
    global S3_CLIENT
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
    source: str = app.current_request.query_params.get('source', 's3')
    if STATIC_BASE_URL is None:
        initialize_base_url(app)
    if source not in ('s3', 'local'):
        app.log.warning('Invalid source provided (requested static resource).')
        return Response(body='Invalid source provided when requesting static resource.', status_code=BadRequestError.STATUS_CODE)
    filename = app.current_request.query_params.get('filename', '/index.template.html')
    try:
        if source.lower() == 's3':
            #: use s3 as source for static files
            if S3_CLIENT is None:
                S3_CLIENT = new_boto3_client('s3')
            s3_filename = filename.lstrip('/')  # remove leading slash for s3
            stage_name = os.getenv('stage', 'error').lower()
            try:
                app.log.info('stage_name: %s', stage_name)
                bucket_name = f'pfun-cma-model-www-{stage_name}'
                response = S3_CLIENT.get_object(Bucket=bucket_name, Key=s3_filename)
            except ClientError:
                app.log.warning(
                    '(stage: %s) Requested static resource does not exist: %s', str(stage_name), str(filename), exc_info=True)
                # app.log.warning('context: %s', str(app.current_request.context))
                # app.log.warning('event: %s', str(app.current_request._event_dict))
                app.log.warning('environment: %s', str({k: v for k, v in os.environ.items() if 'AWS' in k.upper() or 'CHALICE' in k.upper()}))
                return Response(
                    body='Requested static resource does not exist (requested static resource: %s).' % str(filename), status_code=404)
            body = response['Body'].read()
            return Response(body=body,
                            status_code=200,
                            headers={'Content-Type': 'text/*'}
                            )
    except Exception:
        logger.warning('Failed to get static resource from s3: %s', str(filename))
        #: ! attempt to get the file locally if failure
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
    if 'text' in content_type or content_type == '*/*':
        try:
            output = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            output = filepath.read_bytes()
    else:
        output = filepath.read_bytes()
    return Response(body=output,
                    status_code=200, headers={'Content-Type': content_type, 'Access-Control-Allow-Origin': '*'})


BASE_URL = None
STATIC_BASE_URL = None
BODY = None


def initialize_base_url(app):
    global BASE_URL, STATIC_BASE_URL, SDK_CLIENT
    if BASE_URL is not None and STATIC_BASE_URL is not None:
        return BASE_URL
    BASE_URL = app.current_request.headers.get(
        'host', app.current_request.headers.get(
            'origin', app.current_request.headers.get('referer', '')))
    if any([BASE_URL == '', not hasattr(app, 'current_request')]):
        if SDK_CLIENT is None:
            SDK_CLIENT = new_boto3_client('apigateway')
        rest_api_id = next(item for item in SDK_CLIENT.get_rest_apis()['items']
                           if item.get('name') == 'PFun CMA Model Backend')['id']
        BASE_URL = 'https://%s.execute-api.us-east-1.amazonaws.com/api' % (rest_api_id,)
    if '127.0.0.1' in BASE_URL or 'localhost' in BASE_URL:
        #: handle localhost
        BASE_URL = f'http://{BASE_URL}'
    else:
        #: handle non-localhost
        if 'https://' not in BASE_URL:
            BASE_URL = f'https://{BASE_URL}'
        if not BASE_URL.endswith('/api'):
            BASE_URL += '/api'
    STATIC_BASE_URL = f'{BASE_URL}/static'
    return BASE_URL


def get_static_resource(path: str, source: Literal['s3', 'local'] = 's3', content_type: str = 'text/*'):
    global STATIC_BASE_URL
    if STATIC_BASE_URL is None:
        initialize_base_url(app)
    if any(x in STATIC_BASE_URL for x in ('localhost', '127.0.0.1')):
        source = 'local'  # ! overwrite source to local for localhost
    url = urlparse.urljoin(STATIC_BASE_URL, path)
    if path[0] != '/':
        path = '/' + path  # ! add leading slash
    url = utils.add_url_params(STATIC_BASE_URL, {
        'source': source, 'filename': path, 'ContentType': content_type})
    app.log.info('(static resource) GET: %s', url)
    response = None
    with Session() as session:
        response = session.get(url)
    return response


@app.route('/icons/mattepfunlogolighter.png')
def icon_static_resource():
    query_params = app.current_request.query_params
    if query_params is None:
        query_params = {}
    source = query_params.get('source', 's3')
    resp = get_static_resource("icons/mattepfunlogolighter.png", source, "image/png")
    return Response(body=resp.content, status_code=200, headers={'Content-Type': 'image/png', 'x-version': '4'})

def initialize_index_resources():
    global BODY, BASE_URL, STATIC_BASE_URL
    if BODY is not None:
        return BODY
    status_code = 200
    #: initialize base url
    BASE_URL = initialize_base_url(app)
    response = get_static_resource('index.template.html')
    body = response.text
    ROUTES = '\n'.join([
        f'<li><a id="{name}" class="dropdown-item route-link" href="{BASE_URL}{name}">{name}</a></li>'
        for name in PUBLIC_ROUTES])
    app.log.debug('BODY: %s', body)
    app.log.info('BASE_URL: %s', BASE_URL)
    app.log.info('STATIC_BASE_URL: %s', STATIC_BASE_URL)
    source = 's3'
    if any(x in STATIC_BASE_URL for x in ('localhost', '127.0.0.1')):
        source = 'local'  # ! overwrite source to local for localhost
    output_static_base_url = str(
        utils.add_url_params(
            STATIC_BASE_URL, {'source': source, 'filename': ''}))
    PFUN_ICON_PATH = '/icons/mattepfunlogolighter.png'
    if source != 'local':
        if '/api' not in output_static_base_url:
            output_static_base_url = urlparse.urlparse(output_static_base_url)
            output_static_base_url = str(output_static_base_url).replace(
                output_static_base_url.path, '/api' + output_static_base_url.path)
    try:
        BODY = body.format(
            STATIC_BASE_URL=output_static_base_url,
            PFUN_ICON_PATH=PFUN_ICON_PATH,
            ROUTES=ROUTES
        )
    except (ValueError, KeyError) as e:
        logging.warning(f"Error (index): {e}")
        logging.warning(f"STATIC_BASE_URL: {STATIC_BASE_URL}\noutput_static_base_url: {output_static_base_url}\nPFUN_ICON_PATH: {PFUN_ICON_PATH}\nROUTES: {str(ROUTES)}")
    return BODY


@app.route("/")
def index():
    """
    Generates the index page for the web application.

    Returns:
        Response: The HTTP response object containing the index page.
    """
    global BODY
    if BODY is None:
        BODY = initialize_index_resources()
    return Response(
        body=BODY,
        status_code=200,
        headers={'Content-Type': 'text/html'}
    )


@app.route("/routes")
def get_routes():
    routes = json.dumps({k: list(v.keys()) for k, v in app.routes.items()
                         if k in PUBLIC_ROUTES}, indent=4)
    return Response(body=routes, status_code=200)


@app.route("/log", methods=['GET', 'POST'])
def logging_route(level: Literal['info', 'warning', 'error'] = 'info'):
    if app.current_request is None:
        raise RuntimeError("Logging error! No request was provided!")
    if app.current_request.query_params is None:
        raise RuntimeError("Logging error! No query parameters were provided!")
    msg = app.current_request.query_params.get('msg') or \
        app.current_request.query_params.get('message')
    level = app.current_request.query_params.get('level', level)
    if msg is None:
        return Response(body='No message provided.', status_code=BadRequestError.STATUS_CODE)
    loggers = {
        'debug': app.log.debug,
        'info': app.log.info,
        'warning': app.log.warning,
        'error': app.log.error
    }
    loggers[level](msg)
    return {'message': msg, 'level': level}


def get_params(app: Chalice, key: str, default: Any = None, load_json: bool = False) -> Dict:
    if app.current_request is None:
        raise RuntimeError("No request was provided!")
    params = {} if app.current_request.json_body is None else \
        app.current_request.json_body
    if isinstance(params, (str, bytes)):
        params = json.loads(params)
    if key in params:
        params = params[key]
    if app.current_request.query_params is not None:
        params.update(app.current_request.query_params)
    if key in params:
        params = params[key]
    if params is None:
        params = default
    if load_json and isinstance(params, (str, bytes)):
        params = json.loads(params)
    return params


def get_model_config(app: Chalice, key: str = 'model_config') -> Dict:
    return get_params(app, key=key)


CMA_MODEL_INSTANCE = None


def initialize_model():
    global CMA_MODEL_INSTANCE
    if CMA_MODEL_INSTANCE is not None:
        return CMA_MODEL_INSTANCE
    from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
    from chalicelib.engine.cma_model_params import CMAModelParams
    model_config = get_model_config(app)
    if model_config is None:
        model_config = {}
    model_config = CMAModelParams(**model_config)
    model = CMASleepWakeModel(model_config)
    CMA_MODEL_INSTANCE = model
    return CMA_MODEL_INSTANCE


@app.route('/model/results/translate', methods=['POST', 'GET'])
def translate_model_results_by_language():
    results = get_params(app, 'results')
    from_lang = get_params(app, 'from', 'python', load_json=True)
    if from_lang not in ['python', 'javascript']:
        return Response(body='Invalid from language.', status_code=BadRequestError.STATUS_CODE)
    to_lang = get_params(app, 'to', 'javascript', load_json=True)
    if to_lang not in ['python', 'javascript']:
        return Response(body='Invalid to language.', status_code=BadRequestError.STATUS_CODE)
    if from_lang == to_lang:
        return Response(body=json.dumps(results), status_code=200)
    from pandas import DataFrame
    translation_dict = {
        'python': {
            'javascript': lambda x: DataFrame(x).to_json(orient='records'),
        },
        'javascript': {
            'python': lambda x: DataFrame.from_records(x).to_json(orient='columns'),
        }
    }
    return Response(body=translation_dict[from_lang][to_lang](results), status_code=200)


@app.route('/run', methods=["GET", "POST"])
def run_model_route():
    """Runs the CMA model.
    """
    request: Request | None = app.current_request
    if request is None:
        raise RuntimeError("No request was provided!")
    model_config = get_model_config(app)
    model = initialize_model()
    model.update(**model_config)
    df = model.run()
    output = df.to_json()
    response = Response(body=output, status_code=200,
                        headers={'Content-Type': 'application/json',
                                 'Access-Control-Allow-Origin': '*'})
    logger.info('Response: %s', json.dumps(response.to_dict()))
    return response


@app.on_ws_connect(name="run_at_time")
def run_at_time_connect(event):
    logger.info('New connection: %s' % event.connection_id)
    app.websocket_api.send(event.connection_id, "Connected")


def run_at_time_func(app: Chalice) -> str:
    model_config = get_model_config(app)
    calc_params = get_params(app, 'calc_params')
    # pylint-disable=import-outside-toplevel
    from chalicelib.engine.cma_model_params import CMAModelParams
    from pandas import DataFrame
    logger.info('model_config: %s', json.dumps(model_config))
    logger.info('calc_params: %s', json.dumps(calc_params))
    if model_config is None:
        model_config = {}
    model = initialize_model()
    model_config = CMAModelParams(**model_config)
    model.update(model_config)  # ! this occurs inplace !
    if calc_params is None:
        calc_params = {}
    df: DataFrame = model.calc_Gt(**calc_params)
    output = df.to_json()
    return output


@app.on_ws_message(name="run_at_time")
def run_at_time_ws(event):
    logger.info('Received message: %s', event.body)
    output: str = run_at_time_func(app)
    app.websocket_api.send(event.connection_id, output)


@app.route('/run-at-time', methods=["GET", "POST"])
def run_at_time_route():
    try:
        output = run_at_time_func(app)
        return Response(body=output, status_code=200,
                        headers={'Content-Type': 'application/json'})
    except Exception as err:
        app.log.error('failed to run at time.', exc_info=True)
        logger.error('failed to run at time.', exc_info=True)
        error_response = Response(body={"error": "failed to run at time. See error message on server log.", "exception": str(err)}, status_code=500)
        return error_response


@app.route('/fit', methods=['POST'])
def fit_model_to_data():
    from chalicelib.engine.fit import fit_model as cma_fit_model
    from pandas import DataFrame
    data = get_params(app, 'data')
    if data is None:
        raise RuntimeError("no data was provided!")
    if isinstance(data, str):
        data = json.loads(data)
    model_config = get_model_config(app)
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    try:
        df = DataFrame(data)
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
def oauth2_dexcom():
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
        response = post(
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
        response = post(
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
