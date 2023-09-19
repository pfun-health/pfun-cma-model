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
    Dict, Literal
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

app = Chalice(app_name='frontend')

if os.getenv('DEBUG_CHALICE', False) in ['1', 'true']:
    app.debug = True
app.log.setLevel(logging.INFO)

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
    '/routes',
    '/sdk'
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
            try:
                response = S3_CLIENT.get_object(Bucket='pfun-cma-model-www', Key=s3_filename)
            except ClientError:
                app.log.warning('Requested static resource does not exist: %s', str(filename))
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
        if '/api' not in BASE_URL:
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


@app.route('/icons/mattepfunlogolighter.png', content_types=['image/png'])
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