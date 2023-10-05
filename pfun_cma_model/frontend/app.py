"""
PFun CMA Model API routes.
"""
import os
import json
import sys
import uuid
from chalice.app import (
    ConvertToMiddleware,
    Request, BadRequestError, Chalice,
    Response, CORSConfig
)
from requests import post
from pathlib import Path
import urllib.parse as urlparse
from typing import (
    Any, Dict, Literal, Optional
)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pfun_path_helper
pfun_path_helper.append_path(Path(__file__).parent.parent)
import pfun_cma_model.utils as utils
from pfun_cma_model.pathdefs import (
    FRONTEND_ROUTES,
    PUBLIC_ROUTES,
    PRIVATE_ROUTES
)
from pfun_cma_model.secrets import get_secret_func
from pfun_cma_model.sessions import PFunCMASession
from pfun_cma_model.middleware import authorization_required as authreq

BOTO3_SESSION = PFunCMASession.get_boto3_session()
SECRETS_CLIENT = PFunCMASession.get_boto3_client('secretsmanager')
SDK_CLIENT = None
BASE_URL: Optional[str] = None
STATIC_BASE_URL: str | None = None
BODY: Optional[str] = None
S3_CLIENT = PFunCMASession.get_boto3_client('s3')

#: init app, set cors
cors_config = CORSConfig(
    allow_origin='*',
    allow_headers=['Access-Control-Allow-Origin'],
    allow_credentials=True,
    max_age=300,
    expose_headers=['X-API-Key',
                    'Authorization',
                    'Access-Control-Allow-Origin']
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('/tmp/chalice-logs-frontend.log')
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

app = Chalice(app_name='PFun CMA Model Frontend')
if os.getenv('DEBUG_CHALICE', '0') in ['1', 'true']:
    app.debug = True
app.log.setLevel(logging.INFO)

app.api.cors = cors_config


def get_current_request(app: Chalice = app) -> Request:  # pylint: disable=dangerous-default-value
    current_request: Request = app.current_request if app.current_request \
        is not None else Request({})  # to make the linter shut up.
    return current_request


authorization_required = authreq(
    app,
    get_current_request,  # type: ignore
    PRIVATE_ROUTES,
    SECRETS_CLIENT,
    PFunCMASession,
    logger)
app.register_middleware(ConvertToMiddleware(authorization_required), event_type='all')


def get_params(app: Chalice, key: str, default: Any = None, load_json: bool = False) -> Dict:
    current_request = get_current_request(app)
    if current_request is None:
        raise RuntimeError("No request was provided!")
    params = {} if current_request.json_body is None else \
        current_request.json_body
    if isinstance(params, (str, bytes)):
        params = json.loads(params)
    if key in params:
        params = params[key]
    if current_request.query_params is not None:
        params.update(current_request.query_params)
    if key in params:
        params = params[key]
    if params is None:
        params = default
    if load_json and isinstance(params, (str, bytes)):
        params = json.loads(params)
    return params


@app.route('/translate-results', methods=['POST', 'GET'])
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


DexcomEndpoint = Literal["dataRange", "egvs", "alerts", "calibrations",
                         "devices", "events"]


def get_oauth_info():
    oauth_info: Dict[str, str | Any | os.PathLike] = {
        "creds": {},
        "host": "",
        "login_url": "",
        "redirect_uri": "",
        "state": str(uuid.uuid4()),
        "oauth2_tokens": None
    }
    secret: str | bytes = get_secret_func("dexcom_pfun-app_glucose")  # type: ignore
    oauth_info["creds"] = json.loads(secret)
    os.environ['DEXCOM_CLIENT_ID'] = oauth_info['creds']['client_id']
    os.environ['DEXCOM_CLIENT_SECRET'] = oauth_info['creds']['client_secret']
    oauth_info["host"] = os.getenv(
        "DEXCOM_HOST", get_params(app, 'DEXCOM_HOST')) or \
        'https://api.dexcom.com'
    oauth_info["login_url"] = urlparse.urljoin(
        str(oauth_info["host"]), "/v2/oauth2/login")
    oauth_info["redirect_uri"] = os.getenv("DEXCOM_REDIRECT_URI") or \
        '/login-success'
    oauth_info["token_url"] = urlparse.urljoin(
        str(oauth_info["host"]), "v2/oauth2/token")
    oauth_info['refresh_url'] = urlparse.urljoin(
        str(oauth_info["host"]), "v2/oauth2/token")
    oauth_info['endpoint_urls'] = {}
    endpoints: list[str] = vars(DexcomEndpoint)['__args__']
    for endpoint in endpoints:
        oauth_info["endpoint_urls"][endpoint] = urlparse.urljoin(
            str(oauth_info['host']), f"v3/users/{{}}/{endpoint}")
    return oauth_info


oauth_info = None


@app.route('/login-dexcom', methods=['GET', 'POST'])
def oauth2_dexcom():
    """
    Handles the Dexcom login request.
    """

    global oauth_info
    if oauth_info is None:
        oauth_info = get_oauth_info()

    # Get the authorization code from the event.
    authorization_code = get_params(app, 'authorization_code')
    if authorization_code is not None and oauth_info['oauth2_tokens'] is None:
        # Exchange the authorization code for an access token.
        url: str = str(oauth_info['token_url'])
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
        url = str(oauth_info['login_url'])
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
        url = str(oauth_info['refresh_url'])
        payload = {
            'client_id': os.environ['DEXCOM_CLIENT_ID'],
            'client_secret': os.environ['DEXCOM_CLIENT_SECRET'],
            'refresh_token': str(oauth_info['oauth2_tokens']['refresh_token']),  # type: ignore
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


if __name__ == '__main__':
    import chalice.cli.factory as chalice_factory
    cli_factory = chalice_factory.CLIFactory(
        os.path.dirname(__file__),
        debug=True,
        profile='robbie',
        environ=os.environ
    )
    local_server = cli_factory.create_local_server(
        app,
        config=cli_factory.create_config_obj(),
        host='127.0.0.1',
        port=1337
    )
    local_server.serve_forever()
