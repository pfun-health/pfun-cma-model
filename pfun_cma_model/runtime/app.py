"""
PFun CMA Model API Backend Routes.
"""
import os
import json
from chalice.app import (
    ConvertToMiddleware,
    Request, BadRequestError, Chalice,
    Response, CORSConfig
)
from pathlib import Path
from typing import (
    Any, Dict, Literal, Optional
)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pfun_path_helper
pfun_path_helper.append_path(Path(__file__).parent.parent)
from pfun_cma_model.pathdefs import (
    PRIVATE_ROUTES
)
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

file_handler = logging.FileHandler('/tmp/chalice-logs-backend.log')
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

app = Chalice(app_name='PFun CMA Model Backend')
if os.getenv('DEBUG_CHALICE', '0') in ['1', 'true']:
    app.debug = True
app.log.setLevel(logging.INFO)

app.api.cors = cors_config
app.websocket_api.session = PFunCMASession.get_boto3_session()
app.experimental_feature_flags.update([
    'WEBSOCKETS'
])


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


@app.route('/api', methods=['GET', 'POST', 'PUT'])
def dummy_route():
    path = str(get_current_request(app).path).lstrip('/api')
    if path == '':
        path = '/'
    print('path:', path)
    app.log.info('Redirect for path: %s', path)
    return Response(body='redirect...', status_code=303,
                    headers={'Location': path})


@app.route('/params/schema', methods=['GET'])
def params_schema():
    from pfun_cma_model.runtime.src.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return params.model_json_schema()


@app.route('/params/default', methods=['GET'])
def default_params():
    from pfun_cma_model.runtime.src.engine.cma_model_params import CMAModelParams
    params = CMAModelParams()
    return params.model_dump_json()


@app.route("/log", methods=['GET', 'POST'])
def logging_route(level: Literal['info', 'warning', 'error'] = 'info'):
    current_request = get_current_request(app)
    if current_request is None:
        raise RuntimeError("Logging error! No request was provided!")
    if current_request.query_params is None:
        raise RuntimeError("Logging error! No query parameters were provided!")
    msg = current_request.query_params.get('msg') or \
        current_request.query_params.get('message')
    level = current_request.query_params.get('level', level)
    if msg is None:
        return Response(body='No message provided.', status_code=BadRequestError.STATUS_CODE)
    loggers = {
        'debug': app.log.debug,
        'info': app.log.info,
        'warning': app.log.warning,
        'error': app.log.error
    }
    loggers[level](msg)
    return Response(body={'message': msg, 'level': level}, status_code=200)


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


def get_model_config(app: Chalice, key: str = 'model_config') -> Dict:
    return get_params(app, key=key)


CMA_MODEL_INSTANCE = None


def initialize_model():
    global CMA_MODEL_INSTANCE
    if CMA_MODEL_INSTANCE is not None:
        return CMA_MODEL_INSTANCE
    from pfun_cma_model.runtime.src.engine.cma_sleepwake import CMASleepWakeModel
    from pfun_cma_model.runtime.src.engine.cma_model_params import CMAModelParams
    model_config = get_model_config(app)
    if model_config is None:
        model_config = {}
    model_config = CMAModelParams(**model_config)
    model = CMASleepWakeModel(model_config)
    CMA_MODEL_INSTANCE = model
    return CMA_MODEL_INSTANCE


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


@app.route('/run', methods=["GET", "POST"])
def run_model_route():
    """Runs the CMA model.
    """
    request: Request | None = get_current_request(app)
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
    from pfun_cma_model.runtime.src.engine.cma_model_params import CMAModelParams
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
    from pfun_cma_model.runtime.src.engine.fit import fit_model as cma_fit_model
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
