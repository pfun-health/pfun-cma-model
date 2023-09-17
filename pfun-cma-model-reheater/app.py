"""pfun-cma-model-dev-reheater/app.py"""
import logging
import os
import time
import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests import ConnectionError
from requests.sessions import Session
from chalice.app import Chalice


MAX_RETRIES = os.environ.get('MAX_RETRIES', 3)
try:
    MAX_RETRIES = int(MAX_RETRIES)
except ValueError:
    GIVEN_MAX_RETRIES = MAX_RETRIES
    MAX_RETRIES = 3
    logging.warning(
        "Maximum retries must be an integer, not '%s' (using default of %s)",
        GIVEN_MAX_RETRIES, MAX_RETRIES)

BASE_URL = os.environ.get('BASE_URL', 'https://oias8ms59c.execute-api.us-east-1.amazonaws.com/api')

app = Chalice(app_name='pfun-cma-model-dev-reheater')


class DummyResponse:
    def __init__(self, status_code=500, body='{"message": "dummy response."}'):
        self.status_code = status_code
        self.body = body

    def json(self):
        return json.loads(self.body)

    @property
    def raw(self):
        return BytesIO(self.body.encode('utf-8'))


def reheater_func(url_path):
    retries = 0
    success = False
    response = DummyResponse()
    with Session() as session:
        while retries < MAX_RETRIES and not success:
            try:
                response = session.get(BASE_URL + url_path)
                if response.status_code == 200:
                    success = True
            except ConnectionError:
                retries += 1
                time.sleep(2)
    return {
        "message": "Lambda function kept warm!" if success else "Failed to warm up Lambda.",
        "status_code": response.status_code,
        "data": response.raw.read().decode('utf-8')
    }


@app.schedule('rate(15 minutes)')
def keep_warm(event):
    logging.debug('Keeping Lambda warm...')
    logging.debug('Event: %s', json.dumps(event.to_dict()))
    paths = ['/', '/run', '/fit', '/run-at-time']
    futures = []
    results = []
    response = {"message": "empty response?!"}
    with ThreadPoolExecutor(max_workers=10) as executor:
        for path in paths:
            future = executor.submit(reheater_func, path)
            futures.append(future)
        try:
            for future in as_completed(futures, timeout=50):
                result = future.result(timeout=10)
                logging.debug('Result: %s', json.dumps(result))
                results.append(result)
        except TimeoutError as e:
            logging.error('TimeoutError: %s', e)
            response = {
                "error": "TimeoutError: %s" % e,
                "status_code": 502,
                "data": ""
            }
        else:
            response = {
                "message": "Lambda kept warm!",
                "status_code": 200,
                "data": json.dumps({
                    "nr_successes": sum([1 for r in results if r['status_code'] == 200]),
                })
            }
    return response
