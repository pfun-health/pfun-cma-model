"""pfun-cma-model-dev-reheater/app.py"""
import logging
import os
import time
import json
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


@app.schedule('rate(15 minutes)')
def keep_warm(event):
    logging.debug('Keeping Lambda warm...')
    logging.debug('Event: %s', json.dumps(event.to_dict()))
    retries = 0
    success = False
    response = DummyResponse()
    with Session() as session:
        while retries < MAX_RETRIES and not success:
            try:
                response = session.get(BASE_URL)
                if response.status_code == 200:
                    success = True
            except ConnectionError:
                retries += 1
                time.sleep(5)
    return {
        "message": "Lambda function kept warm!" if success else "Failed to warm up Lambda.",
        "status_code": response.status_code,
        "data": response.json()
    }
