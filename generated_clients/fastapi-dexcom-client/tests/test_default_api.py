# coding: utf-8

from fastapi.testclient import TestClient


from datetime import datetime  # noqa: F401
from pydantic import StrictStr, field_validator  # noqa: F401
from typing import Any, List, Optional  # noqa: F401
from openapi_server.models.alert_record import AlertRecord  # noqa: F401
from openapi_server.models.calibration_record import CalibrationRecord  # noqa: F401
from openapi_server.models.egv_record import EGVRecord  # noqa: F401
from openapi_server.models.exchange_authorization_code200_response import ExchangeAuthorizationCode200Response  # noqa: F401
from openapi_server.models.get_data_range200_response import GetDataRange200Response  # noqa: F401
from openapi_server.models.get_devices_v3200_response_inner import GetDevicesV3200ResponseInner  # noqa: F401
from openapi_server.models.get_events_v3200_response_inner import GetEventsV3200ResponseInner  # noqa: F401


def test_exchange_authorization_code(client: TestClient):
    """Test case for exchange_authorization_code

    
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    data = {
        "client_id": 'client_id_example',
        "client_secret": 'client_secret_example',
        "code": 'code_example',
        "grant_type": 'grant_type_example',
        "redirect_uri": 'redirect_uri_example'
    }
    # uncomment below to make a request
    #response = client.request(
    #    "POST",
    #    "/v2/oauth2/token",
    #    headers=headers,
    #    data=data,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_alerts(client: TestClient):
    """Test case for get_alerts

    
    """
    params = [("start_date", '2013-10-20T19:20:30+01:00'),     ("end_date", '2013-10-20T19:20:30+01:00')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v3/users/self/alerts",
    #    headers=headers,
    #    params=params,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_calibrations(client: TestClient):
    """Test case for get_calibrations

    
    """
    params = [("start_date", '2013-10-20T19:20:30+01:00'),     ("end_date", '2013-10-20T19:20:30+01:00')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v3/users/self/calibrations",
    #    headers=headers,
    #    params=params,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_data_range(client: TestClient):
    """Test case for get_data_range

    
    """
    params = [("last_sync_time", '2013-10-20T19:20:30+01:00')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v3/users/self/dataRange",
    #    headers=headers,
    #    params=params,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_devices_v3(client: TestClient):
    """Test case for get_devices_v3

    
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v3/users/self/devices",
    #    headers=headers,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_estimated_glucose_values(client: TestClient):
    """Test case for get_estimated_glucose_values

    
    """
    params = [("start_date", '2013-10-20T19:20:30+01:00'),     ("end_date", '2013-10-20T19:20:30+01:00')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v3/users/self/egvs",
    #    headers=headers,
    #    params=params,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_events_v3(client: TestClient):
    """Test case for get_events_v3

    
    """
    params = [("start_date", '2013-10-20T19:20:30+01:00'),     ("end_date", '2013-10-20T19:20:30+01:00')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v3/users/self/events",
    #    headers=headers,
    #    params=params,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_user_authorization(client: TestClient):
    """Test case for user_authorization

    
    """
    params = [("client_id", 'client_id_example'),     ("redirect_uri", 'redirect_uri_example'),     ("response_type", 'response_type_example'),     ("scope", 'scope_example'),     ("state", 'state_example')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    #response = client.request(
    #    "GET",
    #    "/v2/oauth2/login",
    #    headers=headers,
    #    params=params,
    #)

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200

