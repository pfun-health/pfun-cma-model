# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from openapi_server.apis.default_api_base import BaseDefaultApi
import openapi_server.impl

from fastapi import (  # noqa: F401
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Response,
    Security,
    status,
)

from openapi_server.models.extra_models import TokenModel  # noqa: F401
from datetime import datetime
from pydantic import StrictStr, field_validator
from typing import Any, List, Optional
from openapi_server.models.alert_record import AlertRecord
from openapi_server.models.calibration_record import CalibrationRecord
from openapi_server.models.egv_record import EGVRecord
from openapi_server.models.exchange_authorization_code200_response import ExchangeAuthorizationCode200Response
from openapi_server.models.get_data_range200_response import GetDataRange200Response
from openapi_server.models.get_devices_v3200_response_inner import GetDevicesV3200ResponseInner
from openapi_server.models.get_events_v3200_response_inner import GetEventsV3200ResponseInner
from openapi_server.security_api import get_token_BearerAuth

router = APIRouter()

ns_pkg = openapi_server.impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/v2/oauth2/token",
    responses={
        200: {"model": ExchangeAuthorizationCode200Response, "description": "Access token and refresh token returned"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def exchange_authorization_code(
    client_id: StrictStr = Form(None, description=""),
    client_secret: StrictStr = Form(None, description=""),
    code: StrictStr = Form(None, description=""),
    grant_type: StrictStr = Form(None, description=""),
    redirect_uri: StrictStr = Form(None, description=""),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> ExchangeAuthorizationCode200Response:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().exchange_authorization_code(client_id, client_secret, code, grant_type, redirect_uri)


@router.get(
    "/v3/users/self/alerts",
    responses={
        200: {"model": AlertRecord, "description": "Ok"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def get_alerts(
    start_date: datetime = Query(None, description="", alias="startDate"),
    end_date: datetime = Query(None, description="", alias="endDate"),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> AlertRecord:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().get_alerts(start_date, end_date)


@router.get(
    "/v3/users/self/calibrations",
    responses={
        200: {"model": CalibrationRecord, "description": "Ok"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def get_calibrations(
    start_date: datetime = Query(None, description="", alias="startDate"),
    end_date: datetime = Query(None, description="", alias="endDate"),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> CalibrationRecord:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().get_calibrations(start_date, end_date)


@router.get(
    "/v3/users/self/dataRange",
    responses={
        200: {"model": GetDataRange200Response, "description": "Data range retrieved successfully"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def get_data_range(
    last_sync_time: Optional[datetime] = Query(None, description="", alias="lastSyncTime"),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> GetDataRange200Response:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().get_data_range(last_sync_time)


@router.get(
    "/v3/users/self/devices",
    responses={
        200: {"model": List[GetDevicesV3200ResponseInner], "description": "Ok"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def get_devices_v3(
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> List[GetDevicesV3200ResponseInner]:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().get_devices_v3()


@router.get(
    "/v3/users/self/egvs",
    responses={
        200: {"model": EGVRecord, "description": "Ok"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def get_estimated_glucose_values(
    start_date: datetime = Query(None, description="", alias="startDate"),
    end_date: datetime = Query(None, description="", alias="endDate"),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> EGVRecord:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().get_estimated_glucose_values(start_date, end_date)


@router.get(
    "/v3/users/self/events",
    responses={
        200: {"model": List[GetEventsV3200ResponseInner], "description": "Ok"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def get_events_v3(
    start_date: datetime = Query(None, description="", alias="startDate"),
    end_date: datetime = Query(None, description="", alias="endDate"),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> List[GetEventsV3200ResponseInner]:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().get_events_v3(start_date, end_date)


@router.get(
    "/v2/oauth2/login",
    responses={
        302: {"description": "Redirect to authorization code or access denied error"},
    },
    tags=["default"],
    response_model_by_alias=True,
)
async def user_authorization(
    client_id: StrictStr = Query(None, description="", alias="client_id"),
    redirect_uri: StrictStr = Query(None, description="", alias="redirect_uri"),
    response_type: StrictStr = Query(None, description="", alias="response_type"),
    scope: StrictStr = Query(None, description="", alias="scope"),
    state: Optional[StrictStr] = Query(None, description="", alias="state"),
    token_BearerAuth: TokenModel = Security(
        get_token_BearerAuth
    ),
) -> None:
    if not BaseDefaultApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseDefaultApi.subclasses[0]().user_authorization(client_id, redirect_uri, response_type, scope, state)
