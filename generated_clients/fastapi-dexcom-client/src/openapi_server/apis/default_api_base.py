# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

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

class BaseDefaultApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseDefaultApi.subclasses = BaseDefaultApi.subclasses + (cls,)
    async def exchange_authorization_code(
        self,
        client_id: StrictStr,
        client_secret: StrictStr,
        code: StrictStr,
        grant_type: StrictStr,
        redirect_uri: StrictStr,
    ) -> ExchangeAuthorizationCode200Response:
        ...


    async def get_alerts(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> AlertRecord:
        ...


    async def get_calibrations(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> CalibrationRecord:
        ...


    async def get_data_range(
        self,
        last_sync_time: Optional[datetime],
    ) -> GetDataRange200Response:
        ...


    async def get_devices_v3(
        self,
    ) -> List[GetDevicesV3200ResponseInner]:
        ...


    async def get_estimated_glucose_values(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> EGVRecord:
        ...


    async def get_events_v3(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[GetEventsV3200ResponseInner]:
        ...


    async def user_authorization(
        self,
        client_id: StrictStr,
        redirect_uri: StrictStr,
        response_type: StrictStr,
        scope: StrictStr,
        state: Optional[StrictStr],
    ) -> None:
        ...
