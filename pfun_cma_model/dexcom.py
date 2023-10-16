"""dexcom.py"""
from typing import (
    Literal,
    Dict,
    List,
    Optional,
    Any
)
from pydantic import BaseModel, Field, validator

#: possible dexcom endpoints
DexcomEndpoint = Literal["dataRange", "egvs", "alerts", "calibrations", "devices", "events"]

#: possible EGV units
EgvUnits = Literal["mg/dL", "mmol/L"]


class DexcomAPIRecordModel(BaseModel):
    recordId: str
    systemTime: str
    displayTime: str
    alertName: Optional[str]
    alertState: Optional[str]
    displayDevice: Optional[str]
    transmitterGeneration: Optional[str]
    transmitterId: Optional[str]
    unit: Optional[EgvUnits] = "mg/dL"
    value: Optional[int]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


class DexcomAPIResponseModel:
    recordType: DexcomEndpoint
    recordVersion: Literal["3.0", "2.0"] = "3.0"
    userId: Optional[str]
    tz_offset: Optional[str | float | int | None]
    records: List[DexcomAPIRecordModel]

    class Config:
        allow_extras = True


class EgvsRecordModel(DexcomAPIRecordModel):
    unit: EgvUnits = "mg/dL"
    value: int = Field(ge=30.0, le=400.0)
    status: Literal["unknown", "high", "low", "ok"]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True

    @validator("status", allow_reuse=True)
    def computeStatus(cls, status_value, values):
        if status_value is None or status_value == "unknown":
            egv = values.get("value")
            if not any([isinstance(egv, int), isinstance(egv, float)]):
                return "unknown"
            if egv < 70:
                return "low"
            elif egv > 180:
                return "high"
            else:
                return "ok"
        return status_value


class EgvsResponseModel(DexcomAPIResponseModel, BaseModel):
    recordType = "egvs"
    records: List[EgvsRecordModel] | Dict[Any, EgvsRecordModel] | Any

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


EgvsResponseModel.update_forward_refs()


class AlertsRecordModel(DexcomAPIRecordModel):
    alertName: str


class DatarangeRecordModel(DexcomAPIRecordModel):
    data: Dict[str, Any]
    records: Optional[None | Any]


class DatarangeResponseModel(DexcomAPIResponseModel, BaseModel):
    recordType = "dataRange"
    records: Optional[List[DatarangeRecordModel]]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


class AlertsResponseModel(DexcomAPIResponseModel, BaseModel):
    recordType = "alerts"
    records: List[EgvsRecordModel]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


class CalibrationsResponseModel(DexcomAPIResponseModel, BaseModel):
    recordType = "calibrations"
    records: List[EgvsRecordModel]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


class DevicesResponseModel(DexcomAPIResponseModel, BaseModel):
    recordType = "devices"
    records: List[EgvsRecordModel]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True


class EventsResponseModel:
    recordType = "events"
    records: List[EgvsRecordModel]

    class Config:
        allow_extras = True
        arbitrary_types_allowed = True
