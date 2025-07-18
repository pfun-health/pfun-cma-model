# CalibrationRecord


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_id** | **str** |  | [optional] 
**unit** | **str** |  | [optional] 
**system_time** | **datetime** |  | [optional] 
**display_time** | **datetime** |  | [optional] 
**value** | **int** |  | [optional] 
**display_device** | **str** |  | [optional] 
**transmitter_id** | **str** |  | [optional] 
**transmitter_ticks** | **int** |  | [optional] 
**transmitter_generation** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.calibration_record import CalibrationRecord

# TODO update the JSON string below
json = "{}"
# create an instance of CalibrationRecord from a JSON string
calibration_record_instance = CalibrationRecord.from_json(json)
# print the JSON string representation of the object
print(CalibrationRecord.to_json())

# convert the object into a dict
calibration_record_dict = calibration_record_instance.to_dict()
# create an instance of CalibrationRecord from a dict
calibration_record_from_dict = CalibrationRecord.from_dict(calibration_record_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


