# EGVRecord


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_id** | **str** |  | [optional] 
**system_time** | **datetime** |  | [optional] 
**display_time** | **datetime** |  | [optional] 
**transmitter_id** | **str** |  | [optional] 
**transmitter_ticks** | **int** |  | [optional] 
**value** | **int** |  | [optional] 
**status** | **str** |  | [optional] 
**trend** | **str** |  | [optional] 
**trend_rate** | **float** |  | [optional] 
**unit** | **str** |  | [optional] 
**rate_unit** | **str** |  | [optional] 
**display_device** | **str** |  | [optional] 
**transmitter_generation** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.egv_record import EGVRecord

# TODO update the JSON string below
json = "{}"
# create an instance of EGVRecord from a JSON string
egv_record_instance = EGVRecord.from_json(json)
# print the JSON string representation of the object
print(EGVRecord.to_json())

# convert the object into a dict
egv_record_dict = egv_record_instance.to_dict()
# create an instance of EGVRecord from a dict
egv_record_from_dict = EGVRecord.from_dict(egv_record_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


