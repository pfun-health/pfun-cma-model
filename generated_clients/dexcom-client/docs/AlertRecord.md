# AlertRecord


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_id** | **str** |  | [optional] 
**system_time** | **datetime** |  | [optional] 
**display_time** | **datetime** |  | [optional] 
**alert_name** | **str** |  | [optional] 
**alert_state** | **str** |  | [optional] 
**display_device** | **str** |  | [optional] 
**transmitter_generation** | **str** |  | [optional] 
**transmitter_id** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.alert_record import AlertRecord

# TODO update the JSON string below
json = "{}"
# create an instance of AlertRecord from a JSON string
alert_record_instance = AlertRecord.from_json(json)
# print the JSON string representation of the object
print(AlertRecord.to_json())

# convert the object into a dict
alert_record_dict = alert_record_instance.to_dict()
# create an instance of AlertRecord from a dict
alert_record_from_dict = AlertRecord.from_dict(alert_record_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


