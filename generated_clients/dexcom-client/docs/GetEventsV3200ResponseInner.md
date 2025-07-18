# GetEventsV3200ResponseInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**event_id** | **str** |  | [optional] 
**event_type** | **str** |  | [optional] 
**system_time** | **datetime** |  | [optional] 
**display_time** | **datetime** |  | [optional] 
**value** | **object** |  | [optional] 
**unit** | **object** |  | [optional] 
**transmitter_id** | **object** |  | [optional] 
**transmitter_generation** | **object** |  | [optional] 
**display_device** | **object** |  | [optional] 

## Example

```python
from openapi_client.models.get_events_v3200_response_inner import GetEventsV3200ResponseInner

# TODO update the JSON string below
json = "{}"
# create an instance of GetEventsV3200ResponseInner from a JSON string
get_events_v3200_response_inner_instance = GetEventsV3200ResponseInner.from_json(json)
# print the JSON string representation of the object
print(GetEventsV3200ResponseInner.to_json())

# convert the object into a dict
get_events_v3200_response_inner_dict = get_events_v3200_response_inner_instance.to_dict()
# create an instance of GetEventsV3200ResponseInner from a dict
get_events_v3200_response_inner_from_dict = GetEventsV3200ResponseInner.from_dict(get_events_v3200_response_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


