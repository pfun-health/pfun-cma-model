# GetDevicesV3200ResponseInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**device_id** | **str** |  | [optional] 
**device_type** | **str** |  | [optional] 
**transmitter_generation** | **object** |  | [optional] 
**transmitter_id** | **object** |  | [optional] 
**display_device** | **object** |  | [optional] 
**activation_date** | **datetime** |  | [optional] 
**deactivation_date** | **datetime** |  | [optional] 

## Example

```python
from openapi_client.models.get_devices_v3200_response_inner import GetDevicesV3200ResponseInner

# TODO update the JSON string below
json = "{}"
# create an instance of GetDevicesV3200ResponseInner from a JSON string
get_devices_v3200_response_inner_instance = GetDevicesV3200ResponseInner.from_json(json)
# print the JSON string representation of the object
print(GetDevicesV3200ResponseInner.to_json())

# convert the object into a dict
get_devices_v3200_response_inner_dict = get_devices_v3200_response_inner_instance.to_dict()
# create an instance of GetDevicesV3200ResponseInner from a dict
get_devices_v3200_response_inner_from_dict = GetDevicesV3200ResponseInner.from_dict(get_devices_v3200_response_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


