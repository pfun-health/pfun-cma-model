# GetDataRange200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_type** | **str** |  | [optional] 
**record_version** | **str** |  | [optional] 
**user_id** | **str** |  | [optional] 
**calibrations** | [**GetDataRange200ResponseCalibrations**](GetDataRange200ResponseCalibrations.md) |  | [optional] 
**egvs** | [**GetDataRange200ResponseCalibrations**](GetDataRange200ResponseCalibrations.md) |  | [optional] 
**events** | [**GetDataRange200ResponseCalibrations**](GetDataRange200ResponseCalibrations.md) |  | [optional] 

## Example

```python
from openapi_client.models.get_data_range200_response import GetDataRange200Response

# TODO update the JSON string below
json = "{}"
# create an instance of GetDataRange200Response from a JSON string
get_data_range200_response_instance = GetDataRange200Response.from_json(json)
# print the JSON string representation of the object
print(GetDataRange200Response.to_json())

# convert the object into a dict
get_data_range200_response_dict = get_data_range200_response_instance.to_dict()
# create an instance of GetDataRange200Response from a dict
get_data_range200_response_from_dict = GetDataRange200Response.from_dict(get_data_range200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


