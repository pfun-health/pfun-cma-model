# Config


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**t** | **float** |  | [optional] 
**n** | **int** |  | [optional] 
**d** | **float** |  | [optional] [default to 0.0]
**taup** | **float** |  | [optional] [default to 1.0]
**taug** | **float** |  | [optional] 
**b** | **float** |  | [optional] [default to 0.05]
**cm** | **float** |  | [optional] [default to 0.0]
**toff** | **float** |  | [optional] [default to 0.0]
**t_m** | [**Tm**](Tm.md) |  | [optional] 
**seed** | [**Seed**](Seed.md) |  | [optional] 
**eps** | **float** |  | [optional] 
**lb** | [**Lb**](Lb.md) |  | [optional] 
**ub** | [**Ub**](Ub.md) |  | [optional] 
**bounded_param_keys** | [**BoundedParamKeys**](BoundedParamKeys.md) |  | [optional] 
**midbound** | [**Midbound**](Midbound.md) |  | [optional] 
**bounded_param_descriptions** | [**BoundedParamDescriptions**](BoundedParamDescriptions.md) |  | [optional] 
**bounds** | [**CMAModelParamsBounds**](CMAModelParamsBounds.md) |  | [optional] 

## Example

```python
from openapi_client.models.config import Config

# TODO update the JSON string below
json = "{}"
# create an instance of Config from a JSON string
config_instance = Config.from_json(json)
# print the JSON string representation of the object
print(Config.to_json())

# convert the object into a dict
config_dict = config_instance.to_dict()
# create an instance of Config from a dict
config_from_dict = Config.from_dict(config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


