# CMAModelParamsBounds


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**lb** | **List[float]** |  | 
**ub** | **List[float]** |  | 
**keep_feasible** | **List[bool]** |  | 

## Example

```python
from openapi_client.models.cma_model_params_bounds import CMAModelParamsBounds

# TODO update the JSON string below
json = "{}"
# create an instance of CMAModelParamsBounds from a JSON string
cma_model_params_bounds_instance = CMAModelParamsBounds.from_json(json)
# print the JSON string representation of the object
print(CMAModelParamsBounds.to_json())

# convert the object into a dict
cma_model_params_bounds_dict = cma_model_params_bounds_instance.to_dict()
# create an instance of CMAModelParamsBounds from a dict
cma_model_params_bounds_from_dict = CMAModelParamsBounds.from_dict(cma_model_params_bounds_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


