# CMAModelParams

Represents the parameters for a CMA model.  Args:     t (Optional[array_like], optional): Time vector (decimal hours). Defaults to None.     N (int, optional): Number of time points. Defaults to 24.     d (float, optional): Time zone offset (hours). Defaults to 0.0.     taup (float, optional): Circadian-relative photoperiod length. Defaults to 1.0.     taug (float, optional): Glucose response time constant. Defaults to 1.0.     B (float, optional): Glucose Bias constant. Defaults to 0.05.     Cm (float, optional): Cortisol temporal sensitivity coefficient. Defaults to 0.0.     toff (float, optional): Solar noon offset (latitude). Defaults to 0.0.     tM (Tuple[float, float, float], optional): Meal times (hours). Defaults to (7.0, 11.0, 17.5).     seed (Optional[int], optional): Random seed. Defaults to None.     eps (float, optional): Random noise scale (\"epsilon\"). Defaults to 1e-18.

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
from openapi_client.models.cma_model_params import CMAModelParams

# TODO update the JSON string below
json = "{}"
# create an instance of CMAModelParams from a JSON string
cma_model_params_instance = CMAModelParams.from_json(json)
# print the JSON string representation of the object
print(CMAModelParams.to_json())

# convert the object into a dict
cma_model_params_dict = cma_model_params_instance.to_dict()
# create an instance of CMAModelParams from a dict
cma_model_params_from_dict = CMAModelParams.from_dict(cma_model_params_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


