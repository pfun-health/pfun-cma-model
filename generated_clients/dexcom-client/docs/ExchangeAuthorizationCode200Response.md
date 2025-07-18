# ExchangeAuthorizationCode200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**access_token** | **str** |  | [optional] 
**expires_in** | **int** |  | [optional] 
**token_type** | **str** |  | [optional] 
**refresh_token** | **str** |  | [optional] 

## Example

```python
from openapi_client.models.exchange_authorization_code200_response import ExchangeAuthorizationCode200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ExchangeAuthorizationCode200Response from a JSON string
exchange_authorization_code200_response_instance = ExchangeAuthorizationCode200Response.from_json(json)
# print the JSON string representation of the object
print(ExchangeAuthorizationCode200Response.to_json())

# convert the object into a dict
exchange_authorization_code200_response_dict = exchange_authorization_code200_response_instance.to_dict()
# create an instance of ExchangeAuthorizationCode200Response from a dict
exchange_authorization_code200_response_from_dict = ExchangeAuthorizationCode200Response.from_dict(exchange_authorization_code200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


