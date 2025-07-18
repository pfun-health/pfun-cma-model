# openapi_client.DefaultApi

All URIs are relative to *https://api.dexcom.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**exchange_authorization_code**](DefaultApi.md#exchange_authorization_code) | **POST** /v2/oauth2/token | 
[**get_alerts**](DefaultApi.md#get_alerts) | **GET** /v3/users/self/alerts | 
[**get_calibrations**](DefaultApi.md#get_calibrations) | **GET** /v3/users/self/calibrations | 
[**get_data_range**](DefaultApi.md#get_data_range) | **GET** /v3/users/self/dataRange | 
[**get_devices_v3**](DefaultApi.md#get_devices_v3) | **GET** /v3/users/self/devices | 
[**get_estimated_glucose_values**](DefaultApi.md#get_estimated_glucose_values) | **GET** /v3/users/self/egvs | 
[**get_events_v3**](DefaultApi.md#get_events_v3) | **GET** /v3/users/self/events | 
[**user_authorization**](DefaultApi.md#user_authorization) | **GET** /v2/oauth2/login | 


# **exchange_authorization_code**
> ExchangeAuthorizationCode200Response exchange_authorization_code(client_id, client_secret, code, grant_type, redirect_uri)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.exchange_authorization_code200_response import ExchangeAuthorizationCode200Response
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    client_id = 'client_id_example' # str | 
    client_secret = 'client_secret_example' # str | 
    code = 'code_example' # str | 
    grant_type = 'grant_type_example' # str | 
    redirect_uri = 'redirect_uri_example' # str | 

    try:
        api_response = api_instance.exchange_authorization_code(client_id, client_secret, code, grant_type, redirect_uri)
        print("The response of DefaultApi->exchange_authorization_code:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->exchange_authorization_code: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **client_id** | **str**|  | 
 **client_secret** | **str**|  | 
 **code** | **str**|  | 
 **grant_type** | **str**|  | 
 **redirect_uri** | **str**|  | 

### Return type

[**ExchangeAuthorizationCode200Response**](ExchangeAuthorizationCode200Response.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: application/x-www-form-urlencoded
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Access token and refresh token returned |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_alerts**
> AlertRecord get_alerts(start_date, end_date)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.alert_record import AlertRecord
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    start_date = '2013-10-20T19:20:30+01:00' # datetime | 
    end_date = '2013-10-20T19:20:30+01:00' # datetime | 

    try:
        api_response = api_instance.get_alerts(start_date, end_date)
        print("The response of DefaultApi->get_alerts:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_alerts: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **start_date** | **datetime**|  | 
 **end_date** | **datetime**|  | 

### Return type

[**AlertRecord**](AlertRecord.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Ok |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_calibrations**
> CalibrationRecord get_calibrations(start_date, end_date)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.calibration_record import CalibrationRecord
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    start_date = '2013-10-20T19:20:30+01:00' # datetime | 
    end_date = '2013-10-20T19:20:30+01:00' # datetime | 

    try:
        api_response = api_instance.get_calibrations(start_date, end_date)
        print("The response of DefaultApi->get_calibrations:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_calibrations: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **start_date** | **datetime**|  | 
 **end_date** | **datetime**|  | 

### Return type

[**CalibrationRecord**](CalibrationRecord.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Ok |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_data_range**
> GetDataRange200Response get_data_range(last_sync_time=last_sync_time)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.get_data_range200_response import GetDataRange200Response
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    last_sync_time = '2013-10-20T19:20:30+01:00' # datetime |  (optional)

    try:
        api_response = api_instance.get_data_range(last_sync_time=last_sync_time)
        print("The response of DefaultApi->get_data_range:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_data_range: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **last_sync_time** | **datetime**|  | [optional] 

### Return type

[**GetDataRange200Response**](GetDataRange200Response.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Data range retrieved successfully |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_devices_v3**
> List[GetDevicesV3200ResponseInner] get_devices_v3()

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.get_devices_v3200_response_inner import GetDevicesV3200ResponseInner
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)

    try:
        api_response = api_instance.get_devices_v3()
        print("The response of DefaultApi->get_devices_v3:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_devices_v3: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

[**List[GetDevicesV3200ResponseInner]**](GetDevicesV3200ResponseInner.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Ok |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_estimated_glucose_values**
> EGVRecord get_estimated_glucose_values(start_date, end_date)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.egv_record import EGVRecord
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    start_date = '2013-10-20T19:20:30+01:00' # datetime | 
    end_date = '2013-10-20T19:20:30+01:00' # datetime | 

    try:
        api_response = api_instance.get_estimated_glucose_values(start_date, end_date)
        print("The response of DefaultApi->get_estimated_glucose_values:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_estimated_glucose_values: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **start_date** | **datetime**|  | 
 **end_date** | **datetime**|  | 

### Return type

[**EGVRecord**](EGVRecord.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Ok |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_events_v3**
> List[GetEventsV3200ResponseInner] get_events_v3(start_date, end_date)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.models.get_events_v3200_response_inner import GetEventsV3200ResponseInner
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    start_date = '2013-10-20T19:20:30+01:00' # datetime | 
    end_date = '2013-10-20T19:20:30+01:00' # datetime | 

    try:
        api_response = api_instance.get_events_v3(start_date, end_date)
        print("The response of DefaultApi->get_events_v3:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_events_v3: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **start_date** | **datetime**|  | 
 **end_date** | **datetime**|  | 

### Return type

[**List[GetEventsV3200ResponseInner]**](GetEventsV3200ResponseInner.md)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Ok |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **user_authorization**
> user_authorization(client_id, redirect_uri, response_type, scope, state=state)

### Example

* Bearer (JWT) Authentication (BearerAuth):

```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.dexcom.com
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "https://api.dexcom.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (JWT): BearerAuth
configuration = openapi_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    client_id = 'client_id_example' # str | 
    redirect_uri = 'redirect_uri_example' # str | 
    response_type = 'response_type_example' # str | 
    scope = 'scope_example' # str | 
    state = 'state_example' # str |  (optional)

    try:
        api_instance.user_authorization(client_id, redirect_uri, response_type, scope, state=state)
    except Exception as e:
        print("Exception when calling DefaultApi->user_authorization: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **client_id** | **str**|  | 
 **redirect_uri** | **str**|  | 
 **response_type** | **str**|  | 
 **scope** | **str**|  | 
 **state** | **str**|  | [optional] 

### Return type

void (empty response body)

### Authorization

[BearerAuth](../README.md#BearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**302** | Redirect to authorization code or access denied error |  * Location -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

