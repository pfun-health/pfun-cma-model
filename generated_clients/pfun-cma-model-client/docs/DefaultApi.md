# openapi_client.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**default_params_params_default_get**](DefaultApi.md#default_params_params_default_get) | **GET** /params/default | Default Params
[**fit_model_to_data_fit_post**](DefaultApi.md#fit_model_to_data_fit_post) | **POST** /fit | Fit Model To Data
[**get_sample_dataset_data_sample_get**](DefaultApi.md#get_sample_dataset_data_sample_get) | **GET** /data/sample | Get Sample Dataset
[**params_schema_params_schema_get**](DefaultApi.md#params_schema_params_schema_get) | **GET** /params/schema | Params Schema
[**root_get**](DefaultApi.md#root_get) | **GET** / | Root
[**run_at_time_route_run_at_time_post**](DefaultApi.md#run_at_time_route_run_at_time_post) | **POST** /run-at-time | Run At Time Route
[**run_model_run_post**](DefaultApi.md#run_model_run_post) | **POST** /run | Run Model
[**translate_model_results_by_language_translate_results_post**](DefaultApi.md#translate_model_results_by_language_translate_results_post) | **POST** /translate-results | Translate Model Results By Language


# **default_params_params_default_get**
> object default_params_params_default_get()

Default Params

### Example


```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)

    try:
        # Default Params
        api_response = api_instance.default_params_params_default_get()
        print("The response of DefaultApi->default_params_params_default_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->default_params_params_default_get: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **fit_model_to_data_fit_post**
> object fit_model_to_data_fit_post(body_fit_model_to_data_fit_post)

Fit Model To Data

### Example


```python
import openapi_client
from openapi_client.models.body_fit_model_to_data_fit_post import BodyFitModelToDataFitPost
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    body_fit_model_to_data_fit_post = openapi_client.BodyFitModelToDataFitPost() # BodyFitModelToDataFitPost | 

    try:
        # Fit Model To Data
        api_response = api_instance.fit_model_to_data_fit_post(body_fit_model_to_data_fit_post)
        print("The response of DefaultApi->fit_model_to_data_fit_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->fit_model_to_data_fit_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body_fit_model_to_data_fit_post** | [**BodyFitModelToDataFitPost**](BodyFitModelToDataFitPost.md)|  | 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_sample_dataset_data_sample_get**
> object get_sample_dataset_data_sample_get()

Get Sample Dataset

### Example


```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)

    try:
        # Get Sample Dataset
        api_response = api_instance.get_sample_dataset_data_sample_get()
        print("The response of DefaultApi->get_sample_dataset_data_sample_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_sample_dataset_data_sample_get: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **params_schema_params_schema_get**
> object params_schema_params_schema_get()

Params Schema

### Example


```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)

    try:
        # Params Schema
        api_response = api_instance.params_schema_params_schema_get()
        print("The response of DefaultApi->params_schema_params_schema_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->params_schema_params_schema_get: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **root_get**
> object root_get()

Root

### Example


```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)

    try:
        # Root
        api_response = api_instance.root_get()
        print("The response of DefaultApi->root_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->root_get: %s\n" % e)
```



### Parameters

This endpoint does not need any parameter.

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **run_at_time_route_run_at_time_post**
> object run_at_time_route_run_at_time_post(t0, t1, n, cma_model_params=cma_model_params)

Run At Time Route

### Example


```python
import openapi_client
from openapi_client.models.cma_model_params import CMAModelParams
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    t0 = openapi_client.T0() # T0 | 
    t1 = openapi_client.T1() # T1 | 
    n = 56 # int | 
    cma_model_params = openapi_client.CMAModelParams() # CMAModelParams |  (optional)

    try:
        # Run At Time Route
        api_response = api_instance.run_at_time_route_run_at_time_post(t0, t1, n, cma_model_params=cma_model_params)
        print("The response of DefaultApi->run_at_time_route_run_at_time_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->run_at_time_route_run_at_time_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **t0** | [**T0**](.md)|  | 
 **t1** | [**T1**](.md)|  | 
 **n** | **int**|  | 
 **cma_model_params** | [**CMAModelParams**](CMAModelParams.md)|  | [optional] 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **run_model_run_post**
> object run_model_run_post(cma_model_params=cma_model_params)

Run Model

Runs the CMA model.

### Example


```python
import openapi_client
from openapi_client.models.cma_model_params import CMAModelParams
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    cma_model_params = openapi_client.CMAModelParams() # CMAModelParams |  (optional)

    try:
        # Run Model
        api_response = api_instance.run_model_run_post(cma_model_params=cma_model_params)
        print("The response of DefaultApi->run_model_run_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->run_model_run_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **cma_model_params** | [**CMAModelParams**](CMAModelParams.md)|  | [optional] 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **translate_model_results_by_language_translate_results_post**
> object translate_model_results_by_language_translate_results_post(from_lang, body)

Translate Model Results By Language

### Example


```python
import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = openapi_client.DefaultApi(api_client)
    from_lang = 'from_lang_example' # str | 
    body = None # object | 

    try:
        # Translate Model Results By Language
        api_response = api_instance.translate_model_results_by_language_translate_results_post(from_lang, body)
        print("The response of DefaultApi->translate_model_results_by_language_translate_results_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->translate_model_results_by_language_translate_results_post: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **from_lang** | **str**|  | 
 **body** | **object**|  | 

### Return type

**object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

