from typing import Annotated, Any
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
import numpy as np


def numpy_array_schema(handler: GetCoreSchemaHandler) -> CoreSchema:
    """
    Custom schema for serialization of numpy arrays in pydantic.
    This schema allows for the serialization of numpy arrays to lists,
    while also accepting lists as input.
    """
    return core_schema.json_or_python_schema(
        json_schema=core_schema.list_schema(),
        python_schema=core_schema.union_schema([
            core_schema.is_instance_schema(np.ndarray),
            core_schema.list_schema(),
        ]),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        ),
    )


NumpyArray = Annotated[np.ndarray, numpy_array_schema]


def bounded_cma_model_param_schema(handler: GetCoreSchemaHandler) -> CoreSchema:
    return core_schema.json_or_python_schema(
        json_schema=core_schema.float_schema(),
        python_schema=core_schema.union_schema([
            core_schema.is_instance_schema(np.ndarray),
            core_schema.float_schema(),
        ]),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: x.__json__()
        ),
    )


BoundedModelParamAnnotation = Annotated[Any, bounded_cma_model_param_schema]
