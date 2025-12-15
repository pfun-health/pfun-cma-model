"""
PFun CMA Model - Parameters API Routes
"""
from typing import Literal
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pfun_cma_model.engine.cma_model_params import CMAModelParams
import json
from typing import Mapping, Any

router = APIRouter()


@router.get("/schema")
def params_schema():
    """Get the JSON schema for the model parameters."""
    params = CMAModelParams()
    return Response(
        content=json.dumps(params.model_json_schema()),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@router.get("/default")
def default_params():
    """Get the default model parameters."""
    params = CMAModelParams()
    return Response(
        content=params.model_dump_json(),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@router.post("/describe")
def describe_params(
    params: CMAModelParams | Mapping[str, Any]
):
    """
    Describe a given (single) or set of parameters using CMAModelParams.describe and generate_qualitative_descriptor.
    Args:
        params (CMAModelParams | Mapping[str, Any]): The configuration parameters to describe.
    Returns:
        dict: Dictionary of parameter descriptions and qualitative descriptors.
    """
    if not isinstance(params, CMAModelParams):
        params = CMAModelParams(**params)  # type: ignore

    bounded_keys = list(params.bounded_param_keys)
    result = {}
    for key in bounded_keys:
        try:
            desc = params.describe(key)
            qual = params.generate_qualitative_descriptor(key)
            result[key] = {
                "description": desc,
                "qualitative": qual,
                "value": getattr(params, key, None)
            }
        except Exception as e:
            result[key] = {"error": str(e)}
    return Response(
        content=json.dumps(result),
        status_code=200,
        headers={"Content-Type": "application/json"},
    )


@router.post("/tabulate/{output_fmt}")
def tabulate_params(
    output_fmt: Literal["json", "html", "md"],
    params: CMAModelParams | Mapping[str, Any]
):
    """Generate a markdown table of a given (single) or set of parameters."""
    #: enforce CMAModelParams type
    if not isinstance(params, CMAModelParams):
        params = CMAModelParams(**params)  # type: ignore
    #: generate table
    table = params.generate_markdown_table(output_fmt=output_fmt)
    #: return in requested format
    match output_fmt:
        case "md":
            return Response(
                content=table,
                status_code=200,
                headers={"Content-Type": "text/markdown"},
            )
        case "html":
            return HTMLResponse(
                content=table,
                status_code=200,
                headers={"Content-Type": "text/html"},
            )
        case "json":
            return JSONResponse(
                content=table,
                status_code=200,
                headers={"Content-Type": "application/json"},
            )
