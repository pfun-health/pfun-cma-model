"""
PFun CMA Model - Demo API Routes
"""

import logging
import os
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from pfun_common.settings import Settings, get_settings
from starlette.responses import HTMLResponse

from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.misc.templating import get_templates

router = APIRouter()
logger = logging.getLogger(__name__)


class PFunDemoRoutesContext(BaseModel):
    """Defines the context to include for rendering demo routes (jinja2templates)."""

    model_config = ConfigDict(
        extra='allow',
        arbitrary_types_allowed=True,
    )
    #: Model configuration

    request: Request
    #: The current request for the route

    year: int = Field(default_factory=lambda: datetime.now().year)
    #: Current calendar year (YYYY)


@router.get("/gradio")
def demo_gradio(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    settings: Settings = Depends(get_settings),
):
    """Demo UI endpoint to embed the Gradio interface via an iframe."""
    gradio_url_path = "/gradio/gradio/" if not settings.debug else "/"
    gradio_server_port = f":{settings.gradio_server_port}" if settings.debug else ""
    gradio_url = (
        settings.gradio_server_scheme
        + "://"
        + settings.gradio_server_host
        + gradio_server_port
        + gradio_url_path
    )
    logger.debug("Gradio URL: %s", gradio_url)
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>PFun CMA Model Gradio Demo</title>
            </head>
            <body>
                <h1>Gradio Demo</h1>
                <iframe src="{gradio_url}" width="100%" height="800px" style="border:none;"></iframe>
            </body>
        </html>
        """,
        status_code=200,
    )


@router.get("/dexcom")
def demo_dexcom(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    context = PFunDemoRoutesContext(request=request).model_dump()
    return templates.TemplateResponse(
        "dexcom-demo.html", context=context
    )


@router.get("/data-stream")
def demo_data_stream(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    context = PFunDemoRoutesContext(request=request).model_dump()
    return templates.TemplateResponse(
        "data-stream-demo.html", context=context
    )


@router.get("/run-at-time")
async def demo_run_at_time(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint to run the model at a specific time (using websockets)."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS,
        _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS,
        _MID_DEFAULTS,
        _UB_DEFAULTS,
    )

    default_config = dict(cma_params.bounded_params_dict)
    # formatted parameters to appear in the rendered template
    params = {}
    for ix, pk in enumerate(default_config):
        if pk in default_config:
            params[pk] = {
                "name": _BOUNDED_PARAM_KEYS_DEFAULTS[ix],
                "value": default_config[pk],
                "description": _BOUNDED_PARAM_DESCRIPTIONS[ix],
                "min": _LB_DEFAULTS[ix],
                "max": _UB_DEFAULTS[ix],
                "step": (_UB_DEFAULTS[ix] + _LB_DEFAULTS[ix]) * 0.0125,
                "default": _MID_DEFAULTS[ix],
            }
    # formulate the render context
    rand0, rand1 = os.urandom(16).hex(), os.urandom(16).hex()
    context_dict = {
        "request": request,
        "params": params,
        "cdn": {
            "chartjs": {
                "url": f"https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js?dummy={rand0}"
            },
            "socketio": {
                "url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"
            },
        },
        "year": datetime.now().year
    }
    logger.debug("Demo context: %s", str(context_dict))
    context = PFunDemoRoutesContext(**context_dict)
    context = context.model_dump()
    return templates.TemplateResponse(
        "run-at-time-demo.html",
        context=context,
        headers={"Content-Type": "text/html"},
    )


@router.get("/canvas-wave")
async def demo_canvas_wave(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint for canvas wave demo (using websockets)."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS,
        _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS,
        _MID_DEFAULTS,
        _UB_DEFAULTS,
    )

    default_config = dict(cma_params.bounded_params_dict)
    # formatted parameters to appear in the rendered template
    params = {}
    for ix, pk in enumerate(default_config):
        if pk in default_config:
            params[pk] = {
                "name": _BOUNDED_PARAM_KEYS_DEFAULTS[ix],
                "value": default_config[pk],
                "description": _BOUNDED_PARAM_DESCRIPTIONS[ix],
                "min": _LB_DEFAULTS[ix],
                "max": _UB_DEFAULTS[ix],
                "step": (_UB_DEFAULTS[ix] + _LB_DEFAULTS[ix]) * 0.0125,
                "default": _MID_DEFAULTS[ix],
            }
    # formulate the render context
    rand1 = os.urandom(16).hex()
    context_dict = {
        "request": request,
        "params": params,
        "cdn": {
            "socketio": {
                "url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"
            },
        },
    }
    logger.debug("Demo context: %s", context_dict)
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    return templates.TemplateResponse(
        "canvas-wave-demo.html",
        context=context,
        headers={"Content-Type": "text/html"},
    )


@router.get("/webgl-demo")
async def demo_webgl(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint for the WebGL-based real-time plot."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS,
        _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS,
        _MID_DEFAULTS,
        _UB_DEFAULTS,
    )

    default_config = dict(cma_params.bounded_params_dict)
    # formatted parameters to appear in the rendered template
    params = {}
    for ix, pk in enumerate(default_config):
        if pk in default_config:
            params[pk] = {
                "name": _BOUNDED_PARAM_KEYS_DEFAULTS[ix],
                "value": default_config[pk],
                "description": _BOUNDED_PARAM_DESCRIPTIONS[ix],
                "min": _LB_DEFAULTS[ix],
                "max": _UB_DEFAULTS[ix],
                "default": _MID_DEFAULTS[ix],
            }
    # formulate the render context
    rand0, rand1 = os.urandom(16).hex(), os.urandom(16).hex()
    context_dict = {
        "request": request,
        "params": params,
        "cdn": {
            "webglplot": {
                "url": f"https://cdn.jsdelivr.net/gh/danchitnis/webgl-plot@master/dist/webglplot.umd.min.js?dummy={rand0}"
            },
            "socketio": {
                "url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"
            },
        },
    }
    logger.debug("WebGL Demo context: %s", context_dict)
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    logger.debug("(post-validation) WebGL Demo context: %s", context)
    return templates.TemplateResponse(
        "webgl-demo.html", context=context, headers={"Content-Type": "text/html"}
    )


@router.get("/full-model-run")
async def demo_full_model_run(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint to run the full model (c, m, a) at a specific time (using websockets)."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS,
        _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS,
        _MID_DEFAULTS,
        _UB_DEFAULTS,
    )

    default_config = dict(cma_params.bounded_params_dict)
    # formatted parameters to appear in the rendered template
    params = {}
    for ix, pk in enumerate(default_config):
        if pk in default_config:
            params[pk] = {
                "name": _BOUNDED_PARAM_KEYS_DEFAULTS[ix],
                "value": default_config[pk],
                "description": _BOUNDED_PARAM_DESCRIPTIONS[ix],
                "min": _LB_DEFAULTS[ix],
                "max": _UB_DEFAULTS[ix],
                "step": (_UB_DEFAULTS[ix] + _LB_DEFAULTS[ix]) * 0.0125,
                "default": _MID_DEFAULTS[ix],
            }
    # formulate the render context
    rand0, rand1 = os.urandom(16).hex(), os.urandom(16).hex()
    context_dict = {
        "request": request,
        "params": params,
        "cdn": {
            "chartjs": {
                "url": f"https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js?dummy={rand0}"
            },
            "socketio": {
                "url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"
            },
        },
    }
    logger.debug("Demo context: %s", context_dict)
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    logger.debug("(post-validation) Demo context: %s", context)
    return templates.TemplateResponse(
        "full-model-run-demo.html",
        context=context,
        headers={"Content-Type": "text/html"},
    )
