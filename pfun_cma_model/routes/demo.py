"""
PFun CMA Model - Demo API Routes
"""

import logging
import os
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.misc.templating import get_templates

router = APIRouter()
logger = logging.getLogger(__name__)


class PFunDemoRoutesContext(BaseModel):
    """Defines the context to include for rendering demo routes (jinja2templates)."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )
    #: Model configuration

    request: Request
    #: The current request for the route

    year: int = Field(default_factory=lambda: datetime.now().year)
    #: Current calendar year (YYYY)


@router.get("/llm")
def demo_llm(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    # formulate the render context
    rand0, rand1 = os.urandom(16).hex(), os.urandom(16).hex()
    context_dict = {
        "request": request,
        "params": params,
        "cdn": {
            "bootstrap-css": {
                "hash": "'sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB'",
                "url": f"https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css?dummy={rand0}"
            },
            "inline-script": {
                "hash": "sha256-ZswfTY7H35rbv8WC7NXBoiC7WNu86vSzCDChNWwZZDM="
                "url": None
            },
            "jquery-ui":
                {
                    "url": "https://code.jquery.com/ui/1.14.1/jquery-ui.js"
                }
        },
        "year": datetime.now().year,
    }
    logger.debug("Demo context: %s", str(context_dict))
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    return templates.TemplateResponse("llm-demo.html.jinja2", context=context)


@router.get("/data-stream")
def demo_data_stream(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    context = PFunDemoRoutesContext(request=request).model_dump()
    return templates.TemplateResponse("data-stream-demo.html.jinja2", context=context)


@router.get("/run-at-time")
async def demo_run_at_time(request: Request, templates: Jinja2Templates = Depends(get_templates)):
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
            "chartjs": {"url": f"https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js?dummy={rand0}"},
            "socketio": {"url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"},
        },
        "year": datetime.now().year,
    }
    logger.debug("Demo context: %s", str(context_dict))
    context = PFunDemoRoutesContext(**context_dict)
    context_output = context.model_dump()
    return templates.TemplateResponse(
        "run-at-time-demo.html.jinja2",
        context=context_output,
        headers={"Content-Type": "text/html"},
    )


@router.get("/canvas-wave")
async def demo_canvas_wave(request: Request, templates: Jinja2Templates = Depends(get_templates)):
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
            "socketio": {"url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"},
        },
    }
    logger.debug("Demo context: %s", context_dict)
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    return templates.TemplateResponse(
        "canvas-wave-demo.html.jinja2",
        context=context,
        headers={"Content-Type": "text/html"},
    )


@router.get("/webgl-demo")
async def demo_webgl(request: Request, templates: Jinja2Templates = Depends(get_templates)):
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
            "socketio": {"url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"},
        },
    }
    logger.debug("WebGL Demo context: %s", context_dict)
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    logger.debug("(post-validation) WebGL Demo context: %s", context)
    return templates.TemplateResponse("webgl-demo.html.jinja2", context=context, headers={"Content-Type": "text/html"})


@router.get("/full-model-run")
async def demo_full_model_run(request: Request, templates: Jinja2Templates = Depends(get_templates)):
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
            "chartjs": {"url": f"https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js?dummy={rand0}"},
            "socketio": {"url": f"https://cdn.socket.io/4.7.5/socket.io.min.js?dummy={rand1}"},
        },
    }
    logger.debug("Demo context: %s", context_dict)
    context = PFunDemoRoutesContext(**context_dict).model_dump()
    logger.debug("(post-validation) Demo context: %s", context)
    return templates.TemplateResponse(
        "full-model-run-demo.html.jinja2",
        context=context,
        headers={"Content-Type": "text/html"},
    )
