"""
PFun CMA Model - Demo API Routes
"""
import os
import logging
from datetime import datetime
from fastapi import APIRouter, Request
from pfun_cma_model.engine.cma_model_params import CMAModelParams
from pfun_cma_model.misc.templating import templates

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/dexcom")
def demo_dexcom(request: Request):
    return templates.TemplateResponse("dexcom-demo.html", {
        "request": request,
        "year": datetime.now().year
    })


@router.get("/data-stream")
def demo_data_stream(request: Request):
    return templates.TemplateResponse("data-stream-demo.html", {
        "request": request,
        "year": datetime.now().year
    })


@router.get("/run-at-time")
async def demo_run_at_time(request: Request):
    """Demo UI endpoint to run the model at a specific time (using websockets)."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS, _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS, _MID_DEFAULTS, _UB_DEFAULTS
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
                "default": _MID_DEFAULTS[ix]
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
            }
        }
    }
    logger.debug("Demo context: %s", context_dict)
    return templates.TemplateResponse(
        "run-at-time-demo.html", context=context_dict, headers={"Content-Type": "text/html"})


@router.get("/webgl-demo")
async def demo_webgl(request: Request):
    """Demo UI endpoint for the WebGL-based real-time plot."""
    # load default bounded parameters
    cma_params = CMAModelParams()
    from pfun_cma_model.engine.cma_model_params import (
        _BOUNDED_PARAM_DESCRIPTIONS, _BOUNDED_PARAM_KEYS_DEFAULTS,
        _LB_DEFAULTS, _MID_DEFAULTS, _UB_DEFAULTS
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
                "default": _MID_DEFAULTS[ix]
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
            }
        }
    }
    logger.debug("WebGL Demo context: %s", context_dict)
    return templates.TemplateResponse(
        "webgl-demo.html", context=context_dict, headers={"Content-Type": "text/html"})
