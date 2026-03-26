"""Defines demo routes for the PFun CMA Model application."""

import logging
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, ValidationInfo, field_validator
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from pfun_cma_model.engine.cma_model_params import _DEFAULTS
from pfun_cma_model.misc.templating import get_templates

router = APIRouter()
logger = logging.getLogger(__name__)


def get_formatted_params() -> dict:
    """Helper function to format the model parameters for rendering in the templates."""
    # formatted parameters to appear in the rendered template
    params = {}
    for ix, pk in enumerate(_DEFAULTS.keys):
        if pk in _DEFAULTS.keys:
            params[pk] = {
                "name": _DEFAULTS.keys[ix],
                "value": _DEFAULTS.mids[ix],
                "description": _DEFAULTS.descriptions[ix],
                "min": _DEFAULTS.lbs[ix],
                "max": _DEFAULTS.ubs[ix],
                "step": _DEFAULTS.steps[ix],
                "default": _DEFAULTS.mids[ix],
            }
    return params


class CDNResource(BaseModel):
    """Defines a CDN resource with its URL and integrity hash (if applicable)."""

    # NOTE: Recall that order of fields matters for validation and formatting (e.g. decache should come before url).
    decache: bool = False
    """Whether to append a dummy query parameter for cache busting. Defaults to False."""
    url: str | None = None
    """The URL to the CDN resource (e.g. JS/CSS library). Can be None for inline resources."""
    hash: str | None = None
    """The integrity hash for the CDN resource (e.g. for Subresource Integrity). Optional."""
    nonce: str | None = None
    """The nonce value for the CDN resource (e.g. for Content Security Policy). Optional."""

    @field_validator("url", mode="after")
    @classmethod
    def url_validator(cls, v, info: ValidationInfo):
        """Validates and formats the CDN URL, including cache busting if decache is True."""
        return cls.final_format_url(v, info.data)

    @classmethod
    def final_format_url(cls, v, values):
        """Final formatting for the CDN URL.

        Including:
        + Append the dummy query parameter to the URL for cache busting.
        + Validate that the URL is well-formed (basic check).
        """
        import re

        decache = values.get("decache", False)
        if v is None:
            return v
        # Basic URL validation
        url_regex = re.compile(r"^(https?://|/)[^\s]+$")
        if not url_regex.match(v):
            raise ValueError("Malformed CDN URL")
        # Append dummy query param if decache is True and not already present
        if decache and "dummy=" not in v:
            sep = "&" if "?" in v else "?"
            from os import urandom

            dummy_val = urandom(8).hex()
            v = f"{v}{sep}dummy={dummy_val}"
        return v


class PFunDemoRoutesContext(BaseModel):
    """Defines the context to include for rendering demo routes (jinja2templates)."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )
    #: Model configuration

    request: Request
    #: FastAPI Request object, included for potential use in URL generation or context.

    year: int = Field(default_factory=lambda: datetime.now().year)
    #: Current calendar year (YYYY)

    params: dict = Field(default_factory=dict)
    #: Optional parameters to include in the context (e.g. for model configuration)

    cdn: dict[str, CDNResource] = Field(default_factory=dict)
    #: Optional CDN URLs and integrity hashes for external resources (e.g. JS/CSS libraries)


@router.get("/llm")
def demo_llm(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    """Demo UI endpoint for LLM interactions."""
    # formulate the render context
    demo_route_context = PFunDemoRoutesContext(
        request=request,
        cdn={
            "bootstrap-css": CDNResource(
                hash="'sha384-sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB'",
                url="https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css?",
                decache=True,
            ),
            "inline-script": CDNResource(
                hash="sha256-ZswfTY7H35rbv8WC7NXBoiC7WNu86vSzCDChNWwZZDM=", url=None
            ),
            "jquery-ui": CDNResource(
                url="https://code.jquery.com/ui/1.14.1/jquery-ui.js",
                decache=True,
            ),
        },
        year=datetime.now().year,
    )
    context = demo_route_context.model_dump()
    logger.debug("Demo context: %s", str(context))
    return templates.TemplateResponse(request, "llm-demo.html.jinja2", context=context)


@router.get("/data-stream")
def demo_data_stream(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint for data stream interactions."""
    demo_route_context = PFunDemoRoutesContext(request=request)
    context = demo_route_context.model_dump()
    return templates.TemplateResponse(
        request, "data-stream-demo.html.jinja2", context=context
    )


@router.get("/run-at-time")
async def demo_run_at_time(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint to run the model at a specific time (using websockets)."""
    # load default bounded parameters, formatted parameters to appear in the rendered template.
    params = get_formatted_params()
    # formulate the render context
    demo_route_context = PFunDemoRoutesContext(
        request=request,
        params=params,
        cdn={
            "chartjs": CDNResource(
                url="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js",
                decache=True,
            ),
            "socketio": CDNResource(
                url="https://cdn.socket.io/4.7.5/socket.io.min.js",
                decache=True,
            ),
        },
    )
    context_output = demo_route_context.model_dump()
    return templates.TemplateResponse(
        request,
        "run-at-time-demo.html.jinja2",
        context=context_output,
        headers={"Content-Type": "text/html"},
    )


@router.get("/canvas-wave")
async def demo_canvas_wave(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint for canvas wave demo (using websockets)."""
    # load default bounded parameters
    params = get_formatted_params()
    # formulate the render context
    demo_route_context = PFunDemoRoutesContext(
        request=request,
        params=params,
        cdn={
            "socketio": CDNResource(
                url="https://cdn.socket.io/4.7.5/socket.io.min.js",
                decache=True,
            ),
        },
    )
    context_dict = demo_route_context.model_dump()
    logger.debug("Demo context: %s", context_dict)
    return templates.TemplateResponse(
        request,
        "canvas-wave-demo.html.jinja2",
        context=context_dict,
        headers={"Content-Type": "text/html"},
    )


@router.get("/full-model-run")
async def demo_full_model_run(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint to run the full model (c, m, a) at a specific time (using websockets)."""
    # load default bounded parameters
    params = get_formatted_params()
    # formulate the render context
    demo_route_context = PFunDemoRoutesContext(
        request=request,
        params=params,
        cdn={
            "chartjs": CDNResource(
                url="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js",
                decache=True,
            ),
            "socketio": CDNResource(
                url="https://cdn.socket.io/4.7.5/socket.io.min.js",
                decache=True,
            ),
        },
    )
    context_dict = demo_route_context.model_dump()
    logger.debug("(post-validation) Demo context: %s", context_dict)
    return templates.TemplateResponse(
        request,
        "full-model-run-demo.html.jinja2",
        context=context_dict,
        headers={"Content-Type": "text/html"},
    )


@router.get("/webgl-demo")
async def demo_webgl(
    request: Request, templates: Jinja2Templates = Depends(get_templates)
):
    """Demo UI endpoint for the WebGL-based real-time plot."""
    # formatted parameters to appear in the rendered template
    params = get_formatted_params()
    # formulate the render context
    demo_route_context = PFunDemoRoutesContext(
        request=request,
        params=params,
        cdn={
            "webglplot": CDNResource(
                url="https://cdn.jsdelivr.net/gh/danchitnis/webgl-plot@master/dist/webglplot.umd.min.js",
                decache=True,
            ),
            "socketio": CDNResource(
                url="https://cdn.socket.io/4.7.5/socket.io.min.js",
                decache=True,
            ),
        },
    )
    context_dict = demo_route_context.model_dump()
    logger.debug("WebGL Demo context: %s", context_dict)
    return templates.TemplateResponse(
        request,
        "webgl-demo.html.jinja2",
        context=context_dict,
        headers={"Content-Type": "text/html"},
    )
