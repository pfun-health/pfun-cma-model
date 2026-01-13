"""
PFun CMA Model - Templating utilities
"""
import logging
from pathlib import Path
from typing import Any

from fastapi.templating import Jinja2Templates
from jinja2 import pass_context
import pfun_path_helper as pph  # type: ignore
pph.append_path(Path(__file__).parent.parent)
from pfun_common.settings import get_settings


@pass_context
def https_url_for(context: dict, name: str, **path_params: Any) -> str:
    """Convert http to https.

    ref: https://waylonwalker.com/thoughts-223
    """
    request = context["request"]
    http_url = request.url_for(name, **path_params)
    return str(http_url).replace("http", "https", 1)


def get_templates() -> Jinja2Templates:
    """Get the Jinja2 templates object, include https_url_for filter.

    Returns:
        Jinja2Templates: The Jinja2 templates object.
    """
    debug_mode: bool = get_settings().debug
    templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")
    templates.env.globals["https_url_for"] = https_url_for
    # For DEV, use the default url_for, unless explicitly specified
    # For PROD, use https
    if not debug_mode:
        templates.env.globals["url_for"] = https_url_for
        logging.debug("(not debug mode) Using HTTPS for url_for in templates.")
    elif debug_mode:
        logging.debug("(debug mode) Using HTTP for url_for in templates.")
    return templates
