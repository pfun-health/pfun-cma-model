"""
PFun CMA Model - Templating utilities
"""
import os
import logging
from pathlib import Path
from typing import Any
from jinja2 import pass_context
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)


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
    debug_mode: bool = os.getenv("DEBUG", "0") in ["1", "true"]
    templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

    templates.env.globals["https_url_for"] = https_url_for
    # only use the default url_for for local development, for dev, qa, and prod use https
    if not debug_mode:
        templates.env.globals["url_for"] = https_url_for
        logger.debug("Using HTTPS for url_for in templates.")
    else:
        logger.debug("Using HTTP for url_for in templates.")
    return templates


templates = get_templates()