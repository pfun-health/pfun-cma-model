import logging
from typing import Any, Mapping
from pfun_common.settings import get_settings
from pfun_common.utils import setup_logging
from pfun_cma_model.app import app


def run_app(host: str = "0.0.0.0", port: int = 8001, **kwargs: Any):
    """Run the FastAPI application."""
    import uvicorn
    debug_mode: bool = kwargs.get("debug", get_settings().debug)
    logger = setup_logging(debug=debug_mode)
    # remove unwanted kwargs
    valid_kwargs: Mapping[str, Any] = getattr(
        uvicorn.run, "__kwdefaults__", {}
    )  # ensure a mapping
    for key in list(kwargs.keys()):
        if key in ["extra_args"]:  # handle extra arguments
            logger.debug("(passed to extra_args), %s", str(kwargs.get(key)))
        if key not in valid_kwargs:
            logger.warning(
                f"Unrecognized keyword argument '{key}' for uvicorn.run(). Ignoring it."
            )
            del kwargs[key]
    logger.debug(f"Running FastAPI app on {host}:{port} with kwargs: {kwargs}")
    # must pass the app parameter as a module path to enable hot-reloading
    kwargs.pop("host", None)  # avoid duplicate host/port arguments
    kwargs.pop("port", None)
    if kwargs.get("reload", False):
        # with hot-reloading
        logging.debug("Running with hot-reloading enabled.")
        # remove reload from kwargs to avoid passing it twice
        reload = kwargs.pop("reload", False)
        uvicorn.run(
            "pfun_cma_model.app:app", host=host, port=port, reload=reload, **kwargs
        )
    else:
        # without hot-reloading
        logging.debug("Running without hot-reloading.")
        uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    run_app()
