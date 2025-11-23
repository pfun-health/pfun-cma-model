from pfun_cma_model.app import app, logger
import logging
from typing import Any, Mapping


def run_app(host: str = "0.0.0.0", port: int = 8001, **kwargs: Any):
    """Run the FastAPI application."""
    import uvicorn
    # remove unwanted kwargs
    valid_kwargs: Mapping[str, Any] = getattr(
        uvicorn.run, "__kwdefaults__", {})  # ensure a mapping
    for key in list(kwargs.keys()):
        if key in ["extra_args"]:  # handle extra arguments
            logger.info("(passed to extra_args), %s", str(kwargs.get(key)))
        if key not in valid_kwargs:
            logger.warning(
                f"Unrecognized keyword argument '{key}' for uvicorn.run(). Ignoring it.")
            del kwargs[key]
    logger.info(f"Running FastAPI app on {host}:{port} with kwargs: {kwargs}")
    # must pass the app parameter as a module path to enable hot-reloading
    kwargs.pop("host", None)  # avoid duplicate host/port arguments
    kwargs.pop("port", None)
    if kwargs.get("reload", False):
        # with hot-reloading
        logging.info("Running with hot-reloading enabled.")
        # remove reload from kwargs to avoid passing it twice
        reload = kwargs.pop("reload", False)
        uvicorn.run("pfun_cma_model.app:app", host=host,
                    port=port, reload=reload, **kwargs)
    else:
        # without hot-reloading
        logging.info("Running without hot-reloading.")
        uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    run_app()
