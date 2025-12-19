import os
from datetime import datetime
from dotenv import load_dotenv
from json import dumps
import sys
from pathlib import Path
import logging

try:
    # Python 2 fallback
    from urllib import urlencode, unquote  # type: ignore
    from urlparse import urlparse, parse_qsl, ParseResult  # type: ignore
except ImportError:
    # Python 3 fallback
    from urllib.parse import (
        urlencode, unquote, urlparse, parse_qsl, ParseResult
    )


def setup_logging(logger: logging.Logger, debug_mode: bool = False):
    """Setup logging configuration."""
    # Set the logger to the desired level
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode is enabled. Setting logger level to DEBUG.")
    else:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode is disabled. Setting logger level to INFO.")
    logger.info("...Logging setup complete.")


def load_environment_variables(
        logger: logging.Logger = logging.getLogger(__name__)
    ) -> tuple[bool, Path]:
    """Load environment variables from .env file."""
    logger.debug("Attempting to load environment variables from .env file...")
    env_file = Path(__file__).parent.parent.parent / ".env"
    logger.debug("Checking for .env file at: %s", str(env_file))
    if not env_file.exists():
        logger.warning(
            "No .env file found at '%s'. Using system environment variables.",
            str(env_file))
        return False, env_file
    logger.debug("...env file exists.")
    loaded = load_dotenv(dotenv_path=env_file)
    if not loaded:
        logger.warning(
            f"Failed to load environment variables from {env_file}.")
    logger.debug(f"Loaded environment variables from {env_file}")
    return loaded, env_file
        


def add_url_params(url, params):
    """ Add GET params to provided URL being aware of existing.

    :param url: string of target URL
    :param params: dict containing requested params to be added
    :return: string with updated URL

    ref: https://stackoverflow.com/a/25580545/1871569

    >> url = 'https://stackoverflow.com/test?answers=true'
    >> new_params = {'answers': False, 'data': ['some','values']}
    >> add_url_params(url, new_params)
    'https://stackoverflow.com/test?data=some&data=values&answers=false'
    """
    # Unquoting URL first so we don't lose existing args
    url = unquote(url)
    # Extracting url info
    parsed_url = urlparse(url)
    # Extracting URL arguments from parsed URL
    get_args = parsed_url.query
    # Converting URL arguments to dict
    parsed_get_args = dict(parse_qsl(get_args))
    # Merging URL arguments dict with new params
    parsed_get_args.update(params)

    # Bool and Dict values should be converted to json-friendly values
    # you may throw this part away if you don't like it :)
    parsed_get_args.update(
        {k: dumps(v) for k, v in parsed_get_args.items()
         if isinstance(v, (bool, dict))}
    )

    # Converting URL argument to proper query string
    encoded_get_args = urlencode(parsed_get_args, doseq=True)
    # Creating new parsed result object based on provided with new
    # URL arguments. Same thing happens inside urlparse.
    new_url = ParseResult(
        parsed_url.scheme, parsed_url.netloc, parsed_url.path,
        parsed_url.params, encoded_get_args, parsed_url.fragment
    ).geturl()

    return new_url


def append_root_path():
    """
    adds the root path to the python path
    """
    #: pfun imports (relative)
    root_path = str(Path(__file__).parents[1])
    mod_path = str(Path(__file__).parent)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    if mod_path not in sys.path:
        sys.path.insert(0, mod_path)
    return sys.path
