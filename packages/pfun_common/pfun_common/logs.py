"""Methods to setup the project-level logging configuration."""

import logging
from logging.config import dictConfig
import json
from datetime import datetime, timezone
from typing import Literal, Annotated
from pfun_common.settings import get_settings

LoggingLevel = Literal[logging.DEBUG, logging.INFO]
#: Type definition for expectd logging levels


class JsonFormatter(logging.Formatter):
    """Custom JSON logging formatter.


    ref: https://betterstack.com/community/guides/logging/logging-with-fastapi/#formatting-your-log-records-as-json
    """

    def format(self, record):
        """Define the JSON formatting for each log entry."""
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def get_log_level(**kwds) -> Annotated[int | str, LoggingLevel]:
    """Get the intended logging level.

    NOTE: The returned value should be either logging.DEBUG or logging.INFO,
      depending on the boolean flag provided (attribute 'settings.debug').
    """
    debug_flag_value: bool = kwds.get("debug", get_settings().debug)
    return logging.DEBUG if debug_flag_value else logging.INFO


def get_logger_name(**kwds) -> str:
    """Retrieve the configured logger name."""
    logger_name = kwds.get("logger_name", get_settings().logger_name)
    return logger_name


def init_logging_config(**kwds) -> dict:
    """Define the logging configuration, applied via logging.dictConfig.

    # References:

    + ref: https://betterstack.com/community/guides/logging/logging-with-fastapi/
    + ref: https://betterstack.com/community/guides/logging/logging-with-fastapi/#formatting-your-log-records-as-json

    """

    # retrieve the desired logging verbosity
    log_level = get_log_level(**kwds)

    # retrieve the project-level logger name
    logger_name = get_logger_name(**kwds)

    # define the logging config
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {"()": JsonFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "time_rotating_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": "logs/pfun-cma-model.log",
                "when": "midnight",
                "interval": 1,
                "backupCount": 3,
            },
        },
        "loggers": {
            f"{logger_name}": {
                "handlers": ["console", "time_rotating_file"],
                "level": log_level,
                "propagate": False,
            },
        },
        "root": {"handlers": ["console"], "level": log_level},
    }

    return log_config


def setup_logging(**kwds) -> logging.Logger:
    """Setup logging for the entire pfun project."""

    # define the logging config (as a dict mapping)
    log_config = init_logging_config(**kwds)

    # Apply the logging configuration
    dictConfig(log_config)

    # Create a logger instance with the applied config
    logger_name = get_logger_name(**kwds)
    logger = logging.getLogger(logger_name)

    return logger
