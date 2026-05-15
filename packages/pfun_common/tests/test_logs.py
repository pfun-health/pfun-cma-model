"""Tests for pfun_common.logs module."""

import logging
import json

import pytest

from pfun_common.logs import (
    JsonFormatter,
    get_log_level,
    get_logger_name,
    init_logging_config,
)


class TestJsonFormatter:
    """Test cases for JsonFormatter class."""

    def test_json_formatter_format_creates_valid_json(self):
        """Test that JsonFormatter.format creates valid JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_formatter_includes_required_fields(self):
        """Test that JsonFormatter includes required fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=20,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "module" in parsed
        assert "line" in parsed
        assert "message" in parsed

    def test_json_formatter_level_name(self):
        """Test that JsonFormatter includes correct level name."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=30,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["level"] == "ERROR"


class TestGetLogLevel:
    """Test cases for get_log_level function."""

    def test_get_log_level_returns_int(self):
        """Test that get_log_level returns an integer."""
        level = get_log_level(debug=False)
        assert isinstance(level, int)

    def test_get_log_level_debug_true_returns_debug(self):
        """Test that debug=True returns logging.DEBUG."""
        level = get_log_level(debug=True)
        assert level == logging.DEBUG

    def test_get_log_level_debug_false_returns_info(self):
        """Test that debug=False returns logging.INFO."""
        level = get_log_level(debug=False)
        assert level == logging.INFO


class TestGetLoggerName:
    """Test cases for get_logger_name function."""

    def test_get_logger_name_returns_string(self):
        """Test that get_logger_name returns a string."""
        name = get_logger_name(logger_name="custom-logger")
        assert isinstance(name, str)

    def test_get_logger_name_custom_value(self):
        """Test that get_logger_name returns the provided name."""
        custom_name = "my-custom-logger"
        name = get_logger_name(logger_name=custom_name)
        assert name == custom_name


class TestInitLoggingConfig:
    """Test cases for init_logging_config function."""

    def test_init_logging_config_returns_dict(self):
        """Test that init_logging_config returns a dictionary."""
        config = init_logging_config(debug=False, logger_name="test-logger")
        assert isinstance(config, dict)

    def test_init_logging_config_has_required_keys(self):
        """Test that logging config has required keys."""
        config = init_logging_config(debug=False, logger_name="test-logger")
        assert "version" in config
        assert "formatters" in config
        assert "handlers" in config
        assert "loggers" in config

    def test_init_logging_config_has_console_handler(self):
        """Test that logging config includes console handler."""
        config = init_logging_config(debug=False, logger_name="test-logger")
        assert "console" in config["handlers"]

    def test_init_logging_config_debug_true(self):
        """Test that logging config respects debug flag."""
        config = init_logging_config(debug=True, logger_name="test-logger")
        console_handler = config["handlers"]["console"]
        assert console_handler["level"] == logging.DEBUG
