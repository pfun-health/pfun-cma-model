"""Tests for pfun_common.settings module."""

import pytest

from pfun_common.settings import Settings, get_settings, generate_default_secret_key


class TestSettings:
    """Test cases for Settings class."""

    def test_settings_default_debug_is_false(self):
        """Test that default debug setting is False."""
        settings = Settings()
        assert settings.debug is False

    def test_settings_default_logger_name(self):
        """Test that default logger name is set."""
        settings = Settings()
        assert settings.logger_name == "pfun-app"

    def test_settings_default_server_scheme(self):
        """Test that default server scheme is 'http'."""
        settings = Settings()
        assert settings.server_scheme == "http"

    def test_settings_default_server_host(self):
        """Test that default server host is 'localhost'."""
        settings = Settings()
        assert settings.server_host == "localhost"

    def test_settings_can_be_overridden(self):
        """Test that settings can be overridden."""
        settings = Settings(debug=True, logger_name="custom-logger")
        assert settings.debug is True
        assert settings.logger_name == "custom-logger"


class TestGetSettings:
    """Test cases for get_settings function."""

    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_callable(self):
        """Test that get_settings is callable."""
        assert callable(get_settings)


class TestGenerateDefaultSecretKey:
    """Test cases for generate_default_secret_key function."""

    def test_generate_secret_key_returns_string(self):
        """Test that generate_default_secret_key returns a string."""
        key = generate_default_secret_key()
        assert isinstance(key, str)

    def test_generate_secret_key_is_not_empty(self):
        """Test that generated secret key is not empty."""
        key = generate_default_secret_key()
        assert len(key) > 0

    def test_generate_secret_key_contains_separator(self):
        """Test that generated secret key contains the expected separator."""
        key = generate_default_secret_key()
        assert "-" in key

    def test_generate_secret_key_uniqueness(self):
        """Test that multiple generated keys are unique."""
        key1 = generate_default_secret_key()
        key2 = generate_default_secret_key()
        assert key1 != key2
