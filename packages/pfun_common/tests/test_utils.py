"""Tests for pfun_common.utils module."""

import pytest

from pfun_common.utils import add_url_params, append_root_path


class TestAddUrlParams:
    """Test cases for add_url_params function."""

    def test_add_single_param_to_url_without_params(self):
        """Test adding a single parameter to a URL without existing params."""
        url = "https://example.com/test"
        params = {"key": "value"}
        result = add_url_params(url, params)
        assert "key=value" in result
        assert "https://example.com/test" in result

    def test_add_multiple_params_to_url(self):
        """Test adding multiple parameters to a URL."""
        url = "https://example.com/test"
        params = {"key1": "value1", "key2": "value2"}
        result = add_url_params(url, params)
        assert "key1=value1" in result
        assert "key2=value2" in result

    def test_add_params_to_url_with_existing_params(self):
        """Test adding parameters to a URL that already has query string."""
        url = "https://example.com/test?existing=true"
        params = {"new": "param"}
        result = add_url_params(url, params)
        assert "existing=true" in result or "existing%3Dtrue" in result
        assert "new=param" in result

    def test_add_boolean_param(self):
        """Test adding a boolean parameter to a URL."""
        url = "https://example.com/test"
        params = {"enabled": True}
        result = add_url_params(url, params)
        assert "enabled" in result

    def test_add_dict_param(self):
        """Test adding a dict parameter to a URL."""
        url = "https://example.com/test"
        params = {"config": {"nested": "value"}}
        result = add_url_params(url, params)
        assert "config" in result


class TestAppendRootPath:
    """Test cases for append_root_path function."""

    def test_append_root_path_returns_list(self):
        """Test that append_root_path returns a list."""
        result = append_root_path()
        assert isinstance(result, list)

    def test_append_root_path_contains_paths(self):
        """Test that append_root_path result contains path elements."""
        result = append_root_path()
        assert len(result) > 0
        assert all(isinstance(p, str) for p in result)
