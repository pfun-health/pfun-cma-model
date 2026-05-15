"""Shared pytest configuration and fixtures for pfun-common tests."""

import pytest


@pytest.fixture
def sample_url():
    """Fixture providing a sample URL for testing."""
    return "https://example.com/api/test"


@pytest.fixture
def sample_params():
    """Fixture providing sample parameters for URL manipulation."""
    return {"key": "value", "page": "1"}
