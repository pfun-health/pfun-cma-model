"""Tests for pfun_common.enums module."""

import pytest

from pfun_common.enums import StringEnum


class SampleStringEnum(StringEnum):
    """Sample enum for testing StringEnum functionality."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class TestStringEnumClass:
    """Test cases for StringEnum class."""

    def test_getitem_with_key(self):
        """Test __getitem__ returns the attribute value."""
        result = SampleStringEnum["active"]
        # __getitem__ calls getattr(cls, "ACTIVE") which returns the value
        assert result == "active"

    def test_getitem_with_uppercase_key(self):
        """Test __getitem__ converts key to uppercase."""
        result = SampleStringEnum["ACTIVE"]
        # __getitem__ converts to uppercase then calls getattr
        assert result == "active"

    def test_getitem_multiple_values(self):
        """Test __getitem__ with multiple enum values."""
        assert SampleStringEnum["inactive"] == "inactive"
        assert SampleStringEnum["pending"] == "pending"

    def test_getitem_missing_key_raises_attribute_error(self):
        """Test __getitem__ with missing key raises AttributeError."""
        with pytest.raises(AttributeError):
            SampleStringEnum["nonexistent"]
