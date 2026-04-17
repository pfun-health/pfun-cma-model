"""Unit tests for QtSecureTokenStore.

Tests cover:
  - Store/load round-trip with encryption verification
  - Thread-safe concurrent access
  - Error handling (corrupted tokens, I/O failures)
  - Convenience methods (get_access_token, get_refresh_token)
  - Token clearing and existence checks
  - Edge cases (empty tokens, missing tokens, tampered QSettings)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QSettings

from pfun_qt_gui.auth.secure_token_store import QtSecureTokenStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_key_path(tmp_path: Path) -> str:
    """Fixture providing a temporary key path."""
    return str(tmp_path / "test_key")


@pytest.fixture
def temp_settings(tmp_path: Path) -> QSettings:
    """Fixture providing a temporary QSettings instance."""
    settings_file = tmp_path / "test_settings.ini"
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    settings = QSettings(str(settings_file), QSettings.Format.IniFormat)
    yield settings
    settings.clear()
    settings.sync()


@pytest.fixture
def qt_store(temp_key_path: str, tmp_path: Path) -> QtSecureTokenStore:
    """Fixture providing a QtSecureTokenStore instance with isolated QSettings."""
    # Create an isolated QSettings instance for this test
    settings_file = str(tmp_path / "test_settings.ini")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    settings = QSettings(settings_file, QSettings.Format.IniFormat)

    return QtSecureTokenStore(key_path=temp_key_path, settings=settings)


@pytest.fixture
def sample_tokens() -> tuple[str, str]:
    """Fixture providing sample JWT-like tokens."""
    access_token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJhY2Nlc3MiLCJleHAiOjE2MjM4MjE4MDB9."
        "xW7oV4k1Q2Z9pF3nL8mR5jT6bD7aE8cF9sG0hI1j"
    )
    refresh_token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJyZWZyZXNoIiwiZXhwIjoxNjI0NDI2NjAwfQ."
        "yX8pZ5m2R3a0qF4oL9nS6kU7cE8dF9eG0sH1iJ2k"
    )
    return (access_token, refresh_token)


# ============================================================================
# Basic Store/Load Tests
# ============================================================================


class TestQtSecureTokenStoreBasic:
    """Test basic store/load functionality."""

    def test_store_and_load_roundtrip(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that store followed by load returns original tokens."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        loaded_access, loaded_refresh = qt_store.load_tokens()

        assert loaded_access == access_token
        assert loaded_refresh == refresh_token

    def test_load_nonexistent_tokens_returns_none(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test that load_tokens returns (None, None) when no tokens stored."""
        access_token, refresh_token = qt_store.load_tokens()

        assert access_token is None
        assert refresh_token is None

    def test_tokens_stored_encrypted_in_qsettings(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that tokens are stored encrypted, not plaintext in QSettings."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)

        # Verify tokens are not stored as plaintext
        stored_access = qt_store._settings.value("pfun_auth/access_token")
        stored_refresh = qt_store._settings.value("pfun_auth/refresh_token")

        assert stored_access != access_token
        assert stored_refresh != refresh_token
        assert len(stored_access) > len(access_token)
        assert len(stored_refresh) > len(refresh_token)

    def test_has_tokens_true_after_store(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that has_tokens returns True after storing tokens."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)

        assert qt_store.has_tokens() is True

    def test_has_tokens_false_initially(self, qt_store: QtSecureTokenStore) -> None:
        """Test that has_tokens returns False when no tokens stored."""
        assert qt_store.has_tokens() is False

    def test_has_tokens_false_after_clear(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that has_tokens returns False after clearing tokens."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        qt_store.clear_tokens()

        assert qt_store.has_tokens() is False


# ============================================================================
# Convenience Methods Tests
# ============================================================================


class TestQtSecureTokenStoreConvenienceMethods:
    """Test get_access_token and get_refresh_token convenience methods."""

    def test_get_access_token(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test get_access_token returns access token."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        retrieved = qt_store.get_access_token()

        assert retrieved == access_token

    def test_get_refresh_token(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test get_refresh_token returns refresh token."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        retrieved = qt_store.get_refresh_token()

        assert retrieved == refresh_token

    def test_get_access_token_returns_none_when_missing(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test get_access_token returns None when tokens not stored."""
        assert qt_store.get_access_token() is None

    def test_get_refresh_token_returns_none_when_missing(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test get_refresh_token returns None when tokens not stored."""
        assert qt_store.get_refresh_token() is None


# ============================================================================
# Token Clearing Tests
# ============================================================================


class TestQtSecureTokenStoreClear:
    """Test token clearing functionality."""

    def test_clear_tokens_removes_from_qsettings(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that clear_tokens removes tokens from QSettings."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        qt_store.clear_tokens()

        assert not qt_store._settings.contains("pfun_auth/access_token")
        assert not qt_store._settings.contains("pfun_auth/refresh_token")

    def test_clear_tokens_idempotent(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that clear_tokens can be called multiple times safely."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        qt_store.clear_tokens()
        qt_store.clear_tokens()  # Should not raise

        assert qt_store.has_tokens() is False

    def test_load_returns_none_none_after_clear(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that load_tokens returns (None, None) after clear."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        qt_store.clear_tokens()

        loaded_access, loaded_refresh = qt_store.load_tokens()
        assert loaded_access is None
        assert loaded_refresh is None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestQtSecureTokenStoreErrors:
    """Test error handling and edge cases."""

    def test_store_empty_access_token_raises_error(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test that store_tokens raises ValueError for empty access token."""
        with pytest.raises(ValueError, match="access_token cannot be None"):
            qt_store.store_tokens("", "valid_refresh_token")

    def test_store_none_access_token_raises_error(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test that store_tokens raises ValueError for None access token."""
        with pytest.raises(ValueError, match="access_token cannot be None"):
            qt_store.store_tokens(None, "valid_refresh_token")  # type: ignore

    def test_store_empty_refresh_token_raises_error(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test that store_tokens raises ValueError for empty refresh token."""
        with pytest.raises(ValueError, match="refresh_token cannot be None"):
            qt_store.store_tokens("valid_access_token", "")

    def test_store_none_refresh_token_raises_error(
        self, qt_store: QtSecureTokenStore
    ) -> None:
        """Test that store_tokens raises ValueError for None refresh token."""
        with pytest.raises(ValueError, match="refresh_token cannot be None"):
            qt_store.store_tokens("valid_access_token", None)  # type: ignore

    def test_corrupted_token_raises_error_and_clears(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that corrupted tokens raise ValueError and are cleared."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)

        # Manually corrupt the stored token
        qt_store._settings.setValue("pfun_auth/access_token", "corrupted_data")
        qt_store._settings.sync()

        with pytest.raises(ValueError, match="Failed to decrypt tokens"):
            qt_store.load_tokens()

        # Verify tokens were cleared
        assert qt_store.has_tokens() is False

    def test_tampered_refresh_token_raises_error(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that tampered refresh token is detected."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)

        # Manually tamper with the stored token
        stored = qt_store._settings.value("pfun_auth/refresh_token")
        tampered = stored[:-4] + "xxxx"  # Change last 4 chars
        qt_store._settings.setValue("pfun_auth/refresh_token", tampered)
        qt_store._settings.sync()

        with pytest.raises(ValueError, match="Failed to decrypt tokens"):
            qt_store.load_tokens()

    def test_get_access_token_raises_on_corrupted_token(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that get_access_token raises on corrupted token."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        qt_store._settings.setValue("pfun_auth/access_token", "corrupted")
        qt_store._settings.sync()

        with pytest.raises(ValueError):
            qt_store.get_access_token()

    def test_get_refresh_token_raises_on_corrupted_token(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that get_refresh_token raises on corrupted token."""
        access_token, refresh_token = sample_tokens

        qt_store.store_tokens(access_token, refresh_token)
        qt_store._settings.setValue("pfun_auth/refresh_token", "corrupted")
        qt_store._settings.sync()

        with pytest.raises(ValueError):
            qt_store.get_refresh_token()


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestQtSecureTokenStoreThreadSafety:
    """Test thread safety with concurrent operations."""

    def test_mutex_protects_concurrent_store(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that QMutex protects concurrent store operations."""
        import threading

        access_token, refresh_token = sample_tokens
        results = []

        def store_and_load() -> None:
            try:
                qt_store.store_tokens(access_token, refresh_token)
                loaded = qt_store.load_tokens()
                results.append(loaded)
            except Exception as e:
                results.append(e)

        threads = [threading.Thread(target=store_and_load) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 5
        assert all(r == (access_token, refresh_token) for r in results)

    def test_mutex_protects_concurrent_load(
        self, qt_store: QtSecureTokenStore, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that QMutex protects concurrent load operations."""
        import threading

        access_token, refresh_token = sample_tokens
        qt_store.store_tokens(access_token, refresh_token)

        results = []

        def load_tokens() -> None:
            try:
                loaded = qt_store.load_tokens()
                results.append(loaded)
            except Exception as e:
                results.append(e)

        threads = [threading.Thread(target=load_tokens) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 5
        assert all(r == (access_token, refresh_token) for r in results)


# ============================================================================
# Edge Cases and Special Tokens
# ============================================================================


class TestQtSecureTokenStoreEdgeCases:
    """Test edge cases and special token formats."""

    def test_very_long_tokens(self, qt_store: QtSecureTokenStore) -> None:
        """Test encryption of very long tokens."""
        long_access = "x" * 10000
        long_refresh = "y" * 10000

        qt_store.store_tokens(long_access, long_refresh)
        loaded_access, loaded_refresh = qt_store.load_tokens()

        assert loaded_access == long_access
        assert loaded_refresh == long_refresh

    def test_tokens_with_special_characters(self, qt_store: QtSecureTokenStore) -> None:
        """Test encryption of tokens with special characters."""
        special_access = "token!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        special_refresh = "refresh!@#$%^&*()_+-=[]{}|;:',.<>?/~`"

        qt_store.store_tokens(special_access, special_refresh)
        loaded_access, loaded_refresh = qt_store.load_tokens()

        assert loaded_access == special_access
        assert loaded_refresh == special_refresh

    def test_tokens_with_unicode_characters(self, qt_store: QtSecureTokenStore) -> None:
        """Test encryption of tokens with unicode characters."""
        unicode_access = "token_with_émojis_🔐🎉"
        unicode_refresh = "refresh_with_émojis_🔐🎉"

        qt_store.store_tokens(unicode_access, unicode_refresh)
        loaded_access, loaded_refresh = qt_store.load_tokens()

        assert loaded_access == unicode_access
        assert loaded_refresh == unicode_refresh


# ============================================================================
# Storage Key Prefix Tests
# ============================================================================


class TestQtSecureTokenStoreCustomPrefix:
    """Test custom storage key prefixes."""

    def test_custom_storage_key_prefix(self, temp_key_path: str) -> None:
        """Test that custom storage_key_prefix is used correctly."""
        custom_prefix = "my_custom_auth"
        store = QtSecureTokenStore(
            storage_key_prefix=custom_prefix, key_path=temp_key_path
        )

        access_token = "access_123"
        refresh_token = "refresh_456"

        store.store_tokens(access_token, refresh_token)

        # Verify keys in QSettings use custom prefix
        assert store._settings.contains(f"{custom_prefix}/access_token")
        assert store._settings.contains(f"{custom_prefix}/refresh_token")

    def test_multiple_stores_with_different_prefixes(self, temp_key_path: str) -> None:
        """Test that multiple stores with different prefixes are isolated."""
        store1 = QtSecureTokenStore(storage_key_prefix="auth1", key_path=temp_key_path)
        store2 = QtSecureTokenStore(storage_key_prefix="auth2", key_path=temp_key_path)

        access1 = "access_token_1"
        refresh1 = "refresh_token_1"
        access2 = "access_token_2"
        refresh2 = "refresh_token_2"

        store1.store_tokens(access1, refresh1)
        store2.store_tokens(access2, refresh2)

        loaded1 = store1.load_tokens()
        loaded2 = store2.load_tokens()

        assert loaded1 == (access1, refresh1)
        assert loaded2 == (access2, refresh2)


# ============================================================================
# Key Management Tests
# ============================================================================


class TestQtSecureTokenStoreKeyManagement:
    """Test encryption key initialization and management."""

    def test_initializes_with_backend_store(self, temp_key_path: str) -> None:
        """Test that QtSecureTokenStore initializes backend SecureTokenStore."""
        store = QtSecureTokenStore(key_path=temp_key_path)

        assert store._backend_store is not None
        assert hasattr(store._backend_store, "encrypt")
        assert hasattr(store._backend_store, "decrypt")

    def test_same_key_path_preserves_encryption_key(self, temp_key_path: str) -> None:
        """Test that same key path preserves encryption across instances."""
        store1 = QtSecureTokenStore(key_path=temp_key_path)
        access_token = "test_access_token"
        refresh_token = "test_refresh_token"

        store1.store_tokens(access_token, refresh_token)

        # Create new store with same key path
        store2 = QtSecureTokenStore(key_path=temp_key_path)
        loaded_access, loaded_refresh = store2.load_tokens()

        assert loaded_access == access_token
        assert loaded_refresh == refresh_token

    def test_different_key_paths_cannot_decrypt_tokens(self, tmp_path: Path) -> None:
        """Test that different key paths produce incompatible encryption."""
        key_path1 = str(tmp_path / "key1")
        key_path2 = str(tmp_path / "key2")

        store1 = QtSecureTokenStore(key_path=key_path1)
        access_token = "test_access_token"
        refresh_token = "test_refresh_token"

        store1.store_tokens(access_token, refresh_token)

        # Create store with different key and try to load
        store2 = QtSecureTokenStore(key_path=key_path2)

        with pytest.raises(ValueError, match="Failed to decrypt tokens"):
            store2.load_tokens()


# ============================================================================
# QSettings Persistence Tests
# ============================================================================


class TestQtSecureTokenStoreQSettingsPersistence:
    """Test that tokens persist correctly in QSettings."""

    def test_tokens_persist_in_qsettings_file(
        self, tmp_path: Path, sample_tokens: tuple[str, str]
    ) -> None:
        """Test that tokens persist in QSettings INI file."""
        key_path = str(tmp_path / "key")
        store = QtSecureTokenStore(key_path=key_path)

        access_token, refresh_token = sample_tokens
        store.store_tokens(access_token, refresh_token)

        # Verify tokens are in QSettings
        assert store.has_tokens()

        # Create new store instance (should read from same QSettings)
        store2 = QtSecureTokenStore(key_path=key_path)
        loaded_access, loaded_refresh = store2.load_tokens()

        assert loaded_access == access_token
        assert loaded_refresh == refresh_token
