"""Unit tests for SecureTokenStore encryption/decryption."""

from pathlib import Path

import pytest

from pfun_cma_model.misc.secure_token_store import SecureTokenStore


class TestSecureTokenStoreBasic:
    """Test basic encryption/decryption functionality."""

    def test_encrypt_decrypt_roundtrip(self, temp_key_path: str) -> None:
        """Test that encrypt followed by decrypt returns original token."""
        store = SecureTokenStore(key_path=temp_key_path)
        original_token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )

        encrypted = store.encrypt(original_token)
        decrypted = store.decrypt(encrypted)

        assert decrypted == original_token

    def test_encrypt_produces_ciphertext(self, temp_key_path: str) -> None:
        """Test that encrypt produces different output than input."""
        store = SecureTokenStore(key_path=temp_key_path)
        token = "test_token_12345"

        encrypted = store.encrypt(token)

        assert encrypted != token
        assert len(encrypted) > len(token)
        assert isinstance(encrypted, str)

    def test_decrypt_fails_on_invalid_ciphertext(self, temp_key_path: str) -> None:
        """Test that decrypt raises ValueError for invalid ciphertext."""
        store = SecureTokenStore(key_path=temp_key_path)

        with pytest.raises(ValueError, match="Decryption failed"):
            store.decrypt("invalid_base64_ciphertext")

    def test_decrypt_fails_on_tampered_token(self, temp_key_path: str) -> None:
        """Test that decrypt fails if token is tampered with."""
        store = SecureTokenStore(key_path=temp_key_path)
        token = "original_token"
        encrypted = store.encrypt(token)

        tampered = encrypted[:-4] + "xxxx"

        with pytest.raises(ValueError, match="Decryption failed"):
            store.decrypt(tampered)


class TestSecureTokenStoreEdgeCases:
    """Test edge cases and error handling."""

    def test_encrypt_empty_string_raises_error(self, temp_key_path: str) -> None:
        """Test that encrypt raises ValueError for empty string."""
        store = SecureTokenStore(key_path=temp_key_path)

        with pytest.raises(ValueError, match="Token cannot be None or empty"):
            store.encrypt("")

    def test_encrypt_none_raises_error(self, temp_key_path: str) -> None:
        """Test that encrypt raises ValueError for None."""
        store = SecureTokenStore(key_path=temp_key_path)

        with pytest.raises(ValueError, match="Token cannot be None or empty"):
            store.encrypt(None)  # type: ignore[arg-type]

    def test_decrypt_empty_string_raises_error(self, temp_key_path: str) -> None:
        """Test that decrypt raises ValueError for empty string."""
        store = SecureTokenStore(key_path=temp_key_path)

        with pytest.raises(ValueError, match="Encrypted token cannot be None or empty"):
            store.decrypt("")

    def test_decrypt_none_raises_error(self, temp_key_path: str) -> None:
        """Test that decrypt raises ValueError for None."""
        store = SecureTokenStore(key_path=temp_key_path)

        with pytest.raises(ValueError, match="Encrypted token cannot be None or empty"):
            store.decrypt(None)  # type: ignore[arg-type]

    def test_encrypt_very_long_token(self, temp_key_path: str) -> None:
        """Test encryption of very long token."""
        store = SecureTokenStore(key_path=temp_key_path)
        long_token = "x" * 10000

        encrypted = store.encrypt(long_token)
        decrypted = store.decrypt(encrypted)

        assert decrypted == long_token

    def test_encrypt_special_characters(self, temp_key_path: str) -> None:
        """Test encryption of token with special characters."""
        store = SecureTokenStore(key_path=temp_key_path)
        token_with_special = "token!@#$%^&*()_+-=[]{}|;:',.<>?/~`"

        encrypted = store.encrypt(token_with_special)
        decrypted = store.decrypt(encrypted)

        assert decrypted == token_with_special

    def test_encrypt_unicode_token(self, temp_key_path: str) -> None:
        """Test encryption of token with unicode characters."""
        store = SecureTokenStore(key_path=temp_key_path)
        unicode_token = "token_with_émojis_🔐🎉"

        encrypted = store.encrypt(unicode_token)
        decrypted = store.decrypt(encrypted)

        assert decrypted == unicode_token


class TestSecureTokenStoreKeyManagement:
    """Test key generation and loading."""

    def test_generate_key_creates_valid_fernet_key(self) -> None:
        """Test that generate_key produces valid Fernet key."""
        key = SecureTokenStore.generate_key()

        assert isinstance(key, bytes)
        assert len(key) == 44  # Fernet keys are 44 bytes when base64-encoded

    def test_load_or_create_key_creates_key_file(self, temp_key_path: str) -> None:
        """Test that load_or_create_key creates key file when missing."""
        key_path = Path(temp_key_path)
        assert not key_path.exists()

        key = SecureTokenStore.load_or_create_key(key_path=temp_key_path)

        assert key_path.exists()
        assert len(key) == 44

    def test_load_or_create_key_sets_correct_permissions(
        self, temp_key_path: str
    ) -> None:
        """Test that created key file has 0o600 permissions."""
        key_path = Path(temp_key_path)

        SecureTokenStore.load_or_create_key(key_path=temp_key_path)

        perms = oct(key_path.stat().st_mode)[-3:]
        assert perms == "600"

    def test_load_or_create_key_loads_existing_key(self, temp_key_path: str) -> None:
        """Test that load_or_create_key loads existing key."""
        first_key = SecureTokenStore.load_or_create_key(key_path=temp_key_path)
        second_key = SecureTokenStore.load_or_create_key(key_path=temp_key_path)

        assert first_key == second_key

    def test_load_or_create_key_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that load_or_create_key creates parent directories."""
        nested_path = str(tmp_path / "nested" / "dirs" / "key")

        key = SecureTokenStore.load_or_create_key(key_path=nested_path)

        assert Path(nested_path).exists()
        assert len(key) == 44

    def test_load_or_create_key_fails_on_empty_file(self, temp_key_path: str) -> None:
        """Test that load_or_create_key raises IOError for empty key file."""
        key_path = Path(temp_key_path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(b"")

        with pytest.raises(IOError, match="Key file is empty"):
            SecureTokenStore.load_or_create_key(key_path=temp_key_path)

    def test_load_or_create_key_fails_on_unreadable_file(
        self, temp_key_path: str
    ) -> None:
        """Test that load_or_create_key raises IOError for unreadable file."""
        key_path = Path(temp_key_path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(b"valid_key_data")
        key_path.chmod(0o000)

        try:
            with pytest.raises(IOError, match="Cannot read key file"):
                SecureTokenStore.load_or_create_key(key_path=temp_key_path)
        finally:
            key_path.chmod(0o600)


class TestSecureTokenStoreClear:
    """Test memory clearing functionality."""

    def test_clear_zeros_key(self, temp_key_path: str) -> None:
        """Test that clear() zeros the encryption key."""
        store = SecureTokenStore(key_path=temp_key_path)

        store.clear()

        assert store._key == b"\x00" * 44

    def test_clear_nullifies_cipher(self, temp_key_path: str) -> None:
        """Test that clear() nullifies the cipher."""
        store = SecureTokenStore(key_path=temp_key_path)

        store.clear()

        assert store._cipher is None


class TestSecureTokenStoreMultipleInstances:
    """Test behavior with multiple store instances."""

    def test_multiple_instances_share_key(self, temp_key_path: str) -> None:
        """Test that multiple instances using same key can decrypt each other's tokens."""
        store1 = SecureTokenStore(key_path=temp_key_path)
        store2 = SecureTokenStore(key_path=temp_key_path)

        token = "shared_test_token"
        encrypted_by_store1 = store1.encrypt(token)
        decrypted_by_store2 = store2.decrypt(encrypted_by_store1)

        assert decrypted_by_store2 == token

    def test_different_keys_cannot_decrypt_tokens(self, tmp_path: Path) -> None:
        """Test that tokens encrypted with one key cannot be decrypted with another."""
        key_path1 = str(tmp_path / "key1")
        key_path2 = str(tmp_path / "key2")

        store1 = SecureTokenStore(key_path=key_path1)
        store2 = SecureTokenStore(key_path=key_path2)

        token = "secret_token"
        encrypted = store1.encrypt(token)

        with pytest.raises(ValueError, match="Decryption failed"):
            store2.decrypt(encrypted)


class TestSecureTokenStorePathHandling:
    """Test path expansion and handling."""

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch) -> None:
        """Test that ~ in path is expanded correctly."""
        fake_home = tmp_path
        monkeypatch.setenv("HOME", str(fake_home))

        key_path = "~/.pfun/test_key"
        key = SecureTokenStore.load_or_create_key(key_path=key_path)

        expected_path = fake_home / ".pfun" / "test_key"
        assert expected_path.exists()
        assert len(key) == 44

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Test that absolute paths work correctly."""
        key_path = str(tmp_path / "absolute" / "key")

        key = SecureTokenStore.load_or_create_key(key_path=key_path)

        assert Path(key_path).exists()
        assert len(key) == 44


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_key_path(tmp_path: Path) -> str:
    """Fixture providing a temporary key path."""
    return str(tmp_path / "test_key")
