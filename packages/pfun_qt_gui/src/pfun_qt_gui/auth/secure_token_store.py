"""Qt wrapper for SecureTokenStore with QSettings persistence.

Encrypts and stores JWT tokens (access_token and refresh_token) in Qt's
QSettings while keeping them encrypted at rest. Uses backend SecureTokenStore
for AES-128 Fernet encryption.

Encryption Flow:
  1. PlainText Token
  2. Backend SecureTokenStore.encrypt() → Base64-encoded ciphertext
  3. QSettings storage → Persistent encrypted state
  4. Load from QSettings → Base64-encoded ciphertext
  5. Backend SecureTokenStore.decrypt() → PlainText Token

Thread Safety:
  All encrypt/decrypt operations are protected by QMutex to prevent
  concurrent access during cryptographic operations. This ensures
  atomic read-modify-write operations on QSettings.

Error Recovery:
  If decryption fails (corrupted token or tampered settings), tokens are
  automatically cleared and a warning is logged. Callers should prompt
  for re-login on ValueError.
"""

import logging
from typing import Optional

from PyQt6.QtCore import QMutex, QSettings

from pfun_cma_model.misc.secure_token_store import SecureTokenStore

__all__ = ["QtSecureTokenStore"]

logger = logging.getLogger(__name__)


class QtSecureTokenStore:
    """Secure token storage wrapper for Qt GUI with QSettings persistence.

    Stores encrypted JWT tokens (access_token and refresh_token) in QSettings
    while keeping them encrypted using backend SecureTokenStore (Fernet/AES-128).

    Thread-safe with QMutex protecting all encrypt/decrypt operations.
    """

    def __init__(
        self,
        storage_key_prefix: str = "pfun_auth",
        key_path: str = "~/.pfun/key",
        settings: Optional[QSettings] = None,
    ) -> None:
        """Initialize QtSecureTokenStore with QSettings and backend encryption.

        Creates or loads encryption key from filesystem. QSettings are
        configured with organization="pfun" and application="pfun-cma-model"
        unless a custom settings object is provided (useful for testing).

        Args:
            storage_key_prefix: Prefix for QSettings keys. Tokens stored as
                {storage_key_prefix}/access_token and
                {storage_key_prefix}/refresh_token. Defaults to "pfun_auth".
            key_path: Path to encryption key file. Defaults to ~/.pfun/key.
                Key is created with 0o600 permissions if missing.
            settings: Optional custom QSettings instance for testing.
                If None, creates default QSettings with organization="pfun"
                and application="pfun-cma-model".

        Raises:
            IOError: If encryption key cannot be read or written.
        """
        self._storage_key_prefix = storage_key_prefix
        self._access_token_key = f"{storage_key_prefix}/access_token"
        self._refresh_token_key = f"{storage_key_prefix}/refresh_token"

        # Initialize QSettings with organization/application info (or use provided)
        if settings is None:
            QSettings.setDefaultFormat(QSettings.Format.IniFormat)
            self._settings = QSettings("pfun", "pfun-cma-model")
        else:
            self._settings = settings

        # Initialize backend encryption (loads or creates key)
        self._backend_store = SecureTokenStore(key_path=key_path)

        # Thread safety for encrypt/decrypt operations
        self._crypto_mutex = QMutex()

        logger.debug(
            "Initialized QtSecureTokenStore with prefix=%s, settings_org=%s",
            storage_key_prefix,
            self._settings.organizationName(),
        )

    def store_tokens(self, access_token: str, refresh_token: str) -> None:
        """Encrypt and store both tokens in QSettings.

        Tokens are encrypted using backend SecureTokenStore (Fernet) before
        storing. Operation is mutex-protected for thread safety.

        Args:
            access_token: Plaintext JWT access token.
            refresh_token: Plaintext JWT refresh token.

        Raises:
            ValueError: If either token is None or empty.
            IOError: If QSettings cannot write to persistent storage.
        """
        if not access_token:
            raise ValueError("access_token cannot be None or empty")
        if not refresh_token:
            raise ValueError("refresh_token cannot be None or empty")

        try:
            self._crypto_mutex.lock()
            try:
                encrypted_access = self._backend_store.encrypt(access_token)
                encrypted_refresh = self._backend_store.encrypt(refresh_token)
            finally:
                self._crypto_mutex.unlock()

            self._settings.setValue(self._access_token_key, encrypted_access)
            self._settings.setValue(self._refresh_token_key, encrypted_refresh)
            self._settings.sync()

            logger.debug(
                "Stored encrypted tokens in QSettings (prefix=%s)",
                self._storage_key_prefix,
            )
        except (OSError, IOError) as e:
            raise IOError(f"Cannot write tokens to QSettings: {e}") from e

    def load_tokens(self) -> tuple[str, str] | tuple[None, None]:
        """Load and decrypt tokens from QSettings.

        Returns plaintext tokens if they exist and are valid. If decryption
        fails (corrupted token), clears tokens automatically and raises
        ValueError.

        Operation is mutex-protected for thread safety.

        Returns:
            Tuple of (access_token, refresh_token) if found and valid.
            Tuple of (None, None) if tokens don't exist in QSettings.

        Raises:
            ValueError: If tokens exist but decryption fails (corrupted or
                tampered). Tokens are automatically cleared on failure.
        """
        encrypted_access = self._settings.value(self._access_token_key)
        encrypted_refresh = self._settings.value(self._refresh_token_key)

        # Early exit: no tokens stored (QSettings returns empty string when
        # key doesn't exist)
        if (
            not encrypted_access
            or not encrypted_refresh
            or encrypted_access is None
            or encrypted_refresh is None
        ):
            return (None, None)

        try:
            self._crypto_mutex.lock()
            try:
                access_token = self._backend_store.decrypt(str(encrypted_access))
                refresh_token = self._backend_store.decrypt(str(encrypted_refresh))
            finally:
                self._crypto_mutex.unlock()

            logger.debug(
                "Loaded and decrypted tokens from QSettings (prefix=%s)",
                self._storage_key_prefix,
            )
            return (access_token, refresh_token)
        except ValueError as e:
            # Corruption detected: clear tokens and re-raise
            logger.warning(
                "Failed to decrypt tokens (corrupted or tampered). "
                "Clearing tokens. Error: %s",
                str(e),
            )
            self.clear_tokens()
            raise ValueError(
                "Failed to decrypt tokens: they may be corrupted. "
                "Tokens have been cleared. Please login again."
            ) from e

    def clear_tokens(self) -> None:
        """Delete encrypted tokens from QSettings.

        Removes both access_token and refresh_token entries. Operation is
        mutex-protected for thread safety.
        """
        try:
            self._crypto_mutex.lock()
            try:
                self._settings.remove(self._access_token_key)
                self._settings.remove(self._refresh_token_key)
                self._settings.sync()
            finally:
                self._crypto_mutex.unlock()

            logger.debug(
                "Cleared tokens from QSettings (prefix=%s)",
                self._storage_key_prefix,
            )
        except (OSError, IOError) as e:
            logger.error("Failed to clear tokens from QSettings: %s", str(e))

    def has_tokens(self) -> bool:
        """Check if tokens exist in QSettings.

        Returns:
            True if both access_token and refresh_token exist in QSettings,
            False otherwise.
        """
        return self._settings.contains(
            self._access_token_key
        ) and self._settings.contains(self._refresh_token_key)

    def get_access_token(self) -> str | None:
        """Convenience method to load just the access token.

        Equivalent to calling load_tokens()[0].

        Returns:
            Plaintext access token if found and valid, None otherwise.

        Raises:
            ValueError: If token exists but is corrupted (decryption fails).
                Tokens are automatically cleared on failure.
        """
        access_token, _ = self.load_tokens()
        return access_token

    def get_refresh_token(self) -> str | None:
        """Convenience method to load just the refresh token.

        Equivalent to calling load_tokens()[1].

        Returns:
            Plaintext refresh token if found and valid, None otherwise.

        Raises:
            ValueError: If token exists but is corrupted (decryption fails).
                Tokens are automatically cleared on failure.
        """
        _, refresh_token = self.load_tokens()
        return refresh_token
