"""Secure token storage with Fernet encryption (AES-128 symmetric encryption)."""

import logging
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

__all__ = ["SecureTokenStore"]

logger = logging.getLogger(__name__)


class SecureTokenStore:
    """Encrypts and decrypts JWT tokens using Fernet (AES-128).

    Provides secure, in-memory token storage with encryption/decryption.
    Keys are stored in ~/.pfun/key with restrictive permissions (0o600).
    """

    def __init__(self, key_path: str = "~/.pfun/key") -> None:
        """Initialize SecureTokenStore with key from disk or generate new key.

        Args:
            key_path: Path to store/load encryption key. Defaults to ~/.pfun/key

        Raises:
            IOError: If key file cannot be read or written.
        """
        self._key = self.load_or_create_key(key_path)
        self._cipher = Fernet(self._key)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new Fernet encryption key (32 bytes).

        Returns:
            New Fernet key as bytes.
        """
        return Fernet.generate_key()

    @staticmethod
    def load_or_create_key(key_path: str = "~/.pfun/key") -> bytes:
        """Load existing key or create and save a new one.

        Creates ~/.pfun directory if needed and sets key file permissions
        to 0o600 (readable/writable by owner only).

        Args:
            key_path: Path to encryption key file. Defaults to ~/.pfun/key

        Returns:
            Fernet encryption key as bytes.

        Raises:
            IOError: If key file cannot be read or written.
        """
        expanded_path = Path(key_path).expanduser()
        parent_dir = expanded_path.parent

        if expanded_path.exists():
            try:
                with open(expanded_path, "rb") as f:
                    key = f.read()
                if not key:
                    raise IOError(f"Key file is empty: {expanded_path}")
                return key
            except (OSError, IOError) as e:
                raise IOError(f"Cannot read key file {expanded_path}: {e}") from e

        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise IOError(f"Cannot create directory {parent_dir}: {e}") from e

        key = Fernet.generate_key()

        try:
            with open(expanded_path, "wb") as f:
                f.write(key)
            expanded_path.chmod(0o600)
            logger.warning(
                "Generated new encryption key at %s (permissions 0o600)",
                expanded_path,
            )
            return key
        except (OSError, IOError) as e:
            raise IOError(f"Cannot write key file {expanded_path}: {e}") from e

    def encrypt(self, token: str) -> str:
        """Encrypt a token and return base64-encoded ciphertext.

        Args:
            token: Plaintext JWT token to encrypt.

        Returns:
            Base64-encoded encrypted token (with Fernet timestamp).

        Raises:
            ValueError: If token is None or empty.
        """
        if not token:
            raise ValueError("Token cannot be None or empty")

        encrypted_bytes = self._cipher.encrypt(token.encode())
        return encrypted_bytes.decode()

    def decrypt(self, encrypted_token: str) -> str:
        """Decrypt a ciphertext token and return plaintext.

        Args:
            encrypted_token: Base64-encoded encrypted token from encrypt().

        Returns:
            Plaintext JWT token.

        Raises:
            ValueError: If decryption fails (token tampered or corrupted).
        """
        if not encrypted_token:
            raise ValueError("Encrypted token cannot be None or empty")

        try:
            decrypted_bytes = self._cipher.decrypt(encrypted_token.encode())
            return decrypted_bytes.decode()
        except InvalidToken as e:
            raise ValueError(
                "Decryption failed: token may be tampered or corrupted"
            ) from e

    def clear(self) -> None:
        """Clear encryption key from memory by zeroing it.

        This is a best-effort attempt to overwrite the key in memory.
        Note: Python's garbage collector may still hold references.
        """
        if hasattr(self, "_key") and self._key:
            self._key = b"\x00" * len(self._key)
        if hasattr(self, "_cipher"):
            self._cipher = None  # type: ignore[assignment]
