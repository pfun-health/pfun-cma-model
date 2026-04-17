"""Authentication module for PFun Qt GUI.

Provides:
  * **QtSecureTokenStore** – Qt-specific wrapper for encrypted token storage
    using QSettings with backend SecureTokenStore (AES-128 Fernet encryption).
  * **Thread-safe token persistence** – QMutex-protected encrypt/decrypt operations.
  * **Graceful error recovery** – Corrupted tokens are automatically cleared
    and logged for re-login prompts.
"""

from pfun_qt_gui.auth.secure_token_store import QtSecureTokenStore

__all__ = ["QtSecureTokenStore"]
