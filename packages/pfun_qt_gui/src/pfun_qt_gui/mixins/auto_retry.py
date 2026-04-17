"""Auto-retry mixin for QMainWindow subclasses.

Provides ``AutoRetryMixin``, a cooperative mixin that encapsulates the
exponential-backoff retry logic for failed network requests (HTTP 500).

Usage::

    class MyApp(AutoRetryMixin, HealthCheckMixin, QMainWindow):
        ...

The consuming class must:
  1. Set ``self.network_manager`` to a ``QNetworkAccessManager`` **before**
     any retryable request is sent.
  2. Optionally set ``self._submit_overlay`` to a ``SubmitLoadingOverlay``
     so that retry status text is shown to the user.
  3. Override ``on_retries_exhausted(reply)`` if custom error handling beyond
     the default ``QMessageBox.critical`` is desired.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest

if TYPE_CHECKING:
    from PyQt6.QtNetwork import QNetworkAccessManager
    from pfun_qt_gui.loading_overlay import SubmitLoadingOverlay

logger = logging.getLogger("pfun-qt-gui-app")

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_MS = 1000  # doubles each attempt (exponential backoff)


class AutoRetryMixin:
    """Mixin that adds exponential-backoff auto-retry for POST requests.

    Attributes:
        max_retries: Maximum number of retry attempts.
        retry_base_delay_ms: Base delay between retries (doubles each attempt).
        _retry_count: Number of retries attempted for the current request.
        _pending_request: The ``QNetworkRequest`` being retried (or ``None``).
    """

    # -- configuration (can be overridden by subclass or __init__) --
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_delay_ms: int = DEFAULT_BASE_DELAY_MS

    # -- runtime state --
    _retry_count: int
    _pending_request: QNetworkRequest | None

    # -- expected on the host class --
    network_manager: QNetworkAccessManager
    _submit_overlay: SubmitLoadingOverlay | None

    # ----- public API -----

    def init_retry_state(self) -> None:
        """Initialise (or reset) the retry state.

        Call this once during ``__init__`` of the consuming class.
        """
        self._retry_count = 0
        self._pending_request = None

    def start_retryable_request(self, request: QNetworkRequest) -> None:
        """Store ``request`` and reset the retry counter.

        Call this immediately before ``self.network_manager.post(request, …)``
        so that the mixin can re-issue the same request on failure.
        """
        self._retry_count = 0
        self._pending_request = request

    def clear_pending_request(self) -> None:
        """Discard any stored request and reset the counter."""
        self._pending_request = None
        self._retry_count = 0

    def should_retry(self, reply: QNetworkReply) -> bool:
        """Return ``True`` if the reply is an HTTP 500 and retries remain.

        If ``True`` is returned, a retry has been **scheduled** — the caller
        should ``reply.deleteLater()`` and ``return`` without further
        processing.
        """
        status_code = reply.attribute(
            QNetworkRequest.Attribute.HttpStatusCodeAttribute
        )

        if status_code != 500:
            return False
        if self._retry_count >= self.max_retries:
            return False

        self._retry_count += 1
        delay_ms = self.retry_base_delay_ms * (2 ** (self._retry_count - 1))

        logger.warning(
            "Request returned 500 – scheduling retry %d/%d in %d ms",
            self._retry_count,
            self.max_retries,
            delay_ms,
        )

        # Update the submit overlay so the user knows what's happening
        if hasattr(self, "_submit_overlay") and self._submit_overlay is not None:
            self._submit_overlay.set_status(
                "Server error – retrying…",
                f"Attempt {self._retry_count}/{self.max_retries} "
                f"(waiting {delay_ms / 1000:.0f}s)",
            )

        reply.deleteLater()
        QTimer.singleShot(delay_ms, self._execute_retry)
        return True

    @property
    def retry_count(self) -> int:
        """The number of retries attempted so far."""
        return self._retry_count

    # ----- internal helpers -----

    def _execute_retry(self) -> None:
        """Re-send the stored request.

        Called by ``QTimer.singleShot`` after the backoff delay.
        """
        if self._pending_request is None:
            logger.error("_execute_retry called but no pending request.")
            return

        logger.info(
            "Retrying request (attempt %d/%d)…",
            self._retry_count,
            self.max_retries,
        )

        if hasattr(self, "_submit_overlay") and self._submit_overlay is not None:
            self._submit_overlay.set_status(
                "Retrying…",
                f"Sending attempt {self._retry_count}/{self.max_retries}",
            )

        self.network_manager.post(self._pending_request, b"{}")
