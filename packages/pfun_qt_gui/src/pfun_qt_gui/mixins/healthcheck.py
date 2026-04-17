"""Health-check polling mixin for QMainWindow subclasses.

Provides ``HealthCheckMixin``, a cooperative mixin that encapsulates all
server health-check polling logic:  dedicated ``QNetworkAccessManager``,
retry timer, overlay status updates, and the ``_server_healthy`` flag.

Usage::

    class MyApp(HealthCheckMixin, QMainWindow):
        ...

The consuming class must:
  1. Set ``self.api_url`` **before** calling ``start_health_check()``.
  2. Provide a ``centralWidget()`` (standard ``QMainWindow`` method).
  3. Optionally set ``self._loading_overlay`` to a ``StartupLoadingOverlay``
     before calling ``start_health_check()`` so that status text is updated.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

if TYPE_CHECKING:
    from pfun_qt_gui.loading_overlay import StartupLoadingOverlay, SubmitLoadingOverlay

logger = logging.getLogger("pfun-qt-gui-app")

# Health check polling interval in milliseconds
HEALTH_CHECK_INTERVAL_MS = 2000


class HealthCheckMixin:
    """Mixin that adds server health-check polling to a QMainWindow.

    Attributes:
        _server_healthy: Whether the backend has responded successfully.
        _health_attempts: Number of polling attempts made so far.
    """

    # -- attributes initialised by start_health_check --
    _health_net: QNetworkAccessManager
    _health_timer: QTimer
    _health_attempts: int
    _server_healthy: bool

    # -- expected on the host class (provided by QMainWindow / consumer) --
    api_url: str
    _loading_overlay: StartupLoadingOverlay | SubmitLoadingOverlay | None

    # ----- public API -----

    def start_health_check(self) -> None:
        """Initialise networking and begin polling ``/health``.

        Call this **after** ``self.api_url`` and ``self._loading_overlay``
        have been set.
        """
        self._health_attempts = 0
        self._server_healthy = False

        # Dedicated network manager so that health traffic is isolated
        self._health_net = QNetworkAccessManager(self)  # type: ignore[arg-type]
        self._health_net.finished.connect(self._on_health_reply)

        self._health_timer = QTimer(self)  # type: ignore[arg-type]
        self._health_timer.timeout.connect(self._poll_health)
        self._health_timer.start(HEALTH_CHECK_INTERVAL_MS)

        # Fire the first check immediately instead of waiting for the tick
        QTimer.singleShot(0, self._poll_health)

    @property
    def server_healthy(self) -> bool:
        """Return whether the backend is reachable and ready."""
        return self._server_healthy

    # ----- internal helpers -----

    def _poll_health(self) -> None:
        """Send a GET request to the server ``/health`` endpoint."""
        self._health_attempts += 1
        url = QUrl(f"{self.api_url}/health")
        request = QNetworkRequest(url)
        self._health_net.get(request)

        # Update the overlay status text
        if self._loading_overlay is not None:
            self._loading_overlay.set_status(
                "Connecting to server…",
                f"Polling {self.api_url}/health  (attempt {self._health_attempts})",
            )
        logger.debug(
            "Health check attempt %d → %s/health", self._health_attempts, self.api_url
        )

    def _on_health_reply(self, reply: QNetworkReply) -> None:
        """Handle the health-check response."""
        try:
            if reply.error() == QNetworkReply.NetworkError.NoError:
                body = reply.readAll().data().decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = {}

                if data.get("status") == "ok":
                    self._server_healthy = True
                    self._health_timer.stop()
                    logger.info(
                        "Server healthy after %d attempt(s).", self._health_attempts
                    )

                    if self._loading_overlay is not None:
                        self._loading_overlay.set_status(
                            "Connected ✓",
                            "Server is ready.",
                        )
                        # Short delay so the user sees the success state
                        QTimer.singleShot(
                            600, self._loading_overlay.fade_out_and_remove
                        )
                        self._loading_overlay = None
                    return

            # Server responded but not healthy – update detail
            error_text = (
                reply.errorString()
                if reply.error() != QNetworkReply.NetworkError.NoError
                else "unexpected response"
            )
            logger.debug(
                "Health check attempt %d failed: %s", self._health_attempts, error_text
            )
            if self._loading_overlay is not None:
                self._loading_overlay.set_status(
                    "Waiting for server…",
                    f"Attempt {self._health_attempts} — {error_text}",
                )
        finally:
            reply.deleteLater()
