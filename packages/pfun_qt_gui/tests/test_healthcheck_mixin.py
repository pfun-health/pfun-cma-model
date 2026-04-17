"""Tests for the HealthCheckMixin.

These tests validate the server health-check polling logic in isolation,
with all Qt network objects mocked so the tests run headlessly.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest

from pfun_qt_gui.mixins.healthcheck import HealthCheckMixin, HEALTH_CHECK_INTERVAL_MS


# ---------------------------------------------------------------------------
# Minimal host class
# ---------------------------------------------------------------------------


class _FakeHost(HealthCheckMixin):
    """Minimal stand-in for the QMainWindow-based host class."""

    def __init__(self, api_url: str = "https://example.com:8001"):
        self.api_url = api_url
        self._loading_overlay = None

    def centralWidget(self):  # noqa: N802
        return MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_health_reply(*, ok: bool = True, error: bool = False) -> MagicMock:
    """Create a mock QNetworkReply mimicking the /health endpoint."""
    reply = MagicMock(spec=QNetworkReply)
    if error:
        reply.error.return_value = QNetworkReply.NetworkError.ConnectionRefusedError
        reply.errorString.return_value = "Connection refused"
        reply.readAll.return_value.data.return_value = b""
    else:
        reply.error.return_value = QNetworkReply.NetworkError.NoError
        body = json.dumps({"status": "ok"} if ok else {"status": "degraded"})
        reply.readAll.return_value.data.return_value = body.encode("utf-8")
    return reply


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthCheckInit:
    """start_health_check initialisation."""

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_initial_state(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        assert host._server_healthy is False
        assert host._health_attempts == 0
        assert host.server_healthy is False

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_timer_started(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        # Timer instance's start should have been called
        timer_instance = mock_timer_cls.return_value
        timer_instance.start.assert_called_once_with(HEALTH_CHECK_INTERVAL_MS)


class TestHealthCheckPolling:
    """_poll_health sends a GET to /health."""

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_poll_increments_attempts(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        # Simulate a poll tick
        host._poll_health()
        assert host._health_attempts == 1

        host._poll_health()
        assert host._health_attempts == 2

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_poll_updates_overlay(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host._loading_overlay = MagicMock()
        host.start_health_check()

        host._poll_health()
        host._loading_overlay.set_status.assert_called()
        heading = host._loading_overlay.set_status.call_args[0][0]
        assert "server" in heading.lower() or "connect" in heading.lower()


class TestHealthCheckReply:
    """_on_health_reply handling."""

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_healthy_response_sets_flag(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        reply = _make_health_reply(ok=True)
        host._on_health_reply(reply)

        assert host._server_healthy is True
        assert host.server_healthy is True
        reply.deleteLater.assert_called_once()

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_healthy_response_stops_timer(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        reply = _make_health_reply(ok=True)
        host._on_health_reply(reply)

        host._health_timer.stop.assert_called()

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_unhealthy_response_keeps_polling(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        reply = _make_health_reply(ok=False)
        host._on_health_reply(reply)

        assert host._server_healthy is False
        # timer should NOT have been stopped
        host._health_timer.stop.assert_not_called()

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_network_error_keeps_polling(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        host.start_health_check()

        reply = _make_health_reply(error=True)
        host._on_health_reply(reply)

        assert host._server_healthy is False
        host._health_timer.stop.assert_not_called()
        reply.deleteLater.assert_called_once()

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_overlay_updated_on_success(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        overlay = MagicMock()
        host._loading_overlay = overlay
        host.start_health_check()

        reply = _make_health_reply(ok=True)
        host._on_health_reply(reply)

        overlay.set_status.assert_called()
        heading = overlay.set_status.call_args[0][0]
        assert "connected" in heading.lower() or "✓" in heading

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_overlay_cleared_after_success(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        overlay = MagicMock()
        host._loading_overlay = overlay
        host.start_health_check()

        reply = _make_health_reply(ok=True)
        host._on_health_reply(reply)

        # Overlay reference should be cleared after scheduling fade-out
        assert host._loading_overlay is None

    @patch("pfun_qt_gui.mixins.healthcheck.QTimer")
    @patch("pfun_qt_gui.mixins.healthcheck.QNetworkAccessManager")
    def test_overlay_updated_on_failure(self, mock_nam_cls, mock_timer_cls):
        host = _FakeHost()
        overlay = MagicMock()
        host._loading_overlay = overlay
        host.start_health_check()
        host._health_attempts = 3

        reply = _make_health_reply(error=True)
        host._on_health_reply(reply)

        overlay.set_status.assert_called()
        detail = overlay.set_status.call_args[0][1]
        assert "3" in detail  # attempt number


class TestServerHealthyProperty:
    """server_healthy read-only property."""

    def test_defaults_to_false(self):
        host = _FakeHost()
        host._server_healthy = False
        assert host.server_healthy is False

    def test_reflects_internal_state(self):
        host = _FakeHost()
        host._server_healthy = True
        assert host.server_healthy is True
