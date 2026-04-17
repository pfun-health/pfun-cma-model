"""Tests for the AutoRetryMixin.

These tests validate the exponential-backoff retry logic in isolation,
without requiring a live Qt event loop or display server.  Qt network
objects are mocked so the tests run headlessly in CI.
"""

import pytest
from unittest.mock import MagicMock, patch

from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest


# ---------------------------------------------------------------------------
# Minimal host class that wires up AutoRetryMixin
# ---------------------------------------------------------------------------

from pfun_qt_gui.mixins.auto_retry import AutoRetryMixin, DEFAULT_MAX_RETRIES


class _FakeHost(AutoRetryMixin):
    """Minimal stand-in for the real QMainWindow-based host class."""

    def __init__(self):
        self.network_manager = MagicMock()
        self._submit_overlay = None
        self.init_retry_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reply(status_code: int) -> MagicMock:
    """Create a mock QNetworkReply with the given HTTP status code."""
    reply = MagicMock(spec=QNetworkReply)
    reply.attribute.return_value = status_code
    return reply


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoRetryMixinInit:
    """init_retry_state / clear_pending_request basics."""

    def test_initial_state(self):
        host = _FakeHost()
        assert host.retry_count == 0
        assert host._pending_request is None

    def test_start_retryable_request_stores_request(self):
        host = _FakeHost()
        req = MagicMock(spec=QNetworkRequest)
        host.start_retryable_request(req)
        assert host._pending_request is req
        assert host.retry_count == 0

    def test_clear_pending_request(self):
        host = _FakeHost()
        req = MagicMock(spec=QNetworkRequest)
        host.start_retryable_request(req)
        host.clear_pending_request()
        assert host._pending_request is None
        assert host.retry_count == 0


class TestShouldRetry:
    """should_retry decision and side-effects."""

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_returns_true_on_500(self, mock_timer):
        host = _FakeHost()
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))
        reply = _make_reply(500)

        assert host.should_retry(reply) is True
        assert host.retry_count == 1
        reply.deleteLater.assert_called_once()

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_returns_false_on_200(self, mock_timer):
        host = _FakeHost()
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))
        reply = _make_reply(200)

        assert host.should_retry(reply) is False
        assert host.retry_count == 0

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_returns_false_on_non_500_error(self, mock_timer):
        host = _FakeHost()
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        for code in (400, 401, 403, 404, 422, 502, 503):
            reply = _make_reply(code)
            assert host.should_retry(reply) is False

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_respects_max_retries(self, mock_timer):
        host = _FakeHost()
        host.max_retries = 2
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        # First two 500s should be retried
        assert host.should_retry(_make_reply(500)) is True  # count → 1
        assert host.should_retry(_make_reply(500)) is True  # count → 2

        # Third 500 should NOT be retried (exhausted)
        assert host.should_retry(_make_reply(500)) is False
        assert host.retry_count == 2

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_schedules_timer_with_exponential_delay(self, mock_timer):
        host = _FakeHost()
        host.retry_base_delay_ms = 1000
        host.max_retries = 3
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        # 1st retry → 1000 ms
        host.should_retry(_make_reply(500))
        _, call_args, _ = mock_timer.singleShot.mock_calls[0]
        assert call_args[0] == 1000

        # 2nd retry → 2000 ms
        host.should_retry(_make_reply(500))
        _, call_args, _ = mock_timer.singleShot.mock_calls[1]
        assert call_args[0] == 2000

        # 3rd retry → 4000 ms
        host.should_retry(_make_reply(500))
        _, call_args, _ = mock_timer.singleShot.mock_calls[2]
        assert call_args[0] == 4000


class TestOverlayUpdates:
    """Verify that the submit overlay status is updated on retry."""

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_updates_overlay_on_retry(self, mock_timer):
        host = _FakeHost()
        host._submit_overlay = MagicMock()
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        host.should_retry(_make_reply(500))
        host._submit_overlay.set_status.assert_called_once()

        # Verify the heading mentions "retrying"
        heading = host._submit_overlay.set_status.call_args[0][0]
        assert "retry" in heading.lower()

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_no_crash_without_overlay(self, mock_timer):
        host = _FakeHost()
        host._submit_overlay = None
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        # Should not raise even without an overlay
        host.should_retry(_make_reply(500))
        assert host.retry_count == 1


class TestExecuteRetry:
    """_execute_retry internal helper."""

    def test_posts_pending_request(self):
        host = _FakeHost()
        req = MagicMock(spec=QNetworkRequest)
        host.start_retryable_request(req)
        host._retry_count = 1

        host._execute_retry()
        host.network_manager.post.assert_called_once_with(req, b"{}")

    def test_logs_error_when_no_pending_request(self):
        host = _FakeHost()
        # No start_retryable_request called
        with patch("pfun_qt_gui.mixins.auto_retry.logger") as mock_logger:
            host._execute_retry()
            mock_logger.error.assert_called_once()
        host.network_manager.post.assert_not_called()


class TestCustomRetryConfig:
    """Verify that max_retries and retry_base_delay_ms are configurable."""

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_custom_max_retries(self, mock_timer):
        host = _FakeHost()
        host.max_retries = 5
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        for i in range(5):
            assert host.should_retry(_make_reply(500)) is True

        # 6th should fail
        assert host.should_retry(_make_reply(500)) is False

    @patch("pfun_qt_gui.mixins.auto_retry.QTimer")
    def test_custom_base_delay(self, mock_timer):
        host = _FakeHost()
        host.retry_base_delay_ms = 500
        host.max_retries = 2
        host.start_retryable_request(MagicMock(spec=QNetworkRequest))

        host.should_retry(_make_reply(500))
        _, call_args, _ = mock_timer.singleShot.mock_calls[0]
        assert call_args[0] == 500  # 500 * 2^0

        host.should_retry(_make_reply(500))
        _, call_args, _ = mock_timer.singleShot.mock_calls[1]
        assert call_args[0] == 1000  # 500 * 2^1

    def test_default_values(self):
        host = _FakeHost()
        assert host.max_retries == DEFAULT_MAX_RETRIES
        assert host.retry_base_delay_ms == 1000
