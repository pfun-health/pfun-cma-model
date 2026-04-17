"""Tests for the responsive theme module.

Validates DPI scaling helpers, platform tier detection, and the theme's
stylesheet generation without requiring a running display server.
"""

import pytest
from unittest.mock import MagicMock, patch

from pfun_qt_gui.theme import (
    Palette,
    PlatformTier,
    Theme,
    dpi_scale,
    get_theme,
    platform_tier,
    scale,
)


# ---------------------------------------------------------------------------
# DPI helpers
# ---------------------------------------------------------------------------


class TestDpiScale:
    """dpi_scale() and scale() functions."""

    def test_scale_identity_at_96dpi(self):
        """At 96 dpi the scale factor should be ~1.0, so scale(N) ≈ N."""
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 1.0
            assert scale(100) == 100
            assert scale(16) == 16
            assert scale(0) == 0
        finally:
            _mod._dpi_scale = original

    def test_scale_at_2x(self):
        """At 2× DPI, values should double."""
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 2.0
            assert scale(100) == 200
            assert scale(16) == 32
            assert scale(7) == 14
        finally:
            _mod._dpi_scale = original

    def test_scale_at_fractional(self):
        """At 1.5× DPI, values should round correctly."""
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 1.5
            assert scale(10) == 15
            assert scale(7) == 10  # 7 * 1.5 = 10.5 → rounds to 10
        finally:
            _mod._dpi_scale = original

    def test_dpi_scale_returns_1_without_qapp(self):
        """Without a QApplication, dpi_scale falls back to 1.0."""
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = None  # force re-computation
            with patch.object(_mod, "QApplication") as mock_app_cls:
                mock_app_cls.instance.return_value = None
                result = dpi_scale()
                assert result == 1.0
        finally:
            _mod._dpi_scale = original


# ---------------------------------------------------------------------------
# Platform tier
# ---------------------------------------------------------------------------


class TestPlatformTier:
    """platform_tier() heuristics."""

    def test_android_detected_as_mobile(self):
        with patch("pfun_qt_gui.theme.platform") as mock_platform:
            mock_platform.system.return_value = "Android"
            assert platform_tier() == PlatformTier.MOBILE

    def test_narrow_screen_detected_as_mobile(self):
        import pfun_qt_gui.theme as _mod
        mock_app = MagicMock()
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value.width.return_value = 600
        mock_app.primaryScreen.return_value = mock_screen

        with patch("pfun_qt_gui.theme.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch.object(_mod, "QApplication") as mock_app_cls:
                mock_app_cls.instance.return_value = mock_app
                assert platform_tier() == PlatformTier.MOBILE

    def test_wide_screen_detected_as_tv(self):
        import pfun_qt_gui.theme as _mod
        mock_app = MagicMock()
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value.width.return_value = 3840
        mock_app.primaryScreen.return_value = mock_screen

        with patch("pfun_qt_gui.theme.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch.object(_mod, "QApplication") as mock_app_cls:
                mock_app_cls.instance.return_value = mock_app
                assert platform_tier() == PlatformTier.TV

    def test_normal_screen_detected_as_desktop(self):
        import pfun_qt_gui.theme as _mod
        mock_app = MagicMock()
        mock_screen = MagicMock()
        mock_screen.availableGeometry.return_value.width.return_value = 1920
        mock_app.primaryScreen.return_value = mock_screen

        with patch("pfun_qt_gui.theme.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch.object(_mod, "QApplication") as mock_app_cls:
                mock_app_cls.instance.return_value = mock_app
                assert platform_tier() == PlatformTier.DESKTOP

    def test_no_qapp_defaults_to_desktop(self):
        import pfun_qt_gui.theme as _mod
        with patch("pfun_qt_gui.theme.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            with patch.object(_mod, "QApplication") as mock_app_cls:
                mock_app_cls.instance.return_value = None
                assert platform_tier() == PlatformTier.DESKTOP


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


class TestPalette:
    """Palette dataclass basic validation."""

    def test_palette_has_required_colours(self):
        p = Palette()
        # Spot-check that key colour tokens exist and are strings
        assert isinstance(p.bg_primary, str)
        assert isinstance(p.accent, str)
        assert isinstance(p.text_primary, str)
        assert isinstance(p.error, str)

    def test_palette_is_frozen(self):
        p = Palette()
        with pytest.raises(AttributeError):
            p.bg_primary = "#000"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


class TestTheme:
    """Theme dataclass and stylesheet generation."""

    def test_stylesheet_is_nonempty_string(self):
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 1.0
            t = Theme()
            ss = t.stylesheet()
            assert isinstance(ss, str)
            assert len(ss) > 100  # it's substantial
        finally:
            _mod._dpi_scale = original

    def test_stylesheet_contains_key_selectors(self):
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 1.0
            t = Theme()
            ss = t.stylesheet()
            for selector in [
                "QMainWindow",
                "QLabel",
                "QTextEdit",
                "QTextBrowser",
                "QPushButton#submit_btn",
                "QSplitter",
                "QScrollBar",
            ]:
                assert selector in ss, f"Missing selector: {selector}"
        finally:
            _mod._dpi_scale = original

    def test_stylesheet_respects_dpi(self):
        """Font sizes in the stylesheet should differ at 1× vs 2× DPI."""
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 1.0
            ss_1x = Theme().stylesheet()
            _mod._dpi_scale = 2.0
            ss_2x = Theme().stylesheet()
            # They must not be identical — pixel values differ
            assert ss_1x != ss_2x
        finally:
            _mod._dpi_scale = original

    def test_scale_method(self):
        import pfun_qt_gui.theme as _mod
        original = _mod._dpi_scale
        try:
            _mod._dpi_scale = 1.5
            t = Theme()
            assert t.s(10) == 15
        finally:
            _mod._dpi_scale = original


class TestGetTheme:
    """get_theme singleton."""

    def test_returns_theme_instance(self):
        import pfun_qt_gui.theme as _mod
        original_theme = _mod._theme
        try:
            _mod._theme = None
            t = get_theme()
            assert isinstance(t, Theme)
        finally:
            _mod._theme = original_theme

    def test_returns_same_instance(self):
        import pfun_qt_gui.theme as _mod
        original_theme = _mod._theme
        try:
            _mod._theme = None
            t1 = get_theme()
            t2 = get_theme()
            assert t1 is t2
        finally:
            _mod._theme = original_theme
