"""Centralised responsive theme for the PFun Qt GUI.

Provides:
  * **DPI-aware scaling** – Every pixel value is multiplied by the logical
    DPI ratio so the interface looks consistent across 96-dpi laptops,
    Retina / HiDPI displays, and large-format TVs.
  * **Platform tier detection** – ``mobile``, ``desktop``, or ``tv`` based
    on the primary screen's geometry.
  * **Design tokens** – a single ``Theme`` dataclass that holds all colours,
    radii, spacing, and font sizes.  Values are expressed as *scalable
    integers* that are multiplied by ``dpi_scale`` before use.
  * **Global stylesheet** – ``Theme.stylesheet()`` returns a full QSS
    string that can be applied to the app or window to skin every widget
    in one shot.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# DPI helpers
# ---------------------------------------------------------------------------

_dpi_scale: float | None = None


def dpi_scale() -> float:
    """Return the logical-DPI scaling factor relative to 96 dpi.

    Cached after the first call for the lifetime of the process.
    """
    global _dpi_scale
    if _dpi_scale is not None:
        return _dpi_scale

    app = QApplication.instance()
    if app is None:
        _dpi_scale = 1.0
        return _dpi_scale

    screen = app.primaryScreen()
    if screen is not None:
        _dpi_scale = screen.logicalDotsPerInch() / 96.0
    else:
        _dpi_scale = 1.0
    return _dpi_scale


def scale(px: int | float) -> int:
    """Scale a pixel value by the current DPI factor."""
    return round(px * dpi_scale())


# ---------------------------------------------------------------------------
# Platform tier
# ---------------------------------------------------------------------------

class PlatformTier(Enum):
    MOBILE = auto()
    DESKTOP = auto()
    TV = auto()


def platform_tier() -> PlatformTier:
    """Heuristically classify the runtime environment.

    * **mobile**  – screen width < 800 logical px *or* running on Android.
    * **tv**      – screen width ≥ 2500 logical px.
    * **desktop** – everything else.
    """
    sys_name = platform.system().lower()
    if sys_name == "android":
        return PlatformTier.MOBILE

    app = QApplication.instance()
    if app is None:
        return PlatformTier.DESKTOP

    screen = app.primaryScreen()
    if screen is None:
        return PlatformTier.DESKTOP

    w = screen.availableGeometry().width()
    if w < 800:
        return PlatformTier.MOBILE
    if w >= 2500:
        return PlatformTier.TV
    return PlatformTier.DESKTOP


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    """Colour palette tokens."""

    # Backgrounds
    bg_primary: str = "#0f172a"       # slate-900
    bg_secondary: str = "#1e293b"     # slate-800
    bg_card: str = "#1e293b"          # card surface
    bg_elevated: str = "#334155"      # slate-700
    bg_input: str = "#0f172a"         # input field

    # Accent
    accent: str = "#6366f1"           # indigo-500
    accent_hover: str = "#818cf8"     # indigo-400
    accent_glow: str = "rgba(99, 102, 241, 0.4)"
    accent_gradient_start: str = "#6366f1"
    accent_gradient_end: str = "#8b5cf6"   # violet-500

    # Text
    text_primary: str = "#f1f5f9"     # slate-100
    text_secondary: str = "#94a3b8"   # slate-400
    text_muted: str = "#64748b"       # slate-500
    text_on_accent: str = "#ffffff"

    # Borders
    border: str = "#334155"           # slate-700
    border_hover: str = "#475569"     # slate-600
    border_accent: str = "#6366f1"

    # Semantic
    success: str = "#34d399"          # emerald-400
    warning: str = "#fbbf24"          # amber-400
    error: str = "#f87171"            # red-400
    info: str = "#60a5fa"             # blue-400

    # Recommendation header
    rec_header_bg: str = "#6366f1"
    rec_header_text: str = "#ffffff"

    # Raw output header
    raw_header_bg: str = "#334155"
    raw_header_text: str = "#e2e8f0"


@dataclass
class Theme:
    """Complete theme with tokens scaled to the current DPI."""

    palette: Palette = field(default_factory=Palette)

    # --- fonts ---
    font_family: str = "'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif"
    font_size_h1: int = 26
    font_size_h2: int = 20
    font_size_body: int = 14
    font_size_small: int = 12
    font_size_button: int = 15

    # --- spacing (logical px, pre-scale) ---
    radius_sm: int = 6
    radius_md: int = 10
    radius_lg: int = 16
    spacing_xs: int = 4
    spacing_sm: int = 8
    spacing_md: int = 16
    spacing_lg: int = 24
    spacing_xl: int = 32

    # --- shadows ---
    shadow_card: str = "0px 4px 24px rgba(0, 0, 0, 0.35)"
    shadow_button: str = "0px 2px 8px rgba(99, 102, 241, 0.35)"

    # --- transition feel (ms) ---
    transition_fast: int = 150
    transition_normal: int = 250

    def s(self, px: int | float) -> int:
        """Shorthand for DPI-scaled pixel value."""
        return scale(px)

    # ----- Global QSS -----

    def stylesheet(self) -> str:
        """Return a complete QSS stylesheet for the application."""
        p = self.palette
        s = self.s

        return f"""
        /* ===== Global ===== */
        QMainWindow, QWidget {{
            background-color: {p.bg_primary};
            color: {p.text_primary};
            font-family: {self.font_family};
            font-size: {s(self.font_size_body)}px;
        }}

        /* ===== Scrollbars ===== */
        QScrollBar:vertical {{
            background: {p.bg_secondary};
            width: {s(10)}px;
            margin: 0;
            border-radius: {s(5)}px;
        }}
        QScrollBar::handle:vertical {{
            background: {p.bg_elevated};
            min-height: {s(30)}px;
            border-radius: {s(5)}px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {p.border_hover};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background: {p.bg_secondary};
            height: {s(10)}px;
            margin: 0;
            border-radius: {s(5)}px;
        }}
        QScrollBar::handle:horizontal {{
            background: {p.bg_elevated};
            min-width: {s(30)}px;
            border-radius: {s(5)}px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {p.border_hover};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}

        /* ===== Labels ===== */
        QLabel {{
            background: transparent;
            color: {p.text_primary};
        }}
        QLabel#title_label {{
            font-size: {s(self.font_size_h1)}px;
            font-weight: 700;
            color: {p.text_primary};
        }}
        QLabel#subtitle_label {{
            font-size: {s(self.font_size_body)}px;
            color: {p.text_secondary};
        }}
        QLabel#input_instruction {{
            font-size: {s(self.font_size_small)}px;
            color: {p.info};
            padding: {s(4)}px 0;
        }}
        QLabel#output_title {{
            font-size: {s(self.font_size_h2)}px;
            font-weight: 700;
            color: {p.text_primary};
        }}
        QLabel#output_subtitle {{
            font-size: {s(self.font_size_small)}px;
            color: {p.text_secondary};
        }}
        QLabel#section_header_recs {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {p.accent_gradient_start},
                stop:1 {p.accent_gradient_end}
            );
            color: {p.rec_header_text};
            font-weight: 700;
            font-size: {s(self.font_size_body)}px;
            padding: {s(8)}px {s(12)}px;
            border-radius: {s(self.radius_sm)}px;
        }}
        QLabel#section_header_raw {{
            background: {p.raw_header_bg};
            color: {p.raw_header_text};
            font-weight: 700;
            font-size: {s(self.font_size_body)}px;
            padding: {s(8)}px {s(12)}px;
            border-radius: {s(self.radius_sm)}px;
        }}

        /* ===== Text input ===== */
        QTextEdit {{
            background-color: {p.bg_input};
            color: {p.text_primary};
            border: 1px solid {p.border};
            border-radius: {s(self.radius_md)}px;
            padding: {s(12)}px;
            font-size: {s(self.font_size_body)}px;
            selection-background-color: {p.accent};
        }}
        QTextEdit:focus {{
            border: 1px solid {p.border_accent};
        }}

        /* ===== Output browsers ===== */
        QTextBrowser {{
            background-color: {p.bg_secondary};
            color: {p.text_primary};
            border: 1px solid {p.border};
            border-radius: {s(self.radius_md)}px;
            padding: {s(12)}px;
            font-size: {s(self.font_size_body)}px;
        }}
        QTextBrowser#raw_output {{
            font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
            font-size: {s(self.font_size_small)}px;
            color: {p.text_secondary};
        }}

        /* ===== Push button ===== */
        QPushButton#submit_btn {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {p.accent_gradient_start},
                stop:1 {p.accent_gradient_end}
            );
            color: {p.text_on_accent};
            font-size: {s(self.font_size_button)}px;
            font-weight: 600;
            padding: {s(12)}px {s(32)}px;
            border: none;
            border-radius: {s(self.radius_md)}px;
            min-width: {s(140)}px;
        }}
        QPushButton#submit_btn:hover {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {p.accent_hover},
                stop:1 #a78bfa
            );
        }}
        QPushButton#submit_btn:pressed {{
            background: {p.accent};
            padding-top: {s(13)}px;
            padding-bottom: {s(11)}px;
        }}
        QPushButton#submit_btn:disabled {{
            background: {p.bg_elevated};
            color: {p.text_muted};
        }}

        /* ===== Splitter ===== */
        QSplitter::handle {{
            background: {p.border};
            width: {s(3)}px;
            margin: {s(4)}px 0;
            border-radius: {s(1)}px;
        }}
        QSplitter::handle:hover {{
            background: {p.accent};
        }}

        /* ===== Message box ===== */
        QMessageBox {{
            background-color: {p.bg_card};
            color: {p.text_primary};
        }}
        QMessageBox QPushButton {{
            background: {p.accent};
            color: {p.text_on_accent};
            border: none;
            border-radius: {s(self.radius_sm)}px;
            padding: {s(8)}px {s(20)}px;
            font-weight: 600;
        }}
        QMessageBox QPushButton:hover {{
            background: {p.accent_hover};
        }}
        """


# ---------------------------------------------------------------------------
# Singleton theme instance (create after QApplication exists)
# ---------------------------------------------------------------------------

_theme: Theme | None = None


def get_theme() -> Theme:
    """Return (and lazily create) the global Theme instance."""
    global _theme
    if _theme is None:
        _theme = Theme()
    return _theme
