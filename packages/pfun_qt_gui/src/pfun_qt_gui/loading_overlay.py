"""Loading overlay widgets for the PFun Qt GUI.

Provides a base ``LoadingOverlay`` that renders a full-window semi-transparent
backdrop with a rounded card, animated spinner, heading/detail labels, and
optional rotating quirky messages.  Two concrete subclasses are supplied:

* ``StartupLoadingOverlay`` – shown while the server health-check is pending.
* ``SubmitLoadingOverlay``  – shown while a generation request is in-flight.
"""

from __future__ import annotations

import random
from typing import Sequence

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Quirky loading messages
# ---------------------------------------------------------------------------

_STARTUP_QUIPS: list[str] = [
    "Warming up the stethoscope…",
    "Calibrating the circadian clock…",
    "Stretching before the sprint…",
    "Brewing some green tea…",
    "Checking your vitamin D levels…",
    "Flossing the network cables…",
    "Synchronizing body clocks…",
    "Tuning the biorhythm radio…",
    "Loading electrolytes…",
    "Aligning chakras (and packets)…",
]

_SUBMIT_QUIPS: list[str] = [
    "Consulting Dr. Algorithm…",
    "Crunching the health numbers…",
    "Your cells are in a meeting…",
    "Prescribing 10cc of patience…",
    "Running on the health treadmill…",
    "Analyzing your glucose crystals…",
    "Asking the mitochondria nicely…",
    "Cross-referencing with WebMD (jk)…",
    "Sequencing the fun genome…",
    "Reticulating health splines…",
    "Checking if an apple a day is enough…",
    "Warming up the neural pathways…",
    "Counting sheep for better data…",
    "Hydrating the model…",
]

# How often the quirky sub-message rotates (ms)
_QUIP_ROTATION_INTERVAL_MS = 3500


# ---------------------------------------------------------------------------
# Base overlay
# ---------------------------------------------------------------------------


class LoadingOverlay(QWidget):
    """Full-window translucent overlay with a centred loading card.

    Subclasses should set:
      * ``_initial_heading``  – text shown in the heading label on creation.
      * ``_spinner_color``    – QColor used for the braille spinner.
      * ``_card_accent_color``– QColor used for the card border.
      * ``_quips``            – sequence of quirky messages to rotate through
                                (empty to disable).
    """

    # -- visual constants (shared by all overlays) --
    _BACKDROP_COLOR = QColor(15, 23, 42, 210)
    _CARD_COLOR = QColor(20, 27, 46, 240)
    _CARD_RADIUS = 16

    # -- subclass hooks (defaults) --
    _initial_heading: str = "Loading…"
    _spinner_color: QColor = QColor("#60a5fa")
    _card_accent_color: QColor = QColor(71, 85, 105, 180)
    _quips: Sequence[str] = ()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)

        # --- widget flags ---
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        # --- outer layout (centres the card) ---
        outer_layout = QVBoxLayout(self)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- card container ---
        self._content_box = QWidget(self)
        self._content_box.setStyleSheet("background: transparent;")
        self._content_box.setFixedWidth(420)
        card_layout = QVBoxLayout(self._content_box)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- spinner ---
        self._spinner_label = QLabel("⠋")
        self._spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spinner_label.setStyleSheet(
            f"color: {self._spinner_color.name()}; font-size: 48px; background: transparent;"
        )
        self._spinner_frames = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._advance_spinner)
        self._spinner_timer.start(100)

        # --- heading ---
        self._heading = QLabel(self._initial_heading)
        self._heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading_font = QFont()
        heading_font.setPointSize(18)
        heading_font.setBold(True)
        self._heading.setFont(heading_font)
        self._heading.setStyleSheet("color: #f1f5f9; background: transparent;")

        # --- detail ---
        self._detail = QLabel("")
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setStyleSheet(
            "color: #94a3b8; font-size: 13px; background: transparent;"
        )
        self._detail.setWordWrap(True)

        # --- quirky sub-message ---
        self._quip_label = QLabel("")
        self._quip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._quip_label.setStyleSheet(
            "color: #cbd5e1; font-size: 12px; font-style: italic; background: transparent;"
        )
        self._quip_label.setWordWrap(True)

        # Assemble card layout
        card_layout.addWidget(self._spinner_label)
        card_layout.addSpacing(12)
        card_layout.addWidget(self._heading)
        card_layout.addSpacing(6)
        card_layout.addWidget(self._detail)
        card_layout.addSpacing(10)
        card_layout.addWidget(self._quip_label)

        outer_layout.addWidget(self._content_box, 0, Qt.AlignmentFlag.AlignCenter)

        # --- opacity effect for fade-out ---
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        # --- quirky message rotation ---
        self._quip_timer: QTimer | None = None
        if self._quips:
            self._show_random_quip()
            self._quip_timer = QTimer(self)
            self._quip_timer.timeout.connect(self._show_random_quip)
            self._quip_timer.start(_QUIP_ROTATION_INTERVAL_MS)

    # ----- public API -----

    def set_status(self, heading: str, detail: str = "") -> None:
        """Update the heading and detail text."""
        self._heading.setText(heading)
        self._detail.setText(detail)

    def fade_out_and_remove(self, duration_ms: int = 400) -> None:
        """Animate opacity → 0 then remove the widget from the tree."""
        self._spinner_timer.stop()
        if self._quip_timer is not None:
            self._quip_timer.stop()

        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(duration_ms)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.finished.connect(self._on_fade_finished)
        anim.start()
        # prevent GC before the animation completes
        self._fade_anim = anim

    # ----- painting -----

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        """Draw the full-window backdrop and the content card."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1) full-window semi-transparent backdrop
        painter.fillRect(self.rect(), self._BACKDROP_COLOR)

        # 2) nearly-opaque rounded card behind the content area
        card_rect = self._content_box.geometry().adjusted(-4, -4, 4, 4)
        painter.setPen(QPen(self._card_accent_color, 1.0))
        painter.setBrush(QBrush(self._CARD_COLOR))
        painter.drawRoundedRect(card_rect, self._CARD_RADIUS, self._CARD_RADIUS)

        painter.end()

    # ----- internals -----

    def _advance_spinner(self) -> None:
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self._spinner_label.setText(self._spinner_frames[self._spinner_idx])

    def _show_random_quip(self) -> None:
        if self._quips:
            self._quip_label.setText(random.choice(self._quips))

    def _on_fade_finished(self) -> None:
        self.setParent(None)  # type: ignore[call-overload]
        self.deleteLater()


# ---------------------------------------------------------------------------
# Concrete overlays
# ---------------------------------------------------------------------------


class StartupLoadingOverlay(LoadingOverlay):
    """Overlay shown during application startup while waiting for the server."""

    _initial_heading = "Connecting to server…"
    _spinner_color = QColor("#60a5fa")  # blue-400
    _card_accent_color = QColor(71, 85, 105, 180)  # slate-500
    _quips = _STARTUP_QUIPS


class SubmitLoadingOverlay(LoadingOverlay):
    """Overlay shown after the user submits a query and awaits generation."""

    _initial_heading = "Generating your health tips…"
    _spinner_color = QColor("#34d399")  # emerald-400
    _card_accent_color = QColor(52, 211, 153, 180)  # emerald-400 with alpha
    _quips = _SUBMIT_QUIPS
