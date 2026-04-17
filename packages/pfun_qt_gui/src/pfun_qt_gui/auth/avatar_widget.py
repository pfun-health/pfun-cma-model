"""Avatar Widget for Authenticated Users in PFun Qt GUI.

Provides a circular avatar button that replaces the login button after
authentication. Includes a dropdown menu for health service connections
and account management.

The widget adapts to platform tier (mobile/desktop/tv) for responsive sizing
and follows the established design language from theme.py.
"""

import logging
import json
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QPushButton,
    QMenu,
    QMessageBox,
    QVBoxLayout,
)

from pfun_qt_gui.theme import get_theme, platform_tier, PlatformTier, scale

if TYPE_CHECKING:
    from pfun_qt_gui.auth.secure_token_store import QtSecureTokenStore

logger = logging.getLogger(__name__)


class CircularAvatarButton(QPushButton):
    """Circular avatar button with initials and dropdown menu."""

    # Signal emitted when logout is requested
    logout_requested = pyqtSignal()

    def __init__(self, username: str = "U", parent=None):
        """Initialize the avatar button.

        Args:
            username: User's username or email to extract initials from
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme = get_theme()
        self.username = username
        self.initials = self._extract_initials(username)

        # Configure button appearance
        self.setFixedSize(scale(40), scale(40))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Remove default button styling
        self.setStyleSheet(self._get_stylesheet())

        # Connect click event
        self.clicked.connect(self.show_menu)

    def _extract_initials(self, username: str) -> str:
        """Extract user initials from username/email.

        Args:
            username: Username or email address

        Returns:
            First letter of first name and last name, or first two letters
        """
        if not username:
            return "U"

        # Handle email addresses
        if "@" in username:
            username = username.split("@")[0]

        # Extract first character and last character if space separated
        parts = username.split()
        if len(parts) > 1:
            return f"{parts[0][0]}{parts[-1][0]}".upper()
        elif len(username) >= 2:
            return f"{username[0]}{username[1]}".upper()
        else:
            return username[0].upper()

    def _get_stylesheet(self) -> str:
        """Generate platform-adaptive stylesheet for the avatar button."""
        p = self.theme.palette
        s = self.theme.s

        # Platform-adaptive sizing
        tier = platform_tier()
        if tier == PlatformTier.MOBILE:
            font_size = s(16)
        elif tier == PlatformTier.TV:
            font_size = s(20)
        else:
            font_size = s(14)

        return f"""
            QPushButton {{
                background-color: {p.accent};
                border: 2px solid {p.border_accent};
                border-radius: {s(20)}px;
                color: {p.text_on_accent};
                font-size: {font_size}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {p.accent_hover};
                border: 2px solid {p.text_on_accent};
            }}
            QPushButton:pressed {{
                background-color: {p.accent};
                border: 2px solid {p.text_on_accent};
                padding-top: 1px;
                padding-bottom: 1px;
            }}
        """

    def paintEvent(self, event):
        """Custom painting for circular avatar with initials."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw circular background
        rect = self.rect()
        painter.setBrush(QColor(self.theme.palette.accent))
        painter.setPen(QColor(self.theme.palette.border_accent))
        painter.drawEllipse(rect)

        # Draw initials text
        painter.setPen(QColor(self.theme.palette.text_on_accent))
        font = QFont()
        font.setBold(True)
        font.setPixelSize(self.theme.s(16))
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.initials)

        painter.end()

    def show_menu(self):
        """Create and show the avatar dropdown menu."""
        menu = AvatarDropdownMenu(self.username, self)
        menu.about_to_show.connect(self._on_menu_about_to_show)

        # Position menu below the button
        pos = self.mapToGlobal(QPoint(0, self.height()))
        menu.popup(pos)

        # Emit signal that menu is about to show
        menu.about_to_show.emit()

    def _on_menu_about_to_show(self):
        """Callback when menu is about to show."""
        logger.debug("Avatar dropdown menu opened")


class AvatarDropdownMenu(QMenu):
    """Dropdown menu for authenticated user actions."""

    # Signal emitted when menu is about to show
    about_to_show = pyqtSignal()

    def __init__(self, username: str, parent=None):
        """Initialize the avatar dropdown menu.

        Args:
            username: Authenticated user's username
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme = get_theme()
        self.username = username

        # Configure menu appearance
        self.setStyleSheet(self._get_menu_stylesheet())

        # Build menu items
        self._build_menu()

    def _get_menu_stylesheet(self) -> str:
        """Generate stylesheet for the dropdown menu."""
        p = self.theme.palette
        s = self.theme.s

        return f"""
            QMenu {{
                background-color: {p.bg_card};
                border: 1px solid {p.border};
                border-radius: {s(self.theme.radius_md)}px;
                padding: {s(4)}px 0;
            }}
            QMenu::item {{
                background-color: transparent;
                padding: {s(8)}px {s(20)}px;
                color: {p.text_primary};
                font-size: {s(self.theme.font_size_body)}px;
            }}
            QMenu::item:selected {{
                background-color: {p.accent};
                color: {p.text_on_accent};
            }}
            QMenu::separator {{
                height: 1px;
                background: {p.border};
                margin: {s(4)}px 0;
            }}
        """

    def _build_menu(self):
        """Build the complete dropdown menu."""
        # User info section
        user_action = self.addAction(f"Signed in as: {self.username}")
        user_action.setEnabled(False)
        user_action.setFont(
            QFont(
                user_action.font().family(),
                user_action.font().pointSize(),
                QFont.Weight.Bold,
            )
        )

        self.addSeparator()

        # Health services section
        services_label = self.addAction("Connect Health Services:")
        services_label.setEnabled(False)

        dexcom_action = self.addAction("🔗 Connect to Dexcom")
        dexcom_action.triggered.connect(self._connect_dexcom)

        fitbit_action = self.addAction("🔗 Connect to Fitbit")
        fitbit_action.triggered.connect(self._connect_fitbit)

        google_fit_action = self.addAction("🔗 Connect to Google Fit")
        google_fit_action.triggered.connect(self._connect_google_fit)

        self.addSeparator()

        # Account actions
        profile_action = self.addAction("👤 Profile Settings")
        profile_action.triggered.connect(self._profile_settings)

        logout_action = self.addAction("🚪 Log Out")
        logout_action.triggered.connect(self._logout)

    def _connect_dexcom(self):
        """Handle Dexcom connection request."""
        QMessageBox.information(
            self.parent(),
            "Connection Not Implemented",
            "Dexcom integration is not implemented in this demo.",
        )
        logger.info("User requested Dexcom connection")

    def _connect_fitbit(self):
        """Handle Fitbit connection request."""
        QMessageBox.information(
            self.parent(),
            "Connection Not Implemented",
            "Fitbit integration is not implemented in this demo.",
        )
        logger.info("User requested Fitbit connection")

    def _connect_google_fit(self):
        """Handle Google Fit connection request."""
        QMessageBox.information(
            self.parent(),
            "Connection Not Implemented",
            "Google Fit integration is not implemented in this demo.",
        )
        logger.info("User requested Google Fit connection")

    def _profile_settings(self):
        """Handle profile settings request."""
        QMessageBox.information(
            self.parent(),
            "Profile Settings",
            "Profile settings are not implemented in this demo.",
        )
        logger.info("User accessed profile settings")

    def _logout(self):
        """Handle logout request."""
        # Confirm logout
        reply = QMessageBox.question(
            self.parent(),
            "Confirm Logout",
            "Are you sure you want to log out?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("User logged out")
            # Emit logout signal
            self.logout_requested.emit()


class AvatarWidget(QWidget):
    """Main avatar widget container.

    This widget manages the visibility of either the login button or
    the avatar button based on authentication state. It emits signals
    when authentication state changes.
    """

    # Signal emitted when authentication state changes
    auth_state_changed = pyqtSignal(
        bool
    )  # True = authenticated, False = unauthenticated

    def __init__(self, token_store: "QtSecureTokenStore", parent=None):
        """Initialize the avatar widget.

        Args:
            token_store: QtSecureTokenStore instance for checking auth state
            parent: Parent widget
        """
        super().__init__(parent)
        self.token_store = token_store
        self.theme = get_theme()

        # Current state
        self.avatar_button = None

        # Initialize UI based on auth state
        self.update_auth_state()

    def update_auth_state(self):
        """Update widget visibility based on authentication state."""
        # Clear existing layout if any
        if self.layout():
            # Remove and delete existing widgets
            while self.layout().count():
                child = self.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

        # Create new layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Check if user is authenticated
        if self.token_store.has_tokens():
            try:
                # Get user info from token
                access_token = self.token_store.get_access_token()
                if access_token:
                    # Simple extraction - in real implementation, decode JWT to get user info
                    # For now, we'll use a generic name
                    username = "User"

                    # Try to extract username from token if possible
                    # This is a simplified version - normally we'd decode the JWT
                    if "username" in access_token:
                        # Very basic parsing - replace with proper JWT decoding in production
                        try:
                            import base64

                            # Extract payload part of JWT (middle part)
                            payload = access_token.split(".")[1]
                            # Add padding if needed
                            payload += "=" * ((4 - len(payload) % 4) % 4)
                            decoded_payload = base64.urlsafe_b64decode(payload)
                            token_data = json.loads(decoded_payload)
                            username = token_data.get(
                                "sub", "User"
                            )  # Usually the username/email is in "sub"
                        except Exception:
                            # Fallback to generic username if parsing fails
                            pass

                    # Create avatar button
                    self.avatar_button = CircularAvatarButton(username)
                    # Connect logout signal
                    self.avatar_button.logout_requested = self.token_store
                    layout.addWidget(self.avatar_button)

                    logger.debug("Showing avatar button for authenticated user")
                else:
                    # No access token, fallback to login button
                    self._show_login_button(layout)
            except Exception as e:
                logger.error(f"Error getting user info: {e}")
                # Fallback to login button
                self._show_login_button(layout)
        else:
            # Show login button
            self._show_login_button(layout)

    def _show_login_button(self, layout):
        """Show the login button."""
        login_btn = QPushButton("Login")
        login_btn.setObjectName("submit_btn")
        login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Connect to parent window to show login dialog
        login_btn.clicked.connect(self._show_login_dialog)
        layout.addWidget(login_btn)
        logger.debug("Showing login button for unauthenticated user")

    def _show_login_dialog(self):
        """Show the login dialog via parent window."""
        parent = self.parent()
        if hasattr(parent, "show_login_dialog"):
            parent.show_login_dialog()
