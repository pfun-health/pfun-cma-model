"""Login/Signup Dialog for PFun Qt GUI.

Provides a modal dialog for user authentication with support for:
- Basic username/password authentication
- Single Sign-On (SSO) integration
- JWT token management through QtSecureTokenStore

The dialog adapts to platform tier (mobile/desktop/tv) for responsive sizing
and touch-friendly controls.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFrame,
    QMessageBox,
)

from pfun_qt_gui.theme import get_theme, platform_tier, PlatformTier, scale

if TYPE_CHECKING:
    from pfun_qt_gui.auth.secure_token_store import QtSecureTokenStore

logger = logging.getLogger(__name__)


class LoginDialog(QDialog):
    """Modal login/signup dialog with SSO and basic auth support."""

    # Signal emitted when login is successful
    login_successful = pyqtSignal()

    def __init__(self, token_store: "QtSecureTokenStore", parent=None):
        """Initialize the login dialog.

        Args:
            token_store: QtSecureTokenStore instance for token persistence
            parent: Parent widget (typically main window)
        """
        super().__init__(parent)
        self.token_store = token_store
        self.theme = get_theme()

        # Configure dialog properties
        self.setWindowTitle("Login to PFun Health")
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog)

        # Platform-adaptive sizing
        tier = platform_tier()
        if tier == PlatformTier.MOBILE:
            self.setFixedWidth(scale(300))
        elif tier == PlatformTier.TV:
            self.setFixedWidth(scale(500))
        else:
            self.setFixedWidth(scale(400))

        self.setFixedHeight(scale(400))  # Fixed height for consistent UX

        # Build UI
        self.init_ui()

        # Apply theme
        self.setStyleSheet(self.theme.stylesheet())

    def init_ui(self):
        """Construct the complete UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(scale(24), scale(24), scale(24), scale(24))
        main_layout.setSpacing(scale(16))

        # ── Header ──────────────────────────────────────────────────────
        title_label = QLabel("Welcome to PFun Health")
        title_label.setObjectName("title_label")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle_label = QLabel("Sign in to access personalized health insights")
        subtitle_label.setObjectName("subtitle_label")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setWordWrap(True)

        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # ── Divider ─────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(
            f"background-color: {self.theme.palette.border}; max-height: 1px;"
        )
        main_layout.addWidget(divider)

        # ── Auth Form ───────────────────────────────────────────────────
        # Username field
        username_label = QLabel("Username or Email")
        username_label.setObjectName("input_instruction")
        self.username_input = QLineEdit()
        self.username_input.setObjectName("username_input")
        self.username_input.setPlaceholderText("Enter your username or email")
        self.username_input.returnPressed.connect(self.on_basic_login_clicked)

        # Password field
        password_label = QLabel("Password")
        password_label.setObjectName("input_instruction")
        self.password_input = QLineEdit()
        self.password_input.setObjectName("password_input")
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.returnPressed.connect(self.on_basic_login_clicked)

        main_layout.addWidget(username_label)
        main_layout.addWidget(self.username_input)
        main_layout.addWidget(password_label)
        main_layout.addWidget(self.password_input)

        # ── Buttons ─────────────────────────────────────────────────────
        # Basic auth button
        self.basic_login_btn = QPushButton("Log In")
        self.basic_login_btn.setObjectName("submit_btn")
        self.basic_login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.basic_login_btn.clicked.connect(self.on_basic_login_clicked)

        # SSO buttons
        sso_label = QLabel("Or continue with")
        sso_label.setObjectName("subtitle_label")
        sso_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sso_layout = QHBoxLayout()
        self.google_btn = QPushButton("Google")
        self.google_btn.setObjectName("sso_google_btn")
        self.google_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.google_btn.clicked.connect(self.on_sso_google_clicked)

        self.dexcom_btn = QPushButton("Dexcom")
        self.dexcom_btn.setObjectName("sso_dexcom_btn")
        self.dexcom_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.dexcom_btn.clicked.connect(self.on_sso_dexcom_clicked)

        self.fitbit_btn = QPushButton("Fitbit")
        self.fitbit_btn.setObjectName("sso_fitbit_btn")
        self.fitbit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fitbit_btn.clicked.connect(self.on_sso_fitbit_clicked)

        sso_layout.addWidget(self.google_btn)
        sso_layout.addWidget(self.dexcom_btn)
        sso_layout.addWidget(self.fitbit_btn)

        # Signup option
        signup_layout = QHBoxLayout()
        signup_text = QLabel("Don't have an account?")
        signup_text.setObjectName("subtitle_label")
        self.signup_btn = QPushButton("Sign Up")
        self.signup_btn.setObjectName("link_btn")
        self.signup_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.signup_btn.clicked.connect(self.on_signup_clicked)

        signup_layout.addWidget(signup_text)
        signup_layout.addWidget(self.signup_btn)
        signup_layout.addStretch()

        main_layout.addWidget(self.basic_login_btn)
        main_layout.addWidget(sso_label)
        main_layout.addLayout(sso_layout)
        main_layout.addStretch()
        main_layout.addLayout(signup_layout)

        # Set focus to username field
        self.username_input.setFocus()

    def on_basic_login_clicked(self):
        """Handle basic username/password login."""
        username = self.username_input.text().strip()
        password = self.password_input.text()

        # Validate inputs - require non-empty with minimum length
        if not username:
            logger.warning("Login attempt with empty username")
            QMessageBox.warning(
                self, "Validation Error", "Please enter your username or email."
            )
            return

        if len(username) < 3:
            logger.warning(
                f"Login attempt with username too short: {len(username)} chars"
            )
            QMessageBox.warning(
                self, "Validation Error", "Username must be at least 3 characters."
            )
            return

        if not password:
            logger.warning("Login attempt with empty password")
            QMessageBox.warning(self, "Validation Error", "Please enter your password.")
            return

        if len(password) < 4:
            logger.warning(
                f"Login attempt with password too short: {len(password)} chars"
            )
            QMessageBox.warning(
                self, "Validation Error", "Password must be at least 4 characters."
            )
            return

        # TODO: Implement actual authentication with backend
        # This is a placeholder implementation for demonstration purposes
        try:
            # Here we would call an authentication endpoint
            # and receive JWT tokens back

            # Placeholder tokens for demonstration
            access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            refresh_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

            # Store tokens securely
            self.token_store.store_tokens(access_token, refresh_token)

            logger.info(f"User '{username}' authenticated successfully with basic auth")

            # Emit success signal
            self.login_successful.emit()

            # Close dialog
            self.accept()

        except Exception as e:
            logger.error(f"Authentication failed for user '{username}': {e}")
            QMessageBox.critical(
                self, "Login Failed", f"Unable to authenticate: {str(e)}"
            )

    def on_sso_google_clicked(self):
        """Handle Google SSO login."""
        # TODO: Implement Google OAuth flow
        logger.info("Google SSO login not implemented - feature not yet available")

    def on_sso_dexcom_clicked(self):
        """Handle Dexcom SSO login."""
        # TODO: Implement Dexcom OAuth flow
        logger.info("Dexcom SSO login not implemented - feature not yet available")

    def on_sso_fitbit_clicked(self):
        """Handle Fitbit SSO login."""
        # TODO: Implement Fitbit OAuth flow
        logger.info("Fitbit SSO login not implemented - feature not yet available")

    def on_signup_clicked(self):
        """Handle signup navigation."""
        # TODO: Implement signup flow
        QMessageBox.information(
            self,
            "Signup Not Implemented",
            "Account creation is not implemented in this demo.\nPlease contact support.",
        )
        logger.info("User clicked signup button")
