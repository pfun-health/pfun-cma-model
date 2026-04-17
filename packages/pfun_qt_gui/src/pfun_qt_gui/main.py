import logging
import sys
import os
from pathlib import Path
import json
from dotenv import load_dotenv
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QTextBrowser,
    QMessageBox,
    QSplitter,
    QSizePolicy,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, QUrl, QUrlQuery, QSize
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from pfun_common.settings import get_settings
from pfun_common.logs import setup_logging
from pfun_qt_gui.loading_overlay import StartupLoadingOverlay, SubmitLoadingOverlay
from pfun_qt_gui.mixins import AutoRetryMixin, HealthCheckMixin
from pfun_qt_gui.theme import (
    get_theme,
    platform_tier,
    PlatformTier,
    scale,
)

logger = setup_logging(logger_name="pfun-qt-gui-app")
settings = get_settings()

# env_fpath is the path to the .env file for this application
env_fpath = Path(__file__).parent.parent.parent / ".env"
# root_dir is the directory containing the top-level .env file
root_dir = Path(env_fpath).parent.parent.parent

# import supervisor so that the process group is registered
import supervisor  # noqa: F401

# Breakpoint for switching from horizontal to vertical splitter
_NARROW_BREAKPOINT = 700


class PFunHealthTipsDemo(AutoRetryMixin, HealthCheckMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PFun Health Tips Demo")

        # Apply responsive minimum size based on platform
        tier = platform_tier()
        if tier == PlatformTier.MOBILE:
            self.setMinimumSize(QSize(320, 480))
        elif tier == PlatformTier.TV:
            self.setMinimumSize(QSize(1200, 800))
        else:
            self.setMinimumSize(QSize(640, 480))

        self.load_env()  # load env vars from .env file
        self.api_url = os.environ.get("PFUN_QT_GUI_API_URL", "https://127.0.0.1:8001")
        logging.debug(f"API URL: {self.api_url}")
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_request_finished)

        # Apply the global theme stylesheet
        theme = get_theme()
        self.setStyleSheet(theme.stylesheet())

        # start the UI
        self.init_ui()

        # Active overlay references (at most one visible at a time)
        self._loading_overlay: StartupLoadingOverlay | SubmitLoadingOverlay | None = (
            None
        )
        self._submit_overlay: SubmitLoadingOverlay | None = None

        # Initialise auto-retry state (from AutoRetryMixin)
        self.init_retry_state()

        # Track current splitter orientation for adaptive layout
        self._current_splitter_horizontal = True

        # Show startup loading overlay and begin health polling
        self._loading_overlay = StartupLoadingOverlay(self.centralWidget())
        self._loading_overlay.setGeometry(self.centralWidget().rect())
        self._loading_overlay.show()
        self._loading_overlay.raise_()

        self.start_health_check()

    def load_env(self):
        """Load environment variables from .env file."""
        if not load_dotenv(env_fpath):
            raise RuntimeError(
                f"Failed to load environment variables (from {env_fpath})"
            )
        logging.debug(f"Loaded environment variables from {env_fpath}")

    # ----- resize handling to keep overlay covering the window -----

    def resizeEvent(self, event):
        super().resizeEvent(event)
        central_rect = self.centralWidget().rect()
        if hasattr(self, "_loading_overlay") and self._loading_overlay is not None:
            self._loading_overlay.setGeometry(central_rect)
        if hasattr(self, "_submit_overlay") and self._submit_overlay is not None:
            self._submit_overlay.setGeometry(central_rect)

        # Adaptive splitter orientation: switch to vertical on narrow windows
        if hasattr(self, "splitter"):
            w = event.size().width()
            if w < scale(_NARROW_BREAKPOINT) and self._current_splitter_horizontal:
                self.splitter.setOrientation(Qt.Orientation.Vertical)
                self._current_splitter_horizontal = False
            elif w >= scale(_NARROW_BREAKPOINT) and not self._current_splitter_horizontal:
                self.splitter.setOrientation(Qt.Orientation.Horizontal)
                self._current_splitter_horizontal = True

        # Dynamically adjust query input height based on window height
        if hasattr(self, "query_input"):
            h = event.size().height()
            # Allocate 12–18% of window height to the input area
            input_height = max(scale(80), min(int(h * 0.15), scale(200)))
            self.query_input.setMinimumHeight(scale(60))
            self.query_input.setMaximumHeight(input_height)

    def init_ui(self):
        """Build the complete responsive UI."""
        # Scroll area wrapping the main content so everything is reachable
        # on very small screens / mobile
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(
            scale(24), scale(20), scale(24), scale(20)
        )
        main_layout.setSpacing(scale(4))

        # ── Header ──────────────────────────────────────────────────────
        title_label = QLabel("PFun Health Tips")
        title_label.setObjectName("title_label")
        title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )

        subtitle_label = QLabel("Generate personalised health tips powered by AI")
        subtitle_label.setObjectName("subtitle_label")
        subtitle_label.setWordWrap(True)
        subtitle_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )

        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addSpacing(scale(12))

        # ── Divider ─────────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(
            f"background-color: {get_theme().palette.border}; max-height: 1px;"
        )
        main_layout.addWidget(divider)
        main_layout.addSpacing(scale(12))

        # ── Input section ───────────────────────────────────────────────
        input_instruction = QLabel(
            "Enter a query below, or leave blank for a random health scenario."
        )
        input_instruction.setObjectName("input_instruction")
        input_instruction.setWordWrap(True)
        input_instruction.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        main_layout.addWidget(input_instruction)
        main_layout.addSpacing(scale(4))

        self.query_input = QTextEdit()
        self.query_input.setObjectName("query_input")
        self.query_input.setPlaceholderText(
            "Example: I'm a relatively healthy individual who exercises most "
            "mornings before sunrise. What tips do you have for me?"
        )
        # Flexible height — will be dynamically constrained in resizeEvent
        self.query_input.setMinimumHeight(scale(60))
        self.query_input.setMaximumHeight(scale(140))
        self.query_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        main_layout.addWidget(self.query_input)
        main_layout.addSpacing(scale(8))

        # ── Submit button ───────────────────────────────────────────────
        self.submit_btn = QPushButton("✦  Submit")
        self.submit_btn.setObjectName("submit_btn")
        self.submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.submit_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self.submit_btn.clicked.connect(self.on_submit)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.submit_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        main_layout.addSpacing(scale(16))

        # ── Output header ───────────────────────────────────────────────
        output_title = QLabel("Output")
        output_title.setObjectName("output_title")
        output_title.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        output_subtitle = QLabel("PFun generated information")
        output_subtitle.setObjectName("output_subtitle")
        output_subtitle.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )

        main_layout.addWidget(output_title)
        main_layout.addWidget(output_subtitle)
        main_layout.addSpacing(scale(8))

        # ── Splitter (recommendations | raw JSON) ───────────────────────
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(scale(6))

        # -- Recommendations pane --
        recs_widget = QWidget()
        recs_layout = QVBoxLayout(recs_widget)
        recs_layout.setContentsMargins(0, 0, scale(4), 0)
        recs_layout.setSpacing(scale(6))

        recs_header = QLabel("Recommendations")
        recs_header.setObjectName("section_header_recs")
        recs_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recs_header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.recs_output = QTextBrowser()
        self.recs_output.setObjectName("recs_output")
        self.recs_output.setOpenExternalLinks(True)
        self.recs_output.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        recs_layout.addWidget(recs_header)
        recs_layout.addWidget(self.recs_output)

        # -- Raw output pane --
        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        raw_layout.setContentsMargins(scale(4), 0, 0, 0)
        raw_layout.setSpacing(scale(6))

        raw_header = QLabel("Raw output")
        raw_header.setObjectName("section_header_raw")
        raw_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.raw_output = QTextBrowser()
        self.raw_output.setObjectName("raw_output")
        self.raw_output.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        raw_layout.addWidget(raw_header)
        raw_layout.addWidget(self.raw_output)

        self.splitter.addWidget(recs_widget)
        self.splitter.addWidget(raw_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.splitter, 1)  # stretch factor 1 → fills space

        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)

    def _show_submit_overlay(self) -> None:
        """Show the generation-in-progress overlay."""
        self._submit_overlay = SubmitLoadingOverlay(self.centralWidget())
        self._submit_overlay.setGeometry(self.centralWidget().rect())
        self._submit_overlay.show()
        self._submit_overlay.raise_()

    def _dismiss_submit_overlay(self) -> None:
        """Fade out and clean up the submit overlay."""
        if self._submit_overlay is not None:
            self._submit_overlay.fade_out_and_remove()
            self._submit_overlay = None

    def on_submit(self):
        # Block submission while the server is not yet healthy
        if not self.server_healthy:
            QMessageBox.warning(
                self,
                "Server Unavailable",
                "The server is not yet available. Please wait for the connection to be established.",
            )
            return

        query = self.query_input.toPlainText()

        # Disable button and update text
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("⏳  Generating…")

        # Clear previous outputs
        self.recs_output.clear()
        self.raw_output.clear()

        # Show generation overlay
        self._show_submit_overlay()

        # Build URL with query parameter
        url_str = f"{self.api_url}/llm/generate-scenario"
        url = QUrl(url_str)
        # We need to send as a POST request (even with empty body, but query in params if needed)
        query_params = QUrlQuery()
        query_params.addQueryItem("prompt", query)
        url.setQuery(query_params)

        request = QNetworkRequest(url)
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )

        # Register request for auto-retry (via AutoRetryMixin)
        self.start_retryable_request(request)

        # Send POST request
        self.network_manager.post(request, b"{}")

    def on_request_finished(self, reply: QNetworkReply):
        # Delegate HTTP-500 retries to AutoRetryMixin
        if self.should_retry(reply):
            return

        # ---------- No retry: process the response normally ----------
        status_code = reply.attribute(
            QNetworkRequest.Attribute.HttpStatusCodeAttribute
        )

        # Dismiss the submit overlay
        self._dismiss_submit_overlay()

        # Re-enable button
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("✦  Submit")

        # Clear stored request (AutoRetryMixin)
        self.clear_pending_request()

        if reply.error() != QNetworkReply.NetworkError.NoError:
            error_msg = reply.errorString()
            # Try to read body for more specific error
            body = reply.readAll().data().decode("utf-8")
            if body:
                try:
                    err_json = json.loads(body)
                    if "detail" in err_json:
                        error_msg = err_json["detail"]
                except Exception:
                    pass

            # If this was a 500 that exhausted retries, give a clearer message
            if status_code == 500:
                error_msg = (
                    f"Server returned an internal error after "
                    f"{self.retry_count} retry attempt(s).\n\n{error_msg}"
                )

            logging.error(f"Network error: {error_msg}")
            QMessageBox.critical(
                self, "Error", f"Failed to generate scenario:\n{error_msg}"
            )
            reply.deleteLater()
            return

        # Parse JSON response
        data_bytes = reply.readAll().data()
        try:
            data_str = data_bytes.decode("utf-8")
            data = json.loads(data_str)

            # Format and set Recommendations with themed HTML
            recs_data = data.get("recommendations", {})
            theme = get_theme()
            p = theme.palette
            recs_html = (
                f"<style>"
                f"body {{ color: {p.text_primary}; font-family: {theme.font_family}; "
                f"font-size: {scale(theme.font_size_body)}px; }}"
                f"dt {{ font-weight: 700; color: {p.accent_hover}; "
                f"margin-top: {scale(12)}px; margin-bottom: {scale(4)}px; }}"
                f"dd {{ color: {p.text_secondary}; margin-left: {scale(8)}px; "
                f"margin-bottom: {scale(8)}px; line-height: 1.5; }}"
                f"</style><dl>"
            )
            for key, value in recs_data.items():
                recs_html += (
                    f"<dt>{key}</dt><dd>{value}</dd>"
                )
            recs_html += "</dl>"
            self.recs_output.setHtml(recs_html)

            # Set Raw Output
            pretty_json = json.dumps(data, indent=2)
            self.raw_output.setPlainText(pretty_json)

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Failed to parse response from server.")
        except Exception as e:
            logging.exception(
                "Unexpected error while processing response: %s", str(e), exc_info=True
            )
            QMessageBox.critical(
                self, "Error", f"An unexpected error occurred:\n{str(e)}"
            )

        reply.deleteLater()


def main():
    # Enable high-DPI scaling (important for crisp rendering everywhere)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)

    # Optional: Set an application-wide style or palette
    app.setStyle("Fusion")

    # Apply the global theme stylesheet at the app level as well
    theme = get_theme()
    app.setStyleSheet(theme.stylesheet())

    window = PFunHealthTipsDemo()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
