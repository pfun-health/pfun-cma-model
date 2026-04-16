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
)
from PyQt6.QtCore import Qt, QUrl, QUrlQuery
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from pfun_common.settings import get_settings

settings = get_settings()

env_fpath = Path(__file__).parent.parent.parent / ".env"
if not load_dotenv(env_fpath):
    raise RuntimeError(f"Failed to load environment variables (from {env_fpath})")
logging.debug(f"Loaded environment variables from {env_fpath}")
import supervisor  # noqa: F401 - we just want to ensure it's imported so that the process group is registered


class PFunHealthTipsDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PFun Health Tips Demo")
        self.setMinimumSize(800, 600)

        self.api_url = os.environ.get("PFUN_QT_GUI_API_URL", "https://127.0.0.1:8001")
        logging.debug(f"API URL: {self.api_url}")
        self.network_manager = QNetworkAccessManager(self)
        self.network_manager.finished.connect(self.on_request_finished)

        self.init_ui()

    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Header
        title_label = QLabel("PFun Health Tips Demo")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle_label = QLabel("Generate personalized health tips")
        subtitle_label.setStyleSheet("font-size: 14px; color: gray;")

        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addSpacing(10)

        # Input Section
        input_instruction = QLabel(
            "Please enter a query to generate personalized health tips.\nThe demo will generate a random health scenario if the input is left blank."
        )
        input_instruction.setStyleSheet("color: #0056b3;")
        main_layout.addWidget(input_instruction)

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText(
            "Example: I'm a relatively healthy individual who exercises most mornings before sunrise. What tips do you have for me?"
        )
        self.query_input.setFixedHeight(120)
        main_layout.addWidget(self.query_input)

        # Submit Button
        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(
            "background-color: #0d6efd; color: white; font-size: 16px; padding: 10px; border-radius: 5px;"
        )
        self.submit_btn.clicked.connect(self.on_submit)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.submit_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        main_layout.addSpacing(20)

        # Output Section Header
        output_title = QLabel("Output")
        output_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        output_subtitle = QLabel("PFun generated information")
        output_subtitle.setStyleSheet("font-size: 12px; color: gray;")

        main_layout.addWidget(output_title)
        main_layout.addWidget(output_subtitle)

        # Splitter for outputs
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Recommendations area
        recs_widget = QWidget()
        recs_layout = QVBoxLayout(recs_widget)
        recs_layout.setContentsMargins(0, 0, 5, 0)
        recs_header = QLabel("Recommendations")
        recs_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recs_header.setStyleSheet(
            "background-color: #0d6efd; color: white; font-weight: bold; padding: 5px; border-radius: 3px;"
        )
        self.recs_output = QTextBrowser()
        self.recs_output.setOpenExternalLinks(True)
        recs_layout.addWidget(recs_header)
        recs_layout.addWidget(self.recs_output)

        # Raw Output area
        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        raw_layout.setContentsMargins(5, 0, 0, 0)
        raw_header = QLabel("Raw output")
        raw_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_header.setStyleSheet(
            "background-color: #0dcaf0; color: black; font-weight: bold; padding: 5px; border-radius: 3px;"
        )
        self.raw_output = QTextBrowser()
        self.raw_output.setStyleSheet(
            "background-color: #f8f9fa; font-family: monospace;"
        )
        raw_layout.addWidget(raw_header)
        raw_layout.addWidget(self.raw_output)

        self.splitter.addWidget(recs_widget)
        self.splitter.addWidget(raw_widget)
        self.splitter.setSizes([400, 400])

        main_layout.addWidget(self.splitter)

        self.setCentralWidget(main_widget)

    def on_submit(self):
        query = self.query_input.toPlainText()

        # Disable button and update text
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("Loading...")

        # Clear previous outputs
        self.recs_output.clear()
        self.raw_output.clear()

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

        # Send POST request
        self.network_manager.post(request, b"{}")

    def on_request_finished(self, reply: QNetworkReply):
        # Re-enable button
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("Submit")

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

            # Format and set Recommendations
            recs_data = data.get("recommendations", {})
            recs_html = "<dl>"
            for key, value in recs_data.items():
                recs_html += (
                    f"<dt style='font-weight: bold;'>{key}</dt><dd>{value}</dd>"
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
    app = QApplication(sys.argv)

    # Optional: Set an application-wide style or palette
    app.setStyle("Fusion")

    window = PFunHealthTipsDemo()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
