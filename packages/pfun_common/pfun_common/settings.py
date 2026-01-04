"""pfun_common settings module."""
import logging
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    debug: bool = False
    server_scheme: str = "http"
    server_host: str = "localhost"
    server_port: str | int = "8001"
    gradio_server_scheme: str = "http"
    gradio_server_host: str = "localhost"
    gradio_server_port: str | int = "7860"
    redis_user: str = "default"
    redis_password: str = ""
    redis_host: str = "localhost"
    redis_port: str | int = "6379"
    redis_db: str | int | bool = "0"
    redis_connection_string: str = ""
    perplexity_api_key: str = ""
    secret_key: str = "SChp11HMytLzSj3gaJQAJhq5sqc9Aicnz"
    google_cloud_project_id: str = "pfun-cma-model"
    google_cloud_location: str = "us-central1"

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="allow",
    )

    @field_validator("redis_connection_string", mode="after")
    @classmethod
    def parse_redis_connection_string(cls, v: str, info) -> str:
        """
        Parse REDIS_CONNECTION_STRING and override individual Redis settings.

        Supports URLs in the format: redis://[user[:password]@]host[:port][/db]
        """
        if not v:
            return v

        try:
            parsed = urlparse(v)

            # Extract host (required)
            if parsed.hostname:
                info.data["redis_host"] = parsed.hostname

            # Extract port (optional, defaults to 6379)
            if parsed.port:
                info.data["redis_port"] = parsed.port
            elif parsed.hostname:  # Only set default if we have a hostname
                info.data["redis_port"] = 6379

            # Extract username (optional, defaults to "default")
            if parsed.username:
                info.data["redis_user"] = parsed.username

            # Extract password (optional)
            if parsed.password:
                info.data["redis_password"] = parsed.password

            # Extract database number from path (optional, e.g., "/0")
            if parsed.path and parsed.path != "/":
                db_str = parsed.path.lstrip("/")
                if db_str:
                    try:
                        info.data["redis_db"] = int(db_str)
                    except ValueError:
                        pass  # Keep existing value if db is not a valid integer

        except Exception:
            pass  # Keep existing values if parsing fails

        return v

    @property
    def llm_gen_scenario_endpoint(self) -> str:
        """
        LLM generate-scenario endpoint URL.

        :param self: Description
        :return: Description
        :rtype: str
        """
        return f"{self.server_scheme}://{self.server_host}:{self.server_port}/llm/generate-scenario"

    @property
    def gradio_demo_endpoint(self) -> str:
        """
        Gradio demo endpoint URL.

        :param self: Description
        :return: Description
        :rtype: str
        """
        return f"{self.gradio_server_scheme}://{self.gradio_server_host}:{self.gradio_server_port}/gradio/"


def get_settings() -> Settings:
    """Initialize the settings object (dependency injection helper method)."""
    return Settings()
