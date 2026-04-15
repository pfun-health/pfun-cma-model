"""pfun_common settings module."""

import logging
from base64 import b64encode
from datetime import datetime
from secrets import token_urlsafe
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pfun_llm.backend.ollama import (
    OllamaDefaultModel,
    _OLLAMA_DEFAULT_MODEL,
)


def generate_default_secret_key() -> str:
    """Generate a default secret key. Combines a timestamp and random token for uniqueness."""
    timestamp = datetime.now().isoformat().encode("utf-8")
    timestamp_nonce = b64encode(timestamp).decode("utf-8")
    rand_token = token_urlsafe(16)  # 16 bytes of randomness
    return f"{timestamp_nonce}-{rand_token}"


class Settings(BaseSettings):
    """Settings for the pfun-cma-model application. Values can be overridden via environment variables or a .env file."""

    debug: bool = False
    #: Whether to enable debug mode (e.g. more verbose logging, auto-reload). Can also be set via the DEBUG environment variable.

    logger_name: str = "pfun-app"
    #: The name of the project-level logger.

    guard_passive_mode: bool = False
    #: Whether to enable passive mode for the security guard (i.e. log-only mode without blocking). Can also be set via the GUARD_PASSIVE_MODE environment variable.

    server_scheme: str = "http"
    #: The URL scheme for the server (e.g. "http" or "https"). Can also be set via the SERVER_SCHEME environment variable.

    server_host: str = "localhost"
    #: The host for the server to bind to (e.g. "localhost" or "127.0.0.1").

    server_port: str | int = "8001"
    #: The port for the server to listen on. Can also be set via the SERVER_PORT environment variable.

    production_server_url: str = "https://cloud.tail38611b.ts.net"
    #: The public URL for the production server (e.g. for constructing redirect URIs). Can also be set via the PRODUCTION_SERVER_URL environment variable.

    ssl_server_host: str = "cloud.tail38611b.ts.net"
    #: The host for the SSL server (e.g. for constructing redirect URIs). Can also be set via the SSL_SERVER_HOST environment variable.

    redis_user: str = "default"
    #: The username for Redis authentication. Can also be set via the REDIS_USER environment variable.

    redis_password: str = ""
    #: The password for Redis authentication. Can also be set via the REDIS_PASSWORD environment variable.

    redis_host: str = "localhost"
    #: The host for the Redis server. Can also be set via the REDIS_HOST environment variable.

    redis_port: str | int = "6379"
    #: The port for the Redis server. Can also be set via the REDIS_PORT environment variable.

    redis_db: str | int | bool = "0"
    #: The database number for Redis. Can also be set via the REDIS_DB environment variable.

    redis_connection_string: str = ""
    #: A connection string for Redis. Can also be set via the REDIS_CONNECTION_STRING environment variable.

    perplexity_api_key: str = ""
    google_api_key: str = ""
    ollama_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    ollama_model: OllamaDefaultModel = _OLLAMA_DEFAULT_MODEL
    llm_backend: Literal["google", "perplexity", "ollama", "openai"] = "ollama"
    secret_key: str = Field(default_factory=generate_default_secret_key)
    google_cloud_project_id: str = "pfun-cma-model"
    google_cloud_location: str = "us-central1"
    google_cloud_client_id: str = ""
    google_cloud_client_secret: str = ""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="allow",
    )

    @field_validator("server_port", "redis_port", mode="before")
    @classmethod
    def convert_port_to_int(cls, v: str | int) -> int:
        if isinstance(v, str):
            return int(v)
        return v

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
            # initially, strip any surrounding whitespace
            v = v.strip()
            # somewhat intelligently access the URL itself (without extra params)
            v = [piece for piece in v.split(" ") if "redis://" in piece][0].strip()
            logging.debug("Parsing REDIS_CONNECTION_STRING: %s", v)

            # parse the URL
            parsed = urlparse(v)
            logging.debug("Parsed Redis URL: %s", parsed)

            # Extract host (required)
            if parsed.hostname:
                logging.debug("Parsed Redis host: %s", parsed.hostname)
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

            logging.debug(
                "Parsed Redis settings: host=%s, port=%s, user=%s, db=%s",
                info.data.get("redis_host"),
                info.data.get("redis_port"),
                info.data.get("redis_user"),
                info.data.get("redis_db"),
            )
        except Exception as exc:
            logging.warning(
                "Failed to parse REDIS_CONNECTION_STRING: %s", v, exc_info=exc
            )
            logging.debug("No such REDIS_CONNECTION_STRING: %s", v, exc_info=exc)
            pass  # Keep existing values if parsing fails

        return v

    @property
    def server_url(self) -> str:
        """
        Construct the server URL based on the scheme, host, and port.
        :return: Server URL
        :rtype: str
        """
        return f"{self.server_scheme}://{self.server_host}:{self.server_port}"

    @property
    def redis_url(self) -> str:
        """
        Construct the Redis connection URL.

        :return: Redis connection URL
        :rtype: str
        """
        user = self.redis_user
        password = self.redis_password
        host = self.redis_host
        port = self.redis_port
        db = self.redis_db if isinstance(self.redis_db, int) else 0

        if password:
            return f"redis://{user}:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"


def get_settings() -> Settings:
    """Initialize the settings object (dependency injection helper method)."""
    return Settings()
