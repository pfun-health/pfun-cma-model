import logging
from datetime import datetime, timezone
from typing import Any
from fastapi import (
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from guard import SecurityConfig
from pfun_common.settings import get_settings
from pfun_common.utils import setup_logging

# TODO: Uncomment this when IPInfoManager is implemented
# from guard.handlers.ipinfo_handler import IPInfoManager

# Configure logging
# FastAPI Guard uses its own logger hierarchy under "fastapi_guard" namespace
# This basic config is for the example app's own logging
logger = setup_logging(debug=get_settings().debug)


# Note: FastAPI Guard automatically sets up its own logging via the middleware
# with console output always enabled and optional file logging based on config


# ==================== Response Models ====================


class MessageResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Success",
                "details": {"info": "Additional information"},
            }
        }
    )

    message: str
    details: dict[str, Any] | None = None


class IPInfoResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ip": "8.8.8.8",
                "country": "US",
                "city": "Mountain View",
                "region": "California",
                "is_vpn": False,
                "is_cloud": True,
                "cloud_provider": "Google",
            }
        }
    )

    ip: str
    country: str | None = None
    city: str | None = None
    region: str | None = None
    is_vpn: bool | None = None
    is_cloud: bool | None = None
    cloud_provider: str | None = None


class StatsResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_requests": 1000,
                "blocked_requests": 50,
                "banned_ips": ["192.168.1.100", "10.0.0.50"],
                "rate_limited_ips": {"192.168.1.200": 5},
                "suspicious_activities": [
                    {"ip": "192.168.1.100", "reason": "SQL injection attempt"}
                ],
                "active_rules": {"rate_limit": 10, "auto_ban_threshold": 5},
            }
        }
    )

    total_requests: int
    blocked_requests: int
    banned_ips: list[str]
    rate_limited_ips: dict[str, int]
    suspicious_activities: list[dict[str, Any]]
    active_rules: dict[str, Any]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Access denied",
                "error_code": "ACCESS_DENIED",
                "timestamp": "2024-01-20T10:30:00Z",
            }
        }
    )

    detail: str
    error_code: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuthResponse(BaseModel):
    authenticated: bool
    user: str | None = None
    method: str
    permissions: list[str] = Field(default_factory=list)


class TestPayload(BaseModel):
    input: str | None = Field(None, description="Test input for XSS detection")
    query: str | None = Field(None, description="Test query for SQL injection")
    path: str | None = Field(None, description="Test path for traversal attacks")
    cmd: str | None = Field(None, description="Test command for injection")
    honeypot_field: str | None = Field(
        None, description="Hidden field for bot detection"
    )


# ==================== Custom Hooks ====================


async def custom_request_check(request: Request) -> Response | None:
    """Custom request validation hook."""
    # Example: Block requests with specific query parameters
    if "debug" in request.query_params and request.query_params["debug"] == "true":
        logger.warning(
            "Blocked debug request from %s",
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Debug mode not allowed"},
        )
    return None


async def custom_response_modifier(response: Response) -> Response:
    """Custom response modification hook."""
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ==================== Security Configuration ====================


def setup_security_config() -> SecurityConfig:
    """Produce a security configuration"""
    return SecurityConfig(
        # Proxy Configuration
        trusted_proxies=[
            "127.0.0.1",  # loopback (for local testing)
            "10.0.0.0/8",  # private network
            "168.235.67.32",  # custom proxy IP
            "100.115.68.73",  # tscale proxy
            "192.168.4.0/22",  # private network
            "172.20.0.0/16",  # docker private network
        ],
        trusted_proxy_depth=2,
        trust_x_forwarded_proto=True,
        # Geographical Filtering (requires ipinfo_token OR custom implementation)
        # geo_ip_handler=IPInfoManager("your_token_here"),  # Replace with actual token
        # blocked_countries=["XX"],  # Example: block country code XX
        # whitelist_countries=[],  # Allow all countries by default
        # Cloud Provider Blocking
        # block_cloud_providers={"AWS", "GCP", "Azure"},
        # User Agent Filtering
        blocked_user_agents=["badbot", "evil-crawler", "sqlmap"],
        # Rate Limiting
        enable_rate_limiting=True,
        rate_limit=50,  # nr requests allowed
        rate_limit_window=60,  # per 60 seconds
        # Auto-banning
        enable_ip_banning=True,
        auto_ban_threshold=5,
        auto_ban_duration=300 * 12,  # (300 seconds = 5 minutes) x 12 = 1hr
        # Penetration Detection
        enable_penetration_detection=True,
        # Redis Configuration
        enable_redis=True,
        redis_url=get_settings().redis_url,
        redis_prefix="fastapi_guard:",
        # HTTPS Enforcement
        enforce_https=not get_settings().debug,  # Set to True in production
        # Custom Hooks
        custom_request_check=custom_request_check,  # type: ignore
        custom_response_modifier=custom_response_modifier,  # type: ignore
        # Security Headers Configuration
        security_headers={
            "enabled": True,
            # Content Security Policy
            "csp": {
                "default-src": ["'self'", "https:"],
                "script-src": [
                    "'self'",
                    "buttons.github.io",
                    "cdn.jsdelivr.net",
                    "code.jquery.com",
                    "www.googletagmanager.com",
                    "w3c.github.io",  # includes a compiled trustedtypes module
                    "'sha256-k1Ro88UMqVxp8nnjIuKc9cc3fa0fpR3RvGneepaKUTU='",
                    "'sha256-ZswfTY7H35rbv8WC7NXBoiC7WNu86vSzCDChNWwZZDM='",
                    "'sha256-1jaaODSv58Wmh81mqxA9zy5j99zeo3PLat5wKQplemE='",
                    "'sha256-5ltlQRX7kwLdCPBtTiaRoP1nfpVtU3RWinabSOxIKy8='",
                ],
                # allow github button script
                "style-src": [
                    "'self'",
                    "https://cdn.jsdelivr.net",
                    "https://code.jquery.com",
                    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css",
                    "'sha256-us29Hziqlsx//QRFkxrVzQvfaIvMULlFZ6TCSNoKcP0='",
                    "'sha256-biLFinpqYMtWHmXfkA1BPeCY0/fNt46SAZ+BBk5YUog='",
                    "'sha256-4cUp5Ux03IE6rWs2UU4QWaYmO4rCpmieM4AGGVdgTG8='",
                ],
                "img-src": ["'self'", "data:", "https:"],
                "font-src": ["'self'", "https://fonts.gstatic.com"],
                # WebSocket support
                "connect-src": [
                    "'self'",
                    "wss://localhost:8001",
                    "https://api.github.com/repos/pfun-health/pfun-cma-model",
                    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js.map",
                    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css.map",
                ],
                # require trusted types for
                # #"require-trusted-types-for": ["'script'"],
            },
            # HTTP Strict Transport Security
            "hsts": {
                "max_age": 31536000,  # 1 year
                "include_subdomains": True,
                "preload": False,  # Set to True for production
            },
            # Custom security headers
            "frame_options": "SAMEORIGIN",
            "referrer_policy": "strict-origin-when-cross-origin",
            "permissions_policy": (
                "accelerometer=(), camera=(), geolocation=(), "
                "gyroscope=(), magnetometer=(), microphone=(), "
                "payment=(), usb=()"
            ),
            "custom": {
                "X-App-Name": "pfun-cma-model",
                "X-Security-Contact": "admin@pfun.me",
            },
        },
        # CORS Configuration (works alongside security headers)
        enable_cors=True,
        cors_allow_origins=[
            "https://localhost:8001",
            get_settings().production_server_url,
            f"https://{get_settings().ssl_server_host}",
            "https://pfun.one",
            "https://pfun.app",
            "https://pfun.run",
        ],
        cors_allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        cors_allow_headers=["*"],
        cors_allow_credentials=True,
        cors_expose_headers=["X-Total-Count"],
        cors_max_age=3600,
        # Logging Configuration
        # Console output is always enabled. File logging is optional.
        log_request_level="WARNING",  # Or None to disable request logging
        log_suspicious_level="WARNING",
        custom_log_file="logs/security.log",  # Or remove/set to None for console-only output
        # Excluded Paths
        exclude_paths=[
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/static",
            "/health",
        ],
        # Advanced Configuration
        passive_mode=get_settings().guard_passive_mode,  # Set to True for log-only mode
        # Agent Configuration (optional)
        # enable_agent=True,  # Set to True to enable telemetry
        # agent_api_key="api-test-key",
        # agent_project_id="test-project",
    )
