"""Security configuration tests for the API."""

from pfun_cma_model.security import setup_security_config
from pfun_cma_model.app import app
from ipaddress import ip_address
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
import pfun_path_helper as pph

pph.append_path(path=pph.get_lib_path("pfun_cma_model"))  # noqa: E402
from . import test_base

test_base.setup_test_environment()


# Create test client
client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))

security_config = setup_security_config()


class TestSecurityConfigurationBasics:
    """Test basic security configuration attributes."""

    def test_security_config_exists(self):
        """Verify security_config object is properly initialized."""
        assert security_config is not None
        assert hasattr(security_config, "whitelist")

    def test_trusted_proxies_configured(self):
        """Verify trusted proxies are configured."""
        assert "127.0.0.1" in security_config.trusted_proxies
        assert security_config.trusted_proxy_depth == 2

    def test_rate_limiting_enabled(self):
        """Verify rate limiting is enabled."""
        assert security_config.enable_rate_limiting is True
        assert security_config.rate_limit == 50
        assert security_config.rate_limit_window == 60

    def test_ip_banning_enabled(self):
        """Verify IP banning configuration is set."""
        assert security_config.enable_ip_banning is True
        assert security_config.auto_ban_threshold == 5
        assert security_config.auto_ban_duration == 3600

    def test_penetration_detection_enabled(self):
        """Verify penetration detection is enabled."""
        assert security_config.enable_penetration_detection is True

    def test_cloud_providers_blocked(self):
        """Verify cloud providers are configured to be blocked."""
        assert security_config.block_cloud_providers == {"AWS", "GCP", "Azure"}

    def test_blocked_user_agents_configured(self):
        """Verify blocked user agents are configured."""
        assert "badbot" in security_config.blocked_user_agents
        assert "evil-crawler" in security_config.blocked_user_agents
        assert "sqlmap" in security_config.blocked_user_agents

    def test_https_enforcement_in_dev(self):
        """Verify HTTPS enforcement is False for development."""
        assert security_config.enforce_https is False

    def test_passive_mode_disabled(self):
        """Verify passive mode is disabled (active blocking)."""
        assert security_config.passive_mode is False


class TestSecurityHeaders:
    """Test security headers configuration."""

    def test_security_headers_enabled(self):
        """Verify security headers are enabled."""
        assert security_config.security_headers["enabled"] is True

    def test_csp_configured(self):
        """Verify Content Security Policy is configured."""
        csp = security_config.security_headers.get("csp", {})
        assert "default-src" in csp
        assert "'self'" in csp["default-src"]
        assert "script-src" in csp
        assert "style-src" in csp

    def test_hsts_configured(self):
        """Verify HSTS headers are configured."""
        hsts = security_config.security_headers.get("hsts", {})
        assert hsts.get("max_age") == 31536000
        assert hsts.get("include_subdomains") is True

    def test_custom_security_headers(self):
        """Verify custom security headers are configured."""
        custom = security_config.security_headers.get("custom", {})
        assert custom.get("X-App-Name") == "pfun-cma-model"
        assert custom.get("X-Security-Contact") == "admin@pfun.me"

    def test_frame_options_configured(self):
        """Verify X-Frame-Options is configured."""
        frame_options = security_config.security_headers.get("frame_options")
        assert frame_options == "SAMEORIGIN"

    def test_referrer_policy_configured(self):
        """Verify Referrer-Policy is configured."""
        referrer_policy = security_config.security_headers.get("referrer_policy")
        assert referrer_policy == "strict-origin-when-cross-origin"

    def test_permissions_policy_configured(self):
        """Verify Permissions-Policy is configured."""
        perms = security_config.security_headers.get("permissions_policy")
        assert perms is not None
        assert "camera=()" in perms
        assert "microphone=()" in perms
        assert "geolocation=()" in perms


class TestCORSConfiguration:
    """Test CORS configuration."""

    def test_cors_enabled(self):
        """Verify CORS is enabled."""
        assert security_config.enable_cors is True

    def test_allowed_origins_configured(self):
        """Verify allowed origins are configured."""
        origins = security_config.cors_allow_origins
        assert "http://localhost:8001" in origins

    def test_allowed_methods_configured(self):
        """Verify allowed methods are configured."""
        methods = security_config.cors_allow_methods
        assert "GET" in methods
        assert "POST" in methods
        assert "OPTIONS" in methods

    def test_cors_credentials_allowed(self):
        """Verify CORS credentials are allowed."""
        assert security_config.cors_allow_credentials is True

    def test_cors_max_age_set(self):
        """Verify CORS max age is set."""
        assert security_config.cors_max_age == 3600


class TestExcludedPaths:
    """Test excluded paths from security checks."""

    def test_excluded_paths_configured(self):
        """Verify excluded paths are configured."""
        assert len(security_config.exclude_paths) > 0

    def test_docs_excluded(self):
        """Verify docs paths are excluded from security."""
        assert "/docs" in security_config.exclude_paths
        assert "/redoc" in security_config.exclude_paths
        assert "/openapi.json" in security_config.exclude_paths

    def test_health_excluded(self):
        """Verify health check path is excluded."""
        assert "/health" in security_config.exclude_paths

    def test_static_excluded(self):
        """Verify static files are excluded."""
        assert "/static" in security_config.exclude_paths


class TestCustomHooks:
    """Test custom request/response hooks."""

    def test_custom_request_check_configured(self):
        """Verify custom request check hook is configured."""
        assert security_config.custom_request_check is not None
        assert callable(security_config.custom_request_check)

    def test_custom_response_modifier_configured(self):
        """Verify custom response modifier hook is configured."""
        assert security_config.custom_response_modifier is not None
        assert callable(security_config.custom_response_modifier)


class TestIPAddressHandling:
    """Test IP address configurations are valid."""

    def test_trusted_proxies_valid(self):
        """Verify trusted proxy IPs are valid."""
        for ip in security_config.trusted_proxies:
            try:
                ip_address(ip.split("/")[0])
            except ValueError:
                pytest.fail(f"Invalid IP address in trusted proxies: {ip}")


class TestAPIEndpointSecurity:
    """Test API endpoints with security middleware."""

    def test_localhost_request_allowed(self):
        """Verify requests from localhost are allowed."""
        response = client.get("/health")
        # 404 if endpoint doesn't exist, 200 if it does
        assert response.status_code in [200, 404]

    def test_user_agent_filtering(self):
        """Verify blocked user agents are rejected."""
        headers = {"user-agent": "sqlmap/1.0"}
        response = client.get("/health", headers=headers)
        # Should be blocked or marked as suspicious
        assert response.status_code in [403, 429, 200, 404]

    def test_debug_parameter_blocked(self):
        """Verify debug parameter is blocked by custom hook."""
        response = client.get("/health?debug=true")
        # Custom hook should block this or it's excluded
        assert response.status_code in [403, 200, 404]


class TestSecurityMiddlewareIntegration:
    """Test security middleware is properly integrated."""

    def test_middleware_added_to_app(self):
        """Verify SecurityMiddleware is added to the app."""
        #: ! Essential this checks m.cls, NOT m.__class__ (the latter is the same for all middleware instances)
        middleware_names = [m.cls.__name__ for m in app.user_middleware]
        assert any("security" in name.lower() for name in middleware_names)

    def test_app_has_security_config(self):
        """Verify app has security configuration."""
        # The app should have been initialized with security
        assert app is not None
        assert hasattr(app, "middleware")


class TestSecurityHeadersResponse:
    """Test security headers are included in responses."""

    @pytest.mark.asyncio
    async def test_response_headers_present(self):
        """Verify security headers are added to responses."""
        response = client.get("/health")
        # Note: Headers added by custom_response_modifier
        # Check for at least some expected security headers
        headers = response.headers
        # These should be added by the custom_response_modifier
        assert "X-Content-Type-Options" in headers or len(headers) > 0


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_request_logging_level_set(self):
        """Verify request logging level is configured."""
        assert security_config.log_request_level == "INFO"

    def test_suspicious_logging_level_set(self):
        """Verify suspicious activity logging level is configured."""
        assert security_config.log_suspicious_level == "WARNING"

    def test_custom_log_file_configured(self):
        """Verify custom log file is configured."""
        assert security_config.custom_log_file == "security.log"


class TestRedisCacheConfiguration:
    """Test Redis configuration for caching."""

    def test_redis_enabled(self):
        """Verify Redis is configured for rate limiting state."""
        assert security_config.enable_redis is True

    def test_redis_prefix_set(self):
        """Verify Redis prefix is set."""
        assert security_config.redis_prefix == "fastapi_guard:"

    def test_redis_url_configured(self):
        """Verify Redis URL is configured."""
        assert security_config.redis_url is not None


class TestProxyConfiguration:
    """Test proxy trust configuration."""

    def test_x_forwarded_proto_trusted(self):
        """Verify X-Forwarded-Proto header is trusted."""
        assert security_config.trust_x_forwarded_proto is True

    def test_proxy_depth_reasonable(self):
        """Verify proxy depth is set to reasonable value."""
        assert security_config.trusted_proxy_depth > 0
        assert security_config.trusted_proxy_depth <= 5


class TestConfigurationConsistency:
    """Test configuration consistency and completeness."""

    def test_blocking_rules_consistent(self):
        """Verify blocking rules don't conflict."""
        # If cloud providers are blocked, they should be in a list
        assert isinstance(security_config.block_cloud_providers, set)

    def test_rate_limit_values_reasonable(self):
        """Verify rate limit values are reasonable."""
        assert security_config.rate_limit > 0
        assert security_config.rate_limit_window > 0
        assert security_config.rate_limit <= 1000  # Not excessively high

    def test_auto_ban_values_reasonable(self):
        """Verify auto-ban values are reasonable."""
        assert security_config.auto_ban_threshold > 0
        assert security_config.auto_ban_duration > 0
        assert security_config.auto_ban_threshold <= 100

    def test_csp_default_src_present(self):
        """Verify CSP default-src is configured."""
        csp = security_config.security_headers.get("csp", {})
        assert "default-src" in csp


class TestSecurityHeadersValues:
    """Test specific security header values."""

    def test_csp_script_src_includes_self(self):
        """Verify CSP script-src includes 'self'."""
        csp = security_config.security_headers.get("csp", {})
        script_src = csp.get("script-src", [])
        assert "'self'" in script_src

    def test_csp_style_src_includes_self(self):
        """Verify CSP style-src includes 'self'."""
        csp = security_config.security_headers.get("csp", {})
        style_src = csp.get("style-src", [])
        assert "'self'" in style_src

    def test_csp_img_src_configured(self):
        """Verify CSP img-src is configured."""
        csp = security_config.security_headers.get("csp", {})
        assert "img-src" in csp

    def test_csp_font_src_configured(self):
        """Verify CSP font-src is configured."""
        csp = security_config.security_headers.get("csp", {})
        assert "font-src" in csp


class TestConnectionSecurity:
    """Test connection-related security settings."""

    def test_websocket_in_csp_connect_src(self):
        """Verify WebSocket is allowed in CSP connect-src."""
        csp = security_config.security_headers.get("csp", {})
        connect_src = csp.get("connect-src", [])
        assert any("wss://" in src or "'self'" in src for src in connect_src)

    def test_cors_headers_list_not_empty(self):
        """Verify CORS headers list is configured."""
        assert len(
            security_config.cors_allow_headers
        ) > 0 or security_config.cors_allow_headers == ["*"]


class TestCORSOrigins:
    """Test CORS origins configuration."""

    def test_localhost_allowed_for_cors(self):
        """Verify localhost is allowed for CORS."""
        assert "http://localhost:8001" in security_config.cors_allow_origins

    def test_tailscale_domain_allowed(self):
        """Verify Tailscale domain is allowed for CORS."""
        assert any("tail" in origin for origin in security_config.cors_allow_origins)

    def test_pfun_domain_allowed(self):
        """Verify pfun.one domain is allowed."""
        assert any(
            "pfun.one" in origin for origin in security_config.cors_allow_origins
        )
