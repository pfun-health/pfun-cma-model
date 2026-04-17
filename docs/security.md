---
icon: lucide/shield
---

# Security

## Overview

The PFun CMA Model API uses [`fastapi-guard`](https://pypi.org/project/fastapi-guard/) for comprehensive security middleware, including rate limiting, IP filtering, security headers, and penetration detection.

## Security Middleware

The `SecurityMiddleware` in `pfun_cma_model/security.py` provides:

### IP Filtering

- **Whitelist**: Trusted IPs (localhost, Tailscale)
- **Trusted Proxies**: Reverse proxy IP addresses
- **Cloud Provider Blocking**: Optional blocking of cloud provider IPs

### Rate Limiting

- Redis-backed rate limiting with configurable windows
- Automatic IP banning after threshold violations
- Configurable ban duration

### Security Headers

All responses include hardened security headers:

| Header | Value |
|--------|-------|
| `Content-Security-Policy` | Strict CSP with nonces |
| `Strict-Transport-Security` | HSTS with `max-age` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Permissions-Policy` | Restricted feature access |

### CORS

- Configurable allowed origins (localhost, Tailscale, pfun.one)
- Credential support for authenticated requests
- Configurable methods and max-age

## Excluded Paths

The following paths bypass security middleware:

- `/docs` — Swagger UI
- `/redoc` — ReDoc
- `/openapi.json` — OpenAPI schema
- `/health` — Health check
- `/static/*` — Static files

## Configuration

Security is configured in `pfun_cma_model/security.py` and driven by environment variables:

```python
# Key configuration via SECURITY_POLICY.ini
security_config = SecurityConfig(
    whitelist=["127.0.0.1", "::1"],
    rate_limit=100,
    rate_limit_window=60,
    auto_ban_threshold=10,
    auto_ban_duration=3600,
    enable_https_redirect=True,
)
```

## Testing

The project includes 59 security test cases across 17 test classes:

```bash
# Run all security tests
uv run pytest tests/test_security.py -v

# Run specific test class
uv run pytest tests/test_security.py::TestSecurityHeaders -v

# Run with coverage
uv run pytest tests/test_security.py --cov=pfun_cma_model.security
```

### Test Coverage Areas

| Area | Tests | Description |
|------|-------|-------------|
| Configuration Basics | 9 | Core security settings |
| Security Headers | 8 | CSP, HSTS, custom headers |
| CORS | 5 | Origins, methods, credentials |
| Excluded Paths | 4 | Docs, health, static files |
| Custom Hooks | 2 | Request/response hooks |
| IP Handling | 2 | Whitelist & proxy validation |
| API Endpoint Security | 3 | Localhost, user-agent, debug |
| Middleware Integration | 2 | Middleware attachment |
| Logging | 3 | Request & suspicious activity |
| Redis Cache | 3 | Rate limiting backend |
| Proxy Configuration | 2 | X-Forwarded-Proto, depth |
| Tailscale | 2 | Tailscale IP trust |
| CORS Origins | 3 | Specific origin validation |

For the full test specification, see [SECURITY_TESTS.md](https://github.com/pfun-health/pfun-cma-model/blob/main/SECURITY_TESTS.md).
