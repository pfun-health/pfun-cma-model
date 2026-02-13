# Security Configuration Tests

Comprehensive pytest test suite for verifying the API security configuration in `pfun_cma_model/security.py`.

## Test File Location

`tests/test_security.py` - 392 lines, 59 test cases organized into 17 test classes.

## Test Coverage

### 1. **TestSecurityConfigurationBasics** (9 tests)
Verifies core security configuration attributes:
- Security config initialization
- Whitelist and trusted proxies
- Rate limiting settings
- IP banning configuration
- Penetration detection
- Cloud provider blocking
- User agent filtering
- HTTPS enforcement
- Passive mode status

### 2. **TestSecurityHeaders** (8 tests)
Tests security headers configuration:
- Headers enabled state
- Content Security Policy (CSP)
- HTTP Strict Transport Security (HSTS)
- Custom security headers
- Frame options (X-Frame-Options)
- Referrer Policy
- Permissions Policy

### 3. **TestCORSConfiguration** (5 tests)
Verifies CORS settings:
- CORS enabled
- Allowed origins
- Allowed methods
- Credentials allowance
- Max age setting

### 4. **TestExcludedPaths** (4 tests)
Tests excluded paths from security checks:
- Documentation endpoints
- OpenAPI schema
- Health checks
- Static files

### 5. **TestCustomHooks** (2 tests)
Validates custom request/response hooks:
- Custom request check hook
- Custom response modifier hook

### 6. **TestIPAddressHandling** (2 tests)
Validates IP address configurations:
- Whitelist IP validity
- Trusted proxy IP validity

### 7. **TestAPIEndpointSecurity** (3 tests)
Tests API endpoint security behavior:
- Localhost requests allowed
- User agent filtering
- Debug parameter blocking

### 8. **TestSecurityMiddlewareIntegration** (2 tests)
Verifies middleware integration:
- SecurityMiddleware added to app
- App has security configuration

### 9. **TestSecurityHeadersResponse** (1 test)
Tests security headers in HTTP responses:
- Response headers present

### 10. **TestLoggingConfiguration** (3 tests)
Validates logging setup:
- Request logging level
- Suspicious activity logging level
- Custom log file configuration

### 11. **TestRedisCacheConfiguration** (3 tests)
Tests Redis configuration:
- Redis enabled for rate limiting
- Redis prefix set
- Redis URL configured

### 12. **TestProxyConfiguration** (2 tests)
Verifies proxy trust settings:
- X-Forwarded-Proto trusted
- Proxy depth reasonable

### 13. **TestConfigurationConsistency** (4 tests)
Checks configuration consistency:
- Blocking rules consistency
- Rate limit values reasonableness
- Auto-ban values reasonableness
- CSP default-src present

### 14. **TestSecurityHeadersValues** (4 tests)
Tests specific security header values:
- CSP script-src includes 'self'
- CSP style-src includes 'self'
- CSP img-src configured
- CSP font-src configured

### 15. **TestConnectionSecurity** (2 tests)
Verifies connection-related security:
- WebSocket in CSP connect-src
- CORS headers configured

### 16. **TestTailscaleConfiguration** (2 tests)
Tests Tailscale-specific security setup:
- Tailscale IP in whitelist
- Tailscale IP in trusted proxies

### 17. **TestCORSOrigins** (3 tests)
Validates CORS allowed origins:
- Localhost allowed
- Tailscale domain allowed
- pfun.one domain allowed

## Running the Tests

### Using pytest (with dependencies installed):
```bash
pytest tests/test_security.py -v
```

### Run specific test class:
```bash
pytest tests/test_security.py::TestSecurityHeaders -v
```

### Run specific test:
```bash
pytest tests/test_security.py::TestSecurityHeaders::test_csp_configured -v
```

### Run with coverage:
```bash
pytest tests/test_security.py --cov=pfun_cma_model.security
```

## Test Framework

- **Framework**: pytest
- **Client**: FastAPI TestClient
- **Mock Library**: unittest.mock
- **Async Support**: pytest-asyncio

## Key Features

✅ Comprehensive configuration validation
✅ IP address validity checks
✅ Security header verification
✅ CORS configuration testing
✅ Rate limiting settings validation
✅ Middleware integration tests
✅ Custom hooks verification
✅ Logging configuration checks
✅ Redis cache configuration tests
✅ Proxy trust settings validation
✅ Tailscale integration verification

## Configuration Sections Tested

1. **IP Filtering**: Whitelist, trusted proxies, IP validity
2. **Rate Limiting**: Enabled, limits, window sizes
3. **IP Banning**: Threshold, duration, enabled state
4. **Security Headers**: CSP, HSTS, custom headers, permissions policy
5. **CORS**: Origins, methods, credentials, max-age
6. **User Agent**: Blocked agents list
7. **Cloud Providers**: Blocking list
8. **Logging**: Levels, file output
9. **Redis**: URL, prefix, enabled state
10. **Proxy**: Trust settings, depth
11. **Custom Hooks**: Request check, response modifier
12. **Excluded Paths**: Documentation, health, static files

## Notes

- Tests follow the existing test structure from `tests/test_app.py`
- Test file includes proper path setup for the pfun_cma_model package
- All tests are designed to verify configuration without requiring live security middleware execution
- IP address validation uses Python's `ipaddress` module
- Tests support both individual runs and suite execution
