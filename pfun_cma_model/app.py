"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""
from pfun_common.settings import get_settings
import pfun_cma_model.api as api_core

# Export the FastAPI app from api_core
app = api_core.app

# # #
# setup the security layer
# # #
from guard import SecurityMiddleware
from guard.models import SecurityConfig
# Configure rate limiting
config = SecurityConfig(
    rate_limit=100,               # Max 100 requests
    rate_limit_window=120,         # over X seconds
    enable_rate_limiting=True,    # Enable rate limiting (true by default)
    enable_redis=True,            # Use Redis for distributed setup (true by default)
    redis_url=get_settings().redis_url
)
# Add middleware with rate limiting
app.add_middleware(SecurityMiddleware, config=config)

# # #
# Setup MCP
# # #
from fastapi_mcp import FastApiMCP
mcp = FastApiMCP(app)
mcp.mount()

# # #
# Export the socket-io session from api_core
# # #
socketio_session = api_core.socketio_session
