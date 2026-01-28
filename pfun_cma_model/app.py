"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""
from pfun_common.settings import get_settings
import pfun_cma_model.api as api_core

# Export the FastAPI app from api_core
app = api_core.app

# # #
# Export the socket-io session from api_core
# # #
socketio_session = api_core.socketio_session
