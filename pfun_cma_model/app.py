"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""
import pfun_cma_model.api as api_core

# Export the FastAPI app from api_core
app = api_core.app

# Export the socket-io session from api_core
socketio_session = api_core.socketio_session
