"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""
import pfun_cma_model.api as api_core

# monkey-patch for the fastapi app
app = api_core.app

# socket-io session
socketio_session = api_core.socketio_session
