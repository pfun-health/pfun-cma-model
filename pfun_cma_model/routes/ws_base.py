import logging
import socketio
from fastapi import FastAPI


class NoPrefixNamespace(socketio.AsyncNamespace):
    """Custom Socket.IO namespace without a prefix.
    This allows handling messages without a prefix.
    Inherits from socketio.AsyncNamespace to handle messages directly.

    Arguments:
        namespace (str): The namespace for this Socket.IO server.
        sio (socketio.AsyncServer): The Socket.IO server instance.
        app (FastAPI): The FastAPI application instance.
    """

    def __init__(self, namespace: str = "/", app: FastAPI = None):
        super().__init__(namespace=namespace)
        self.app = app
        
    @property
    def sio(self):
        """Return the Socket.IO server instance."""
        return self.server

    def on_connect(self, sid, environ):
        logging.info("connect ", sid)

    async def on_message(self, sid, data):
        logging.info("message ", data)
        await self.server.emit("response", "hi " + data)

    def on_disconnect(self, sid):
        logging.info("disconnect ", sid)
