import json
import logging
from collections.abc import Mapping
from typing import Optional
from fastapi import FastAPI
import socketio
from pfun_cma_model.app import run_at_time_func


async def get_logger():
    """Asynchronously get a logger instance."""
    return logging.getLogger("pfun_cma_model")


class NoPrefixNamespace(socketio.AsyncNamespace):
    """Custom Socket.IO namespace without a prefix.
    This allows handling messages without a prefix.
    Inherits from socketio.AsyncNamespace to handle messages directly.

    Arguments:
        namespace (str): The namespace for this Socket.IO server.
        sio (socketio.AsyncServer): The Socket.IO server instance.
        app (FastAPI): The FastAPI application instance.
    """

    def __init__(self, namespace: str = "/", app: Optional[FastAPI] = None):
        super().__init__(namespace=namespace)
        self._logger = None
        self.app = app

    @property
    async def logger(self):
        """Asynchronously get the logger instance."""
        if self._logger is None:
            self._logger = await get_logger()
        return self._logger

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


class PFunWebsocketNamespace(NoPrefixNamespace):
    """
    Custom Socket.IO namespace for handling websocket events, namely for the expedient 'run-at-time' functionality.
    Inherits from the base NoPrefixNamespace to handle messages without a prefix.
    This namespace is specifically designed to handle 'run-at-time' events and communicate with the Socket.IO server.
    """

    def on_connect(self, sid: str, environ: Mapping[str, str]):
        self.logger.debug(f"SocketIO client connected: {sid}")
        super().on_connect(sid, environ)

    async def on_message(self, sid, data):
        self.logger.debug(f"Received message from {sid}: {data}")
        await super().on_message(sid, data)

    def on_disconnect(self, sid):
        self.logger.debug(f"SocketIO client disconnected: {sid}")
        super().on_disconnect(sid)

    async def on_run(self, sid, data):
        """Handle 'run' event from client, run model, and emit result as 'message'."""
        try:
            # Accept both dict and JSON string
            run_args = data if isinstance(data, dict) else json.loads(data)
            t0 = run_args.get("t0", 0)
            t1 = run_args.get("t1", 100)
            n = run_args.get("n", 100)
            config = run_args.get("config", {})
            self.logger.debug(
                f"Received run event: t0={t0}, t1={t1}, n={n}, config={config}")
            output = await run_at_time_func(t0, t1, n, **config)
            # Always emit as JSON string
            await self.sio.emit("message", output, to=sid)
            self.logger.debug(f"Sent output to {sid}")
        except Exception as e:
            self.logger.error(f"Error in handle_run: {e}", exc_info=True)
            await self.sio.emit("message", json.dumps({"error": str(e)}), to=sid)
