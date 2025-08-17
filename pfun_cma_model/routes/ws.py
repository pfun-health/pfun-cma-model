import json
import logging
from collections.abc import Mapping
from typing import Optional, List
from fastapi import FastAPI
import socketio
from pfun_cma_model.app import run_at_time_func
from pfun_cma_model.routes.ws_base import NoPrefixNamespace
logger = logging.getLogger()


class PFunWebsocketNamespace(NoPrefixNamespace):
    """
    Custom Socket.IO namespace for handling websocket events, namely for the expedient 'run-at-time' functionality.
    Inherits from the base NoPrefixNamespace to handle messages without a prefix.
    This namespace is specifically designed to handle 'run-at-time' events and communicate with the Socket.IO server.
    """

    def on_connect(self, sid: str, environ: Mapping[str, str]):
        logger.debug(f"SocketIO client connected: {sid}")
        super().on_connect(sid, environ)

    async def on_message(self, sid, data):
        logger.debug(f"Received message from {sid}: {data}")
        await super().on_message(sid, data)

    def on_disconnect(self, sid):
        logger.debug(f"SocketIO client disconnected: {sid}")
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
            logger.debug(
                f"Received run event: t0={t0}, t1={t1}, n={n}, config={config}")
            output = await run_at_time_func(t0, t1, n, **config)
            # Always emit as JSON string
            await self.sio.emit("message", output, to=sid)
            logger.debug(f"Sent output to {sid}")
        except Exception as e:
            logger.error(f"Error in handle_run: {e}", exc_info=True)
            await self.sio.emit("message", json.dumps({"error": str(e)}), to=sid)
