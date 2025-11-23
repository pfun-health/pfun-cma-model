import json
import logging
from collections.abc import Mapping
from typing import Optional
from fastapi import FastAPI
import socketio
from pfun_cma_model.engine.cma import CMASleepWakeModel
from pfun_cma_model.stream import stream_run_at_time_func


async def get_logger():
    """Asynchronously get a logger instance."""
    return logging.getLogger("pfun_cma_model")


class AsyncLoggerWrapper:
    """Wrapper for an async logger that can be awaited."""

    def __init__(self, logger_coroutine):
        self._logger_coroutine = logger_coroutine
        self._logger = None

    async def __call__(self):
        if self._logger is None:
            self._logger = await self._logger_coroutine
        return self._logger

    async def debug(self, msg, *args, **kwargs):
        logger_instance = await self()
        try:
            await logger_instance.debug(msg, *args, **kwargs)
        except TypeError:
            # ignore 'NoneType' in async expression error
            pass

    async def info(self, msg, *args, **kwargs):
        logger_instance = await self()
        try:
            await logger_instance.info(msg, *args, **kwargs)
        except TypeError:
            # ignore 'NoneType' in async expression error
            pass

    async def warning(self, msg, *args, **kwargs):
        logger_instance = await self()
        await logger_instance.warning(msg, *args, **kwargs)

    async def error(self, msg, *args, **kwargs):
        logger_instance = await self()
        await logger_instance.error(msg, *args, **kwargs)

    async def critical(self, msg, *args, **kwargs):
        logger_instance = await self()
        await logger_instance.critical(msg, *args, **kwargs)


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
            self._logger = AsyncLoggerWrapper(get_logger())
        return self._logger

    @property
    def sio(self):
        """Return the Socket.IO server instance."""
        return self.server

    async def on_connect(self, sid, environ):
        await (await self.logger).debug("connect %s", sid)

    async def on_message(self, sid, data):
        await (await self.logger).debug("message %s", data)
        await self.server.emit("response", "hi " + data)

    async def on_disconnect(self, sid):
        await (await self.logger).info("disconnect %s", sid)


class PFunWebsocketNamespace(NoPrefixNamespace):
    """
    Custom Socket.IO namespace for handling websocket events, namely for the expedient 'run-at-time' functionality.
    Inherits from the base NoPrefixNamespace to handle messages without a prefix.
    This namespace is specifically designed to handle 'run-at-time' events and communicate with the Socket.IO server.
    """

    async def on_connect(self, sid: str, environ: Mapping[str, str]):
        await (await self.logger).debug(f"SocketIO client connected: {sid}")
        await super().on_connect(sid, environ)

    async def on_message(self, sid, data):
        await (await self.logger).debug(f"Received message from {sid}: {data}")
        await super().on_message(sid, data)

    async def on_disconnect(self, sid):
        await (await self.logger).debug(f"SocketIO client disconnected: {sid}")
        await super().on_disconnect(sid)

    async def on_run(self, sid, data):
        """Handle 'run' event from client, run model, and stream results as 'message' events."""
        try:
            # Accept both dict and JSON string
            run_args = data if isinstance(data, dict) else json.loads(data)
            t0 = run_args.get("t0", 0)
            t1 = run_args.get("t1", 100)
            n = run_args.get("n", 100)
            config = run_args.get("config", {})

            model = CMASleepWakeModel()
            # Use an async for loop to iterate over the async generator
            async for point in stream_run_at_time_func(model, t0, t1, n, **config):
                await self.sio.emit("message", point, to=sid)

        except Exception as e:
            logging.error(f"Error in handle_run: {e}", exc_info=True)
            await self.sio.emit("message", json.dumps({"error": str(e)}), to=sid)
