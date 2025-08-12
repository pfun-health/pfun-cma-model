from typing import (
    Optional, List
)
from collections.abc import Iterable
from fastapi.applications import FastAPI
import socketio


class PFunSocketIOSession:
    """
    A session for interacting with Socket.IO.
    """

    def __init__(self, app: FastAPI, ns: Optional[socketio.AsyncNamespace | List[socketio.AsyncNamespace]] = None):
        """
        Initialize the PFunSocketIOSession with a FastAPI app.
        This sets up the Socket.IO server with the provided app.
        """
        self.app = app
        self.ns = ns if isinstance(ns, Iterable) and not isinstance(
            ns, (str, bytes)) else [ns] if ns else []
        self.sio: Optional[socketio.AsyncServer] = None
        self.mgr = None
        # Setup Socket.IO server and mount it to the FastAPI app
        self.setup_redis_manager()
        self.setup_socketio()
        self.mount_socketio()

    def __getattr__(self, name):
        """Pass any non-matching attributes to the Socket.IO server.
        """
        # First, determine if the attribute exists in the Socket.IO server
        if hasattr(self.sio, name):
            # if it does, return it
            return getattr(self.sio, name)
        # if it doesn't, check if it's a method of the PFunSocketIOSession class
        elif hasattr(super(), name):
            # if it is, return it
            return getattr(super(), name)
        # otherwise, raise an AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def setup_socketio(self, mgr: Optional[socketio.AsyncRedisManager] = None) -> socketio.AsyncServer:
        """
        Setup Socket.IO server with FastAPI app.
        This is useful for handling WebSocket connections.
        """
        # see https://python-socketio.readthedocs.io/en/latest/server.html#using-a-message-queue
        # Create a Socket.IO server instance with Redis manager
        if mgr is None:
            self.setup_redis_manager()
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
            connection_manager=self.mgr
        )
        # Register namespaces if provided
        if self.ns:
            for namespace in self.ns:
                if isinstance(namespace, socketio.AsyncNamespace):
                    self.sio.register_namespace(namespace)  # type: ignore
                    namespace._set_server(self.sio)  # set the server instance
                else:
                    raise TypeError(
                        "Namespaces must be instances of socketio.AsyncNamespace")
        # initialize the Socket.IO server with the FastAPI app
        self.sio_app = socketio.ASGIApp(
            socketio_server=self.sio, other_asgi_app=self.app)
        return self.sio

    def mount_socketio(self, path: str = "/socket.io/") -> None:
        """
        Mount the Socket.IO ASGI app to the FastAPI app at a specified path.
        This allows WebSocket connections to be handled at the given path.
        """
        self.app.add_route(
            path, route=self.sio_app, methods=["GET", "POST"], include_in_schema=False)
        self.app.add_websocket_route(path, route=self.sio_app)

    def setup_redis_manager(self):
        """
        Setup Redis manager for Socket.IO.
        """
        self.mgr = socketio.AsyncRedisManager(url="redis://localhost:6379/0")
        return self.mgr


class PFunBotoSession:
    """@todo: Implement a session for interacting with AWS Boto3."""
    pass
