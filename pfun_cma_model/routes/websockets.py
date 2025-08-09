# websockets.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import List


class ConnectionManager:
    """Manages WebSocket connections for the FastAPI application.

    This class handles the connections, allowing for sending messages to individual or all connected clients.

    Attributes:
        active_connections (List[WebSocket]): A list of currently active WebSocket connections.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
