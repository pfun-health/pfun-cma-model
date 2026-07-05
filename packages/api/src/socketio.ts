/**
 * Socket.IO WebSocket handler.
 * Implements the socket.io contract from CLEANROOM_INSTRUCTIONS.
 */

import { Server as SocketIOServer } from "socket.io";
import type { Server as HttpServer } from "http";
import { CMASleepWakeModel } from "@pfun/core";

let ioInstance: SocketIOServer | null = null;

export function isSocketIoActive(): boolean {
  return ioInstance !== null;
}

export function setupSocketIO(httpServer: HttpServer): SocketIOServer {
  const io = new SocketIOServer(httpServer, {
    path: "/socket.io/",
    cors: {
      origin: "*",
      methods: ["GET", "POST"],
    },
  });

  ioInstance = io;

  io.on("connection", (socket) => {
    console.log(`[Socket.IO] Client connected: ${socket.id}`);

    socket.on("disconnect", () => {
      console.log(`[Socket.IO] Client disconnected: ${socket.id}`);
    });

    // Echo-style message handler
    socket.on("message", (data) => {
      socket.emit("response", data);
    });

    // Run event: stream run-at-time results
    socket.on("run", (data) => {
      try {
        let payload: {
          t0?: number;
          t1?: number;
          n?: number;
          config?: Record<string, unknown>;
        };

        if (typeof data === "string") {
          payload = JSON.parse(data);
        } else {
          payload = data ?? {};
        }

        const { t0 = 0, t1 = 100, n = 100, config = {} } = payload;
        const model = new CMASleepWakeModel();

        for (const point of model.runAtTimeStream(t0, t1, n, config)) {
          socket.emit("message", JSON.stringify(point));
        }
      } catch (err) {
        socket.emit(
          "message",
          JSON.stringify({ error: String(err) }),
        );
      }
    });

    // Run full event: stream full model results
    socket.on("run_full", (data) => {
      try {
        let payload: {
          t0?: number;
          t1?: number;
          n?: number;
          config?: Record<string, unknown>;
        };

        if (typeof data === "string") {
          payload = JSON.parse(data);
        } else {
          payload = data ?? {};
        }

        const { t0 = 0, t1 = 24, n = 100, config = {} } = payload;
        const model = new CMASleepWakeModel();

        for (const point of model.runFullStream(t0, t1, n, config)) {
          socket.emit("message", JSON.stringify(point));
        }
      } catch (err) {
        socket.emit(
          "message",
          JSON.stringify({ error: String(err) }),
        );
      }
    });
  });

  return io;
}

export function shutdownSocketIO(): void {
  if (ioInstance) {
    ioInstance.close();
    ioInstance = null;
  }
}
