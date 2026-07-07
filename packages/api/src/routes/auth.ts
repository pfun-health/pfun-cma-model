/**
 * Auth routes: /auth/*
 * JWT-based authentication.
 */

import { Hono } from "hono";
import jwt from "jsonwebtoken";
import type { AppConfig } from "../config.js";

export function createAuthRoutes(config: AppConfig): Hono {
  const app = new Hono();

  const JWT_SECRET = config.jwtSecretKey;
  const JWT_EXPIRY_MINUTES = config.jwtExpirationMinutes;

  /**
   * Extract and verify token from Authorization header.
   */
  function verifyToken(authHeader: string | undefined): {
    valid: boolean;
    payload?: jwt.JwtPayload;
    error?: string;
  } {
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return { valid: false, error: "Missing or invalid authorization header" };
    }

    const token = authHeader.slice(7);
    try {
      const payload = jwt.verify(token, JWT_SECRET) as jwt.JwtPayload;
      return { valid: true, payload };
    } catch (err) {
      return { valid: false, error: "Invalid token" };
    }
  }

  // POST /auth/token/refresh
  app.post("/token/refresh", async (c) => {
    const auth = verifyToken(c.req.header("authorization"));
    if (!auth.valid) {
      return c.json({ detail: auth.error }, 401, {
        "WWW-Authenticate": "Bearer",
      });
    }

    const newToken = jwt.sign(
      { sub: auth.payload!.sub, provider: auth.payload!.provider ?? "local" },
      JWT_SECRET,
      { expiresIn: `${JWT_EXPIRY_MINUTES}m`, algorithm: "HS256" },
    );

    return c.json({
      access_token: newToken,
      token_type: "bearer",
      expires_in: JWT_EXPIRY_MINUTES * 60,
    });
  });

  // POST /auth/token/verify
  app.post("/token/verify", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const token = body.token ?? c.req.header("authorization")?.slice(7);

    if (!token) {
      return c.json({ valid: false, detail: "No token provided" });
    }

    try {
      const payload = jwt.verify(token, JWT_SECRET) as jwt.JwtPayload;
      return c.json({
        valid: true,
        identity: payload.sub,
        issued_at: payload.iat,
        expires_at: payload.exp,
      });
    } catch {
      return c.json({ valid: false, detail: "Invalid token" });
    }
  });

  // GET /auth/user/me
  app.get("/user/me", (c) => {
    const auth = verifyToken(c.req.header("authorization"));
    if (!auth.valid) {
      return c.json({ detail: auth.error }, 401, {
        "WWW-Authenticate": "Bearer",
      });
    }

    return c.json({
      id: auth.payload!.sub,
      first_name: auth.payload!.first_name ?? null,
      display_name: auth.payload!.display_name ?? null,
      picture: auth.payload!.picture ?? null,
      provider: auth.payload!.provider ?? "local",
    });
  });

  // POST /auth/logout
  app.post("/logout", (c) => {
    return c.json({ message: "Logged out successfully", success: true });
  });

  // GET /auth/health
  app.get("/health", (c) => {
    return c.json({
      status: "ok",
      jwt_algorithm: "HS256",
      expiry_minutes: JWT_EXPIRY_MINUTES,
    });
  });

  // GET /auth/health/verify
  app.get("/health/verify", (c) => {
    const token = c.req.query("token");
    if (!token) {
      return c.json({ status: "no_token" });
    }

    try {
      jwt.verify(token, JWT_SECRET);
      return c.json({ status: "valid" });
    } catch {
      return c.json({ status: "invalid" });
    }
  });

  return app;
}
