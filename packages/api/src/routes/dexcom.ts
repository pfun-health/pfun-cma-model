/**
 * Dexcom routes: /dexcom/*
 */

import { Hono } from "hono";
import type { AppConfig } from "../config.js";

// Simple in-memory session store (production would use Redis)
const sessions = new Map<string, Record<string, string>>();

function getSessionId(c: { req: { header: (name: string) => string | undefined } }): string {
  return c.req.header("x-session-id") ?? "default";
}

export function createDexcomRoutes(config: AppConfig): Hono {
  const app = new Hono();

  const DEXCOM_BASE = "https://sandbox-api.dexcom.com";

  // GET /dexcom/test
  app.get("/test", (c) => {
    return c.json({ message: "Dexcom router is working" });
  });

  // POST /dexcom/token
  app.post("/token", async (c) => {
    const body = await c.req.json();
    const { code, redirect_uri } = body;

    if (!code) {
      return c.json({ detail: "Missing authorization code" }, 400);
    }

    try {
      const tokenRes = await fetch(`${DEXCOM_BASE}/v2/oauth2/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          client_id: config.dexcomClientId,
          client_secret: config.dexcomClientSecret,
          code,
          grant_type: "authorization_code",
          redirect_uri: redirect_uri ?? config.dexcomRedirectUri,
        }),
      });

      if (!tokenRes.ok) {
        const errData = await tokenRes.json().catch(() => ({}));
        return c.json({ detail: errData }, (tokenRes.status >= 400 && tokenRes.status < 600 ? tokenRes.status : 500) as 400);
      }

      const data = (await tokenRes.json()) as {
        access_token: string;
        refresh_token: string;
      };

      // Store tokens in session
      const sessionId = getSessionId(c);
      const session = sessions.get(sessionId) ?? {};
      session.dexcom_access_token = data.access_token;
      session.dexcom_refresh_token = data.refresh_token;
      sessions.set(sessionId, session);

      return c.json(data);
    } catch (err) {
      return c.json({ detail: "Token exchange failed", error: String(err) }, 500);
    }
  });

  // GET /dexcom/auth/callback
  app.get("/auth/callback", (c) => {
    const code = c.req.query("code");
    if (!code) {
      return c.json({ detail: "Missing code parameter" }, 400);
    }

    const sessionId = getSessionId(c);
    const session = sessions.get(sessionId) ?? {};
    session.dexcom_auth_code = code;
    sessions.set(sessionId, session);

    return c.redirect("/demo/dexcom");
  });

  // GET /dexcom/users/self/egvs
  app.get("/users/self/egvs", async (c) => {
    const sessionId = getSessionId(c);
    const session = sessions.get(sessionId);
    const accessToken = session?.dexcom_access_token;

    if (!accessToken) {
      return c.json({ detail: "No access token. Please authenticate first." }, 401);
    }

    const startDate = c.req.query("startDate") ?? "";
    const endDate = c.req.query("endDate") ?? "";

    try {
      const authHeader = "Bearer " + accessToken;
      const res = await fetch(
        `${DEXCOM_BASE}/v2/users/self/egvs?startDate=${startDate}&endDate=${endDate}`,
        { headers: { Authorization: authHeader } },
      );
      const data = await res.json();
      return c.json(data, 200);
    } catch (err) {
      return c.json({ detail: "Failed to fetch EGVs", error: String(err) }, 500);
    }
  });

  // GET /dexcom/users/self/devices
  app.get("/users/self/devices", async (c) => {
    const sessionId = getSessionId(c);
    const session = sessions.get(sessionId);
    const accessToken = session?.dexcom_access_token;

    if (!accessToken) {
      return c.json({ detail: "No access token. Please authenticate first." }, 401);
    }

    try {
      const authHeader = "Bearer " + accessToken;
      const res = await fetch(`${DEXCOM_BASE}/v2/users/self/devices`, {
        headers: { Authorization: authHeader },
      });
      const data = await res.json();
      return c.json(data, 200);
    } catch (err) {
      return c.json({ detail: "Failed to fetch devices", error: String(err) }, 500);
    }
  });

  return app;
}
