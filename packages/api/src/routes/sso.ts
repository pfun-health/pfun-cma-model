/**
 * SSO routes: /sso/*
 * Google SSO integration.
 */

import { Hono } from "hono";
import jwt from "jsonwebtoken";
import type { AppConfig } from "../config.js";

export function createSsoRoutes(config: AppConfig): Hono {
  const app = new Hono();

  const JWT_SECRET = config.jwtSecretKey;
  const JWT_EXPIRY_MINUTES = config.jwtExpirationMinutes;

  // GET /sso/protected
  app.get("/protected", (c) => {
    const html = `<html><body>
      <h1>Hello, authenticated user!</h1>
      <meta http-equiv="refresh" content="2;url=/admin/" />
      <p>Redirecting to admin...</p>
    </body></html>`;
    return c.html(html);
  });

  // GET /sso/auth/login
  app.get("/auth/login", (c) => {
    const clientId = config.googleClientId;
    const redirectUri = encodeURIComponent(
      `${c.req.url.split("/sso")[0]}/sso/auth/callback`,
    );
    const authUrl =
      `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${clientId}&redirect_uri=${redirectUri}` +
      `&response_type=code&scope=openid+email+profile`;

    return c.redirect(authUrl);
  });

  // GET /sso/auth/logout
  app.get("/auth/logout", (c) => {
    // Clear session cookie
    c.header("Set-Cookie", "token=; Path=/; Max-Age=0");
    return c.redirect("/");
  });

  // GET /sso/auth/callback
  app.get("/auth/callback", async (c) => {
    const code = c.req.query("code");
    if (!code) {
      return c.json({ detail: "Missing authorization code" }, 400);
    }

    try {
      // Exchange code for token with Google
      const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          code,
          client_id: config.googleClientId,
          client_secret: config.googleClientSecret,
          redirect_uri: `${c.req.url.split("/sso")[0]}/sso/auth/callback`,
          grant_type: "authorization_code",
        }),
      });

      if (!tokenRes.ok) {
        return c.json({ detail: "SSO token exchange failed" }, 400);
      }

      const tokenData = (await tokenRes.json()) as { id_token?: string };
      const idToken = tokenData.id_token;

      if (!idToken) {
        return c.json({ detail: "No ID token received" }, 400);
      }

      // Decode Google ID token (in production, verify signature)
      const decoded = jwt.decode(idToken) as Record<string, string> | null;

      // Create our own JWT
      const appToken = jwt.sign(
        {
          sub: decoded?.sub ?? "unknown",
          email: decoded?.email,
          display_name: decoded?.name,
          picture: decoded?.picture,
          provider: "google",
        },
        JWT_SECRET,
        { expiresIn: `${JWT_EXPIRY_MINUTES}m`, algorithm: "HS256" },
      );

      // Set cookie and redirect
      c.header(
        "Set-Cookie",
        `token=${appToken}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${JWT_EXPIRY_MINUTES * 60}`,
      );
      return c.redirect("/user/");
    } catch (err) {
      return c.json({ detail: "SSO callback failed", error: String(err) }, 500);
    }
  });

  return app;
}
