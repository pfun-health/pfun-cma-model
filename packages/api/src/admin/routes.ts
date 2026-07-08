/**
 * Admin panel routes: /admin/*
 * Minimal CRUD UI for User and Site entities, protected by JWT auth.
 * Mirrors pfun_cma_model/admin/ (sqladmin panel) functionality.
 */

import { Hono } from "hono";
import jwt from "jsonwebtoken";
import type { AppConfig } from "../config.js";
import {
  listUsers,
  getUserById,
  createUser,
  updateUser,
  deleteUser,
  listSites,
  getSiteById,
  createSite,
  deleteSite,
  getUserByNameOrEmail,
  verifyPassword,
} from "./db.js";

/**
 * JWT auth middleware for admin routes.
 */
function requireAdmin(jwtSecret: string) {
  return async (c: import("hono").Context, next: import("hono").Next) => {
    const authHeader = c.req.header("authorization");
    const cookieToken = getCookieToken(c.req.header("cookie"));
    const token = authHeader?.startsWith("Bearer ")
      ? authHeader.slice(7)
      : cookieToken;

    if (!token) {
      return c.json({ detail: "Authentication required" }, 401);
    }

    try {
      const payload = jwt.verify(token, jwtSecret) as jwt.JwtPayload;
      if (!payload.is_admin && !payload.sub?.startsWith("admin")) {
        return c.json({ detail: "Admin access required" }, 403);
      }
      c.set("adminUser", payload);
    } catch {
      return c.json({ detail: "Invalid or expired token" }, 401);
    }

    await next();
  };
}

function getCookieToken(cookieHeader: string | undefined): string | null {
  if (!cookieHeader) return null;
  const match = cookieHeader.match(/(?:^|;\s*)token=([^;]+)/);
  return match ? match[1] : null;
}

export function createAdminRoutes(config: AppConfig): Hono {
  const app = new Hono();
  const auth = requireAdmin(config.jwtSecretKey);

  // --- Login page ---
  app.get("/login", (c) => {
    return c.html(renderLoginPage());
  });

  app.post("/login", async (c) => {
    const body = await c.req.parseBody();
    const username = String(body.username ?? "");
    const password = String(body.password ?? "");

    const user = getUserByNameOrEmail(username);
    if (!user || !verifyPassword(password, user.hashed_password)) {
      return c.html(renderLoginPage("Invalid credentials"), 401);
    }

    const token = jwt.sign(
      {
        sub: String(user.id),
        name: user.name,
        email: user.email,
        is_admin: user.is_admin,
      },
      config.jwtSecretKey,
      { expiresIn: `${config.jwtExpirationMinutes}m` },
    );

    c.header("Set-Cookie", `token=${token}; HttpOnly; Path=/; SameSite=Strict`);
    c.header("Location", "/admin/");
    return c.body(null, 302);
  });

  app.get("/logout", (c) => {
    c.header("Set-Cookie", "token=; HttpOnly; Path=/; Max-Age=0; SameSite=Strict");
    c.header("Location", "/admin/login");
    return c.body(null, 302);
  });

  // --- Dashboard (index) ---
  app.get("/", auth, (c) => {
    const users = listUsers(10);
    const sites = listSites(10);
    return c.html(renderDashboard(users, sites));
  });

  // Redirect /admin (no trailing slash) to /admin/
  app.get("", auth, (c) => {
    const users = listUsers(10);
    const sites = listSites(10);
    return c.html(renderDashboard(users, sites));
  });

  // -------- User CRUD --------

  app.get("/users", auth, (c) => {
    const limit = parseInt(c.req.query("limit") ?? "100", 10);
    const offset = parseInt(c.req.query("offset") ?? "0", 10);
    return c.json(listUsers(limit, offset));
  });

  app.get("/users/:id", auth, (c) => {
    const user = getUserById(parseInt(c.req.param("id") ?? "0", 10));
    if (!user) return c.json({ detail: "User not found" }, 404);
    const { hashed_password: _, ...safeUser } = user;
    return c.json(safeUser);
  });

  app.post("/users", auth, async (c) => {
    const body = await c.req.json();
    const {
      name,
      email,
      password,
      age = 0,
      bio,
      site_id,
      is_admin = false,
    } = body as {
      name: string;
      email: string;
      password: string;
      age?: number;
      bio?: string;
      site_id?: number;
      is_admin?: boolean;
    };

    if (!name || !email || !password) {
      return c.json({ detail: "name, email and password are required" }, 400);
    }

    try {
      const user = createUser(name, email, password, age, bio, site_id, is_admin);
      const { hashed_password: _, ...safeUser } = user;
      return c.json(safeUser, 201);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("UNIQUE")) {
        return c.json({ detail: "Email already exists" }, 409);
      }
      return c.json({ detail: msg }, 500);
    }
  });

  app.patch("/users/:id", auth, async (c) => {
    const id = parseInt(c.req.param("id") ?? "0", 10);
    const body = await c.req.json();
    const user = updateUser(id, body);
    if (!user) return c.json({ detail: "User not found" }, 404);
    const { hashed_password: _, ...safeUser } = user;
    return c.json(safeUser);
  });

  app.delete("/users/:id", auth, (c) => {
    const id = parseInt(c.req.param("id") ?? "0", 10);
    const ok = deleteUser(id);
    if (!ok) return c.json({ detail: "User not found" }, 404);
    return c.json({ detail: "Deleted" });
  });

  // -------- Site CRUD --------

  app.get("/sites", auth, (c) => {
    const limit = parseInt(c.req.query("limit") ?? "100", 10);
    const offset = parseInt(c.req.query("offset") ?? "0", 10);
    return c.json(listSites(limit, offset));
  });

  app.get("/sites/:id", auth, (c) => {
    const site = getSiteById(parseInt(c.req.param("id") ?? "0", 10));
    if (!site) return c.json({ detail: "Site not found" }, 404);
    return c.json(site);
  });

  app.post("/sites", auth, async (c) => {
    const body = await c.req.json();
    const { name } = body as { name: string };
    if (!name) return c.json({ detail: "name is required" }, 400);
    const site = createSite(name);
    return c.json(site, 201);
  });

  app.delete("/sites/:id", auth, (c) => {
    const id = parseInt(c.req.param("id") ?? "0", 10);
    const ok = deleteSite(id);
    if (!ok) return c.json({ detail: "Site not found" }, 404);
    return c.json({ detail: "Deleted" });
  });

  return app;
}

// --- Minimal HTML helpers ---

function renderLoginPage(error?: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Admin Login — PFun CMA Model</title>
  <style>
    body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #f5f5f5; }
    .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); min-width: 320px; }
    h2 { margin-top: 0; }
    label { display: block; margin-bottom: 0.25rem; font-weight: 600; }
    input { width: 100%; padding: 0.5rem; margin-bottom: 1rem; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button { width: 100%; padding: 0.6rem; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
    .error { color: red; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Admin Login</h2>
    ${error ? `<p class="error">${error}</p>` : ""}
    <form method="post" action="/admin/login">
      <label for="username">Username or Email</label>
      <input type="text" id="username" name="username" required autocomplete="username">
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required autocomplete="current-password">
      <button type="submit">Sign In</button>
    </form>
  </div>
</body>
</html>`;
}

function renderDashboard(
  users: import("./db.js").User[],
  sites: import("./db.js").Site[],
): string {
  const userRows = users
    .map(
      (u) =>
        `<tr>
          <td>${u.id}</td>
          <td>${esc(u.name)}</td>
          <td>${esc(u.email)}</td>
          <td>${u.is_admin ? "✅" : ""}</td>
          <td>${u.age}</td>
          <td>${u.site_id ?? "—"}</td>
          <td>
            <a href="/admin/users/${u.id}">View</a>
            <form method="post" action="/admin/users/${u.id}/delete" style="display:inline">
              <button type="submit" onclick="return confirm('Delete?')">Delete</button>
            </form>
          </td>
        </tr>`,
    )
    .join("\n");

  const siteRows = sites
    .map(
      (s) =>
        `<tr>
          <td>${s.id}</td>
          <td>${esc(s.name)}</td>
          <td>
            <form method="post" action="/admin/sites/${s.id}/delete" style="display:inline">
              <button type="submit" onclick="return confirm('Delete?')">Delete</button>
            </form>
          </td>
        </tr>`,
    )
    .join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Admin — PFun CMA Model</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
    th, td { border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }
    th { background: #f0f0f0; }
    a.btn, button { padding: 0.3rem 0.7rem; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; font-size: 0.875rem; }
    .logout { float: right; background: #cc3300; }
  </style>
</head>
<body>
  <h1>PFun CMA Model — Admin Panel <a class="btn logout" href="/admin/logout">Logout</a></h1>

  <h2>Users</h2>
  <table>
    <thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Admin</th><th>Age</th><th>Site</th><th>Actions</th></tr></thead>
    <tbody>${userRows || "<tr><td colspan='7'>No users</td></tr>"}</tbody>
  </table>

  <h2>Sites</h2>
  <table>
    <thead><tr><th>ID</th><th>Name</th><th>Actions</th></tr></thead>
    <tbody>${siteRows || "<tr><td colspan='3'>No sites</td></tr>"}</tbody>
  </table>

  <p><em>Use the REST API endpoints (<code>POST /admin/users</code>, <code>POST /admin/sites</code>, etc.) to create records.</em></p>
</body>
</html>`;
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
