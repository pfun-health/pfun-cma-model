import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createApp } from "../src/index.js";
import { initAdminDb, closeAdminDb } from "../src/admin/db.js";
import { initResultsStore } from "../src/results.js";
import type { Hono } from "hono";

let app: ReturnType<typeof createApp>["app"];

beforeAll(() => {
  initAdminDb({ debug: true, port: 0, host: "", redisUrl: null, redisHost: "", redisPort: 0, redisDb: 0, redisPassword: null, jwtSecretKey: "test-secret", jwtExpirationMinutes: 30, sessionSecret: "test", dexcomClientId: "", dexcomClientSecret: "", dexcomRedirectUri: "", googleClientId: "", googleClientSecret: "", corsOrigins: [], trustedHosts: ["*"], staticDir: "static", templateDir: "templates", version: "1.0.0" });
  initResultsStore(true);
  const result = createApp();
  app = result.app;
});

afterAll(() => {
  closeAdminDb();
});

describe("Health routes", () => {
  it("GET /health should return 200 with status ok", async () => {
    const res = await app.request("/health");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.message).toBe("PFun CMA Model API is running.");
  });

  it("GET /health/ws/run-at-time should return 503 when no socket.io", async () => {
    const res = await app.request("/health/ws/run-at-time");
    expect(res.status).toBe(503);
    const body = await res.json();
    expect(body.status).toBe("error");
    expect(body.message).toContain("NOT running");
  });
});

describe("OpenAPI routes", () => {
  it("GET /openapi.json should return OpenAPI schema", async () => {
    const res = await app.request("/openapi.json");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.openapi).toBe("3.1.0");
    expect(body.info.title).toBe("PFun CMA Model Routing API");
    expect(body.info.description).toContain("PFun CMA model");
  });

  it("GET /docs should return HTML", async () => {
    const res = await app.request("/docs");
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("swagger-ui");
  });

  it("GET /redoc should return HTML", async () => {
    const res = await app.request("/redoc");
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("redoc");
  });
});

describe("Security middleware", () => {
  it("should reject ?debug=true", async () => {
    const res = await app.request("/health?debug=true");
    expect(res.status).toBe(403);
    const body = await res.json();
    expect(body.detail).toBe("Debug mode not allowed");
  });

  it("should block bad user agents", async () => {
    const res = await app.request("/health", {
      headers: { "User-Agent": "sqlmap/1.0" },
    });
    expect(res.status).toBe(403);
  });

  it("should set security headers", async () => {
    const res = await app.request("/health");
    expect(res.headers.get("X-Frame-Options")).toBe("DENY");
    expect(res.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(res.headers.get("Referrer-Policy")).toBe(
      "strict-origin-when-cross-origin",
    );
    expect(res.headers.get("Content-Security-Policy")).toContain("default-src");
    expect(res.headers.get("Strict-Transport-Security")).toContain("max-age");
    expect(res.headers.get("Permissions-Policy")).toContain("camera=()");
  });
});

describe("Template routes", () => {
  it("GET / should return HTML", async () => {
    const res = await app.request("/");
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("PFun CMA Model");
  });

  it("GET /about should return HTML", async () => {
    const res = await app.request("/about");
    expect(res.status).toBe(200);
  });

it("GET /login should return HTML", async () => {
    const res = await app.request("/login");
    expect(res.status).toBe(200);
  });
});
