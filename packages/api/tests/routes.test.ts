import { describe, it, expect, beforeAll } from "vitest";
import { createApp } from "../src/index.js";

let app: ReturnType<typeof createApp>["app"];

beforeAll(() => {
  app = createApp().app;
});

describe("Data routes", () => {
  it("GET /data/sample/download should return HTML by default", async () => {
    const res = await app.request("/data/sample/download");
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("<table>");
  });

  it("GET /data/sample/download?media_type=json should return JSON", async () => {
    const res = await app.request("/data/sample/download?media_type=json&nrows=5");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBe(5);
  });

  it("GET /data/sample/download?media_type=text should return CSV", async () => {
    const res = await app.request("/data/sample/download?media_type=text&nrows=3");
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Type")).toContain("text/csv");
    const text = await res.text();
    expect(text).toContain(",");
  });

  it("GET /data/sample/download?media_type=octet-stream should return 501", async () => {
    const res = await app.request("/data/sample/download?media_type=octet-stream");
    expect(res.status).toBe(501);
  });

  it("should reject nrows < -1", async () => {
    const res = await app.request("/data/sample/download?nrows=-2");
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.detail).toContain("nrows must be -1");
  });

  it("should reject pct0 out of range", async () => {
    const res = await app.request("/data/sample/download?pct0=1.5");
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.detail).toContain("pct0 must be between");
  });

  it("GET /data/sample/stream should return streaming data", async () => {
    const res = await app.request("/data/sample/stream?nrows=5&media_type=json");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
  });

  it("GET /data/sample/download?nrows=-1 should return all from pct0", async () => {
    const res = await app.request("/data/sample/download?nrows=-1&pct0=0.9&media_type=json");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.length).toBeGreaterThan(0);
    // Should be less than full dataset since pct0=0.9
    expect(body.length).toBeLessThan(1024);
  });
});

describe("Auth routes", () => {
  it("GET /auth/health should return ok", async () => {
    const res = await app.request("/auth/health");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.jwt_algorithm).toBe("HS256");
  });

  it("GET /auth/health/verify without token should return no_token", async () => {
    const res = await app.request("/auth/health/verify");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("no_token");
  });

  it("GET /auth/health/verify with invalid token should return invalid", async () => {
    const res = await app.request("/auth/health/verify?token=invalidtoken");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("invalid");
  });

  it("GET /auth/user/me without token should return 401", async () => {
    const res = await app.request("/auth/user/me");
    expect(res.status).toBe(401);
    expect(res.headers.get("WWW-Authenticate")).toBe("Bearer");
  });

  it("POST /auth/logout should return success", async () => {
    const res = await app.request("/auth/logout", { method: "POST" });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.success).toBe(true);
  });

  it("POST /auth/token/refresh without token should return 401", async () => {
    const res = await app.request("/auth/token/refresh", { method: "POST" });
    expect(res.status).toBe(401);
  });
});

describe("Dexcom routes", () => {
  it("GET /dexcom/test should return ok", async () => {
    const res = await app.request("/dexcom/test");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.message).toBe("Dexcom router is working");
  });

  it("POST /dexcom/token without code should return 400", async () => {
    const res = await app.request("/dexcom/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(400);
  });

  it("GET /dexcom/users/self/egvs without token should return 401", async () => {
    const res = await app.request("/dexcom/users/self/egvs");
    expect(res.status).toBe(401);
  });

  it("GET /dexcom/users/self/devices without token should return 401", async () => {
    const res = await app.request("/dexcom/users/self/devices");
    expect(res.status).toBe(401);
  });

  it("GET /dexcom/auth/callback without code should return 400", async () => {
    const res = await app.request("/dexcom/auth/callback");
    expect(res.status).toBe(400);
  });
});

describe("LLM routes", () => {
  it("POST /llm/generate-scenario should return scenario", async () => {
    const res = await app.request("/llm/generate-scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "test", include_recommendations: true }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("forecasted_events");
    expect(body).toHaveProperty("qualitative_description");
    expect(body).toHaveProperty("parameters");
    expect(body).toHaveProperty("recommendations");
  });

  it("POST /llm/generate-scenario without recommendations", async () => {
    const res = await app.request("/llm/generate-scenario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "test", include_recommendations: false }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.recommendations).toBeUndefined();
  });
});

describe("Demo routes", () => {
  it("GET /demo/llm should return HTML", async () => {
    const res = await app.request("/demo/llm");
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("html");
  });

  it("GET /demo/data-stream should return HTML", async () => {
    const res = await app.request("/demo/data-stream");
    expect(res.status).toBe(200);
  });

  it("GET /demo/run-at-time should return HTML", async () => {
    const res = await app.request("/demo/run-at-time");
    expect(res.status).toBe(200);
  });

  it("GET /demo/canvas-wave should return HTML", async () => {
    const res = await app.request("/demo/canvas-wave");
    expect(res.status).toBe(200);
  });

  it("GET /demo/full-model-run should return HTML", async () => {
    const res = await app.request("/demo/full-model-run");
    expect(res.status).toBe(200);
  });

  it("GET /demo/webgl-demo should return HTML", async () => {
    const res = await app.request("/demo/webgl-demo");
    expect(res.status).toBe(200);
  });
});

describe("SSO routes", () => {
  it("GET /sso/protected should return HTML", async () => {
    const res = await app.request("/sso/protected");
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("admin");
  });
});
