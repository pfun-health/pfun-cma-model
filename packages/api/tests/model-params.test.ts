import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createApp } from "../src/index.js";
import { initAdminDb, closeAdminDb } from "../src/admin/db.js";
import { initResultsStore } from "../src/results.js";

let app: ReturnType<typeof createApp>["app"];

beforeAll(() => {
  initAdminDb({ debug: true, port: 0, host: "", redisUrl: null, redisHost: "", redisPort: 0, redisDb: 0, redisPassword: null, jwtSecretKey: "test-secret", jwtExpirationMinutes: 30, sessionSecret: "test", dexcomClientId: "", dexcomClientSecret: "", dexcomRedirectUri: "", googleClientId: "", googleClientSecret: "", corsOrigins: [], trustedHosts: ["*"], staticDir: "static", templateDir: "templates", version: "1.0.0" });
  initResultsStore(true);
  app = createApp().app;
});

afterAll(() => {
  closeAdminDb();
});

describe("Model routes", () => {
  it("POST /model/run should return model output", async () => {
    const res = await app.request("/model/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBeGreaterThan(0);
    expect(body[0]).toHaveProperty("t");
    expect(body[0]).toHaveProperty("G");
    expect(body[0]).toHaveProperty("c");
    expect(body[0]).toHaveProperty("m");
    // Check CORS header
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("*");
  });

  it("POST /model/run with config should use custom params", async () => {
    const res = await app.request("/model/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ B: 0.5, N: 50 }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.length).toBe(50);
  });

  it("POST /model/run-at-time should return time series", async () => {
    const res = await app.request("/model/run-at-time", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ t0: 0, t1: 24, n: 50 }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(Array.isArray(body)).toBe(true);
    expect(body.length).toBe(50);
    expect(body[0]).toHaveProperty("x");
    expect(body[0]).toHaveProperty("y");
  });

  it("POST /model/run-at-time/stream should return ndjson", async () => {
    const res = await app.request("/model/run-at-time/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ t0: 0, t1: 24, n: 10 }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    const lines = text.trim().split("\n").filter(Boolean);
    expect(lines.length).toBe(10);
    const firstLine = JSON.parse(lines[0]);
    expect(firstLine).toHaveProperty("x");
    expect(firstLine).toHaveProperty("y");
  });

  it("POST /model/fit should return fit result", async () => {
    const res = await app.request("/model/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: {} }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("params");
    expect(body).toHaveProperty("residual");
    expect(body).toHaveProperty("success");
  });

  it("POST /model/fit with invalid JSON string data should return 400", async () => {
    const res = await app.request("/model/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: "not valid json{{{" }),
    });
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body).toHaveProperty("error");
    expect(body.exception_type).toBe("JSONDecodeError");
  });
});

describe("Params routes", () => {
  it("GET /params/schema should return JSON schema", async () => {
    const res = await app.request("/params/schema");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.title).toBe("CMAModelParams");
    expect(body.properties).toBeDefined();
  });

  it("GET /params/default should return default params", async () => {
    const res = await app.request("/params/default");
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.N).toBe(1024);
    expect(body.d).toBe(0.0);
    expect(body.taup).toBe(1.0);
    expect(body.tM).toEqual([7.0, 11.0, 17.5]);
  });

  it("POST /params/describe should return descriptions", async () => {
    const res = await app.request("/params/describe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ d: 5.0, B: 0.5 }),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.d).toHaveProperty("description");
    expect(body.d).toHaveProperty("qualitative");
    expect(body.d).toHaveProperty("value");
    expect(body.d.value).toBe(5.0);
  });

  it("POST /params/tabulate/md should return markdown", async () => {
    const res = await app.request("/params/tabulate/md", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("| Parameter");
  });

  it("POST /params/tabulate/html should return HTML table", async () => {
    const res = await app.request("/params/tabulate/html", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("<table>");
  });

  it("POST /params/tabulate/json should return JSON with table", async () => {
    const res = await app.request("/params/tabulate/json", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.table).toContain("| Parameter");
  });
});
