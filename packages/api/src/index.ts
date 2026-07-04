/**
 * Main application entry point.
 * PFun CMA Model Routing API - TypeScript Clean-Room Implementation.
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { createServer } from "http";
import { loadConfig, getVersionString } from "./config.js";
import {
  securityHeaders,
  rateLimiter,
  userAgentFilter,
  debugQueryRejection,
  requestTracker,
} from "./middleware/security.js";
import { createHealthRoutes } from "./routes/health.js";
import { createModelRoutes } from "./routes/model.js";
import { createParamsRoutes } from "./routes/params.js";
import { createDataRoutes } from "./routes/data.js";
import { createAuthRoutes } from "./routes/auth.js";
import { createSsoRoutes } from "./routes/sso.js";
import { createDexcomRoutes } from "./routes/dexcom.js";
import { createLlmRoutes } from "./routes/llm.js";
import { createDemoRoutes } from "./routes/demo.js";
import { setupSocketIO, isSocketIoActive, shutdownSocketIO } from "./socketio.js";

// Redis client (optional)
let redisClient: unknown | null = null;

/**
 * Simple template renderer (Nunjucks-compatible stub).
 * In production, use full Nunjucks with template directory.
 */
function createTemplateRenderer(templateDir: string) {
  return (name: string, ctx: Record<string, unknown>): string => {
    // Minimal template rendering - returns basic HTML structure
    const year = ctx.year ?? new Date().getFullYear();
    const title = name.replace(/\.html\.jinja2$/, "").replace(/-/g, " ");

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PFun CMA Model - ${title}</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header><h1>PFun CMA Model</h1></header>
  <main>
    <h2>${title}</h2>
    <p>Year: ${year}</p>
    ${ctx.access_message ? `<p>${ctx.access_message}</p>` : ""}
    ${ctx.bounded_params ? renderBoundedParams(ctx.bounded_params as Array<Record<string, unknown>>) : ""}
  </main>
  <footer><p>&copy; ${year} PFun Health</p></footer>
</body>
</html>`;
  };
}

function renderBoundedParams(params: Array<Record<string, unknown>>): string {
  if (!params || params.length === 0) return "";
  const rows = params
    .map(
      (p) =>
        `<tr><td>${p.name}</td><td>${p.value}</td><td>${p.min}</td><td>${p.max}</td><td>${p.step}</td></tr>`,
    )
    .join("\n");
  return `<table><thead><tr><th>Name</th><th>Value</th><th>Min</th><th>Max</th><th>Step</th></tr></thead><tbody>${rows}</tbody></table>`;
}

export function createApp() {
  const config = loadConfig();
  const versionString = getVersionString(config);
  const templateRenderer = createTemplateRenderer(config.templateDir);

  const app = new Hono();

  // Apply middleware
  app.use("*", cors({ origin: config.corsOrigins }));
  app.use("*", securityHeaders());
  app.use("*", rateLimiter());
  app.use("*", userAgentFilter());
  app.use("*", debugQueryRejection());
  app.use("*", requestTracker(redisClient));

  // OpenAPI metadata routes
  app.get("/openapi.json", (c) => {
    return c.json({
      openapi: "3.1.0",
      info: {
        title: "PFun CMA Model Routing API",
        description:
          "Server-side operations for operating the PFun CMA model; schema definitions, data IO, model execution.",
        version: versionString,
      },
      paths: {},
    });
  });

  app.get("/docs", (c) => {
    const html = `<!DOCTYPE html>
<html><head><title>API Docs</title>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<script>SwaggerUIBundle({ url: '/openapi.json', dom_id: '#swagger-ui' })</script>
</body></html>`;
    return c.html(html);
  });

  app.get("/redoc", (c) => {
    const html = `<!DOCTYPE html>
<html><head><title>ReDoc</title>
<script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"></script>
</head><body>
<redoc spec-url="/openapi.json"></redoc>
</body></html>`;
    return c.html(html);
  });

  // Mount route groups
  const healthRoutes = createHealthRoutes(
    isSocketIoActive,
    templateRenderer,
    config.staticDir,
  );
  app.route("/", healthRoutes);
  app.route("/model", createModelRoutes());
  app.route("/params", createParamsRoutes());
  app.route("/data", createDataRoutes());
  app.route("/auth", createAuthRoutes(config));
  app.route("/sso", createSsoRoutes(config));
  app.route("/dexcom", createDexcomRoutes(config));
  app.route("/llm", createLlmRoutes());
  app.route("/demo", createDemoRoutes(templateRenderer));

  return { app, config };
}

// Main entry point
if (
  process.argv[1] &&
  (process.argv[1].endsWith("index.ts") || process.argv[1].endsWith("index.js"))
) {
  const { app, config } = createApp();

  const server = serve(
    {
      fetch: app.fetch,
      port: config.port,
      hostname: config.host,
    },
    (info) => {
      console.log(
        `🚀 PFun CMA Model API listening on http://${config.host}:${info.port}`,
      );
      console.log(`   Version: ${getVersionString(config)}`);
    },
  );

  // Setup Socket.IO on the HTTP server
  setupSocketIO(server as unknown as import("http").Server);

  // Try Redis connection (non-fatal)
  tryRedisConnection(config).catch(() => {
    console.log("⚠️  Redis not available, continuing without caching.");
  });

  // Graceful shutdown
  process.on("SIGTERM", () => shutdown(server));
  process.on("SIGINT", () => shutdown(server));
}

async function tryRedisConnection(config: ReturnType<typeof loadConfig>) {
  try {
    const { default: Redis } = await import("ioredis");
    const client = new Redis({
      host: config.redisHost,
      port: config.redisPort,
      db: config.redisDb,
      password: config.redisPassword ?? undefined,
      connectTimeout: 3000,
      retryStrategy: () => null, // Don't retry
    });

    await new Promise<void>((resolve, reject) => {
      client.on("ready", () => {
        redisClient = client;
        console.log("✅ Redis connected.");
        resolve();
      });
      client.on("error", reject);
      setTimeout(reject, 3000);
    });
  } catch {
    redisClient = null;
  }
}

function shutdown(server: unknown) {
  console.log("\n🛑 Shutting down...");
  shutdownSocketIO();
  if (redisClient && typeof (redisClient as Record<string, unknown>).quit === "function") {
    (redisClient as { quit: () => void }).quit();
  }
  if (server && typeof (server as Record<string, unknown>).close === "function") {
    (server as { close: () => void }).close();
  }
  process.exit(0);
}

export { createApp as default };
