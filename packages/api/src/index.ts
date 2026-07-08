/**
 * Main application entry point.
 * PFun CMA Model Routing API - TypeScript Clean-Room Implementation.
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import path from "path";
import nunjucks from "nunjucks";
import { loadConfig, getVersionString } from "./config.js";
import {
  securityHeaders,
  rateLimiter,
  userAgentFilter,
  trustedHostMiddleware,
  penetrationDetection,
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
import { createAdminRoutes } from "./admin/routes.js";
import { initAdminDb, closeAdminDb } from "./admin/db.js";
import { initResultsStore } from "./results.js";
import { setupSocketIO, isSocketIoActive, shutdownSocketIO } from "./socketio.js";

// Redis client (optional)
let redisClient: unknown | null = null;

/**
 * Create a Nunjucks-based template renderer.
 * Falls back to a minimal stub if the template directory does not exist.
 * Mirrors pfun_cma_model/misc/templating.py (Jinja2 → Nunjucks).
 */
function createTemplateRenderer(templateDir: string) {
  // Resolve relative to the current working directory (package root)
  const resolvedDir = path.resolve(templateDir);

  let env: nunjucks.Environment | null = null;
  try {
    // Configure nunjucks with the template directory; autoescape HTML
    env = nunjucks.configure(resolvedDir, {
      autoescape: true,
      noCache: process.env.NODE_ENV !== "production",
    });
  } catch {
    console.warn(`[templates] Could not configure Nunjucks from "${resolvedDir}", using stub.`);
  }

  return (name: string, ctx: Record<string, unknown>): string => {
    if (!env) {
      return stubRender(name, ctx);
    }
    try {
      return env.render(name, ctx);
    } catch (err) {
      console.warn(`[templates] Render failed for "${name}": ${String(err)}, using stub.`);
      return stubRender(name, ctx);
    }
  };
}

function stubRender(name: string, ctx: Record<string, unknown>): string {
  const year = ctx.year ?? new Date().getFullYear();
  const title = name.replace(/\.html\.jinja2$/, "").replace(/-/g, " ");
  const params = ctx.bounded_params as Array<Record<string, unknown>> | undefined;
  // Spec §3.6 demo context requires name/value/description/min/max/step per bounded key.
  const paramsHtml = params?.length
    ? `<table><thead><tr><th>Name</th><th>Value</th><th>Min</th><th>Max</th><th>Step</th></tr></thead><tbody>
        ${params.map((p) => `<tr><td>${p.name}</td><td>${p.value}</td><td>${p.min}</td><td>${p.max}</td><td>${p.step}</td></tr>`).join("")}
       </tbody></table>`
    : "";

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
    ${ctx.access_message ? `<p>${ctx.access_message}</p>` : ""}
    ${paramsHtml}
  </main>
  <footer><p>&copy; ${year} PFun Health</p></footer>
</body>
</html>`;
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
  app.use("*", trustedHostMiddleware(config.trustedHosts));
  app.use("*", penetrationDetection());
  app.use("*", requestTracker(redisClient));

  // Serve static files
  app.use(
    "/static/*",
    serveStatic({ root: path.resolve(config.staticDir, "..") }),
  );

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
  app.route("/admin", createAdminRoutes(config));

  return { app, config };
}

// Main entry point
const isMainModule = process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/^.*[\\/]/, ""));
if (isMainModule) {
  const { app, config } = createApp();

  // --- Startup hooks ---
  // Initialize admin DB schema (mirrors Python lifespan init_models)
  initAdminDb(config);
  console.log("✅ Admin DB initialized.");

  // Initialize results store (mirrors Python duckdb background task setup)
  initResultsStore(config.debug);
  console.log("✅ Results store initialized.");

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
  closeAdminDb();
  if (redisClient && typeof (redisClient as Record<string, unknown>).quit === "function") {
    (redisClient as { quit: () => void }).quit();
  }
  if (server && typeof (server as Record<string, unknown>).close === "function") {
    (server as { close: () => void }).close();
  }
  process.exit(0);
}

export { createApp as default };
