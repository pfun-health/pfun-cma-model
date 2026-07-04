/**
 * Health and top-level routes.
 */

import { Hono } from "hono";
import { readFile } from "fs/promises";
import { join } from "path";

export function createHealthRoutes(
  socketIoActive: () => boolean,
  renderTemplate: (name: string, ctx: Record<string, unknown>) => string,
  staticDir: string,
): Hono {
  const app = new Hono();

  app.get("/health", (c) => {
    return c.json({ status: "ok", message: "PFun CMA Model API is running." });
  });

  app.get("/health/ws/run-at-time", (c) => {
    if (socketIoActive()) {
      return c.json({
        status: "ok",
        message: "'run-at-time' WebSocket is running.",
      });
    }
    return c.json(
      { status: "error", message: "'run-at-time' WebSocket is NOT running." },
      503,
    );
  });

  app.get("/", (c) => {
    const html = renderTemplate("index.html.jinja2", {
      year: new Date().getFullYear(),
      access_message: "Welcome to the PFun CMA Model API.",
    });
    return c.html(html);
  });

  app.get("/about", (c) => {
    const html = renderTemplate("about-doc.html.jinja2", {});
    return c.html(html);
  });

  app.get("/pitch", (c) => {
    const html = renderTemplate("pitch-doc.html.jinja2", {});
    return c.html(html);
  });

  app.get("/login", (c) => {
    const html = renderTemplate("sqladmin/login.html", {});
    return c.html(html);
  });

  app.get("/favicon.ico", async (c) => {
    try {
      const faviconPath = join(staticDir, "favicon.ico");
      const data = await readFile(faviconPath);
      return new Response(data, {
        headers: { "Content-Type": "image/x-icon" },
      });
    } catch {
      return c.notFound();
    }
  });

  return app;
}
