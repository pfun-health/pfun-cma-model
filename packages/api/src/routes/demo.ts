/**
 * Demo routes: /demo/*
 * HTML template-rendered demo pages.
 */

import { Hono } from "hono";
import {
  BOUNDED_PARAM_KEYS,
  getBoundedParamInfo,
  getDefaultParams,
} from "@pfun/core";

export function createDemoRoutes(
  renderTemplate: (name: string, ctx: Record<string, unknown>) => string,
): Hono {
  const app = new Hono();

  function getDemoContext(): Record<string, unknown> {
    const params = getDefaultParams();
    const boundedParamsMeta = BOUNDED_PARAM_KEYS.map((key) =>
      getBoundedParamInfo(key, params[key] as number),
    );

    return {
      year: new Date().getFullYear(),
      bounded_params: boundedParamsMeta,
    };
  }

  app.get("/llm", (c) => {
    const html = renderTemplate("llm-demo.html.jinja2", getDemoContext());
    return c.html(html);
  });

  app.get("/data-stream", (c) => {
    const html = renderTemplate(
      "data-stream-demo.html.jinja2",
      getDemoContext(),
    );
    return c.html(html);
  });

  app.get("/run-at-time", (c) => {
    const html = renderTemplate(
      "run-at-time-demo.html.jinja2",
      getDemoContext(),
    );
    return c.html(html);
  });

  app.get("/canvas-wave", (c) => {
    const html = renderTemplate(
      "canvas-wave-demo.html.jinja2",
      getDemoContext(),
    );
    return c.html(html);
  });

  app.get("/full-model-run", (c) => {
    const html = renderTemplate(
      "full-model-run-demo.html.jinja2",
      getDemoContext(),
    );
    return c.html(html);
  });

  app.get("/webgl-demo", (c) => {
    const html = renderTemplate("webgl-demo.html.jinja2", getDemoContext());
    return c.html(html);
  });

  return app;
}
