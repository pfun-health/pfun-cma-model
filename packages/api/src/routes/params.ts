/**
 * Parameters routes: /params/*
 */

import { Hono } from "hono";
import {
  CMAModelParamsSchema,
  getDefaultParams,
  getParamsJsonSchema,
  generateParamsTable,
  BOUNDED_PARAM_KEYS,
  calcSerr,
  getQualitativeDescriptor,
  BOUNDED_PARAM_DESCRIPTIONS,
  type BoundedParamKey,
} from "@pfun/core";

export function createParamsRoutes(): Hono {
  const app = new Hono();

  // GET /params/schema
  app.get("/schema", (c) => {
    return c.json(getParamsJsonSchema());
  });

  // GET /params/default
  app.get("/default", (c) => {
    const defaults = getDefaultParams();
    return c.json(defaults);
  });

  // POST /params/describe
  app.post("/describe", async (c) => {
    const body = await c.req.json();
    const params = CMAModelParamsSchema.parse(body);

    const descriptions: Record<
      string,
      { description: string; qualitative: string; value: number }
    > = {};

    for (const key of BOUNDED_PARAM_KEYS) {
      const value = params[key] as number;
      const serr = calcSerr(key, value);
      const qualitative = getQualitativeDescriptor(serr);
      descriptions[key] = {
        description: BOUNDED_PARAM_DESCRIPTIONS[key],
        qualitative,
        value,
      };
    }

    return c.json(descriptions);
  });

  // POST /params/tabulate/:output_fmt
  app.post("/tabulate/:output_fmt", async (c) => {
    const outputFmt = c.req.param("output_fmt") as "json" | "html" | "md";
    if (!["json", "html", "md"].includes(outputFmt)) {
      return c.json({ detail: "Invalid output format" }, 400);
    }

    const body = await c.req.json();
    const params = CMAModelParamsSchema.parse(body);
    const table = generateParamsTable(params, outputFmt);

    if (outputFmt === "md") {
      return new Response(table, {
        headers: { "Content-Type": "text/markdown" },
      });
    } else if (outputFmt === "html") {
      return c.html(table);
    } else {
      return c.json(JSON.parse(table));
    }
  });

  return app;
}
