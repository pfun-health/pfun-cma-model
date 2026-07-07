/**
 * Model execution routes: /model/*
 */

import { Hono } from "hono";
import { stream } from "hono/streaming";
import {
  CMASleepWakeModel,
  CMAModelParamsSchema,
  type CMAModelParams,
} from "@pfun/core";

export function createModelRoutes(): Hono {
  const app = new Hono();

  // POST /model/run
  app.post("/run", async (c) => {
    try {
      const body = await c.req.json().catch(() => ({}));
      const config = body && Object.keys(body).length > 0 ? body : undefined;

      const model = new CMASleepWakeModel();
      const result = model.run(config);
      const jsonStr = JSON.stringify(result);

      return new Response(jsonStr, {
        status: 200,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (err) {
      return c.json(
        {
          error: "failed to run model",
          exception: String(err),
        },
        500,
      );
    }
  });

  // POST /model/run-at-time
  app.post("/run-at-time", async (c) => {
    try {
      const body = await c.req.json();
      const { t0 = 0, t1 = 100, n = 100, config } = body;

      const model = new CMASleepWakeModel();
      const results = model.runAtTime(t0, t1, n, config);
      return c.json(results);
    } catch (err) {
      return c.json(
        {
          error: "failed to run at time. See error message on server log.",
          exception: String(err),
        },
        500,
      );
    }
  });

  // POST /model/run-at-time/stream
  app.post("/run-at-time/stream", async (c) => {
    try {
      const body = await c.req.json();
      const { t0 = 0, t1 = 100, n = 100, config } = body;

      const model = new CMASleepWakeModel();

      return stream(c, async (s) => {
        c.header("Content-Type", "application/x-ndjson");
        try {
          for (const point of model.runAtTimeStream(t0, t1, n, config)) {
            await s.write(JSON.stringify(point) + "\n");
          }
        } catch (err) {
          await s.write(
            JSON.stringify({
              error: "failed to run at time. See error message on server log.",
              status_code: 500,
            }) + "\n",
          );
        }
      });
    } catch (err) {
      return c.json(
        {
          error: "failed to run at time. See error message on server log.",
          exception: String(err),
        },
        500,
      );
    }
  });

  // POST /model/fit
  app.post("/fit", async (c) => {
    try {
      const body = await c.req.json();
      let { data, config } = body;

      // Load sample data if empty
      if (!data || (typeof data === "object" && Object.keys(data).length === 0)) {
        // Use synthetic sample data
        data = generateSampleData();
      }

      if (typeof data === "string") {
        try {
          data = JSON.parse(data);
        } catch (parseErr) {
          return c.json(
            {
              error: "Invalid data format",
              exception: String(parseErr),
              exception_type: "JSONDecodeError",
            },
            400,
          );
        }
      }

      // Parse config if string
      let parsedConfig: Partial<CMAModelParams> | undefined;
      if (typeof config === "string") {
        try {
          parsedConfig = CMAModelParamsSchema.partial().parse(JSON.parse(config));
        } catch {
          parsedConfig = undefined;
        }
      } else if (config) {
        parsedConfig = config;
      }

      const { fitModel } = await import("@pfun/core");
      const result = fitModel(
        { t: data.t ?? data.time ?? [], G: data.G ?? data.glucose ?? [] },
        parsedConfig,
      );

      return c.json(result);
    } catch (err) {
      const isValidation =
        err instanceof Error &&
        (err.name === "ZodError" || err.message.includes("validation"));
      return c.json(
        {
          error: isValidation ? "Validation error" : "Fit failed",
          exception: String(err),
          exception_type: err instanceof Error ? err.constructor.name : "Error",
        },
        isValidation ? 400 : 500,
      );
    }
  });

  return app;
}

/**
 * Generate synthetic sample data for fitting.
 */
function generateSampleData(): { t: number[]; G: number[] } {
  const model = new CMASleepWakeModel();
  const results = model.runAtTime(0, 24, 100);
  return {
    t: results.map((r) => parseFloat(r.x)),
    G: results.map((r) => parseFloat(r.y) + (Math.random() - 0.5) * 0.1),
  };
}
