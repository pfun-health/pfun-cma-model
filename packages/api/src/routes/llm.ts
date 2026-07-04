/**
 * LLM routes: /llm/*
 */

import { Hono } from "hono";
import { stream } from "hono/streaming";

export interface GeneratedScenario {
  forecasted_events: string;
  qualitative_description: string;
  parameters: Record<string, { value: number; description: string; stderr: number }>;
  recommendations?: Record<string, string>;
}

export function createLlmRoutes(): Hono {
  const app = new Hono();

  // POST /llm/generate-scenario
  app.post("/generate-scenario", async (c) => {
    try {
      const body = await c.req.json();
      const {
        prompt,
        include_sample_trace = false,
        include_recommendations = true,
      } = body;

      const scenario = await generateScenario(
        prompt,
        include_sample_trace,
        include_recommendations,
      );

      return c.json(scenario, 200, {
        "Content-Type": "application/json",
      });
    } catch (err) {
      // Retry once on serialization failure
      try {
        const body = await c.req.json().catch(() => ({}));
        const scenario = await generateScenario(
          (body as Record<string, unknown>).prompt as string ?? "",
          false,
          true,
        );
        return c.json(scenario);
      } catch (retryErr) {
        return c.json(
          { error: "Scenario generation failed", detail: String(retryErr) },
          500,
        );
      }
    }
  });

  // POST /llm/generate-scenarios (SSE)
  app.post("/generate-scenarios", async (c) => {
    const body = await c.req.json();
    const {
      prompts = [],
      include_sample_trace = false,
      include_recommendations = true,
    } = body as {
      prompts: string[];
      include_sample_trace: boolean;
      include_recommendations: boolean;
    };

    return stream(c, async (s) => {
      c.header("Content-Type", "text/event-stream");
      c.header("Cache-Control", "no-cache");
      c.header("Connection", "keep-alive");

      let id = 1;
      for (const prompt of prompts) {
        try {
          const scenario = await generateScenario(
            prompt,
            include_sample_trace,
            include_recommendations,
          );
          await s.write(`id: ${id}\n`);
          await s.write(`event: generated_scenario\n`);
          await s.write(`retry: 2300\n`);
          await s.write(`data: ${JSON.stringify(scenario)}\n\n`);
        } catch (err) {
          await s.write(`id: ${id}\n`);
          await s.write(`event: generated_scenario\n`);
          await s.write(`retry: 2300\n`);
          await s.write(
            `data: ${JSON.stringify({ error: String(err) })}\n\n`,
          );
        }
        id++;
      }
    });
  });

  return app;
}

/**
 * Generate a scenario (stub implementation - in production connects to LLM).
 */
async function generateScenario(
  prompt: string,
  includeSampleTrace: boolean,
  includeRecommendations: boolean,
): Promise<GeneratedScenario> {
  // This would call an actual LLM in production
  const scenario: GeneratedScenario = {
    forecasted_events: `Based on the prompt "${prompt}", glucose levels are expected to remain stable with normal circadian patterns.`,
    qualitative_description:
      "Normal metabolic state with standard circadian glucose regulation.",
    parameters: {
      d: { value: 0.0, description: "Time zone offset", stderr: 0.1 },
      taup: { value: 1.0, description: "Photoperiod duration", stderr: 0.05 },
      taug: {
        value: 1.0,
        description: "Glucose response time constant",
        stderr: 0.08,
      },
      B: { value: 0.05, description: "Glucose baseline", stderr: 0.01 },
      Cm: {
        value: 0.0,
        description: "Cortisol sensitivity",
        stderr: 0.02,
      },
      toff: { value: 0.0, description: "Solar noon offset", stderr: 0.1 },
    },
  };

  if (includeRecommendations) {
    scenario.recommendations = {
      dietary:
        "Maintain regular meal timing aligned with circadian rhythm.",
      activity:
        "Moderate exercise during daylight hours supports glucose regulation.",
      sleep: "Consistent sleep schedule supports metabolic health.",
    };
  }

  return scenario;
}
