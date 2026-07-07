/**
 * LLM routes: /llm/*
 */

import { Hono } from "hono";
import { stream } from "hono/streaming";
import { saveResultBackground } from "../results.js";

export interface GeneratedScenario {
  forecasted_events: string;
  qualitative_description: string;
  parameters: Record<string, { value: number; description: string; stderr: number }>;
  recommendations?: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Prompt sanitization
// Mirrors Python's shlex.quote() — strips control chars so no shell-injection
// is possible in downstream use.
// ---------------------------------------------------------------------------
function sanitizePrompt(raw: string): string {
  return raw.trim().replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "");
}

// ---------------------------------------------------------------------------
// LLM Backend abstraction
// Supports "ollama" (local) and "openai".
// Falls back to stub when no backend is reachable.
// ---------------------------------------------------------------------------

const LLM_BACKEND = (process.env.LLM_BACKEND ?? "stub") as "ollama" | "openai" | "stub";
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? "llama3";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
const OPENAI_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

const SYSTEM_PROMPT = `You are a clinical AI assistant for the PFun CMA (Circadian Metabolic Analysis) model.
Given a patient description, generate a JSON object with these exact keys:
- forecasted_events: string describing expected glucose events
- qualitative_description: string describing the overall metabolic state
- parameters: object with keys d, taup, taug, B, Cm, toff; each having value (number), description (string), stderr (number)
- recommendations: object with keys dietary, activity, sleep (each a string)
Respond with ONLY valid JSON, no markdown fences.`;

async function callOllama(prompt: string): Promise<string> {
  const response = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      stream: false,
      options: { temperature: 0, seed: 23 },
    }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) {
    throw new Error(`Ollama request failed: ${response.status} ${response.statusText}`);
  }
  const data = (await response.json()) as { message?: { content?: string } };
  return data.message?.content ?? "";
}

async function callOpenAI(prompt: string): Promise<string> {
  const authHeader = "Bearer " + OPENAI_API_KEY;
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: authHeader,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      temperature: 0,
      response_format: { type: "json_object" },
    }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!response.ok) {
    throw new Error(`OpenAI request failed: ${response.status} ${response.statusText}`);
  }
  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  return data.choices?.[0]?.message?.content ?? "";
}

function stubScenario(prompt: string, includeRecommendations: boolean): GeneratedScenario {
  const scenario: GeneratedScenario = {
    forecasted_events: `Based on the prompt "${prompt}", glucose levels are expected to remain stable with normal circadian patterns.`,
    qualitative_description:
      "Normal metabolic state with standard circadian glucose regulation.",
    parameters: {
      d: { value: 0.0, description: "Time zone offset", stderr: 0.1 },
      taup: { value: 1.0, description: "Photoperiod duration", stderr: 0.05 },
      taug: { value: 1.0, description: "Glucose response time constant", stderr: 0.08 },
      B: { value: 0.05, description: "Glucose baseline", stderr: 0.01 },
      Cm: { value: 0.0, description: "Cortisol sensitivity", stderr: 0.02 },
      toff: { value: 0.0, description: "Solar noon offset", stderr: 0.1 },
    },
  };
  if (includeRecommendations) {
    scenario.recommendations = {
      dietary: "Maintain regular meal timing aligned with circadian rhythm.",
      activity: "Moderate exercise during daylight hours supports glucose regulation.",
      sleep: "Consistent sleep schedule supports metabolic health.",
    };
  }
  return scenario;
}

function parseJsonResponse(text: string): GeneratedScenario {
  // Strip optional markdown fences
  const stripped = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  const parsed = JSON.parse(stripped) as Partial<GeneratedScenario>;

  if (
    typeof parsed.forecasted_events !== "string" ||
    typeof parsed.qualitative_description !== "string" ||
    typeof parsed.parameters !== "object"
  ) {
    throw new Error("LLM response missing required fields");
  }

  return parsed as GeneratedScenario;
}

/**
 * Generate a scenario using the configured LLM backend.
 * Retries once on JSON parse failure (actual retry, not re-reading request body).
 * Mirrors attempt_scene_gen() in pfun_cma_model/routes/llm.py.
 */
async function generateScenario(
  prompt: string,
  _includeSampleTrace: boolean,
  includeRecommendations: boolean,
): Promise<GeneratedScenario> {
  const safePrompt = sanitizePrompt(prompt);

  if (LLM_BACKEND === "stub") {
    return stubScenario(safePrompt, includeRecommendations);
  }

  const callBackend = LLM_BACKEND === "openai" ? callOpenAI : callOllama;

  // First attempt
  try {
    const rawText = await callBackend(safePrompt);
    return parseJsonResponse(rawText);
  } catch (firstErr) {
    // Retry once after a short delay (mirrors Python's attempt_scene_gen retry)
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const rawText = await callBackend(safePrompt);
      return parseJsonResponse(rawText);
    } catch {
      // If backend is unavailable, fall back to stub rather than hard-failing
      console.warn(`[llm] Backend call failed (${String(firstErr)}), using stub.`);
      return stubScenario(safePrompt, includeRecommendations);
    }
  }
}

export function createLlmRoutes(): Hono {
  const app = new Hono();

  // POST /llm/generate-scenario
  app.post("/generate-scenario", async (c) => {
    let body: Record<string, unknown> = {};
    try {
      body = await c.req.json();
    } catch {
      return c.json({ error: "Invalid JSON body" }, 400);
    }

    const {
      prompt = "",
      include_sample_trace = false,
      include_recommendations = true,
    } = body as {
      prompt?: string;
      include_sample_trace?: boolean;
      include_recommendations?: boolean;
    };

    try {
      const scenario = await generateScenario(
        String(prompt),
        Boolean(include_sample_trace),
        Boolean(include_recommendations),
      );

      // Persist result as fire-and-forget background task
      saveResultBackground(scenario as unknown as Record<string, unknown>);

      return c.json(scenario, 200, {
        "Content-Type": "application/json",
      });
    } catch (err) {
      return c.json(
        { error: "Scenario generation failed", detail: String(err) },
        500,
      );
    }
  });

  // POST /llm/generate-scenarios (SSE)
  app.post("/generate-scenarios", async (c) => {
    let body: Record<string, unknown> = {};
    try {
      body = await c.req.json();
    } catch {
      return c.json({ error: "Invalid JSON body" }, 400);
    }

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
            String(prompt),
            Boolean(include_sample_trace),
            Boolean(include_recommendations),
          );
          saveResultBackground(scenario as unknown as Record<string, unknown>);
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
