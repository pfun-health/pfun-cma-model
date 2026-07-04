import { z } from "zod";
import { Bounds } from "./bounds.js";

/**
 * Bounded parameter keys and their metadata.
 */
export const BOUNDED_PARAM_KEYS = [
  "d",
  "taup",
  "taug",
  "B",
  "Cm",
  "toff",
] as const;

export type BoundedParamKey = (typeof BOUNDED_PARAM_KEYS)[number];

export const BOUNDED_PARAM_LB: Record<BoundedParamKey, number> = {
  d: -12.0,
  taup: 0.5,
  taug: 0.1,
  B: 0.0,
  Cm: 0.0,
  toff: -3.0,
};

export const BOUNDED_PARAM_UB: Record<BoundedParamKey, number> = {
  d: 14.0,
  taup: 3.0,
  taug: 3.0,
  B: 1.0,
  Cm: 2.0,
  toff: 3.0,
};

export const BOUNDED_PARAM_MID: Record<BoundedParamKey, number> = {
  d: 0.0,
  taup: 1.0,
  taug: 1.0,
  B: 0.05,
  Cm: 0.0,
  toff: 0.0,
};

export const BOUNDED_PARAM_STEPS: Record<BoundedParamKey, number> = {
  d: (14.0 + -12.0) * 0.0125,
  taup: (3.0 + 0.5) * 0.0125,
  taug: (3.0 + 0.1) * 0.0125,
  B: (1.0 + 0.0) * 0.0125,
  Cm: (2.0 + 0.0) * 0.0125,
  toff: (3.0 + -3.0) * 0.0125,
};

export const BOUNDED_PARAM_DESCRIPTIONS: Record<BoundedParamKey, string> = {
  d: "Time zone offset; scalar hours[time]; Estimated effects of photoperiod offset; correlates with peak light exposure time relative to solar noon",
  taup: "Photoperiod duration; scalar hours[time]; Estimated number of hours of light exposure (relative to darkness) in a 24-hour period; correlates with light exposure duration",
  taug: "Glucose meal-response time constant; dimensionless[time]; correlates with the rate of postprandial glucose metabolism; higher values indicate slower return to baseline glucose levels after meals, which can mitigate hypoglycemia risk by increasing the time until glucose levels drop dangerously low",
  B: "Glucose baseline constant; dimensionless[Glucose]; correlates with basal glucose levels; correlates with A1C-- values higher than 0.05 indicate elevated baseline glucose levels, which can increase hyperglycemia risk",
  Cm: "Cortisol sensitivity coefficient; dimensionless[Cortisol]; correlates with the influence of cortisol on glucose variability; higher values indicate greater cortisol sensitivity, which can increase glucose variability, thereby increasing hyperglycemia/hypoglycemia risk",
  toff: "Solar-noon offset; hours[time]; correlates with the timing of solar noon relative to the individual's circadian phase; can reflect chronotype and influence the alignment of circadian rhythms with the external light-dark cycle, which can impact glucose metabolism and overall metabolic health",
};

export const DEFAULT_BOUNDS = new Bounds(
  BOUNDED_PARAM_KEYS.map((k) => BOUNDED_PARAM_LB[k]),
  BOUNDED_PARAM_KEYS.map((k) => BOUNDED_PARAM_UB[k]),
  BOUNDED_PARAM_KEYS.map(() => true),
);

/**
 * CMA Model Parameters schema with Zod.
 */
export const CMAModelParamsSchema = z.object({
  N: z.number().int().default(1024),
  d: z.number().default(0.0),
  taup: z.number().default(1.0),
  taug: z.union([z.number(), z.array(z.number())]).default(1.0),
  B: z.number().default(0.05),
  Cm: z.number().default(0.0),
  toff: z.number().default(0.0),
  tM: z.array(z.number()).default([7.0, 11.0, 17.5]),
  seed: z.number().nullable().default(null),
  eps: z.number().default(1e-18),
});

export type CMAModelParams = z.infer<typeof CMAModelParamsSchema>;

/**
 * Get default CMA model parameters.
 */
export function getDefaultParams(): CMAModelParams {
  return CMAModelParamsSchema.parse({});
}

/**
 * Get JSON schema for CMAModelParams (simplified representation).
 */
export function getParamsJsonSchema(): Record<string, unknown> {
  return {
    title: "CMAModelParams",
    type: "object",
    properties: {
      N: { type: "integer", default: 1024, description: "Number of time points" },
      d: { type: "number", default: 0.0, description: "Time zone offset (hours)" },
      taup: {
        type: "number",
        default: 1.0,
        description: "Circadian-relative photoperiod length",
      },
      taug: {
        anyOf: [{ type: "number" }, { type: "array", items: { type: "number" } }],
        default: 1.0,
        description: "Glucose response time constant",
      },
      B: {
        type: "number",
        default: 0.05,
        description: "Glucose Bias constant",
      },
      Cm: {
        type: "number",
        default: 0.0,
        description: "Cortisol temporal sensitivity coefficient",
      },
      toff: {
        type: "number",
        default: 0.0,
        description: "Solar noon offset (latitude)",
      },
      tM: {
        type: "array",
        items: { type: "number" },
        default: [7.0, 11.0, 17.5],
        description: "Meal times (hours)",
      },
      seed: {
        anyOf: [{ type: "number" }, { type: "null" }],
        default: null,
        description: "Random seed",
      },
      eps: {
        type: "number",
        default: 1e-18,
        description: "Random noise scale (epsilon)",
      },
    },
    required: [],
  };
}

/**
 * Qualitative descriptor based on standardized error.
 */
export function getQualitativeDescriptor(serr: number): string {
  const EPS = 0.1 + 1e-8;
  const parts: string[] = [];

  const isVery = Math.abs(serr) >= 0.23;
  const isLow = serr <= -EPS;
  const isHigh = serr >= EPS;
  const isNormal = serr >= -EPS && serr <= EPS;

  if (isVery) parts.push("Very");
  if (isLow) parts.push("Low");
  if (isNormal) parts.push("Normal");
  if (isHigh) parts.push("High");

  return parts.join(" ") || "Normal";
}

/**
 * Calculate standardized error for a bounded parameter.
 */
export function calcSerr(paramKey: BoundedParamKey, value: number): number {
  const mid = BOUNDED_PARAM_MID[paramKey];
  const range = BOUNDED_PARAM_UB[paramKey] - BOUNDED_PARAM_LB[paramKey];
  return (value - mid) / range;
}

/**
 * Describe a parameter with its qualitative descriptor.
 */
export function describeParam(paramKey: BoundedParamKey, value: number): string {
  const serr = calcSerr(paramKey, value);
  const qual = getQualitativeDescriptor(serr);
  return `${BOUNDED_PARAM_DESCRIPTIONS[paramKey]} (${qual})`;
}

/**
 * Get bounded parameter info for a given key and value.
 */
export function getBoundedParamInfo(
  paramKey: BoundedParamKey,
  value: number,
): {
  name: string;
  value: number;
  description: string;
  step: number;
  min: number;
  max: number;
  default: number;
} {
  return {
    name: paramKey,
    value,
    description: BOUNDED_PARAM_DESCRIPTIONS[paramKey],
    step: BOUNDED_PARAM_STEPS[paramKey],
    min: BOUNDED_PARAM_LB[paramKey],
    max: BOUNDED_PARAM_UB[paramKey],
    default: BOUNDED_PARAM_MID[paramKey],
  };
}

/**
 * Generate markdown/html/json table of bounded parameters.
 */
export function generateParamsTable(
  params: CMAModelParams,
  outputFmt: "md" | "html" | "json",
): string {
  const headers = [
    "Parameter",
    "Type",
    "Value",
    "Default",
    "Lower Bound",
    "Upper Bound",
    "Description",
  ];

  const rows = BOUNDED_PARAM_KEYS.map((key) => {
    const value = params[key] as number;
    const serr = calcSerr(key, value);
    const qual = getQualitativeDescriptor(serr);
    const desc = `${BOUNDED_PARAM_DESCRIPTIONS[key]} (${qual})`;
    return [
      key,
      "float",
      String(value),
      String(BOUNDED_PARAM_MID[key]),
      String(BOUNDED_PARAM_LB[key]),
      String(BOUNDED_PARAM_UB[key]),
      desc,
    ];
  });

  if (outputFmt === "md") {
    return formatMarkdownTable(headers, rows);
  } else if (outputFmt === "html") {
    return formatHtmlTable(headers, rows);
  } else {
    // json - return JSON containing markdown table string
    return JSON.stringify({ table: formatMarkdownTable(headers, rows) });
  }
}

function formatMarkdownTable(headers: string[], rows: string[][]): string {
  const sep = headers.map((h) => "-".repeat(h.length));
  const lines = [
    `| ${headers.join(" | ")} |`,
    `| ${sep.join(" | ")} |`,
    ...rows.map((r) => `| ${r.join(" | ")} |`),
  ];
  return lines.join("\n");
}

function formatHtmlTable(headers: string[], rows: string[][]): string {
  const headerRow = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  const bodyRows = rows
    .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("\n");
  return `<table>\n<thead>\n${headerRow}\n</thead>\n<tbody>\n${bodyRows}\n</tbody>\n</table>`;
}
