/**
 * Wire contract between the time2move page and its Worker.
 *
 * The Worker is a stateless facade over the PFun CMA engine. A request carries only
 * what the engine needs: the local time of day, a metabolic-profile id, meal times
 * and a binned movement-intensity trace. Nothing that locates or identifies a person
 * has a field here: no coordinates, speed, distance, heart rate, age, weight or
 * readings. Those stay on the device, where the reach estimate is computed.
 */

/** Version of this contract; sent as `v` in both directions. */
export const API_VERSION = 1;

/** The model day is sampled every 2 minutes from 00:00 to 24:00 inclusive. */
export const BIN_MINUTES = 2;
export const BINS_PER_DAY = 720;
export const GRID_POINTS = BINS_PER_DAY + 1;
export const MINUTES_PER_DAY = 1440;

/** Request limits, enforced by the Worker's validator. */
export const MAX_BODY_BYTES = 8192;
export const MAX_MEALS = 4;
export const MAX_WHAT_IF = 3;
export const MAX_WHAT_IF_MINUTES = 360;
/** Movement intensity travels as integers 0..100 (hundredths of the engine's raw activity input). */
export const INTENSITY_SCALE = 100;
/** Hormone and activity signals are returned on a coarser grid (every 10 minutes). */
export const SIGNAL_STEP_MINUTES = 10;
export const SIGNAL_POINTS = MINUTES_PER_DAY / SIGNAL_STEP_MINUTES + 1;
/** The post-meal window counts as open when the forecast reaches its threshold within this lead time. */
export const WINDOW_LEAD_MINUTES = 20;

export const MEAL_SIZES = ["light", "usual", "large"] as const;
export type MealSize = (typeof MEAL_SIZES)[number];

export type Meal = {
  /** Local minutes since midnight, integer 0..1439. Meals are strictly increasing. */
  tMin: number;
  size: MealSize;
};

export type ActivityTrace = {
  /** Index of the first bin in `q` (bin i covers minutes [2i, 2i + 2)), integer 0..719. */
  start: number;
  /**
   * Movement intensity per bin, integers 0..100. May be empty. The last bin may not lie
   * after the bin that contains `nowMin`: the future can only enter through `whatIf`.
   */
  q: number[];
};

export type WhatIf = {
  /** Caller's label, echoed back. `[a-z0-9_-]{1,16}`, unique within a request. */
  id: string;
  /** Intensity to hold from the current bin, integer 1..100. */
  q: number;
  /** How long to hold it, integer minutes 2..360; the model rests afterwards. */
  minutes: number;
};

export type SimulateRequest = {
  v: typeof API_VERSION;
  /** Local minutes since midnight, integer 0..1439. No date and no time zone. */
  nowMin: number;
  profile: {
    /** One of the ids listed by `GET /api/v1/health`. */
    id: string;
    /** Usual middle of sleep, local minutes since midnight (0..1439). Sets the circadian phase. */
    midSleepMin?: number;
  };
  /** 1..4 meals of the model day, eaten or planned. */
  meals: Meal[];
  activity: ActivityTrace;
  /** 0..3 "what if I kept moving" forecasts. */
  whatIf?: WhatIf[];
};

/** Names of the coarse signal rows in a response. All are dimensionless model signals. */
export const SIGNAL_NAMES = ["c", "c_mod", "m", "a", "I_E_eff", "s_A", "A"] as const;
export type SignalName = (typeof SIGNAL_NAMES)[number];

export type SimulateResponse = {
  v: typeof API_VERSION;
  engine: { revision: string | null; sha256: string };
  grid: { stepMin: typeof BIN_MINUTES; n: typeof GRID_POINTS };
  /** Index of the bin that contains `nowMin`. */
  nowIndex: number;
  /** Reference levels in model units, for axis lines and the post-meal window. */
  levels: {
    /** Lowest value of this profile's resting day: "fasting level (model)". */
    fasting: number;
    /** Lowest value of the reference healthy day. */
    referenceFasting: number;
    /** 95th percentile of the reference healthy day: "typical healthy peak (model)". */
    referencePeak: number;
    /** `fasting + 0.25 * (referencePeak - referenceFasting)`: a meal counts as "still in the curve" above this. */
    threshold: number;
  };
  /** Modelled glucose in relative model units (not mg/dL or mmol/L), 721 values each. */
  glucose: {
    /** The same day with no movement at all. */
    rest: number[];
    /** With the movement observed so far, then rest from now. */
    actual: number[];
  };
  /** Model internals from the `actual` run, every 10 minutes (145 values each). */
  signals: { stepMin: typeof SIGNAL_STEP_MINUTES; n: typeof SIGNAL_POINTS } & Record<SignalName, number[]>;
  now: {
    rest: number;
    actual: number;
    /** The engine's normalised activity for the current bin, 0..1. */
    A: number;
    /** The engine's post-activity sensitisation state for the current bin. */
    sA: number;
  };
  /**
   * Post-meal window from the rest-from-now forecast: the span during which modelled
   * glucose stays above `levels.threshold`. Times are local minutes since midnight.
   */
  window: {
    /** True when the forecast is above the threshold now or reaches it within WINDOW_LEAD_MINUTES. */
    open: boolean;
    /** When the forecast next rises above the threshold; null when it is above it already or never does today. */
    opensAtMin: number | null;
    /** When it then falls back below (1440 = stays above until the end of the model day); null without a window. */
    closesAtMin: number | null;
  };
  whatIf: Array<WhatIf & {
    /** Index of the first value in `G` (the current bin). */
    fromIndex: number;
    /** Modelled glucose from `fromIndex` to the end of the day. */
    G: number[];
  }>;
  notice: string;
};

export type HealthProfile = { id: string; name: string; description: string };

export type HealthResponse = {
  ok: boolean;
  api: typeof API_VERSION;
  engine: { loaded: boolean; revision: string | null; sha256: string | null; maxPoints: number | null; maxMeals: number | null };
  /** Metabolic profiles this deployment's engine build offers (illustrative presets, not diagnoses). */
  profiles: HealthProfile[];
  /** Id of the reference profile; always present in `profiles`. */
  defaultProfile: string;
  /**
   * The engine's intensity response: its normalised activity for raw intensities
   * 0, 0.01, ..., 1. The page uses it to average stop-and-go movement within a bin
   * the way the engine would have seen it. Null when the engine is not loaded.
   */
  intensityResponse: { step: number; a: number[] } | null;
};

export const ERROR_CODES = [
  "invalid_json",
  "invalid_request",
  "unsupported_version",
  "not_found",
  "method_not_allowed",
  "payload_too_large",
  "unsupported_media_type",
  "rate_limited",
  "engine_error",
  "engine_unavailable",
] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

export type ErrorResponse = {
  error: string;
  code: ErrorCode;
  /** Where validation failed, e.g. `meals[1].tMin`. */
  path?: string;
};

export const NOTICE = "Model estimates from the PFun CMA engine. Not measurements, not medical advice.";

/** Bin index that contains a local minute of the day. */
export function binOf(minute: number): number {
  return Math.min(BINS_PER_DAY - 1, Math.max(0, Math.floor(minute / BIN_MINUTES)));
}
