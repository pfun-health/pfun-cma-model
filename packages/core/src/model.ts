/**
 * CMASleepWakeModel - The Cortisol-Melatonin-Adiponectin Sleep-Wake model.
 * Clean-room TypeScript implementation.
 */

import { E, Light, exp, linspace, computeG, mealDistr } from "./calc.js";
import {
  CMAModelParams,
  CMAModelParamsSchema,
  BOUNDED_PARAM_KEYS,
  type BoundedParamKey,
} from "./params.js";

export interface ModelRunRow {
  t: number;
  c: number;
  m: number;
  a: number;
  I_S: number;
  I_E: number;
  L: number;
  G: number;
  is_meal: boolean;
  [key: string]: number | boolean;
}

export interface RunAtTimeResult {
  x: string;
  y: string;
}

/**
 * The CMA Sleep-Wake Model.
 */
export class CMASleepWakeModel {
  private _params: CMAModelParams;

  constructor(params?: Partial<CMAModelParams>) {
    this._params = CMAModelParamsSchema.parse(params ?? {});
  }

  get params(): CMAModelParams {
    return { ...this._params };
  }

  /**
   * Update model parameters.
   */
  updateParams(updates: Partial<CMAModelParams>): void {
    this._params = CMAModelParamsSchema.parse({ ...this._params, ...updates });
  }

  /**
   * Solve/run the model in place, retaining the latest run output.
   * CLI compatibility alias; equivalent to {@link run} but discards the
   * returned rows (the CLI only needs the side-effect of execution).
   */
  solve(): void {
    this.run();
  }

  /**
   * Generate the time vector.
   */
  getTimeVector(t0: number = 0, t1: number = 24, n?: number): number[] {
    return linspace(t0, t1, n ?? this._params.N);
  }

  /**
   * Run the full CMA model.
   * Returns array of rows with t, c, m, a, I_S, I_E, L, G, g_0..g_n, is_meal columns.
   */
  run(config?: Partial<CMAModelParams>): ModelRunRow[] {
    if (config) {
      this.updateParams(config);
    }

    const p = this._params;
    const t = this.getTimeVector();
    const n = t.length;
    const dt = t.length > 1 ? t[1] - t[0] : 1;

    // Initialize arrays
    const c = new Array<number>(n).fill(0); // cortisol
    const m = new Array<number>(n).fill(0); // melatonin
    const a = new Array<number>(n).fill(0); // adiponectin
    const I_S = new Array<number>(n).fill(0); // sleep pressure (insulin sensitivity proxy)
    const I_E = new Array<number>(n).fill(0); // extracellular insulin
    const L_arr = new Array<number>(n).fill(0); // light

    // Random noise
    const rng = p.seed !== null ? createSeededRandom(p.seed) : () => 0;
    const eps = p.eps ?? 1e-18;

    // Compute light, cortisol, melatonin over time
    for (let i = 0; i < n; i++) {
      const ti = t[i];
      const hour = ((ti % 24) + 24) % 24; // normalize to [0, 24)

      // Light function based on photoperiod
      const lightPhase = (hour - 12 + p.d) / p.taup;
      L_arr[i] = Light(lightPhase);

      // Cortisol: circadian with photoperiod
      const cortPhase = (2 * Math.PI * (hour - 6 + p.d)) / 24;
      c[i] = 0.5 * (1.0 + Math.cos(cortPhase)) + p.Cm * mealDistr(p.Cm, ti, p.toff);

      // Melatonin: inverse of light with phase delay
      const melPhase = (2 * Math.PI * (hour - 2 + p.d)) / 24;
      m[i] = E(-Light(lightPhase)) * (0.5 * (1.0 + Math.cos(melPhase)));

      // Adiponectin: sleep-dependent
      a[i] = 0.5 * (1.0 - E(c[i] - 0.5)) + eps * rng();

      // Sleep pressure (cumulative process S)
      I_S[i] =
        i > 0
          ? I_S[i - 1] + dt * (L_arr[i] > 0.5 ? -0.1 * I_S[i - 1] : 0.05 * (1 - I_S[i - 1]))
          : 0.5;

      // Extracellular insulin proxy
      I_E[i] = 0.1 * c[i] + 0.05 * I_S[i] + eps * rng();
    }

    // Compute glucose for each meal
    const gMeals = computeG(t, I_E, p.tM, p.taug, p.B, p.Cm, p.toff, true);

    // Aggregate glucose
    const G = t.map((_, i) => {
      let sum = 0;
      for (let j = 0; j < gMeals.length; j++) {
        sum += gMeals[j][i];
      }
      return sum;
    });

    // Determine meal proximity
    const isMeal = t.map((ti) => {
      return p.tM.some((tm) => Math.abs(ti - tm) < 0.5);
    });

    // Build result rows
    const results: ModelRunRow[] = [];
    for (let i = 0; i < n; i++) {
      const row: ModelRunRow = {
        t: t[i],
        c: c[i],
        m: m[i],
        a: a[i],
        I_S: I_S[i],
        I_E: I_E[i],
        L: L_arr[i],
        G: G[i],
        is_meal: isMeal[i],
      };
      // Add individual meal glucose columns
      for (let j = 0; j < gMeals.length; j++) {
        row[`g_${j}`] = gMeals[j][i];
      }
      results.push(row);
    }

    return results;
  }

  /**
   * Run model at specific time points and return glucose values.
   * Used for the /model/run-at-time endpoint.
   */
  runAtTime(t0: number, t1: number, n: number, config?: Partial<CMAModelParams>): RunAtTimeResult[] {
    if (config) {
      this.updateParams(config);
    }

    const t = linspace(t0, t1, n);
    const p = this._params;
    const I_E = new Array<number>(n).fill(0.05); // simplified constant I_E for point query

    const gMeals = computeG(t, I_E, p.tM, p.taug, p.B, p.Cm, p.toff, true);

    return t.map((ti, i) => {
      let G = 0;
      for (let j = 0; j < gMeals.length; j++) {
        G += gMeals[j][i];
      }
      return { x: String(ti), y: String(G) };
    });
  }

  /**
   * Run at time as a generator for streaming.
   */
  *runAtTimeStream(
    t0: number,
    t1: number,
    n: number,
    config?: Partial<CMAModelParams>,
  ): Generator<RunAtTimeResult> {
    if (config) {
      this.updateParams(config);
    }

    const t = linspace(t0, t1, n);
    const p = this._params;
    const I_E = new Array<number>(n).fill(0.05);
    const gMeals = computeG(t, I_E, p.tM, p.taug, p.B, p.Cm, p.toff, true);

    for (let i = 0; i < t.length; i++) {
      let G = 0;
      for (let j = 0; j < gMeals.length; j++) {
        G += gMeals[j][i];
      }
      yield { x: String(t[i]), y: String(G) };
    }
  }

  /**
   * Run full model as generator for streaming via Socket.IO.
   */
  *runFullStream(
    t0: number = 0,
    t1: number = 24,
    n: number = 100,
    config?: Partial<CMAModelParams>,
  ): Generator<{ t: string; c: string; m: string; a: string }> {
    if (config) {
      this.updateParams(config);
    }

    const rows = this.run();
    // Resample to n points
    const step = Math.max(1, Math.floor(rows.length / n));
    for (let i = 0; i < rows.length; i += step) {
      const row = rows[i];
      yield {
        t: String(row.t),
        c: String(row.c),
        m: String(row.m),
        a: String(row.a),
      };
    }
  }
}

/**
 * Simple seeded PRNG (xorshift32) for reproducibility.
 */
function createSeededRandom(seed: number): () => number {
  let state = seed | 0 || 1;
  return () => {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };
}
