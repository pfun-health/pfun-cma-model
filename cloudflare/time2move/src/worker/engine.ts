/**
 * Thin, runtime-agnostic (Cloudflare Workers and Node) wrapper around the PFun CMA
 * engine compiled to WebAssembly (`cma.wasm`, built by `make wasm-worker` in
 * pfun-cma-engine-c from `src/wasm/worker_shim.c`).
 *
 * The module is libc-free: the host supplies `exp`, `log`, `pow` and `cos`. Inputs are
 * written into static buffers inside the module's linear memory, `cma_run` is called,
 * and the outputs are read back through typed-array views.
 *
 * Instantiate once per isolate and reuse; every call overwrites the static buffers.
 */

/** Circadian / glucose parameters of the CMA model. */
export type ModelParams = {
  /** Circadian phase offset in hours; the light signal peaks near t = 12 + d. */
  d: number;
  /** Photoperiod width / cortisol tone. */
  taup: number;
  /** Basal glucose bias. The fasting floor of G is 2 * B when Cm = 0. */
  B: number;
  /** Frequency multiplier of the circadian meal-distribution term. */
  Cm: number;
  /** Phase offset (hours) of the circadian meal-distribution term. */
  toff: number;
  /** Noise scale; also added to taup inside the light term. Keep at 1e-18. */
  eps: number;
};

/** Gains of the activity pathways (see pfun_cma_engine.h). */
export type ActivityGains = {
  /** Slope of the activity sigmoid. */
  k_A: number;
  /** Half-activation threshold of the activity sigmoid. */
  A_thresh: number;
  /** Contraction-mediated glucose uptake gain. */
  gamma_A: number;
  /** Saturation of contraction-mediated uptake. */
  kappa_A: number;
  /** Activity amplification of cortisol. */
  beta_A: number;
  /** Post-activity insulin sensitisation gain. */
  eta_A: number;
  /** Decay time (hours) of post-activity sensitisation; <= 0 disables it. */
  tau_A: number;
};

export type EngineInput = {
  /** Hours of the local day, strictly increasing, within [0, 24]. */
  t: Float64Array;
  /** Meal onset times (hours), 1..maxMeals entries. */
  meals: ArrayLike<number>;
  /** Width of each meal response (hours), same length as `meals`. */
  mealWidths: ArrayLike<number>;
  /** Raw activity in [0, 1] per time point, or null for the no-activity model. */
  activity: Float64Array | null;
  model: ModelParams;
  gains: ActivityGains;
};

/** Names of the engine's per-time-point output rows, in memory order. */
export const SIGNALS = ["L", "m", "c", "c_mod", "A", "a", "I_S", "I_E", "I_E_eff", "s_A", "G"] as const;
export type Signal = (typeof SIGNALS)[number];
export type EngineOutput = Record<Signal, Float64Array>;

export type Engine = {
  readonly maxPoints: number;
  readonly maxMeals: number;
  /** Run the model and copy out every signal. */
  run(input: EngineInput): EngineOutput;
  /** Run the model and copy out glucose only (cheaper for what-if sweeps). */
  runG(input: EngineInput): Float64Array;
};

type Exports = {
  memory: WebAssembly.Memory;
  cma_max_points(): number;
  cma_max_meals(): number;
  cma_in_t(): number;
  cma_in_A_raw(): number;
  cma_in_taug(): number;
  cma_in_tM(): number;
  cma_out_G(): number;
  cma_run(...args: number[]): number;
};

/** libm supplied by the host; matches the native build to about 1e-12. */
export const WASM_IMPORTS = {
  env: { exp: Math.exp, log: Math.log, pow: Math.pow, cos: Math.cos, sin: Math.sin },
};

const G_ROW = SIGNALS.indexOf("G");

export function createEngine(module: WebAssembly.Module): Engine {
  const x = new WebAssembly.Instance(module, WASM_IMPORTS).exports as unknown as Exports;
  for (const name of ["memory", "cma_max_points", "cma_max_meals", "cma_in_t", "cma_in_A_raw", "cma_in_taug", "cma_in_tM", "cma_out_G", "cma_run"]) {
    if (!(name in x)) throw new Error(`cma.wasm is missing the export "${name}"; rebuild it with \`make wasm-worker\` in pfun-cma-engine-c`);
  }
  const maxPoints = x.cma_max_points();
  const maxMeals = x.cma_max_meals();
  const view = (ptr: number, len: number) => new Float64Array(x.memory.buffer, ptr, len);

  /* The shim keeps its eleven output rows in one static `double[11][maxPoints]`
     and exports a pointer to the last row (G); row k therefore starts
     (10 - k) * maxPoints doubles before it. `verifyLayout` below proves that
     assumption against closed-form values before the engine is handed out. */
  const rowPtr = (k: number) => x.cma_out_G() - (G_ROW - k) * maxPoints * 8;

  function load(input: EngineInput): number {
    const n = input.t.length;
    const nMeals = input.meals.length;
    if (n < 2 || n > maxPoints) throw new RangeError(`t must have 2..${maxPoints} points (got ${n})`);
    if (nMeals < 1 || nMeals > maxMeals) throw new RangeError(`meals must have 1..${maxMeals} entries (got ${nMeals})`);
    if (input.mealWidths.length !== nMeals) throw new RangeError("mealWidths and meals must have equal length");
    if (input.activity && input.activity.length !== n) throw new RangeError("activity and t must have equal length");
    view(x.cma_in_t(), n).set(input.t);
    view(x.cma_in_tM(), nMeals).set(input.meals);
    view(x.cma_in_taug(), nMeals).set(input.mealWidths);
    if (input.activity) view(x.cma_in_A_raw(), n).set(input.activity);
    const { model: p, gains: a } = input;
    const rc = x.cma_run(
      n, nMeals, input.activity ? 1 : 0, p.d, p.taup, p.B, p.Cm, p.toff, p.eps,
      a.k_A, a.A_thresh, a.gamma_A, a.kappa_A, a.beta_A, a.eta_A, a.tau_A,
    );
    if (rc !== 0) throw new RangeError("cma_run rejected the input sizes");
    return n;
  }

  const engine: Engine = {
    maxPoints,
    maxMeals,
    run(input) {
      const n = load(input);
      const out = {} as EngineOutput;
      SIGNALS.forEach((name, k) => {
        out[name] = Float64Array.from(view(rowPtr(k), n));
      });
      return out;
    },
    runG(input) {
      const n = load(input);
      return Float64Array.from(view(x.cma_out_G(), n));
    },
  };

  verifyLayout(engine);
  return engine;
}

/**
 * Check the assumed output layout on a tiny input, using only relations that hold
 * between rows of the published model (light is a closed form of t; I_S and I_E are
 * simple combinations of c, m and a) and the fact that the activity rows are inert
 * without activity and respond to it with. Any other row order fails at least one.
 */
function verifyLayout(engine: Engine): void {
  const t = Float64Array.of(6, 9, 12, 15, 18);
  // Arbitrary in-range values: the relations below hold for any parameters.
  const model: ModelParams = { d: 0, taup: 1, B: 0.05, Cm: 0, toff: 0, eps: 1e-18 };
  const gains: ActivityGains = { k_A: 4, A_thresh: 0.25, gamma_A: 0.1, kappa_A: 1, beta_A: 0.1, eta_A: 0.5, tau_A: 1 };
  const base = { t, meals: [7], mealWidths: [1], model, gains };
  const rest = engine.run({ ...base, activity: null });
  const moved = engine.run({ ...base, activity: Float64Array.of(0, 0.9, 0.9, 0, 0) });

  const close = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
  const light = (ti: number) => 2 / (1 + Math.exp(2 * ((0.025 * (ti - 12 - model.d) ** 2) / (model.eps + model.taup)) ** 2));
  const idx = [0, 1, 2, 3, 4];
  const ok =
    idx.every((i) => close(rest.L[i], light(t[i]))) &&
    idx.every((i) => close(rest.I_S[i], 1 - 0.23 * rest.c[i] - 0.97 * rest.m[i])) &&
    idx.every((i) => close(rest.I_E[i], rest.a[i] * rest.I_S[i])) &&
    // Without activity the activity rows are inert.
    idx.every((i) => rest.A[i] === 0 && rest.s_A[i] === 0 && rest.c_mod[i] === rest.c[i] && rest.I_E_eff[i] === rest.I_E[i]) &&
    // With activity at samples 1 and 2: A is positive exactly there, cortisol is
    // amplified there, sensitisation starts one sample later and outlasts the bout.
    idx.every((i) => (i === 1 || i === 2 ? moved.A[i] > 0 && moved.A[i] <= 1 && moved.c_mod[i] > moved.c[i] : moved.A[i] === 0)) &&
    moved.s_A[1] === 0 && moved.s_A[2] > 0 && moved.s_A[3] > 0 && moved.s_A[4] > 0 && moved.s_A[4] < moved.s_A[3] &&
    idx.every((i) => close(moved.I_E_eff[i], moved.I_E[i] * (1 + moved.s_A[i]))) &&
    idx.every((i) => Number.isFinite(moved.G[i]) && moved.G[i] > 0 && moved.G[i] <= rest.G[i] * (1 + 1e-12));
  if (!ok) {
    throw new Error(
      "cma.wasm output layout does not match this wrapper (expected the worker_shim.c layout: " +
        `${SIGNALS.join(", ")}); rebuild both from the same pfun-cma-engine-c revision`,
    );
  }
}
