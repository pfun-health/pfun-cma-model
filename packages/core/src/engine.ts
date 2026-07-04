// LCG random number generator (linear congruential generator)
function pfun_rand(seed: Int32Array | null): number {
  if (seed === null) return 0.0;
  seed[0] = (seed[0] * 1103515245 + 12345) & 0x7fffffff;
  return seed[0] / 0x7fffffff;
}

function pfun_uniform(seed: Int32Array | null, low: number, high: number): number {
  return low + (high - low) * pfun_rand(seed);
}

// Clipped exponential to avoid overflow/underflow
export function exp_clipped(x: number): number {
  if (x < -709.0) x = -709.0;
  if (x > 709.0) x = 709.0;
  return Math.exp(x);
}

export function expit(x: number): number {
  return 1.0 / (1.0 + exp_clipped(-2.0 * x));
}

export function calc_vdep_current(v: number, v1: number, v2: number, A: number, B: number): number {
  return A * expit(B * (v - v1) / v2);
}

export function E_pfun(x: number): number {
  return 1.0 / (1.0 + exp_clipped(-2.0 * x));
}

export function Light_pfun(x: number): number {
  return 2.0 / (1.0 + exp_clipped(2.0 * x * x));
}

export function K_pfun(x: number): number {
  if (x > 0.0) {
    const logVal = Math.log(2.0 * x);
    return exp_clipped(-(logVal * logVal));
  }
  return 0.0;
}

export function meal_distr_pfun(Cm: number, t: number, toff: number): number {
  const cosVal = Math.cos(2.0 * Math.PI * Cm * (t + toff) / 24.0);
  return cosVal * cosVal;
}

export function calc_L(
  t: Float64Array,
  N: number,
  d: number,
  taup: number,
  eps: number,
  out: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    const diff = t[i] - 12.0 - d;
    out[i] = Light_pfun(0.025 * (diff * diff) / (eps + taup));
  }
}

export function calc_M(
  t: Float64Array,
  N: number,
  L: Float64Array,
  d: number,
  eps: number,
  seed: Int32Array | null,
  out: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    const cosVal = Math.cos(-(t[i] - 3.0 - d) * Math.PI / 24.0);
    let mVal = (1.0 - L[i]) ** 3 * (cosVal * cosVal);
    if (seed !== null && seed[0] !== 0) {
      mVal += pfun_uniform(seed, -eps, eps);
    }
    out[i] = mVal;
  }
}

export function calc_c(
  t: Float64Array,
  N: number,
  L: Float64Array,
  m: Float64Array,
  d: number,
  taup: number,
  out: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    out[i] =
      (4.9 / (1.0 + taup)) *
      Math.PI *
      E_pfun((L[i] - 0.88) ** 3) *
      E_pfun(0.05 * (8.0 - t[i] + d)) *
      E_pfun(2.0 * (-m[i]) ** 3);
  }
}

export function calc_a(
  t: Float64Array,
  N: number,
  c: Float64Array,
  m: Float64Array,
  L: Float64Array,
  d: number,
  taup: number,
  eps: number,
  out: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    const tAlt = 0.7 * (27.0 - t[i] + d);
    const diffAlt = tAlt - 12.0 - d;
    const LAlt = Light_pfun(0.025 * (diffAlt * diffAlt) / (eps + taup));
    const diffT = t[i] - 13.0 - d;
    out[i] =
      (E_pfun((-c[i] * m[i]) ** 3) +
        exp_clipped(-0.025 * (diffT * diffT)) * LAlt) /
      2.0;
  }
}

export function calc_I_S(
  N: number,
  c: Float64Array,
  m: Float64Array,
  out: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    out[i] = 1.0 - 0.23 * c[i] - 0.97 * m[i];
  }
}

export function calc_I_E(
  N: number,
  a: Float64Array,
  I_S: Float64Array,
  out: Float64Array,
): void {
  for (let i = 0; i < N; i++) {
    out[i] = a[i] * I_S[i];
  }
}

export function calc_G(
  t: Float64Array,
  N: number,
  I_E: Float64Array,
  tM: Float64Array,
  n_meals: number,
  taug: Float64Array,
  B: number,
  Cm: number,
  toff: number,
  include_bias_in_components: number,
  out_G_instant: Float64Array | null,
  out_g_components: Float64Array | null,
): void {
  // baseline glucose
  if (out_G_instant !== null) {
    for (let i = 0; i < N; i++) {
      out_G_instant[i] = B * (1.0 + meal_distr_pfun(Cm, t[i], toff));
    }
  }

  // per-meal contributions
  for (let j = 0; j < n_meals; j++) {
    const tm_j = tM[j];
    const taug_j = taug[j];
    const taug_j_sq = taug_j * taug_j;

    for (let i = 0; i < N; i++) {
      const k_G = K_pfun((t[i] - tm_j) / taug_j_sq);
      const g_val = 1.3 * k_G / (1.0 + I_E[i]);

      if (out_g_components !== null) {
        out_g_components[j * N + i] = g_val;
        if (include_bias_in_components !== 0) {
          out_g_components[j * N + i] +=
            B * (1.0 + meal_distr_pfun(Cm, t[i], toff));
        }
      }

      if (out_G_instant !== null) {
        out_G_instant[i] += g_val;
      }
    }
  }
}

export function runCMAModel(
  t: Float64Array,
  N: number,
  d: number,
  taup: number,
  taugVal: number,
  taugVec: Float64Array | null,
  B: number,
  Cm: number,
  toff: number,
  tM: Float64Array,
  nMeals: number,
  seed: Int32Array | null,
  eps: number,
  outL: Float64Array,
  outM: Float64Array,
  outC: Float64Array,
  outA: Float64Array,
  outIS: Float64Array,
  outIE: Float64Array,
  outG: Float64Array,
  outGComponents: Float64Array | null,
): void {
  calc_L(t, N, d, taup, eps, outL);
  calc_M(t, N, outL, d, eps, seed, outM);
  calc_c(t, N, outL, outM, d, taup, outC);
  calc_a(t, N, outC, outM, outL, d, taup, eps, outA);
  calc_I_S(N, outC, outM, outIS);
  calc_I_E(N, outA, outIS, outIE);

  // taugVec fallback: if null, use taugVal for all meals
  let actualTaug: Float64Array;
  if (taugVec === null) {
    const tempTaug = new Float64Array(nMeals);
    for (let j = 0; j < nMeals; j++) {
      tempTaug[j] = taugVal;
    }
    actualTaug = tempTaug;
  } else {
    actualTaug = taugVec;
  }

  calc_G(t, N, outIE, tM, nMeals, actualTaug, B, Cm, toff, 0, outG, outGComponents);
}
