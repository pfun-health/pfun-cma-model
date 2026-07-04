/**
 * Core mathematical functions for the CMA model.
 * Clean-room implementation based on the CLEANROOM_INSTRUCTIONS spec.
 */

/**
 * Safe exponential: clips input to avoid overflow.
 */
export function exp(x: number): number {
  const clipped = Math.max(-709, Math.min(709, x));
  return Math.exp(clipped);
}

/**
 * PFun sigmoid (expit variant with scale factor 2).
 */
export function expitPfun(x: number): number {
  return 1.0 / (1.0 + exp(-2.0 * x));
}

/**
 * Normalized E function: maps to [-1, 1] range.
 */
export function E_norm(x: number): number {
  return 2.0 * (expitPfun(2.0 * x) - 0.5);
}

/**
 * Light intensity function.
 */
export function Light(x: number): number {
  return 2.0 / (1.0 + exp(2.0 * Math.pow(x, 2)));
}

/**
 * Core E function (sigmoid).
 */
export function E(x: number): number {
  return 1.0 / (1.0 + exp(-2.0 * x));
}

/**
 * Glucose response function K(x).
 * Piecewise: K(x) = exp(-log(2x)^2) for x > 0, else 0.
 */
export function K(x: number): number {
  if (x > 0.0) {
    return exp(-Math.pow(Math.log(2.0 * x), 2));
  }
  return 0.0;
}

/**
 * Vectorized K over an array.
 */
export function K_vec(arr: number[]): number[] {
  return arr.map(K);
}

/**
 * Meal distribution function.
 */
export function mealDistr(Cm: number, t: number, toff: number): number {
  return Math.pow(Math.cos((2 * Math.PI * Cm * (t + toff)) / 24), 2);
}

/**
 * Vectorized glucose computation G for a single meal time.
 */
export function computeG_single(
  t: number[],
  I_E: number[],
  tm: number,
  taug: number,
): number[] {
  return t.map((ti, i) => {
    const kG = K((ti - tm) / Math.pow(taug, 2));
    return (1.3 * kG) / (1.0 + I_E[i]);
  });
}

/**
 * Full vectorized glucose computation G across all meal times.
 */
export function computeG(
  t: number[],
  I_E: number[],
  tM: number[],
  taug: number | number[],
  B: number,
  Cm: number,
  toff: number,
  includeBias: boolean = false,
): number[][] {
  const taugArr = Array.isArray(taug) ? taug : Array(tM.length).fill(taug);
  const result: number[][] = [];

  for (let j = 0; j < tM.length; j++) {
    const g = computeG_single(t, I_E, tM[j], taugArr[j]);
    result.push(g);
  }

  if (includeBias) {
    for (let j = 0; j < result.length; j++) {
      for (let i = 0; i < t.length; i++) {
        result[j][i] += B * (1.0 + mealDistr(Cm, t[i], toff));
      }
    }
  }

  return result;
}

/**
 * Generate a linear time vector.
 */
export function linspace(t0: number, t1: number, n: number): number[] {
  if (n <= 1) return [t0];
  const step = (t1 - t0) / (n - 1);
  return Array.from({ length: n }, (_, i) => t0 + i * step);
}

/**
 * Normalize array to [a, b] range.
 */
export function normalize(x: number[], a: number = 0.0, b: number = 1.0): number[] {
  const xmin = Math.min(...x);
  const xmax = Math.max(...x);
  if (xmax === xmin) return x.map(() => (a + b) / 2);
  return x.map((v) => a + ((b - a) * (v - xmin)) / (xmax - xmin));
}
