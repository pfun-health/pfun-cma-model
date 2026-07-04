/**
 * Model fitting via least-squares optimization.
 * Clean-room implementation.
 */

import { CMASleepWakeModel } from "./model.js";
import {
  CMAModelParams,
  CMAModelParamsSchema,
  BOUNDED_PARAM_KEYS,
  BOUNDED_PARAM_LB,
  BOUNDED_PARAM_UB,
  type BoundedParamKey,
} from "./params.js";
import { linspace } from "./calc.js";

export interface CMAFitResult {
  params: CMAModelParams;
  residual: number;
  iterations: number;
  success: boolean;
  message: string;
}

/**
 * Simple Nelder-Mead-like optimizer for fitting model params.
 */
export function fitModel(
  data: { t: number[]; G: number[] },
  config?: Partial<CMAModelParams>,
  maxIter: number = 200,
  tol: number = 1e-6,
): CMAFitResult {
  const baseParams = CMAModelParamsSchema.parse(config ?? {});
  const model = new CMASleepWakeModel(baseParams);

  // We optimize only bounded parameters
  const keys = [...BOUNDED_PARAM_KEYS];
  let x = keys.map((k) => baseParams[k] as number);

  // Objective: sum of squared residuals between model G and data G
  function objective(params: number[]): number {
    const updates: Partial<CMAModelParams> = {};
    keys.forEach((k, i) => {
      (updates as Record<string, number>)[k] = clampParam(k, params[i]);
    });

    const m = new CMASleepWakeModel({ ...baseParams, ...updates });
    const results = m.runAtTime(
      Math.min(...data.t),
      Math.max(...data.t),
      data.t.length,
    );

    let sse = 0;
    for (let i = 0; i < data.G.length; i++) {
      const predicted = parseFloat(results[i]?.y ?? "0");
      sse += Math.pow(data.G[i] - predicted, 2);
    }
    return sse;
  }

  // Simple coordinate descent optimization
  let bestCost = objective(x);
  let improved = true;
  let iter = 0;

  while (improved && iter < maxIter) {
    improved = false;
    iter++;

    for (let i = 0; i < keys.length; i++) {
      const lb = BOUNDED_PARAM_LB[keys[i]];
      const ub = BOUNDED_PARAM_UB[keys[i]];
      const stepSize = (ub - lb) * 0.05 * Math.pow(0.95, iter);

      // Try positive step
      const xPlus = [...x];
      xPlus[i] = clampParam(keys[i], x[i] + stepSize);
      const costPlus = objective(xPlus);

      if (costPlus < bestCost - tol) {
        x = xPlus;
        bestCost = costPlus;
        improved = true;
        continue;
      }

      // Try negative step
      const xMinus = [...x];
      xMinus[i] = clampParam(keys[i], x[i] - stepSize);
      const costMinus = objective(xMinus);

      if (costMinus < bestCost - tol) {
        x = xMinus;
        bestCost = costMinus;
        improved = true;
      }
    }
  }

  const finalParams: Partial<CMAModelParams> = { ...baseParams };
  keys.forEach((k, i) => {
    (finalParams as Record<string, number>)[k] = x[i];
  });

  return {
    params: CMAModelParamsSchema.parse(finalParams),
    residual: bestCost,
    iterations: iter,
    success: bestCost < tol || iter < maxIter,
    message: improved ? "Converged" : "Optimization completed (no further improvement)",
  };
}

function clampParam(key: BoundedParamKey, value: number): number {
  return Math.max(BOUNDED_PARAM_LB[key], Math.min(BOUNDED_PARAM_UB[key], value));
}
