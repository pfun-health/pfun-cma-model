/**
 * @pfun/core - CMA Sleep-Wake Model Engine
 *
 * Clean-room TypeScript implementation of the PFun CMA model.
 */

// Implementation B (primary) - exports used by tests and external consumers
export { CMASleepWakeModel } from "./cma.js";
export {
  CMAModelParamsSchema,
  type CMAModelParams,
  CMAModelParamsDefaults,
  CMAModelParamsKeys,
  getCMADefaultParams,
} from "./cma_model_params.js";
export { PFunCMAParamsGrid } from "./grid.js";
export { generateScenario } from "./llm.js";
export {
  runCMAModel,
  exp_clipped,
  expit,
  calc_vdep_current,
  Light_pfun,
  E_pfun,
  K_pfun,
  meal_distr_pfun,
  calc_L,
  calc_M,
  calc_c,
  calc_a,
  calc_I_S,
  calc_I_E,
  calc_G,
} from "./engine.js";

// Implementation A (secondary) - higher-level API and metadata
export { Bounds } from "./bounds.js";
export {
  getDefaultParams,
  getParamsJsonSchema,
  getQualitativeDescriptor,
  calcSerr,
  describeParam,
  getBoundedParamInfo,
  generateParamsTable,
  BOUNDED_PARAM_KEYS,
  BOUNDED_PARAM_LB,
  BOUNDED_PARAM_UB,
  BOUNDED_PARAM_MID,
  BOUNDED_PARAM_STEPS,
  BOUNDED_PARAM_DESCRIPTIONS,
  DEFAULT_BOUNDS,
  type BoundedParamKey,
} from "./params.js";
export {
  exp,
  expitPfun,
  E_norm,
  Light,
  E,
  K,
  K_vec,
  mealDistr,
  computeG_single,
  computeG,
  linspace,
  normalize,
} from "./calc.js";
export {
  type ModelRunRow,
  type RunAtTimeResult,
} from "./model.js";
export { fitModel, type CMAFitResult } from "./fit.js";
