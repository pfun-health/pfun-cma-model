/**
 * @pfun/core - CMA Sleep-Wake Model Engine
 *
 * Clean-room TypeScript implementation of the PFun CMA model.
 */

export { Bounds } from "./bounds.js";
export {
  CMAModelParamsSchema,
  type CMAModelParams,
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
  CMASleepWakeModel,
  type ModelRunRow,
  type RunAtTimeResult,
} from "./model.js";
export { fitModel, type CMAFitResult } from "./fit.js";
