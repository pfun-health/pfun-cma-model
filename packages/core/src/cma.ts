import { runCMAModel } from './engine.js';
import { CMAModelParams, CMAModelParamsSchema } from './cma_model_params.js';

export class CMASleepWakeModel {
  private _params: CMAModelParams;
  private _solution: Record<string, number[]> | null = null;

  constructor(params: Partial<CMAModelParams> = {}) {
    this._params = CMAModelParamsSchema.parse(params);
  }

  get params(): CMAModelParams {
    return { ...this._params };
  }

  get solution(): Record<string, number[]> | null {
    return this._solution;
  }

  update(updates: Partial<CMAModelParams>): void {
    this._params = CMAModelParamsSchema.parse({ ...this._params, ...updates });
  }

  solve(): void {
    const { t: paramsT, N, d, taup, taug, B, Cm, toff, tM, seed, eps } = this._params;
    const t = (paramsT && paramsT.length > 0)
        ? paramsT
        : Array.from({ length: N }, (_, i) => i * (24.0 / (N - 1)));

    // Ensure all TypedArrays are sized appropriately
    const tArray = new Float64Array(t);
    const tMArray = new Float64Array(tM);
    const seedPtr = new Int32Array([seed ?? Math.floor(Math.random() * 100000)]);
    const out_L = new Float64Array(N);
    const out_m = new Float64Array(N);
    const out_c = new Float64Array(N);
    const out_a = new Float64Array(N);
    const out_I_S = new Float64Array(N);
    const out_I_E = new Float64Array(N);
    const out_G = new Float64Array(N);
    const out_g = new Float64Array(N * tM.length);

    runCMAModel(
      tArray, N, d, taup, taug, null, B, Cm, toff,
      tMArray, tM.length, seedPtr, eps,
      out_L, out_m, out_c, out_a, out_I_S, out_I_E, out_G, out_g
    );

    this._solution = {
      t: Array.from(tArray),
      L: Array.from(out_L),
      M: Array.from(out_m),
      C: Array.from(out_c),
      A: Array.from(out_a),
      I_S: Array.from(out_I_S),
      I_E: Array.from(out_I_E),
      G: Array.from(out_G),
    };
  }
}
