import { PLS } from 'ml-pls';
import { CMASleepWakeModel } from './cma.js';
import { CMAModelParams } from './cma_model_params.js';

export interface DataPoint {
    t?: number;
    G?: number;
    [key: string]: unknown;
}

export interface FitOptions {
    N?: number;
    latentVariables?: number;
    [key: string]: unknown;
}

export interface FitResult {
    formatted_data: DataPoint[];
    model_params: CMAModelParams;
    model_dump_json: () => string;
}

export function fitModel(data: DataPoint[], opts?: FitOptions): FitResult {
    if (!data || data.length === 0) {
        throw new Error("Cannot fit model on empty dataset.");
    }

    // Extract time and glucose values from input data
    const t_values = data.map((d: DataPoint) => d.t ?? 0);
    const g_values = data.map((d: DataPoint) => d.G ?? 0);

    // Prepare regression tensors: time as predictor, glucose as response
    const x = t_values.map((val: number) => [val]);
    const y = g_values.map((val: number) => [val]);

    // Fit a PLS model to capture the glucose dynamics
    const latentVariables = opts?.latentVariables ?? 1;
    // The PLS type declaration incorrectly requires a second argument.
    // The JavaScript implementation accepts a single options argument.
    const pls = new (PLS as unknown as new (options: { latentVectors?: number }) => PLS)({ latentVectors: latentVariables });
    pls.train(x, y);

    // Extract PLS regression coefficient to inform CMA model parameters.
    // pls.B is the regression coefficient matrix (set by train()).
    const coefficient = pls.B?.get(0, 0);
    const taugHint = coefficient !== undefined && coefficient !== 0
        ? Math.abs(coefficient) * 0.5 + 1.0
        : 1.0;
    const taupHint = taugHint * 1.5;

    const fittedModel = new CMASleepWakeModel();
    fittedModel.update({
        taug: Math.min(Math.max(taugHint, 0.1), 3.0),
        taup: Math.min(Math.max(taupHint, 0.5), 3.0),
        B: 0.05,
        N: opts?.N ?? data.length,
    });

    fittedModel.solve();

    return {
        formatted_data: data,
        model_params: fittedModel.params,
        model_dump_json: () => JSON.stringify(fittedModel.params),
    };
}
