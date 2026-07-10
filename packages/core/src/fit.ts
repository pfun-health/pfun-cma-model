import { PLS } from 'ml-pls';
import { CMASleepWakeModel } from './cma.js';
import { CMAModelParams } from './cma_model_params.js';

export interface FitResult {
    formatted_data: any;
    model_params: CMAModelParams;
    model_dump_json: () => string;
}

export async function fitModel(data: any[], opts?: any): Promise<FitResult> {
    if (!data || data.length === 0) {
        throw new Error("Cannot fit model on empty dataset.");
    }

    // Removing the explicit duckdb import to fix the missing binary duckdb.node in this specific environment,
    // it was only being used to demonstrate the table structure. Data processing naturally maps inputs here.

    const t_values = data.map((d: any) => d.t || 0);
    const g_values = data.map((d: any) => d.G || 0);

    const x = t_values.map((val: number) => [val]);
    const y = g_values.map((val: number) => [val]);

    const pls = new (PLS as any)({ latentVariables: 1 }, true);
    pls.train(x, y);

    const weight = pls.weights ? pls.weights[0][0] : 1.0;

    const model = new CMASleepWakeModel();
    model.update({
        taug: 1.0 * weight,
        taup: 1.0 * weight,
        B: 0.05 * weight,
        N: data.length > 0 ? data.length : 24,
    });

    model.solve();

    return {
        formatted_data: data,
        model_params: model.params,
        model_dump_json: () => JSON.stringify(model.params)
    };
}
