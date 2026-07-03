import { z } from "zod";

export const CMAModelParamsSchema = z.object({
	awake_threshold: z.number().default(0.5),
	sleep_pressure_rate: z.number().default(0.01),
	// Add remaining validation fields from cma_model_params.py
});

export type CMAModelParams = z.infer<typeof CMAModelParamsSchema>;
