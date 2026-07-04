import { z } from 'zod';

export const CMAModelParamsSchema = z.object({
  t: z.array(z.number()).nullable().default(null),
  N: z.number().int().min(2).default(24),
  d: z.number().default(0.0),
  taup: z.number().default(1.0),
  taug: z.number().default(1.0),
  B: z.number().default(0.05),
  Cm: z.number().default(0.0),
  toff: z.number().default(0.0),
  tM: z.tuple([z.number(), z.number(), z.number()]).default([7.0, 11.0, 17.5]),
  seed: z.number().int().nullable().default(null),
  eps: z.number().default(1e-18),
});

export type CMAModelParams = z.infer<typeof CMAModelParamsSchema>;
