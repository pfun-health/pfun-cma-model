import { describe, it, expect } from 'vitest';
import {
  CMASleepWakeModel,
  CMAModelParamsSchema,
  PFunCMAParamsGrid,
  generateScenario,
  runCMAModel,
  exp_clipped,
  expit,
  Light_pfun,
  E_pfun,
  K_pfun,
} from '../index.js';

describe('CMASleepWakeModel', () => {
  it('should construct with default parameters', () => {
    const model = new CMASleepWakeModel();
    expect(model.params).toBeDefined();
    expect(model.params.N).toBe(24);
    expect(model.params.taup).toBe(1.0);
  });

  it('should accept partial parameters', () => {
    const model = new CMASleepWakeModel({ N: 48, taup: 2.0 });
    expect(model.params.N).toBe(48);
    expect(model.params.taup).toBe(2.0);
  });

  it('should solve and produce a solution', () => {
    const model = new CMASleepWakeModel({ N: 24, d: 0.0 });
    model.solve();
    const solution = model.solution;
    expect(solution).not.toBeNull();
    expect(solution!.t).toHaveLength(24);
    expect(solution!.L).toHaveLength(24);
    expect(solution!.M).toHaveLength(24);
    expect(solution!.C).toHaveLength(24);
    expect(solution!.A).toHaveLength(24);
    expect(solution!.I_S).toHaveLength(24);
    expect(solution!.I_E).toHaveLength(24);
    expect(solution!.G).toHaveLength(24);
  });

  it('should update parameters between solves', () => {
    const model = new CMASleepWakeModel({ d: 1.0 });
    model.update({ d: -1.0, taup: 0.5 });
    expect(model.params.d).toBe(-1.0);
    expect(model.params.taup).toBe(0.5);
  });

  it('should reject invalid parameters via the schema', () => {
    expect(() => new CMASleepWakeModel({ N: -1 as unknown as number })).toThrow();
  });
});

describe('CMAModelParamsSchema', () => {
  it('should parse valid params', () => {
    const result = CMAModelParamsSchema.parse({ N: 24, d: 1.0 });
    expect(result.N).toBe(24);
    expect(result.d).toBe(1.0);
  });

  it('should apply defaults for missing fields', () => {
    const result = CMAModelParamsSchema.parse({});
    expect(result.N).toBe(24);
    expect(result.taup).toBe(1.0);
    expect(result.taug).toBe(1.0);
  });

  it('should reject invalid N', () => {
    expect(() => CMAModelParamsSchema.parse({ N: 1 })).toThrow();
    expect(() => CMAModelParamsSchema.parse({ N: 1.5 })).toThrow();
  });
});

describe('PFunCMAParamsGrid', () => {
  it('should build a grid with default options', () => {
    const grid = new PFunCMAParamsGrid();
    expect(grid.pgrid.length).toBeGreaterThan(0);
    expect(grid.keys).toEqual(['taug', 'taup', 'B', 'Cm']);
  });

  it('should run and produce a collection', () => {
    const grid = new PFunCMAParamsGrid({ N: 4, m: 2, keys: ['taug', 'taup'] });
    grid.run();
    expect(grid.collection.length).toBeGreaterThan(0);
    for (const entry of grid.collection) {
      expect(entry).toHaveProperty('N');
      expect(entry).toHaveProperty('taug');
      expect(entry).toHaveProperty('taup');
      expect(entry).toHaveProperty('t');
    }
  });
});

describe('generateScenario', () => {
  it('should generate a standard scenario by default', () => {
    const result = generateScenario('A healthy individual');
    expect(result).toHaveProperty('qualitative_description');
    expect(result).toHaveProperty('parameters');
    expect(result.parameters.d).toBe(0.0);
  });

  it('should generate a night owl scenario', () => {
    const result = generateScenario('night owl');
    expect(result.parameters.toff).toBe(2.5);
  });

  it('should generate an early bird scenario', () => {
    const result = generateScenario('early bird');
    expect(result.parameters.toff).toBe(-2.0);
  });

  it('should detect unhealthy keywords', () => {
    const result = generateScenario('person has diabetes');
    expect(result.parameters.B).toBe(0.2);
    expect(result.parameters.taug).toBe(2.5);
  });
});

describe('Engine utility functions', () => {
  it('exp_clipped should clip extreme values', () => {
    expect(exp_clipped(-1000)).toBeCloseTo(Math.exp(-709.0));
    expect(exp_clipped(1000)).toBeCloseTo(Math.exp(709.0));
  });

  it('expit should produce values in (0, 1)', () => {
    expect(expit(0)).toBeCloseTo(0.5);
    expect(expit(10)).toBeGreaterThan(0.5);
    expect(expit(-10)).toBeLessThan(0.5);
  });

  it('Light_pfun should return non-negative values', () => {
    const val = Light_pfun(0);
    expect(val).toBeGreaterThanOrEqual(0);
  });

  it('E_pfun should match expit', () => {
    expect(E_pfun(0)).toBe(expit(0));
    expect(E_pfun(2)).toBe(expit(2));
  });

  it('K_pfun should return 0 for non-positive x', () => {
    expect(K_pfun(0)).toBe(0);
    expect(K_pfun(-1)).toBe(0);
  });

  it('K_pfun should return non-negative for positive x', () => {
    const val = K_pfun(1);
    expect(val).toBeGreaterThanOrEqual(0);
  });
});

describe('runCMAModel (low-level)', () => {
  it('should fill output arrays of expected length', () => {
    const N = 24;
    const t = new Float64Array(Array.from({ length: N }, (_, i) => i * (24.0 / (N - 1))));
    const tM = new Float64Array([7.0, 11.0, 17.5]);
    const seed = new Int32Array([42]);
    const outL = new Float64Array(N);
    const outM = new Float64Array(N);
    const outC = new Float64Array(N);
    const outA = new Float64Array(N);
    const outIS = new Float64Array(N);
    const outIE = new Float64Array(N);
    const outG = new Float64Array(N);
    const outGComp = new Float64Array(N * tM.length);

    runCMAModel(
      t, N, 0.0, 1.0, 1.0, null, 0.05, 0.0, 0.0,
      tM, tM.length, seed, 1e-18,
      outL, outM, outC, outA, outIS, outIE, outG, outGComp,
    );

    expect(Array.from(outL).some(v => v > 0)).toBe(true);
    expect(Array.from(outG).some(v => v > 0)).toBe(true);
  });
});
