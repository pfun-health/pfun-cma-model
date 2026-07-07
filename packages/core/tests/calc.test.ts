import { describe, it, expect } from "vitest";
import {
  exp,
  expitPfun,
  E,
  E_norm,
  K,
  K_vec,
  Light,
  mealDistr,
  linspace,
  normalize,
  computeG_single,
  computeG,
} from "../src/calc.js";

describe("exp", () => {
  it("should compute exponential", () => {
    expect(exp(0)).toBe(1);
    expect(exp(1)).toBeCloseTo(Math.E);
  });

  it("should clip to avoid overflow", () => {
    expect(exp(1000)).toBe(Math.exp(709));
    expect(exp(-1000)).toBe(Math.exp(-709));
  });
});

describe("expitPfun", () => {
  it("should be 0.5 at x=0", () => {
    expect(expitPfun(0)).toBeCloseTo(0.5);
  });

  it("should approach 1 for large positive x", () => {
    expect(expitPfun(10)).toBeCloseTo(1, 5);
  });

  it("should approach 0 for large negative x", () => {
    expect(expitPfun(-10)).toBeCloseTo(0, 5);
  });
});

describe("E", () => {
  it("should be sigmoid at 0.5 for x=0", () => {
    expect(E(0)).toBeCloseTo(0.5);
  });

  it("should be bounded [0,1]", () => {
    expect(E(-100)).toBeGreaterThanOrEqual(0);
    expect(E(100)).toBeLessThanOrEqual(1);
  });
});

describe("E_norm", () => {
  it("should be 0 at x=0", () => {
    expect(E_norm(0)).toBeCloseTo(0);
  });

  it("should be bounded [-1,1]", () => {
    expect(E_norm(-100)).toBeGreaterThanOrEqual(-1);
    expect(E_norm(100)).toBeLessThanOrEqual(1);
  });
});

describe("K", () => {
  it("should return 0 for x <= 0", () => {
    expect(K(0)).toBe(0);
    expect(K(-1)).toBe(0);
  });

  it("should be positive for x > 0", () => {
    expect(K(0.5)).toBeGreaterThan(0);
    expect(K(1)).toBeGreaterThan(0);
  });

  it("should peak near x=0.5", () => {
    expect(K(0.5)).toBeGreaterThan(K(0.1));
    expect(K(0.5)).toBeGreaterThan(K(2));
  });
});

describe("K_vec", () => {
  it("should vectorize K", () => {
    const result = K_vec([-1, 0, 0.5, 1, 2]);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0);
    expect(result[2]).toBeGreaterThan(0);
  });
});

describe("Light", () => {
  it("should be ~2 at x=0", () => {
    expect(Light(0)).toBeCloseTo(2 / (1 + 1), 5); // 2/(1+exp(0)) = 1
    // Actually: 2/(1+exp(0)) = 2/2 = 1
    expect(Light(0)).toBeCloseTo(1);
  });

  it("should decrease for large |x|", () => {
    expect(Light(5)).toBeLessThan(Light(0));
    expect(Light(-5)).toBeLessThan(Light(0));
  });
});

describe("mealDistr", () => {
  it("should return 1 at t=0, Cm=0", () => {
    expect(mealDistr(0, 0, 0)).toBe(1);
  });

  it("should be bounded [0,1]", () => {
    for (let t = 0; t < 24; t++) {
      const val = mealDistr(1.0, t, 0);
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(1);
    }
  });
});

describe("linspace", () => {
  it("should generate correct number of points", () => {
    const t = linspace(0, 10, 11);
    expect(t.length).toBe(11);
    expect(t[0]).toBe(0);
    expect(t[10]).toBe(10);
  });

  it("should be evenly spaced", () => {
    const t = linspace(0, 1, 5);
    expect(t[1] - t[0]).toBeCloseTo(0.25);
    expect(t[2] - t[1]).toBeCloseTo(0.25);
  });

  it("should handle n=1", () => {
    const t = linspace(5, 10, 1);
    expect(t).toEqual([5]);
  });
});

describe("normalize", () => {
  it("should normalize to [0,1] by default", () => {
    const result = normalize([0, 5, 10]);
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0.5);
    expect(result[2]).toBe(1);
  });

  it("should normalize to custom range", () => {
    const result = normalize([0, 10], -1, 1);
    expect(result[0]).toBe(-1);
    expect(result[1]).toBe(1);
  });
});

describe("computeG_single", () => {
  it("should return glucose response for single meal", () => {
    const t = linspace(0, 24, 100);
    const I_E = new Array(100).fill(0.05);
    const result = computeG_single(t, I_E, 7.0, 1.0);
    expect(result.length).toBe(100);
    // Should have peak near meal time
    const peakIdx = result.indexOf(Math.max(...result));
    const peakTime = t[peakIdx];
    expect(peakTime).toBeGreaterThan(5);
    expect(peakTime).toBeLessThan(10);
  });
});

describe("computeG", () => {
  it("should compute glucose for multiple meals", () => {
    const t = linspace(0, 24, 100);
    const I_E = new Array(100).fill(0.05);
    const result = computeG(t, I_E, [7.0, 11.0, 17.5], 1.0, 0.05, 0.0, 0.0, true);
    expect(result.length).toBe(3); // one row per meal
    expect(result[0].length).toBe(100);
  });

  it("should handle taug as array", () => {
    const t = linspace(0, 24, 50);
    const I_E = new Array(50).fill(0.05);
    const result = computeG(t, I_E, [7.0, 11.0, 17.5], [1.0, 1.5, 2.0], 0.05, 0.0, 0.0, false);
    expect(result.length).toBe(3);
  });
});
