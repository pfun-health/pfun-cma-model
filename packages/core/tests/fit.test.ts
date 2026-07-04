import { describe, it, expect } from "vitest";
import { fitModel } from "../src/fit.js";
import { CMASleepWakeModel } from "../src/model.js";

describe("fitModel", () => {
  it("should return a fit result", () => {
    // Generate synthetic data
    const model = new CMASleepWakeModel({ B: 0.1 });
    const data = model.runAtTime(0, 24, 50);
    const t = data.map((d) => parseFloat(d.x));
    const G = data.map((d) => parseFloat(d.y));

    const result = fitModel({ t, G }, {}, 50);
    expect(result).toHaveProperty("params");
    expect(result).toHaveProperty("residual");
    expect(result).toHaveProperty("iterations");
    expect(result).toHaveProperty("success");
    expect(result).toHaveProperty("message");
  });

  it("should produce finite residual", () => {
    const model = new CMASleepWakeModel();
    const data = model.runAtTime(0, 24, 30);
    const t = data.map((d) => parseFloat(d.x));
    const G = data.map((d) => parseFloat(d.y));

    const result = fitModel({ t, G }, {}, 20);
    expect(Number.isFinite(result.residual)).toBe(true);
  });

  it("should return valid params", () => {
    const model = new CMASleepWakeModel({ B: 0.2 });
    const data = model.runAtTime(0, 24, 30);
    const t = data.map((d) => parseFloat(d.x));
    const G = data.map((d) => parseFloat(d.y));

    const result = fitModel({ t, G }, {}, 30);
    expect(result.params.N).toBe(1024);
    expect(result.params.d).toBeGreaterThanOrEqual(-12);
    expect(result.params.d).toBeLessThanOrEqual(14);
    expect(result.params.B).toBeGreaterThanOrEqual(0);
    expect(result.params.B).toBeLessThanOrEqual(1);
  });
});
