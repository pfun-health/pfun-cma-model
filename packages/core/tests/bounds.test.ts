import { describe, it, expect } from "vitest";
import {
  Bounds,
} from "../src/bounds.js";

describe("Bounds", () => {
  it("should create bounds with arrays", () => {
    const b = new Bounds([-1, 0], [1, 2], [true, true]);
    expect(b.lb).toEqual([-1, 0]);
    expect(b.ub).toEqual([1, 2]);
    expect(b.length).toBe(2);
  });

  it("should create scalar bounds expanded to arrays", () => {
    const b = new Bounds(-5, 5, true);
    expect(b.lb).toEqual([-5]);
    expect(b.ub).toEqual([5]);
  });

  it("should clip values to bounds", () => {
    const b = new Bounds([-1, 0], [1, 10]);
    expect(b.clip(0, 5)).toBe(1);
    expect(b.clip(0, -5)).toBe(-1);
    expect(b.clip(1, 5)).toBe(5);
    expect(b.clip(1, 15)).toBe(10);
  });

  it("should clip all values", () => {
    const b = new Bounds([0, 0], [1, 1]);
    expect(b.clipAll([2, -1])).toEqual([1, 0]);
    expect(b.clipAll([0.5, 0.5])).toEqual([0.5, 0.5]);
  });

  it("should compute residuals", () => {
    const b = new Bounds([0, 0], [10, 10]);
    const { sl, sb } = b.residual([5, 8]);
    expect(sl).toEqual([5, 8]);
    expect(sb).toEqual([5, 2]);
  });

  it("should serialize to JSON", () => {
    const b = new Bounds([0], [1], [true]);
    const json = b.toJSON();
    expect(json.lb).toEqual([0]);
    expect(json.ub).toEqual([1]);
    expect(json.keepFeasible).toEqual([true]);
  });

  it("should throw on mismatched lengths", () => {
    expect(() => new Bounds([0, 1], [1])).toThrow();
  });
});
