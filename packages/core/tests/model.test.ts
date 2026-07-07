import { describe, it, expect } from "vitest";
import { CMASleepWakeModel } from "../src/model.js";

describe("CMASleepWakeModel", () => {
  describe("constructor", () => {
    it("should create model with default params", () => {
      const model = new CMASleepWakeModel();
      const params = model.params;
      expect(params.N).toBe(1024);
      expect(params.d).toBe(0.0);
      expect(params.tM).toEqual([7.0, 11.0, 17.5]);
    });

    it("should accept custom params", () => {
      const model = new CMASleepWakeModel({ d: 2.0, B: 0.1 });
      expect(model.params.d).toBe(2.0);
      expect(model.params.B).toBe(0.1);
    });
  });

  describe("getTimeVector", () => {
    it("should generate default time vector", () => {
      const model = new CMASleepWakeModel({ N: 100 });
      const t = model.getTimeVector();
      expect(t.length).toBe(100);
      expect(t[0]).toBe(0);
      expect(t[t.length - 1]).toBe(24);
    });

    it("should generate custom time vector", () => {
      const model = new CMASleepWakeModel();
      const t = model.getTimeVector(5, 10, 50);
      expect(t.length).toBe(50);
      expect(t[0]).toBe(5);
      expect(t[t.length - 1]).toBeCloseTo(10);
    });
  });

  describe("run", () => {
    it("should return array of model output rows", () => {
      const model = new CMASleepWakeModel({ N: 100 });
      const results = model.run();
      expect(results.length).toBe(100);
    });

    it("should include all required columns", () => {
      const model = new CMASleepWakeModel({ N: 10 });
      const results = model.run();
      const row = results[0];
      expect(row).toHaveProperty("t");
      expect(row).toHaveProperty("c");
      expect(row).toHaveProperty("m");
      expect(row).toHaveProperty("a");
      expect(row).toHaveProperty("I_S");
      expect(row).toHaveProperty("I_E");
      expect(row).toHaveProperty("L");
      expect(row).toHaveProperty("G");
      expect(row).toHaveProperty("is_meal");
    });

    it("should include per-meal glucose columns", () => {
      const model = new CMASleepWakeModel({ N: 10 });
      const results = model.run();
      const row = results[0];
      expect(row).toHaveProperty("g_0");
      expect(row).toHaveProperty("g_1");
      expect(row).toHaveProperty("g_2");
    });

    it("should produce finite values", () => {
      const model = new CMASleepWakeModel({ N: 50 });
      const results = model.run();
      for (const row of results) {
        expect(Number.isFinite(row.t)).toBe(true);
        expect(Number.isFinite(row.c)).toBe(true);
        expect(Number.isFinite(row.m)).toBe(true);
        expect(Number.isFinite(row.G)).toBe(true);
      }
    });

    it("should accept config override", () => {
      const model = new CMASleepWakeModel({ N: 50 });
      const results1 = model.run();
      const results2 = model.run({ B: 0.5 });
      // Different B should produce different G values
      expect(results1[25].G).not.toBeCloseTo(results2[25].G, 1);
    });

    it("should be deterministic with same seed", () => {
      const model1 = new CMASleepWakeModel({ N: 50, seed: 42 });
      const model2 = new CMASleepWakeModel({ N: 50, seed: 42 });
      const r1 = model1.run();
      const r2 = model2.run();
      for (let i = 0; i < r1.length; i++) {
        expect(r1[i].G).toBeCloseTo(r2[i].G);
      }
    });
  });

  describe("runAtTime", () => {
    it("should return array of {x, y} results", () => {
      const model = new CMASleepWakeModel();
      const results = model.runAtTime(0, 24, 50);
      expect(results.length).toBe(50);
      expect(results[0]).toHaveProperty("x");
      expect(results[0]).toHaveProperty("y");
    });

    it("should return string values for x and y", () => {
      const model = new CMASleepWakeModel();
      const results = model.runAtTime(0, 24, 10);
      expect(typeof results[0].x).toBe("string");
      expect(typeof results[0].y).toBe("string");
    });

    it("should cover requested time range", () => {
      const model = new CMASleepWakeModel();
      const results = model.runAtTime(5, 20, 100);
      const firstT = parseFloat(results[0].x);
      const lastT = parseFloat(results[results.length - 1].x);
      expect(firstT).toBeCloseTo(5);
      expect(lastT).toBeCloseTo(20);
    });
  });

  describe("runAtTimeStream", () => {
    it("should yield results as generator", () => {
      const model = new CMASleepWakeModel();
      const gen = model.runAtTimeStream(0, 24, 10);
      const results = [...gen];
      expect(results.length).toBe(10);
      expect(results[0]).toHaveProperty("x");
      expect(results[0]).toHaveProperty("y");
    });
  });

  describe("runFullStream", () => {
    it("should yield full model tuples", () => {
      const model = new CMASleepWakeModel({ N: 50 });
      const gen = model.runFullStream(0, 24, 20);
      const results = [...gen];
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty("t");
      expect(results[0]).toHaveProperty("c");
      expect(results[0]).toHaveProperty("m");
      expect(results[0]).toHaveProperty("a");
    });
  });

  describe("updateParams", () => {
    it("should update parameters", () => {
      const model = new CMASleepWakeModel();
      model.updateParams({ d: 5.0, B: 0.2 });
      expect(model.params.d).toBe(5.0);
      expect(model.params.B).toBe(0.2);
    });
  });
});
