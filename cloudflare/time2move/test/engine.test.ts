import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SIGNALS, type ActivityGains, type EngineInput, type ModelParams } from "../src/worker/engine.ts";
import { needsEngine, testEngine, testManifest } from "./helpers.ts";

/** The engine's own calibration (from engine.json); only read inside tests that have the engine. */
const gains = (): ActivityGains => testManifest().calibration.gains;
const model = (): ModelParams => testManifest().calibration.model;

function grid(n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => (24 * i) / (n - 1));
}

function input(overrides: Partial<EngineInput> = {}): EngineInput {
  return { t: grid(721), meals: [7, 11, 17.5], mealWidths: [1, 1, 1], activity: null, model: model(), gains: gains(), ...overrides };
}

type Fixture = {
  cases: Record<
    string,
    { params: ModelParams & { taug: number; tM: number[] }; t: number[] } & Record<"L" | "m" | "c" | "a" | "I_S" | "I_E" | "G", number[]>
  >;
};

test("wasm engine reproduces the public Python engine (no activity)", { skip: needsEngine }, () => {
  const fixture = JSON.parse(readFileSync(new URL("./fixtures/python-engine.json", import.meta.url), "utf8")) as Fixture;
  const engine = testEngine();
  let worst = 0;
  for (const [name, c] of Object.entries(fixture.cases)) {
    const p = c.params;
    const out = engine.run({
      t: Float64Array.from(c.t),
      meals: p.tM,
      mealWidths: p.tM.map(() => p.taug),
      activity: null,
      model: { d: p.d, taup: p.taup, B: p.B, Cm: p.Cm, toff: p.toff, eps: 1e-18 },
      gains: gains(),
    });
    for (const key of ["L", "m", "c", "a", "I_S", "I_E", "G"] as const) {
      for (let i = 0; i < c.t.length; i++) {
        const diff = Math.abs(out[key][i] - c[key][i]) / Math.max(1, Math.abs(c[key][i]));
        worst = Math.max(worst, diff);
        assert.ok(diff < 1e-9, `${name}: ${key}[${i}] wasm ${out[key][i]} vs python ${c[key][i]}`);
      }
    }
  }
  assert.ok(worst < 1e-9, `max relative difference ${worst}`);
});

test("every signal is returned, finite, and the right length", { skip: needsEngine }, () => {
  const engine = testEngine();
  const activity = new Float64Array(721);
  activity.fill(0.55, 360, 390);
  const out = engine.run(input({ activity }));
  for (const name of SIGNALS) {
    assert.equal(out[name].length, 721, name);
    assert.ok(out[name].every(Number.isFinite), `${name} has a non-finite value`);
  }
  assert.deepEqual(Array.from(engine.runG(input({ activity }))), Array.from(out.G));
});

test("activity lowers glucose, raises cortisol and leaves a sensitisation tail", { skip: needsEngine }, () => {
  const engine = testEngine();
  const t = grid(721);
  const activity = new Float64Array(721);
  for (let i = 0; i < t.length; i++) if (t[i] >= 12 && t[i] < 13) activity[i] = 0.55;
  const rest = engine.run(input());
  const moved = engine.run(input({ activity }));
  const at = (hour: number) => Math.round((hour / 24) * 720);

  // During the bout: less glucose, more cortisol, activity normalised by the engine's sigmoid.
  assert.ok(moved.G[at(12.5)] < rest.G[at(12.5)] * 0.95);
  assert.ok(moved.c_mod[at(12.5)] > rest.c_mod[at(12.5)]);
  assert.ok(moved.A[at(12.5)] > 0.5 && moved.A[at(12.5)] <= 1);
  assert.equal(moved.A[at(11.5)], 0);
  // After it: sensitisation persists and decays; before it: nothing changed.
  assert.ok(moved.s_A[at(13.5)] > moved.s_A[at(16)] && moved.s_A[at(16)] > 0);
  assert.ok(moved.I_E_eff[at(13.5)] > rest.I_E_eff[at(13.5)]);
  for (let i = 0; i < at(12); i++) assert.equal(moved.G[i], rest.G[i]);
  // Uptake is proportional, so glucose never goes negative.
  assert.ok(moved.G.every((g) => g > 0));
});

test("all-zero activity equals the no-activity model", { skip: needsEngine }, () => {
  const engine = testEngine();
  const a = engine.runG(input());
  const b = engine.runG(input({ activity: new Float64Array(721) }));
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-12);
});

test("input sizes are validated against the module's limits", { skip: needsEngine }, () => {
  const engine = testEngine();
  assert.equal(engine.maxPoints, 768);
  assert.equal(engine.maxMeals, 4);
  assert.throws(() => engine.run(input({ t: grid(769) })), RangeError);
  assert.throws(() => engine.run(input({ t: grid(1) })), RangeError);
  assert.throws(() => engine.run(input({ meals: [], mealWidths: [] })), RangeError);
  assert.throws(() => engine.run(input({ meals: [6, 9, 12, 15, 18], mealWidths: [1, 1, 1, 1, 1] })), RangeError);
  assert.throws(() => engine.run(input({ meals: [7, 12], mealWidths: [1] })), RangeError);
  assert.throws(() => engine.run(input({ activity: new Float64Array(10) })), RangeError);
});
