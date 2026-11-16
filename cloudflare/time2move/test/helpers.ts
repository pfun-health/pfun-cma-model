import { existsSync, readFileSync } from "node:fs";
import { createEngine, type ActivityGains, type Engine, type ModelParams } from "../src/worker/engine.ts";

/** Engine files copied in by `npm run engine` (absent in a fresh public checkout). */
export const WASM_PATH = new URL("../src/worker/cma.wasm", import.meta.url).pathname;
export const MANIFEST_PATH = new URL("../src/worker/engine.json", import.meta.url).pathname;
export const HAVE_ENGINE = existsSync(WASM_PATH) && existsSync(MANIFEST_PATH);

/** node:test `skip` option for tests that need the engine. */
export const needsEngine = HAVE_ENGINE ? false : "engine files are missing (run `npm run engine`)";

export type EngineManifest = {
  repository: string;
  revision: string | null;
  source: string;
  bytes: number;
  sha256: string;
  calibration: { model: ModelParams; gains: ActivityGains; meals: number[]; mealWidths: number[] };
};

let cachedEngine: Engine | undefined;
export function testEngine(): Engine {
  cachedEngine ??= createEngine(new WebAssembly.Module(readFileSync(WASM_PATH)));
  return cachedEngine;
}

let cachedManifest: EngineManifest | undefined;
export function testManifest(): EngineManifest {
  cachedManifest ??= JSON.parse(readFileSync(MANIFEST_PATH, "utf8")) as EngineManifest;
  return cachedManifest;
}
