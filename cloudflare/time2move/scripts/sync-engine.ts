/**
 * Bring the PFun CMA engine (WebAssembly) into this demo.
 *
 * The engine lives in the pfun-cma-engine-c repository and is not committed here.
 * This script looks for a checkout of that repository, optionally rebuilds the
 * module from source, and copies it to src/worker/cma.wasm. It also writes
 * src/worker/engine.json: where the module came from plus the engine's own
 * calibration (baseline model parameters and activity gains from demos/scenarios.json),
 * so this demo never carries a second copy of those numbers.
 *
 *   node scripts/sync-engine.ts            copy the committed cloudflare/time2eat/cma.wasm
 *   node scripts/sync-engine.ts --build    rebuild from C first (needs clang + wasm-ld)
 *   node scripts/sync-engine.ts --check    exit non-zero if the engine files are absent
 *
 * Checkout lookup order: $PFUN_CMA_ENGINE_C, then a sibling of this repository
 * (../pfun-cma-engine-c next to pfun-cma-model).
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "src/worker/cma.wasm");
const manifest = join(root, "src/worker/engine.json");
const args = new Set(process.argv.slice(2));

function fail(message: string): never {
  console.error(`engine: ${message}`);
  process.exit(1);
}

if (args.has("--check")) {
  for (const file of [target, manifest]) {
    if (!existsSync(file)) fail(`${file} is missing; run \`npm run engine\` (see README.md, "Getting the engine")`);
  }
  console.log(`engine: ${target} and engine.json present`);
  process.exit(0);
}

const candidates = [process.env.PFUN_CMA_ENGINE_C, resolve(root, "../../../pfun-cma-engine-c")].filter(
  (p): p is string => Boolean(p),
);
const engineRepo = candidates.find((p) => existsSync(join(p, "src/pfun_cma_engine.c")));
if (!engineRepo) {
  fail(
    "could not find a pfun-cma-engine-c checkout. Clone https://github.com/pfun-health/pfun-cma-engine-c " +
      "next to pfun-cma-model, or set PFUN_CMA_ENGINE_C to its path.\n  looked in: " +
      candidates.join(", "),
  );
}

let source = join(engineRepo, "cloudflare/time2eat/cma.wasm");
if (args.has("--build")) {
  // Same recipe as `make wasm-worker` in pfun-cma-engine-c, written to a temp file.
  source = join(mkdtempSync(join(tmpdir(), "time2move-engine-")), "cma.wasm");
  try {
    execFileSync(
      "clang",
      [
        "--target=wasm32", "-O2", "-ffreestanding", "-nostdlib", "-fvisibility=hidden",
        `-I${join(engineRepo, "include")}`, `-I${join(engineRepo, "src/wasm/include")}`, `-I${join(engineRepo, "src")}`,
        "-Wl,--no-entry", "-Wl,--allow-undefined", "-Wl,--export-dynamic",
        "-Wl,--initial-memory=262144", "-Wl,--max-memory=262144", "-Wl,-z,stack-size=32768",
        join(engineRepo, "src/wasm/worker_shim.c"), join(engineRepo, "src/pfun_cma_engine.c"),
        "-o", source,
      ],
      { stdio: "inherit" },
    );
  } catch {
    fail("clang could not build the engine (needs clang with the wasm32 target and wasm-ld); run without --build to copy the committed module");
  }
} else if (!existsSync(source)) {
  fail(`${source} does not exist; run with --build to compile it from source`);
}

function gitRevision(dir: string): string | null {
  try {
    return execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

type Scenario = { modelParams: Record<string, number>; activityParams: Record<string, number>; meals: number[]; mealWidths: number[] };

function calibration(): Scenario {
  const file = join(engineRepo!, "demos/scenarios.json");
  if (!existsSync(file)) fail(`${file} does not exist; this demo needs the engine's baseline calibration`);
  const baseline = (JSON.parse(readFileSync(file, "utf8")) as Record<string, Scenario>).baseline;
  const need = (group: Record<string, number> | undefined, keys: string[], label: string) => {
    for (const key of keys) {
      if (typeof group?.[key] !== "number" || !Number.isFinite(group[key])) fail(`demos/scenarios.json baseline.${label}.${key} is missing or not a number`);
    }
  };
  need(baseline?.modelParams, ["d", "taup", "B", "Cm", "toff", "eps"], "modelParams");
  need(baseline?.activityParams, ["k_A", "A_thresh", "gamma_A", "kappa_A", "beta_A", "eta_A", "tau_A"], "activityParams");
  if (!Array.isArray(baseline.meals) || baseline.meals.length !== baseline.mealWidths?.length) fail("demos/scenarios.json baseline meals/mealWidths are malformed");
  return baseline;
}

const base = calibration();
copyFileSync(source, target);
const bytes = readFileSync(target);
const info = {
  repository: "pfun-health/pfun-cma-engine-c",
  revision: gitRevision(engineRepo),
  source: args.has("--build") ? "built from src/wasm/worker_shim.c + src/pfun_cma_engine.c" : "cloudflare/time2eat/cma.wasm",
  bytes: bytes.length,
  sha256: createHash("sha256").update(bytes).digest("hex"),
  calibration: {
    model: base.modelParams,
    gains: base.activityParams,
    meals: base.meals,
    mealWidths: base.mealWidths,
  },
};
writeFileSync(manifest, JSON.stringify(info, null, 2) + "\n");
console.log(`engine: copied ${info.bytes} bytes from ${engineRepo} (${info.revision?.slice(0, 12) ?? "unknown revision"}) and wrote engine.json`);
