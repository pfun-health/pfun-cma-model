/**
 * Lightweight persistence layer for LLM scenario results.
 * Appends each generated scenario as a JSON-line to a .jsonl file,
 * mirroring the DuckDB background task in pfun_cma_model/routes/llm.py.
 */

import fs from "fs";
import path from "path";

let dbPath: string | null = null;

export function initResultsStore(debug = false): void {
  const resultsDir = path.resolve("results");
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
  }
  dbPath = debug
    ? path.resolve("results/cma_recs-local.jsonl")
    : path.resolve("results/cma_recs.jsonl");
}

/**
 * Append a scenario result record asynchronously (fire-and-forget).
 * Mirrors save2duckdb() in pfun_cma_model/db.py.
 */
export function saveResultBackground(record: Record<string, unknown>): void {
  if (!dbPath) return;
  const line = JSON.stringify({ ...record, _saved_at: new Date().toISOString() }) + "\n";
  // Non-blocking append
  fs.appendFile(dbPath, line, (err) => {
    if (err) {
      console.error("[results] Failed to persist result:", err.message);
    }
  });
}
