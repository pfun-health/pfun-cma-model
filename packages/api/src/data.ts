/**
 * Sample data reader.
 * Reads the bundled valid_data.csv and returns parsed records.
 * Mirrors pfun_cma_model/data.py (read_sample_data) and pathdefs.py.
 */

import fs from "fs";
import path from "path";
import { CMASleepWakeModel } from "@pfun/core";

export interface SampleRow {
  t?: number;
  G?: number;
  c?: number;
  m?: number;
  [key: string]: number | boolean | string | undefined;
}

// Path to the bundled CSV (relative to this source file: src/data.ts → ../data/)
const DEFAULT_CSV_PATH = path.resolve(
  new URL("../data/valid_data.csv", import.meta.url).pathname,
);

let cachedData: SampleRow[] | null = null;

/**
 * Parse a CSV string into an array of record objects.
 */
function parseCsv(csv: string): SampleRow[] {
  const lines = csv.trim().split("\n");
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: SampleRow = {};
    headers.forEach((header, i) => {
      const raw = (values[i] ?? "").trim();
      if (raw === "" || raw === "NaN" || raw === "None") {
        row[header] = undefined;
      } else if (raw === "True" || raw === "False") {
        row[header] = raw === "True";
      } else {
        const num = Number(raw);
        row[header] = isNaN(num) ? raw : num;
      }
    });
    return row;
  });
}

/**
 * Read sample data from the bundled CSV file.
 * Falls back to synthetic model data if the file is not available.
 * Mirrors PFunDataPaths.read_sample_data() in Python.
 */
export function readSampleData(csvPath?: string): SampleRow[] {
  if (cachedData) return cachedData;

  const filePath = csvPath ?? DEFAULT_CSV_PATH;

  try {
    const content = fs.readFileSync(filePath, "utf-8");
    cachedData = parseCsv(content);
    return cachedData;
  } catch {
    console.warn("[data] Sample CSV not found, falling back to synthetic model data.");
    return generateSyntheticData();
  }
}

/**
 * Generate synthetic sample data using the CMA model as fallback.
 */
function generateSyntheticData(): SampleRow[] {
  const model = new CMASleepWakeModel();
  const results = model.run();
  return results.map((r) => ({
    t: r.t,
    G: r.G,
    c: r.c,
    m: r.m,
  }));
}
