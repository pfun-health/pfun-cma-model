/**
 * Data routes: /data/*
 */

import { Hono } from "hono";
import { stream } from "hono/streaming";
import { readSampleData, type SampleRow } from "../data.js";

type MediaType = "json" | "text" | "html" | "octet-stream";
export function createDataRoutes(): Hono {
  const app = new Hono();

  // GET /data/sample/download
  app.get("/sample/download", (c) => {
    const nrows = parseInt(c.req.query("nrows") ?? "23", 10);
    const mediaType = (c.req.query("media_type") ?? "html") as MediaType;
    const pct0 = parseFloat(c.req.query("pct0") ?? "0");

    // Validate
    if (nrows < -1) {
      return c.json(
        { detail: "nrows must be -1 (for full dataset) or a non-negative integer." },
        400,
      );
    }
    if (pct0 < 0.0 || pct0 > 1.0) {
      return c.json({ detail: "pct0 must be between 0.0 and 1.0." }, 400);
    }

    if (mediaType === "octet-stream") {
      return c.json(
        { message: "Octet-stream download not implemented in non-streaming endpoint." },
        501,
      );
    }

const allData = readSampleData();
    const data = selectRows(allData, pct0, nrows);

    if (mediaType === "json") {
      return c.json(data, 200, {
        "Content-Type": "application/json",
      });
    } else if (mediaType === "text") {
      const csv = toCsv(data);
      return new Response(csv, {
        headers: { "Content-Type": "text/csv" },
      });
    } else {
      const html = toHtmlTable(data);
      return c.html(html);
    }
  });

  // GET /data/sample/stream
  app.get("/sample/stream", (c) => {
    const nrows = parseInt(c.req.query("nrows") ?? "10", 10);
    const mediaType = (c.req.query("media_type") ?? "text") as MediaType;
    const pct0 = parseFloat(c.req.query("pct0") ?? "0.5");

    // Validate
    if (nrows < -1) {
      return c.json(
        { detail: "nrows must be -1 (for full dataset) or a non-negative integer." },
        400,
      );
    }
    if (pct0 < 0.0 || pct0 > 1.0) {
      return c.json({ detail: "pct0 must be between 0.0 and 1.0." }, 400);
    }

const allData = readSampleData();
    const data = selectRows(allData, pct0, nrows);

    if (mediaType === "octet-stream") {
      const csv = toCsv(data, false, false);
      return stream(c, async (s) => {
        c.header("Content-Type", "application/octet-stream");
        c.header("Transfer-Encoding", "chunked");
        const chunkSize = 512;
        for (let i = 0; i < csv.length; i += chunkSize) {
          await s.write(csv.slice(i, i + chunkSize));
        }
      });
    }

    if (mediaType === "json") {
      return stream(c, async (s) => {
        c.header("Content-Type", "application/json");
        await s.write(JSON.stringify(data));
      });
    } else if (mediaType === "text") {
      return stream(c, async (s) => {
        c.header("Content-Type", "text/csv");
        await s.write(toCsv(data));
      });
    } else {
      return stream(c, async (s) => {
        c.header("Content-Type", "text/html");
        await s.write(toHtmlTable(data));
      });
    }
  });

  return app;
}

function selectRows(allData: SampleRow[], pct0: number, nrows: number): SampleRow[] {
  const totalRows = allData.length;
  const row0 = Math.floor(pct0 * totalRows);

  if (nrows === -1) {
    return allData.slice(row0);
  }

  // Wrap-around indexing
  const result: SampleRow[] = [];
  for (let i = 0; i < nrows; i++) {
    const idx = (row0 + i) % totalRows;
    result.push(allData[idx]);
  }
  return result;
}

function toCsv(data: SampleRow[], includeHeader: boolean = true, includeIndex: boolean = true): string {
  if (data.length === 0) return "";
  const keys = Object.keys(data[0]);
  const lines: string[] = [];
  if (includeHeader) {
    const header = includeIndex ? ["", ...keys] : keys;
    lines.push(header.join(","));
  }
  data.forEach((row, i) => {
    const values = keys.map((k) => String(row[k]));
    const line = includeIndex ? [String(i), ...values] : values;
    lines.push(line.join(","));
  });
  return lines.join("\n");
}

function toHtmlTable(data: SampleRow[]): string {
  if (data.length === 0) return "<table></table>";
  const keys = Object.keys(data[0]);
  const headerRow = `<tr>${keys.map((k) => `<th>${k}</th>`).join("")}</tr>`;
  const bodyRows = data
    .map((row) => `<tr>${keys.map((k) => `<td>${row[k]}</td>`).join("")}</tr>`)
    .join("\n");
  return `<table>\n<thead>\n${headerRow}\n</thead>\n<tbody>\n${bodyRows}\n</tbody>\n</table>`;
}
