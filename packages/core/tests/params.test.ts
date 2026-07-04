import { describe, it, expect } from "vitest";
import {
  CMAModelParamsSchema,
  getDefaultParams,
  getParamsJsonSchema,
  calcSerr,
  getQualitativeDescriptor,
  describeParam,
  getBoundedParamInfo,
  generateParamsTable,
  BOUNDED_PARAM_KEYS,
  BOUNDED_PARAM_LB,
  BOUNDED_PARAM_UB,
  BOUNDED_PARAM_MID,
} from "../src/params.js";

describe("CMAModelParams", () => {
  it("should produce correct defaults", () => {
    const params = getDefaultParams();
    expect(params.N).toBe(1024);
    expect(params.d).toBe(0.0);
    expect(params.taup).toBe(1.0);
    expect(params.taug).toBe(1.0);
    expect(params.B).toBe(0.05);
    expect(params.Cm).toBe(0.0);
    expect(params.toff).toBe(0.0);
    expect(params.tM).toEqual([7.0, 11.0, 17.5]);
    expect(params.seed).toBeNull();
    expect(params.eps).toBe(1e-18);
  });

  it("should parse partial inputs", () => {
    const params = CMAModelParamsSchema.parse({ d: 5.0, B: 0.1 });
    expect(params.d).toBe(5.0);
    expect(params.B).toBe(0.1);
    expect(params.N).toBe(1024); // default
  });

  it("should handle taug as array", () => {
    const params = CMAModelParamsSchema.parse({ taug: [1.0, 1.5, 2.0] });
    expect(params.taug).toEqual([1.0, 1.5, 2.0]);
  });

  it("should validate bounded param keys match spec", () => {
    expect(BOUNDED_PARAM_KEYS).toEqual(["d", "taup", "taug", "B", "Cm", "toff"]);
  });

  it("should have correct bounds", () => {
    expect(BOUNDED_PARAM_LB.d).toBe(-12.0);
    expect(BOUNDED_PARAM_UB.d).toBe(14.0);
    expect(BOUNDED_PARAM_LB.taup).toBe(0.5);
    expect(BOUNDED_PARAM_UB.taup).toBe(3.0);
    expect(BOUNDED_PARAM_LB.taug).toBe(0.1);
    expect(BOUNDED_PARAM_UB.taug).toBe(3.0);
    expect(BOUNDED_PARAM_LB.B).toBe(0.0);
    expect(BOUNDED_PARAM_UB.B).toBe(1.0);
    expect(BOUNDED_PARAM_LB.Cm).toBe(0.0);
    expect(BOUNDED_PARAM_UB.Cm).toBe(2.0);
    expect(BOUNDED_PARAM_LB.toff).toBe(-3.0);
    expect(BOUNDED_PARAM_UB.toff).toBe(3.0);
  });
});

describe("Qualitative descriptors", () => {
  it("should return Normal for midpoint values", () => {
    const serr = calcSerr("d", BOUNDED_PARAM_MID.d);
    expect(serr).toBeCloseTo(0);
    expect(getQualitativeDescriptor(serr)).toBe("Normal");
  });

  it("should return High for high values", () => {
    const serr = calcSerr("d", 10.0);
    expect(serr).toBeGreaterThan(0);
    expect(getQualitativeDescriptor(serr)).toContain("High");
  });

  it("should return Low for low values", () => {
    const serr = calcSerr("d", -10.0);
    expect(serr).toBeLessThan(0);
    expect(getQualitativeDescriptor(serr)).toContain("Low");
  });

  it("should include Very for extreme values", () => {
    const serr = calcSerr("B", 0.9);
    expect(Math.abs(serr)).toBeGreaterThan(0.23);
    expect(getQualitativeDescriptor(serr)).toContain("Very");
  });
});

describe("describeParam", () => {
  it("should include description and qualifier", () => {
    const desc = describeParam("d", 0.0);
    expect(desc).toContain("Time zone offset");
    expect(desc).toContain("Normal");
  });
});

describe("getBoundedParamInfo", () => {
  it("should return full metadata", () => {
    const info = getBoundedParamInfo("B", 0.05);
    expect(info.name).toBe("B");
    expect(info.value).toBe(0.05);
    expect(info.min).toBe(0.0);
    expect(info.max).toBe(1.0);
    expect(info.default).toBe(0.05);
    expect(info.step).toBeGreaterThan(0);
    expect(info.description).toContain("Glucose baseline");
  });
});

describe("generateParamsTable", () => {
  const params = getDefaultParams();

  it("should generate markdown table", () => {
    const table = generateParamsTable(params, "md");
    expect(table).toContain("| Parameter");
    expect(table).toContain("| d |");
    expect(table).toContain("| B |");
  });

  it("should generate HTML table", () => {
    const table = generateParamsTable(params, "html");
    expect(table).toContain("<table>");
    expect(table).toContain("<th>Parameter</th>");
    expect(table).toContain("<td>d</td>");
  });

  it("should generate JSON containing markdown table", () => {
    const json = generateParamsTable(params, "json");
    const parsed = JSON.parse(json);
    expect(parsed.table).toContain("| Parameter");
  });
});

describe("getParamsJsonSchema", () => {
  it("should return valid schema object", () => {
    const schema = getParamsJsonSchema();
    expect(schema.title).toBe("CMAModelParams");
    expect(schema.type).toBe("object");
    expect(schema.properties).toBeDefined();
    const props = schema.properties as Record<string, unknown>;
    expect(props.N).toBeDefined();
    expect(props.d).toBeDefined();
  });
});
