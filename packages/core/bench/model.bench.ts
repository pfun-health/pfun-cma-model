import { bench, describe } from "vitest";
import { CMASleepWakeModel } from "../src/model.js";
import { fitModel } from "../src/fit.js";
import { computeG, linspace, K, E, Light } from "../src/calc.js";

describe("CMA Model Benchmarks", () => {
  bench("model.run() N=1024 (default)", () => {
    const model = new CMASleepWakeModel({ N: 1024 });
    model.run();
  });

  bench("model.run() N=4096", () => {
    const model = new CMASleepWakeModel({ N: 4096 });
    model.run();
  });

  bench("model.runAtTime() 1000 points", () => {
    const model = new CMASleepWakeModel();
    model.runAtTime(0, 24, 1000);
  });

  bench("model.runAtTime() 100 points", () => {
    const model = new CMASleepWakeModel();
    model.runAtTime(0, 24, 100);
  });

  bench("computeG 3 meals x 1024 points", () => {
    const t = linspace(0, 24, 1024);
    const I_E = new Array(1024).fill(0.05);
    computeG(t, I_E, [7.0, 11.0, 17.5], 1.0, 0.05, 0.0, 0.0, true);
  });

  bench("K function x 10000", () => {
    for (let i = 0; i < 10000; i++) {
      K(i / 10000);
    }
  });

  bench("E function x 10000", () => {
    for (let i = -5000; i < 5000; i++) {
      E(i / 1000);
    }
  });

  bench("Light function x 10000", () => {
    for (let i = -5000; i < 5000; i++) {
      Light(i / 1000);
    }
  });

  bench("linspace 10000 points", () => {
    linspace(0, 24, 10000);
  });
});

describe("Model Fitting Benchmarks", () => {
  const model = new CMASleepWakeModel({ B: 0.1 });
  const data = model.runAtTime(0, 24, 50);
  const t = data.map((d) => parseFloat(d.x));
  const G = data.map((d) => parseFloat(d.y));

  bench("fitModel 20 iterations, 50 data points", () => {
    fitModel({ t, G }, {}, 20);
  });

  bench("fitModel 10 iterations, 30 data points", () => {
    const shortData = model.runAtTime(0, 24, 30);
    fitModel(
      {
        t: shortData.map((d) => parseFloat(d.x)),
        G: shortData.map((d) => parseFloat(d.y)),
      },
      {},
      10,
    );
  });
});

describe("Streaming Benchmarks", () => {
  bench("runAtTimeStream generator 1000 points", () => {
    const model = new CMASleepWakeModel();
    const gen = model.runAtTimeStream(0, 24, 1000);
    for (const _ of gen) {
      // consume
    }
  });

  bench("runFullStream generator 100 points", () => {
    const model = new CMASleepWakeModel({ N: 200 });
    const gen = model.runFullStream(0, 24, 100);
    for (const _ of gen) {
      // consume
    }
  });
});
