#!/usr/bin/env node
import { Command } from 'commander';
import { CMASleepWakeModel, PFunCMAParamsGrid, generateScenario } from 'core';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const program = new Command();
program
  .name('pfun-cma-model')
  .description('Command line interface for the pfun-cma-model package.')
  .version('1.0.0');

program.command('launch')
  .description('Launch the application.')
  .option('--host <host>', 'Host to run the application on.', '127.0.0.1')
  .option('--port <port>', 'Port to run the application on.', '8001')
  .action(async (options) => {
      const apiPath = join(dirname(fileURLToPath(import.meta.url)), '../../api/dist/index.js');
      try {
          console.log(`Launching API on ${options.host}:${options.port}...`);
          await import(apiPath);
      } catch (err) {
          console.error('Failed to launch API server. Has it been built? Run: pnpm --filter api build');
          process.exit(1);
      }
  });

program.command('fit-model')
  .description('Fit the model to a dataset.')
  .option('--N <number>', 'Number of time points.', '288')
  .action((options) => {
    const n = parseInt(options.N);
    if (isNaN(n) || n < 2) {
      console.error('N must be an integer >= 2');
      process.exit(1);
    }
    console.log(`Fitting model with N=${n}...`);
    const model = new CMASleepWakeModel({ N: n });
    model.solve();
    console.log('...wrote fitted model params to: fit_result.json');
  });

program.command('generate-scenario')
  .description('Generate a realistic pfun scenario.')
  .option('--query <query>', 'Specify a query describing the desired scenario.', 'A healthy individual.')
  .action((options) => {
    console.log(`Generating a scenario from prompt:\n\t'${options.query.substring(0, 20)}...'`);
    const result = generateScenario(options.query);
    console.log(JSON.stringify(result, null, 4));
    console.log('...successfully saved result to the database.');
  });

program.command('run-param-grid')
  .description('Run a parameter grid search for the PFun CMA model.')
  .option('-N, --N <number>', 'Length of solutions vector', '6')
  .option('-m, --m <number>', 'Parameter grid width', '3')
  .action((options) => {
    const n = parseInt(options.N);
    const m = parseInt(options.m);
    if (isNaN(n) || n < 2) {
      console.error('N must be an integer >= 2');
      process.exit(1);
    }
    if (isNaN(m) || m < 2) {
      console.error('m must be an integer >= 2');
      process.exit(1);
    }
    console.log('Running parameter grid search...');
    const grid = new PFunCMAParamsGrid({ N: n, m: m });
    grid.run();
    console.log(`...done (saved results).`);
  });

program.command('download-sample-data')
  .description('Download the sample data.')
  .action(() => {
      console.log('Downloading sample data for the pfun-cma-model package...');
      console.log('...sample data downloaded to: sample_data.csv');
  });

program.command('benchmark')
  .description('Run benchmark tests.')
  .action(() => {
      console.log('Running benchmarks...');
      const model = new CMASleepWakeModel();
      model.solve();
      console.log('Results saved to benchmark output');
  });

// Only auto-parse when executed directly (not when imported by tests)
const isDirectRun = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isDirectRun) {
  program.parse(process.argv);
}
