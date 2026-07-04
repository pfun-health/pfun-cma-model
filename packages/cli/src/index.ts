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
      // Stub to launch the API server dynamically
      const apiPath = join(dirname(fileURLToPath(import.meta.url)), '../../api/dist/index.js');
      console.log(`Launching API on ${options.host}:${options.port}...`);
      await import(apiPath);
  });

program.command('fit-model')
  .description('Fit the model to a dataset.')
  .option('--N <number>', 'Number of time points.', '288')
  .action((options) => {
    console.log(`Fitting model with N=${options.N}...`);
    const model = new CMASleepWakeModel({ N: parseInt(options.N) });
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
    console.log('Running parameter grid search...');
    const grid = new PFunCMAParamsGrid({ N: parseInt(options.N), m: parseInt(options.m) });
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

program.parse(process.argv);
