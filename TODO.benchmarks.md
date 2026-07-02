# Benchmark Improvements TODO

## High Priority (Immediate value) — COMPLETE ✓

### 1. Command-line arguments for configurability
- [x] Add CLI options for benchmark sizes (`--sizes`)
- [x] Add option for number of repetitions (`--repeats`)
- [x] Add option to specify which implementations to test (`--implementations python,c`)
- [x] Add option to output results to file (`--output-file`)
- [x] Add option to specify model parameters (`--params`)

### 2. Enhanced performance metrics
- [x] Add mean, min, max, and standard deviation measurements
- [x] Add warmup runs to exclude cold start effects (`--warmup`)

### 3. Better error handling and validation
- [x] Add validation of input parameters (sizes, repeats, warmup, threshold)
- [x] Add better error messages for missing C library (graceful in Python-only mode)
- [x] Add customizable correctness threshold (`--threshold`)

### 3b. Missing separator section header
- [x] Restored missing `_SEPARATOR` section comment

## Medium Priority (Significant value)

### 4. Configuration management
- [ ] Create a configuration class for benchmark parameters
- [ ] Support loading configurations from JSON/YAML files
- [ ] Add environment variable support for configuration

### 5. Extended benchmark scenarios
- [ ] Add parameter sweep functionality (test different parameter combinations)
- [ ] Add meal time variation testing
- [ ] Add different time vector configurations (non-uniform spacing)
- [ ] Add random seed testing for stochastic scenarios

### 6. Improved reporting
- [ ] Add CSV output format option
- [ ] Add visualization of results (plots)
- [ ] Add comparison with previous runs
- [ ] Add confidence intervals to reporting

## Low Priority (Nice to have)

### 7. Parallel execution
- [ ] Add option to run different sizes in parallel
- [ ] Add option to run Python and C implementations in parallel
- [ ] Add process-level isolation for more accurate measurements

### 8. Documentation improvements
- [ ] Add detailed docstrings for all functions
- [ ] Add usage examples in help text
- [ ] Add explanation of what each metric means
- [ ] Add guidance on interpreting results

### 9. Resource monitoring
- [ ] Add CPU usage monitoring
- [ ] Add memory consumption tracking
- [ ] Add system load monitoring

### 10. Test suite for benchmark itself
- [ ] Add unit tests for benchmark functions
- [ ] Add integration tests for the CLI interface
- [ ] Add validation tests for correctness checking