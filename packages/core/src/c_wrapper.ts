import { load } from 'koffi';
import { join } from 'path';

// Load the compiled shared library (adjust extension based on OS: .so, .dylib, .dll)
const libPath = join(__dirname, '../../pfun-cma-engine-c/build/libpfun_cma_engine.so');
const engine = load(libPath);

// Replicate the C wrapper from c_wrapper.py
export const cma_init = engine.func('cma_init', 'int', []);
export const cma_step = engine.func('cma_step', 'void', ['int']);
export const cma_cleanup = engine.func('cma_cleanup', 'void', []);
