import koffi from 'koffi';
import { join, dirname } from 'path';
import { existsSync } from 'fs';
import { platform } from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ext = platform() === 'win32' ? 'dll' : platform() === 'darwin' ? 'dylib' : 'so';
let libPath = join(__dirname, '../../../pfun-cma-engine-c/pfun_cma_engine/libpfun_cma_engine.' + ext);

if (!existsSync(libPath)) {
  libPath = join(__dirname, '../../pfun-cma-engine-c/pfun_cma_engine/libpfun_cma_engine.' + ext);
}

if (!existsSync(libPath)) {
  throw new Error(`Could not find the compiled C engine at ${libPath}. Did you run build?`);
}

/** Expected C engine function signature for ABI compatibility checks */
export const EXPECTED_ENGINE_SIGNATURE = {
  name: 'run_cma_model',
  returnType: 'void',
  paramTypes: [
    'const double*', 'int', 'double', 'double', 'double',
    'const double*', 'double', 'double', 'double',
    'const double*', 'int', 'int*', 'double',
    'double*', 'double*', 'double*', 'double*',
    'double*', 'double*', 'double*', 'double*'
  ],
  paramCount: 21
} as const;

const lib = koffi.load(libPath);

export const run_cma_model = lib.func('run_cma_model', 'void', [
  'const double*',
  'int',
  'double',
  'double',
  'double',
  'const double*',
  'double',
  'double',
  'double',
  'const double*',
  'int',
  'int*',
  'double',
  'double*',
  'double*',
  'double*',
  'double*',
  'double*',
  'double*',
  'double*',
  'double*'
]);
