import { execSync } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const engineDir = join(__dirname, '../../pfun-cma-engine-c');

console.log('Building C engine...');
try {
  execSync('make', { cwd: engineDir, stdio: 'inherit' });
  console.log('C engine built successfully.');
} catch (error) {
  console.error('Failed to build C engine:', error);
  process.exit(1);
}
