import { CMASleepWakeModel } from './src/cma';

const model = new CMASleepWakeModel({ N: 288 });
model.solve();
console.log('Keys in solution:', Object.keys(model.solution || {}));
console.log('G array length:', model.solution?.G.length);
console.log('G values sample:', model.solution?.G.slice(0, 5));
