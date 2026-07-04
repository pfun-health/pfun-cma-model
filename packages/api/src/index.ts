import app from './app.js';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config({ path: resolve(__dirname, '../.env') });

const PORT = process.env.PORT || 8001;

app.listen(PORT, () => {
    console.log(`CMA Model API listening on port ${PORT}`);
});
