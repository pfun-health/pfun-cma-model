import app from './app.js';
import dotenv from 'dotenv';

dotenv.config();

const PORT = process.env.PORT || 8001;

app.listen(PORT, () => {
    console.log(`CMA Model API listening on port ${PORT}`);
});
