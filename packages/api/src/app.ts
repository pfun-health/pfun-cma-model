import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import path from 'path';
import { fileURLToPath } from 'url';
import rateLimit from 'express-rate-limit';
import { CMASleepWakeModel, CMAModelParamsSchema } from 'core';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(cors());
app.use(helmet());
app.use(morgan('dev'));
app.use(express.json());

// Set up rate limiter: maximum of 100 requests per 15 minutes
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: "Too many requests from this IP, please try again after 15 minutes."
});

// Apply rate limiter to all routes
app.use(limiter);

// Basic health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Serve static assets and templates
app.use('/static', express.static(path.join(__dirname, '../static')));

// Demo route
app.get('/demo', (req, res) => {
    // Basic static serving for now
    res.sendFile(path.join(__dirname, '../templates/demo.html'));
});

// Fit model endpoint
app.post('/model/fit', (req, res) => {
    try {
        const params = CMAModelParamsSchema.parse(req.body);
        const model = new CMASleepWakeModel(params);
        model.solve();
        res.json({
            params: model.params,
            solution: model.solution
        });
    } catch (error: any) {
        res.status(400).json({ error: error.message });
    }
});

// Streaming endpoint simulating time-based yield
app.post('/model/run-at-time/stream', async (req, res) => {
    res.setHeader('Content-Type', 'application/x-ndjson');
    try {
        const params = CMAModelParamsSchema.parse(req.body);
        const model = new CMASleepWakeModel(params);
        model.solve();

        const N = model.solution?.G.length || 0;

        // Simulating yielding chunks dynamically
        for (let i = 0; i < N; i += Math.max(1, Math.floor(N / 10))) {
            const chunk = {
                t: model.solution?.t.slice(i, i + 10),
                G: model.solution?.G.slice(i, i + 10)
            };
            res.write(JSON.stringify(chunk) + '\n');
            // Small delay to simulate streaming
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        res.end();
    } catch (error: any) {
        res.write(JSON.stringify({ error: error.message }) + '\n');
        res.end();
    }
});

export default app;
