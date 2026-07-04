import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import path from 'path';
import { fileURLToPath } from 'url';
import { CMASleepWakeModel, CMAModelParamsSchema } from 'core';

const app = express();

app.use(cors({
    origin: process.env.CORS_ORIGIN || 'http://localhost:8001',
    methods: ['GET', 'POST']
}));
app.use(helmet());
app.use(morgan('dev'));
app.use(express.json());

// Basic health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
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

// Streaming endpoint stub
app.post('/model/run-at-time/stream', (req, res) => {
    res.setHeader('Content-Type', 'application/x-ndjson');
    try {
        const params = CMAModelParamsSchema.parse(req.body);
        const model = new CMASleepWakeModel(params);
        model.solve();
        const output = {
            params: model.params,
            solution: model.solution
        };
        res.write(JSON.stringify(output) + '\n');
        res.end();
    } catch (error: any) {
        res.write(JSON.stringify({ error: error.message }) + '\n');
        res.end();
    }
});

// Global error handler — must be registered after all routes
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

export default app;

// Serve static assets and templates
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use('/static', express.static(path.join(__dirname, '../static')));

// Demo route
app.get('/demo', (req, res) => {
    // Basic static serving for now
    res.sendFile(path.join(__dirname, '../templates/demo.html'));
});
