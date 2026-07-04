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
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        res.status(400).json({ error: message });
    }
});

// Run model at given parameters
app.post('/model/run-at-time', (req, res) => {
    try {
        const params = CMAModelParamsSchema.parse(req.body);
        const model = new CMASleepWakeModel(params);
        model.solve();
        res.json({
            params: model.params,
            solution: model.solution
        });
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        res.status(400).json({ error: message });
    }
});

// Serve static assets and templates
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use('/static', express.static(path.join(__dirname, '../static')));

// Demo route
app.get('/demo', (req, res) => {
    // Basic static serving for now
    res.sendFile(path.join(__dirname, '../templates/demo.html'));
});

// Global error handler — must be registered after all routes
app.use((err: unknown, req: express.Request, res: express.Response, next: express.NextFunction) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

export default app;
