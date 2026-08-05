/**
 * Application instance for testing and external consumption.
 * Re-exports the Hono app created by createApp().
 */
import { createApp } from './index.js';

const { app } = createApp();

export default app;
