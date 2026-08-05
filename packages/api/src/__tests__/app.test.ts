import { describe, it, expect } from 'vitest';
import app from '../app.js';

describe('API Application', () => {
  it('should export the Hono app', () => {
    expect(app).toBeDefined();
    expect(typeof app.fetch).toBe('function');
    expect(app.routes).toBeDefined();
  });
});
