import { describe, it, expect } from 'vitest';
import app from '../app.js';

describe('API Application', () => {
  it('should export the express app', () => {
    expect(app).toBeDefined();
    expect(typeof app.listen).toBe('function');
    expect(typeof app.get).toBe('function');
    expect(typeof app.post).toBe('function');
    expect(typeof app.use).toBe('function');
  });
});
