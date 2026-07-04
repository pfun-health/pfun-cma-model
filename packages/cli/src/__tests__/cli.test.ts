import { describe, it, expect } from 'vitest';

describe('CLI Package', () => {
  it('should export the CLI entry point as an ES module', async () => {
    // The CLI is a script that parses process.argv; we just verify it can be imported
    const cli = await import('../index.js');
    expect(cli).toBeDefined();
  });
});
