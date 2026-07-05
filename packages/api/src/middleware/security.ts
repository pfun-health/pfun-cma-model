/**
 * Security middleware: rate limiting, headers, penetration detection, IP banning.
 */

import type { Context, Next } from "hono";

export interface SecurityConfig {
  rateLimit: number;
  rateLimitWindow: number; // seconds
  autoBanThreshold: number;
  autoBanDuration: number; // seconds
  blockedUserAgents: string[];
  enablePenetrationDetection: boolean;
}

const DEFAULT_SECURITY_CONFIG: SecurityConfig = {
  rateLimit: 50,
  rateLimitWindow: 60,
  autoBanThreshold: 5,
  autoBanDuration: 3600,
  blockedUserAgents: ["badbot", "evil-crawler", "sqlmap"],
  enablePenetrationDetection: true,
};

// In-memory rate limiter (Redis-backed version available when Redis connected)
const requestCounts = new Map<string, { count: number; resetAt: number }>();
const bannedIps = new Map<string, number>(); // ip -> unban timestamp

/**
 * Get client IP from request.
 */
function getClientIp(c: Context): string {
  return (
    c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ??
    c.req.header("x-real-ip") ??
    "unknown"
  );
}

/**
 * Security headers middleware.
 */
export function securityHeaders() {
  return async (c: Context, next: Next) => {
    await next();

    c.res.headers.set(
      "Content-Security-Policy",
      "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'",
    );
    c.res.headers.set(
      "Strict-Transport-Security",
      "max-age=63072000; includeSubDomains; preload",
    );
    c.res.headers.set("X-Content-Type-Options", "nosniff");
    c.res.headers.set("X-Frame-Options", "DENY");
    c.res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
    c.res.headers.set(
      "Permissions-Policy",
      "camera=(), microphone=(), geolocation=(), interest-cohort=()",
    );
    c.res.headers.set("X-XSS-Protection", "1; mode=block");
  };
}

/**
 * Rate limiting and IP ban middleware.
 */
export function rateLimiter(config: SecurityConfig = DEFAULT_SECURITY_CONFIG) {
  return async (c: Context, next: Next) => {
    const ip = getClientIp(c);
    const now = Date.now();

    // Check if IP is banned
    const banExpiry = bannedIps.get(ip);
    if (banExpiry && now < banExpiry) {
      return c.json({ detail: "IP banned" }, 403);
    } else if (banExpiry) {
      bannedIps.delete(ip);
    }

    // Check rate limit
    const entry = requestCounts.get(ip);
    if (entry && now < entry.resetAt) {
      entry.count++;
      if (entry.count > config.rateLimit) {
        // Check if should ban
        if (entry.count > config.rateLimit * config.autoBanThreshold) {
          bannedIps.set(ip, now + config.autoBanDuration * 1000);
        }
        return c.json({ detail: "Rate limit exceeded" }, 429);
      }
    } else {
      requestCounts.set(ip, {
        count: 1,
        resetAt: now + config.rateLimitWindow * 1000,
      });
    }

    await next();
  };
}

/**
 * User-agent blocking middleware.
 */
export function userAgentFilter(
  blockedAgents: string[] = DEFAULT_SECURITY_CONFIG.blockedUserAgents,
) {
  return async (c: Context, next: Next) => {
    const ua = c.req.header("user-agent")?.toLowerCase() ?? "";
    for (const blocked of blockedAgents) {
      if (ua.includes(blocked.toLowerCase())) {
        return c.json({ detail: "Blocked user agent" }, 403);
      }
    }
    await next();
  };
}

/**
 * Debug query rejection middleware.
 */
export function debugQueryRejection() {
  return async (c: Context, next: Next) => {
    const debugParam = c.req.query("debug");
    if (debugParam === "true") {
      return c.json({ detail: "Debug mode not allowed" }, 403);
    }
    await next();
  };
}

/**
 * Request tracking middleware (stores request metadata).
 */
export function requestTracker(redisClient: unknown | null) {
  return async (c: Context, next: Next) => {
    const ip = getClientIp(c);
    const timestamp = new Date().toISOString();
    const metadata = {
      ip,
      path: c.req.path,
      method: c.req.method,
      query: c.req.query(),
      headers: Object.fromEntries(c.req.raw.headers.entries()),
      timestamp,
    };

    // Store in Redis if available (non-blocking)
    if (redisClient && typeof (redisClient as Record<string, unknown>).setex === "function") {
      const key = `client_request:${ip}:${timestamp}`;
      try {
        await (redisClient as { setex: (k: string, ttl: number, v: string) => Promise<void> }).setex(
          key,
          3600,
          JSON.stringify(metadata),
        );
      } catch {
        // Redis failure should not block request
      }
    }

    await next();
  };
}
