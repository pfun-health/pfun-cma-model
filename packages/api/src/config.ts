/**
 * Application configuration from environment variables.
 */

export interface AppConfig {
  port: number;
  host: string;
  debug: boolean;
  redisUrl: string | null;
  redisHost: string;
  redisPort: number;
  redisDb: number;
  redisPassword: string | null;
  jwtSecretKey: string;
  jwtExpirationMinutes: number;
  sessionSecret: string;
  dexcomClientId: string;
  dexcomClientSecret: string;
  dexcomRedirectUri: string;
  googleClientId: string;
  googleClientSecret: string;
  corsOrigins: string[];
  trustedHosts: string[];
  staticDir: string;
  templateDir: string;
  version: string;
}

export function loadConfig(): AppConfig {
  const debug = process.env.DEBUG === "true" || process.env.DEBUG === "1";

  return {
    port: parseInt(process.env.PORT ?? "8000", 10),
    host: process.env.HOST ?? "0.0.0.0",
    debug,
    redisUrl: process.env.REDIS_URL ?? null,
    redisHost: process.env.REDIS_HOST ?? "localhost",
    redisPort: parseInt(process.env.REDIS_PORT ?? "6379", 10),
    redisDb: parseInt(process.env.REDIS_DB ?? "0", 10),
    redisPassword: process.env.REDIS_PASSWORD ?? null,
    jwtSecretKey: process.env.JWT_SECRET_KEY ?? "insecure-default-secret",
    jwtExpirationMinutes: parseInt(process.env.JWT_EXPIRATION_MINUTES ?? "1440", 10),
    sessionSecret: process.env.SESSION_SECRET ?? "session-secret-key",
    dexcomClientId: process.env.DEXCOM_CLIENT_ID ?? "",
    dexcomClientSecret: process.env.DEXCOM_CLIENT_SECRET ?? "",
    dexcomRedirectUri: process.env.DEXCOM_REDIRECT_URI ?? "",
    googleClientId: process.env.GOOGLE_CLIENT_ID ?? "",
    googleClientSecret: process.env.GOOGLE_CLIENT_SECRET ?? "",
    corsOrigins: (process.env.CORS_ORIGINS ?? "*").split(","),
// "*" allows all hosts when debug=true; restrict via TRUSTED_HOSTS in production.
    trustedHosts: (process.env.TRUSTED_HOSTS ?? (debug ? "*" : "localhost,127.0.0.1")).split(","),
    staticDir: process.env.STATIC_DIR ?? "static",
    templateDir: process.env.TEMPLATE_DIR ?? "templates",
    version: process.env.npm_package_version ?? "1.0.0",
  };
}

/**
 * Generate versioned string like `<version>-dev.<timestamp>`.
 */
export function getVersionString(config: AppConfig): string {
  const now = new Date();
  const ts = now.toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `${config.version}-dev.${ts}`;
}
