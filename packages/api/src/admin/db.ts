/**
 * Admin database layer using better-sqlite3.
 * Mirrors pfun_cma_model/admin/models.py and admin/core.py.
 */

import Database from "better-sqlite3";
import bcrypt from "bcrypt";
import path from "path";
import fs from "fs";
import type { AppConfig } from "../config.js";

export interface User {
  id: number;
  name: string;
  email: string;
  is_admin: boolean;
  site_id: number | null;
  age: number;
  bio: string | null;
  hashed_password: string;
}

export interface Site {
  id: number;
  name: string;
}

let db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (!db) {
    throw new Error("Admin database not initialized. Call initAdminDb() first.");
  }
  return db;
}

/**
 * Initialize admin SQLite database, create tables if they don't exist.
 * Mirrors init_models() / setup_admin_backend() in Python.
 */
export function initAdminDb(config: AppConfig): void {
  const resultsDir = path.resolve("results");
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
  }

  // Already initialized - idempotent
  if (db) return;

  const dbPath = config.debug
    ? path.resolve("results/admin-local.db")
    : path.resolve("results/admin.db");

  db = new Database(dbPath);

  // Enable WAL for better concurrency
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");

  db.exec(`
    CREATE TABLE IF NOT EXISTS sites (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      is_admin INTEGER NOT NULL DEFAULT 0,
      site_id INTEGER REFERENCES sites(id),
      age INTEGER NOT NULL DEFAULT 0,
      bio TEXT,
      hashed_password TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
  `);
}

export function closeAdminDb(): void {
  if (db) {
    db.close();
    db = null;
  }
}

// --- User CRUD ---

export function getUserByNameOrEmail(nameOrEmail: string): User | undefined {
  const d = getDb();
  const row = d
    .prepare(
      "SELECT * FROM users WHERE name = ? OR email = ? LIMIT 1",
    )
    .get(nameOrEmail, nameOrEmail) as User | undefined;
  if (row) row.is_admin = Boolean(row.is_admin);
  return row;
}

export function getUserById(id: number): User | undefined {
  const d = getDb();
  const row = d.prepare("SELECT * FROM users WHERE id = ?").get(id) as User | undefined;
  if (row) row.is_admin = Boolean(row.is_admin);
  return row;
}

export function listUsers(limit = 100, offset = 0): User[] {
  const d = getDb();
  return (d.prepare("SELECT * FROM users LIMIT ? OFFSET ?").all(limit, offset) as User[]).map(
    (u) => ({ ...u, is_admin: Boolean(u.is_admin) }),
  );
}

export function createUser(
  name: string,
  email: string,
  password: string,
  age: number,
  bio?: string,
  siteId?: number,
  isAdmin = false,
): User {
  const d = getDb();
  const hashedPassword = bcrypt.hashSync(password, 12);
  const stmt = d.prepare(
    "INSERT INTO users (name, email, is_admin, site_id, age, bio, hashed_password) VALUES (?, ?, ?, ?, ?, ?, ?)",
  );
  const result = stmt.run(
    name,
    email,
    isAdmin ? 1 : 0,
    siteId ?? null,
    age,
    bio ?? null,
    hashedPassword,
  );
  return getUserById(result.lastInsertRowid as number)!;
}

export function updateUser(
  id: number,
  fields: Partial<Omit<User, "id" | "hashed_password">>,
): User | undefined {
  const d = getDb();
  const allowed = ["name", "email", "is_admin", "site_id", "age", "bio"] as const;
  const updates: string[] = [];
  const values: unknown[] = [];
  for (const key of allowed) {
    if (key in fields) {
      updates.push(`${key} = ?`);
      values.push(key === "is_admin" ? (fields[key] ? 1 : 0) : fields[key]);
    }
  }
  if (updates.length === 0) return getUserById(id);
  values.push(id);
  d.prepare(`UPDATE users SET ${updates.join(", ")} WHERE id = ?`).run(...values);
  return getUserById(id);
}

export function deleteUser(id: number): boolean {
  const d = getDb();
  const result = d.prepare("DELETE FROM users WHERE id = ?").run(id);
  return result.changes > 0;
}

export function verifyPassword(plainPassword: string, hashedPassword: string): boolean {
  return bcrypt.compareSync(plainPassword, hashedPassword);
}

export function hashPassword(password: string): string {
  return bcrypt.hashSync(password, 12);
}

// --- Site CRUD ---

export function listSites(limit = 100, offset = 0): Site[] {
  const d = getDb();
  return d.prepare("SELECT * FROM sites LIMIT ? OFFSET ?").all(limit, offset) as Site[];
}

export function getSiteById(id: number): Site | undefined {
  const d = getDb();
  return d.prepare("SELECT * FROM sites WHERE id = ?").get(id) as Site | undefined;
}

export function createSite(name: string): Site {
  const d = getDb();
  const result = d.prepare("INSERT INTO sites (name) VALUES (?)").run(name);
  return getSiteById(result.lastInsertRowid as number)!;
}

export function deleteSite(id: number): boolean {
  const d = getDb();
  const result = d.prepare("DELETE FROM sites WHERE id = ?").run(id);
  return result.changes > 0;
}
