import Database from "better-sqlite3";
import fs from "fs";

import { SERVER_CONFIG } from "../config/server";

if (!fs.existsSync(SERVER_CONFIG.storageDir)) {
  fs.mkdirSync(SERVER_CONFIG.storageDir, { recursive: true });
}

export const db = new Database(SERVER_CONFIG.dbFile);

// Initialize Schema
db.exec(`
  CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT,
    data TEXT,
    position_x REAL,
    position_y REAL,
    parent_id TEXT,
    width REAL,
    height REAL
  );

  CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source TEXT,
    target TEXT,
    source_handle TEXT,
    target_handle TEXT,
    data TEXT
  );

  CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
  );

  CREATE TABLE IF NOT EXISTS mutations (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    payload BLOB,
    timestamp INTEGER,
    source INTEGER,
    description TEXT,
    user_id TEXT
  );

  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    tree_id TEXT,
    role TEXT,
    parts TEXT,
    metadata TEXT,
    timestamp INTEGER,
    node_id TEXT,
    FOREIGN KEY(parent_id) REFERENCES conversations(id)
  );
`);

// Migration for existing tables
try {
  db.exec("ALTER TABLE conversations ADD COLUMN tree_id TEXT;");
} catch {
  // Column might already exist
}
try {
  db.exec("ALTER TABLE conversations RENAME COLUMN content TO parts;");
} catch {
  // Column might already be renamed
}
