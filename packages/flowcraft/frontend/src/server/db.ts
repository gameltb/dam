import { EventEmitter } from "events";
import { type AppNode, type Edge } from "../types";
import fs from "fs";
import path from "path";
import Database from "better-sqlite3";
import { generateGallery } from "./generators";

const STORAGE_DIR =
  process.env.FLOWCRAFT_STORAGE_DIR ?? path.join(process.cwd(), "storage");
if (!fs.existsSync(STORAGE_DIR)) {
  fs.mkdirSync(STORAGE_DIR, { recursive: true });
}

const DB_FILE = path.join(STORAGE_DIR, "flowcraft.db");
const db = new Database(DB_FILE);

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
`);

export let serverVersion = 0;

export const incrementVersion = () => {
  serverVersion++;
  syncToDB();
};

export const serverGraph: {
  nodes: AppNode[];
  edges: Edge[];
} = {
  nodes: [],
  edges: [],
};

export const eventBus = new EventEmitter();

function syncToDB() {
  const transaction = db.transaction(() => {
    // Sync Version
    db.prepare(
      "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
    ).run("version", serverVersion.toString());

    // Sync Nodes
    db.prepare("DELETE FROM nodes").run();
    const insertNode = db.prepare(`
      INSERT INTO nodes (id, type, data, position_x, position_y, parent_id, width, height)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const node of serverGraph.nodes) {
      insertNode.run(
        node.id,
        node.type ?? "dynamic",
        JSON.stringify(node.data),
        node.position.x,
        node.position.y,
        node.parentId ?? null,
        node.measured?.width ?? null,
        node.measured?.height ?? null,
      );
    }

    // Sync Edges
    db.prepare("DELETE FROM edges").run();
    const insertEdge = db.prepare(`
      INSERT INTO edges (id, source, target, source_handle, target_handle, data)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    for (const edge of serverGraph.edges) {
      insertEdge.run(
        edge.id,
        edge.source,
        edge.target,
        edge.sourceHandle ?? null,
        edge.targetHandle ?? null,
        JSON.stringify(edge.data ?? {}),
      );
    }
  });

  try {
    transaction();
  } catch (err) {
    console.error("Failed to sync to database:", err);
  }
}

interface NodeRow {
  id: string;
  type: string;
  data: string;
  position_x: number;
  position_y: number;
  parent_id: string | null;
  width: number | null;
  height: number | null;
}

interface EdgeRow {
  id: string;
  source: string;
  target: string;
  source_handle: string | null;
  target_handle: string | null;
  data: string;
}

export function loadFromDisk() {
  try {
    // Load Version
    const versionRow = db
      .prepare("SELECT value FROM metadata WHERE key = ?")
      .get("version") as { value: string } | undefined;
    serverVersion = versionRow ? parseInt(versionRow.value, 10) : 0;

    // Load Nodes
    const nodesRows = db.prepare("SELECT * FROM nodes").all() as NodeRow[];
    serverGraph.nodes = nodesRows.map((row) => ({
      id: row.id,
      type: row.type,
      data: JSON.parse(row.data),
      position: { x: row.position_x, y: row.position_y },
      parentId: row.parent_id ?? undefined,
      measured:
        row.width !== null || row.height !== null
          ? { width: row.width ?? 0, height: row.height ?? 0 }
          : undefined,
    })) as AppNode[];

    // Load Edges
    const edgesRows = db.prepare("SELECT * FROM edges").all() as EdgeRow[];
    serverGraph.edges = edgesRows.map((row) => ({
      id: row.id,
      source: row.source,
      target: row.target,
      sourceHandle: row.source_handle ?? undefined,
      targetHandle: row.target_handle ?? undefined,
      data: JSON.parse(row.data),
    })) as Edge[];

    console.log(`[DB] Loaded ${serverGraph.nodes.length} nodes from database.`);

    if (serverGraph.nodes.length === 0) {
      console.log("[DB] Initializing empty graph with gallery showcase.");
      const gallery = generateGallery();
      serverGraph.nodes = gallery.nodes;
      serverGraph.edges = gallery.edges;
      serverVersion = 1;
      syncToDB();
    }
  } catch (err) {
    console.error("Failed to load from database:", err);
  }
}
