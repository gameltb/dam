import { EventEmitter } from "events";

import { type AppNode, type Edge } from "@/types";
import { db } from "./Database";
import { generateGallery } from "./Generators";

export let serverVersion = 0;

export const incrementVersion = () => {
  serverVersion++;
  syncToDB();
};

export const serverGraph: {
  edges: Edge[];
  nodes: AppNode[];
} = {
  edges: [],
  nodes: [],
};

export const eventBus = new EventEmitter();

interface EdgeRow {
  data: string;
  id: string;
  source: string;
  source_handle: null | string;
  target: string;
  target_handle: null | string;
}

interface NodeRow {
  data: string;
  height: null | number;
  id: string;
  parent_id: null | string;
  position_x: number;
  position_y: number;
  type: string;
  width: null | number;
}

export function getMutations(fromSeq: number, toSeq?: number) {
  let query = "SELECT * FROM mutations WHERE seq >= ?";
  const params: unknown[] = [fromSeq];
  if (toSeq) {
    query += " AND seq <= ?";
    params.push(toSeq);
  }
  query += " ORDER BY seq ASC";
  return db.prepare(query).all(...params) as {
    description: null | string;
    payload: Buffer;
    seq: number;
    source: number;
    timestamp: number;
    type: string;
    user_id: null | string;
  }[];
}

export function loadFromDisk() {
  try {
    // Load Version
    const versionRow = db
      .prepare("SELECT value FROM metadata WHERE key = ?")
      .get("version") as undefined | { value: string };
    serverVersion = versionRow ? parseInt(versionRow.value, 10) : 0;

    // Load Nodes
    const nodesRows = db.prepare("SELECT * FROM nodes").all() as NodeRow[];
    serverGraph.nodes = nodesRows.map((row) => ({
      data: JSON.parse(row.data) as AppNode["data"],
      id: row.id,
      measured:
        row.width !== null || row.height !== null
          ? { height: row.height ?? 0, width: row.width ?? 0 }
          : undefined,
      parentId: row.parent_id ?? undefined,
      position: { x: row.position_x, y: row.position_y },
      type: row.type,
    })) as AppNode[];

    // Load Edges
    const edgesRows = db.prepare("SELECT * FROM edges").all() as EdgeRow[];
    serverGraph.edges = edgesRows.map((row) => ({
      data: JSON.parse(row.data) as Edge["data"],
      id: row.id,
      source: row.source,
      sourceHandle: row.source_handle ?? undefined,
      target: row.target,
      targetHandle: row.target_handle ?? undefined,
    })) as Edge[];

    console.log(
      `[Persistence] Loaded ${serverGraph.nodes.length.toString()} nodes from database.`,
    );

    if (serverGraph.nodes.length === 0) {
      console.log(
        "[Persistence] Initializing empty graph with gallery showcase.",
      );
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

export function logMutation(
  type: string,
  payloadBinary: Uint8Array,
  source: number,
  description?: string,
) {
  const stmt = db.prepare(`
    INSERT INTO mutations (type, payload, timestamp, source, description)
    VALUES (?, ?, ?, ?, ?)
  `);
  const result = stmt.run(
    type,
    Buffer.from(payloadBinary),
    Date.now(),
    source,
    description ?? null,
  );
  return result.lastInsertRowid;
}

export function syncToDB() {
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
        node.type,
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

export {
  addChatMessage,
  clearChatHistory,
  duplicateBranch,
  getChatHistory,
  getMessage,
} from "./ChatService";
