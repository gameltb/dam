import crypto from "crypto";

import { type ChatMessagePart } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { db } from "./Database";

export interface ChatMessage {
  id: string;
  metadata: Record<string, unknown>;
  parentId: null | string;
  parts: ChatMessagePart[];
  role: string;
  siblingIds: string[];
  timestamp: number;
  treeId: string;
}

interface ConversationRow {
  id: string;
  metadata: string;
  parent_id: null | string;
  parts: string;
  role: string;
  timestamp: number;
  tree_id: string;
}

export function addChatMessage(params: {
  id: string;
  metadata?: unknown;
  nodeId?: string;
  parentId: null | string;
  parts: unknown[];
  role: string;
  treeId: string;
}) {
  const stmt = db.prepare(`
    INSERT INTO conversations (id, parent_id, tree_id, role, parts, metadata, timestamp, node_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    params.id,
    params.parentId,
    params.treeId,
    params.role,
    JSON.stringify(params.parts),
    JSON.stringify(params.metadata ?? {}),
    Date.now(),
    params.nodeId ?? null,
  );
}

export function clearChatHistory(nodeId: string) {
  const stmt = db.prepare("DELETE FROM conversations WHERE node_id = ?");
  stmt.run(nodeId);
}

export function duplicateBranch(params: {
  newParentId: null | string;
  newTreeId: string;
  nodeId?: string;
  startMessageId: string;
}): string {
  let currentSourceId: null | string = params.startMessageId;
  let lastNewId: null | string = params.newParentId;

  while (currentSourceId) {
    const original = getMessage(currentSourceId);
    if (!original) break;

    const newId = crypto.randomUUID();
    addChatMessage({
      id: newId,
      metadata: original.metadata,
      nodeId: params.nodeId,
      parentId: lastNewId,
      parts: original.parts,
      role: original.role,
      treeId: params.newTreeId,
    });

    lastNewId = newId;

    const nextChild = db
      .prepare("SELECT id FROM conversations WHERE parent_id = ? LIMIT 1")
      .get(currentSourceId) as undefined | { id: string };
    currentSourceId = nextChild?.id ?? null;
  }

  return lastNewId ?? params.startMessageId; // Fallback to start if nothing cloned
}

export function getChatHistory(headId: string): ChatMessage[] {
  const history: ChatMessage[] = [];
  let currentId: null | string = headId;

  while (currentId) {
    const msg = getMessage(currentId);
    if (!msg) break;
    history.unshift(msg);
    currentId = msg.parentId;
  }
  return history;
}

export function getMessage(id: string): ChatMessage | undefined {
  const row = db.prepare("SELECT * FROM conversations WHERE id = ?").get(id) as
    | ConversationRow
    | undefined;
  if (!row) return undefined;

  let siblings: { id: string }[];
  if (row.parent_id === null) {
    siblings = db
      .prepare(
        "SELECT id FROM conversations WHERE parent_id IS NULL AND tree_id = ? AND id != ?",
      )
      .all(row.tree_id, row.id) as { id: string }[];
  } else {
    siblings = db
      .prepare("SELECT id FROM conversations WHERE parent_id = ? AND id != ?")
      .all(row.parent_id, row.id) as { id: string }[];
  }

  return {
    id: row.id,
    metadata: JSON.parse(row.metadata) as Record<string, unknown>,
    parentId: row.parent_id,
    parts: JSON.parse(row.parts) as ChatMessagePart[],
    role: row.role,
    siblingIds: siblings.map((s) => s.id),
    timestamp: row.timestamp,
    treeId: row.tree_id,
  };
}
