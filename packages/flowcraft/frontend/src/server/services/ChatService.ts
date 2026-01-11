import { fromJson, toJson } from "@bufbuild/protobuf";
import crypto from "crypto";

import {
  type ChatMessagePart,
  ChatMessagePartSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";

import { getSpacetimeConn } from "../spacetimeClient";

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

export function addChatMessage(params: {
  contentId?: string; // Reuse existing content if provided
  id: string;
  metadata?: unknown;
  nodeId?: string;
  parentId: null | string;
  parts: unknown[];
  role: string;
  treeId: string;
}) {
  // Sync to SpacetimeDB
  try {
    const conn = getSpacetimeConn();
    if (conn) {
      const partsJson = JSON.stringify(
        (params.parts as ChatMessagePart[]).map((p) =>
          toJson(ChatMessagePartSchema, p),
        ),
      );

      // Generate a contentId if none provided (representing this specific unique content)
      const contentId = params.contentId || crypto.randomUUID();

      conn.reducers.addChatMessage({
        contentId,
        id: params.id,
        modelId: (params.metadata as any)?.modelId ?? "",
        nodeId: params.nodeId ?? "",
        parentId: params.parentId ?? "",
        partsJson: partsJson,
        role: params.role,
        timestamp: BigInt(Date.now()),
        treeId: params.treeId,
      });
    } else {
      console.warn(
        "[ChatService] No SpacetimeDB connection for addChatMessage",
      );
    }
  } catch (e) {
    console.error("[ChatService] Failed to sync to SpacetimeDB:", e);
  }
}

export function branchAndEditMessage(params: {
  messageId: string;
  newParts: ChatMessagePart[];
  nodeId?: string;
  treeId: string;
}): string {
  const conn = getSpacetimeConn();
  if (!conn) throw new Error("No SpacetimeDB connection");

  const original = getMessage(params.messageId);
  if (!original) throw new Error("Message not found");

  // 1. Create the new edited message as a sibling (shares same parent)
  const newId = crypto.randomUUID();
  addChatMessage({
    id: newId,
    metadata: original.metadata,
    nodeId: params.nodeId,
    parentId: original.parentId,
    parts: params.newParts,
    role: original.role,
    treeId: params.treeId,
  });

  // 2. COW: Clone all descendants from original to new branch
  const copyDescendants = (oldParentId: string, newParentId: string) => {
    // Collect immediate children from SpacetimeDB
    const children: ChatMessage[] = [];
    for (const msg of conn.db.chatMessages) {
      if (msg.parentId === oldParentId) {
        children.push(mapSpacetimeMessage(msg));
      }
    }
    children.sort((a, b) => a.timestamp - b.timestamp);

    for (const child of children) {
      const childNewId = crypto.randomUUID();
      const stMsg = conn.db.chatMessages.id.find(child.id);

      addChatMessage({
        contentId: stMsg?.contentId, // STRUCTURAL SHARING: Reuse content!
        id: childNewId,
        metadata: child.metadata,
        nodeId: params.nodeId,
        parentId: newParentId,
        parts: child.parts,
        role: child.role,
        treeId: params.treeId,
      });
      copyDescendants(child.id, childNewId);
    }
  };

  copyDescendants(params.messageId, newId);
  return newId;
}

export function clearChatHistory(nodeId: string) {
  // CoW: We no longer perform physical deletion.
  // The frontend handles this by resetting the conversationHeadId.
  console.log(
    `[ChatService] clearChatHistory ignored for nodeId: ${nodeId} to preserve CoW`,
  );
}

export function duplicateBranch(params: {
  newParentId: null | string;
  newTreeId: string;
  nodeId?: string;
  startMessageId: string;
}): string {
  const conn = getSpacetimeConn();
  if (!conn) throw new Error("No SpacetimeDB connection");

  const original = getMessage(params.startMessageId);
  if (!original) return params.startMessageId;

  const newId = crypto.randomUUID();
  const stOrig = conn.db.chatMessages.id.find(params.startMessageId);

  addChatMessage({
    contentId: stOrig?.contentId, // STRUCTURAL SHARING
    id: newId,
    metadata: original.metadata,
    nodeId: params.nodeId,
    parentId: params.newParentId,
    parts: original.parts,
    role: original.role,
    treeId: params.newTreeId,
  });

  const copyDescendants = (oldParentId: string, newParentId: string) => {
    const children: ChatMessage[] = [];
    for (const msg of conn.db.chatMessages) {
      if (msg.parentId === oldParentId) {
        children.push(mapSpacetimeMessage(msg));
      }
    }
    children.sort((a, b) => a.timestamp - b.timestamp);

    for (const child of children) {
      const childNewId = crypto.randomUUID();
      const stMsg = conn.db.chatMessages.id.find(child.id);

      addChatMessage({
        contentId: stMsg?.contentId, // STRUCTURAL SHARING
        id: childNewId,
        metadata: child.metadata,
        nodeId: params.nodeId,
        parentId: newParentId,
        parts: child.parts,
        role: child.role,
        treeId: params.newTreeId,
      });
      copyDescendants(child.id, childNewId);
    }
  };

  copyDescendants(params.startMessageId, newId);
  return newId;
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
  const conn = getSpacetimeConn();
  if (!conn) return undefined;

  const stMsg = conn.db.chatMessages.id.find(id);
  if (!stMsg) return undefined;

  const msg = mapSpacetimeMessage(stMsg);

  // Compute siblings
  const siblings: string[] = [];
  for (const m of conn.db.chatMessages) {
    if (
      m.id !== msg.id &&
      m.parentId === msg.parentId &&
      m.treeId === msg.treeId
    ) {
      siblings.push(m.id);
    }
  }

  return {
    ...msg,
    siblingIds: siblings,
  };
}

// Helper to map SpacetimeDB row to ChatMessage
function mapSpacetimeMessage(msg: any): ChatMessage {
  const conn = getSpacetimeConn();
  let parts: ChatMessagePart[] = [];

  // Structural Sharing: Join with chat_contents
  const content = conn?.db.chatContents.id.find(msg.contentId);
  const targetPartsJson = content ? content.partsJson : msg.partsJson;
  const targetRole = content ? content.role : msg.role;

  try {
    const rawParts = JSON.parse(targetPartsJson || "[]");
    if (Array.isArray(rawParts)) {
      parts = rawParts.map((p: any) => fromJson(ChatMessagePartSchema, p));
    }
  } catch (e) {
    console.error("Failed to parse chat message parts", e);
  }

  return {
    id: msg.id,
    metadata: { modelId: msg.modelId },
    parentId: msg.parentId || null,
    parts,
    role: targetRole,
    siblingIds: [], // Computed separately if needed
    timestamp: Number(msg.timestamp),
    treeId: msg.treeId,
  };
}
