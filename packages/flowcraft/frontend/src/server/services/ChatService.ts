import { create } from "@bufbuild/protobuf";

import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { wrapReducers } from "@/utils/pb-client";

import { getSpacetimeConn } from "../spacetimeClient";

export async function addChatMessage(nodeId: string, role: string, content: string, parentId?: string) {
  const conn = getSpacetimeConn();
  if (!conn) return;

  const pbConn = wrapReducers(conn as any);
  const msgId = crypto.randomUUID();

  pbConn.pbreducers.addChatMessage({
    message: create(ChatSyncMessageSchema, {
      id: msgId,
      parentId: parentId || "",
      parts: [{ part: { case: "text", value: content } }],
      role: role as any,
      timestamp: BigInt(Date.now()),
    }),
    nodeId: nodeId,
  });

  return msgId;
}

export function branchAndEditMessage() {}

export async function clearChatHistory() {}

export function duplicateBranch() {}

export async function getChatHistory(treeId: string): Promise<any[]> {
  const conn = getSpacetimeConn();
  if (!conn) return [];

  const allMessages = Array.from(conn.db.chatMessages.iter());
  return allMessages
    .filter((m: any) => m.state?.treeId === treeId)
    .map((m: any) => m.state)
    .sort((a: any, b: any) => Number(a.timestamp - b.timestamp));
}

export function getMessage() {}
