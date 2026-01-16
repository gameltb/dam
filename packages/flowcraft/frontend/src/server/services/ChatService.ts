import { getSpacetimeConn } from "../spacetimeClient";

export async function addChatMessage(_nodeId: string, _role: string, _content: string) {
  // Implementation via reducers
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
