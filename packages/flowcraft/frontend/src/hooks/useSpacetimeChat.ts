import { fromJson, toJson } from "@bufbuild/protobuf";
import { useEffect, useMemo } from "react";
import { useSpacetimeDB, useTable } from "spacetimedb/react";

import { type ChatMessage } from "@/components/media/chat/types";
import {
  type ChatMessagePart,
  ChatMessagePartSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { DbConnection, tables } from "@/generated/spacetime";

export const useSpacetimeChat = (
  nodeId: string,
  treeId?: string,
  headId?: string,
) => {
  const { getConnection } = useSpacetimeDB();
  const [stMessages] = useTable(tables.chatMessages);
  const [stContents] = useTable(tables.chatContents);

  useEffect(() => {
    const conn = getConnection<DbConnection>();
    if (conn) {
      conn
        .subscriptionBuilder()
        .subscribe(["SELECT * FROM chat_messages", "SELECT * FROM chat_contents"]);
    }
  }, [getConnection]);

  const { messages } = useMemo(() => {
    // 0. Create content lookup map
    const contentMap = new Map(stContents.map((c: any) => [c.id, c]));

    const filtered = stMessages
      .filter((msg: any) =>
        treeId ? msg.treeId === treeId : msg.nodeId === nodeId,
      )
      .map((msg: any) => {
        const content = contentMap.get(msg.contentId);
        let parts: ChatMessagePart[] = [];

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
          createdAt: Number(msg.timestamp),
          id: msg.id,
          modelId: msg.modelId,
          parentId: msg.parentId || undefined,
          parts,
          role: targetRole as "assistant" | "system" | "user",
          siblingIds: [],
          treeId: msg.treeId,
        } as ChatMessage;
      });

    // 1. Build parent mapping for sibling calculation
    const parentMap = new Map<string, string[]>();
    filtered.forEach((m) => {
      const pid = m.parentId || "root";
      if (!parentMap.has(pid)) parentMap.set(pid, []);
      parentMap.get(pid)?.push(m.id);
    });

    const enriched = filtered.map((m) => ({
      ...m,
      siblingIds:
        parentMap.get(m.parentId || "root")?.filter((id) => id !== m.id) || [],
    }));

    // 2. Trace path from headId upwards to build the current branch
    const branch: ChatMessage[] = [];
    let currentId = headId;

    // Sort all by time initially just to be safe
    const msgMap = new Map(enriched.map((m) => [m.id, m]));

    while (currentId) {
      const msg = msgMap.get(currentId);
      if (!msg) break;
      branch.unshift(msg);
      currentId = msg.parentId;
    }

    return {
      allMessagesInTree: enriched,
      messages:
        branch.length > 0
          ? branch
          : enriched.sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0)),
    };
  }, [stMessages, nodeId, treeId, headId]);

  const addMessage = (msg: ChatMessage) => {
    const conn = getConnection<DbConnection>();
    if (conn) {
      const partsJson = JSON.stringify(
        (msg.parts || []).map((p) => toJson(ChatMessagePartSchema, p)),
      );

      const contentId = (msg as any).contentId || crypto.randomUUID();

      conn.reducers.addChatMessage({
        contentId,
        id: msg.id,
        modelId: (msg as any).modelId || "",
        nodeId: nodeId,
        parentId: msg.parentId || "",
        partsJson,
        role: msg.role,
        timestamp: BigInt(msg.createdAt || Date.now()),
        treeId: msg.treeId || treeId || "",
      });
    }
  };

  const clearHistory = () => {
    const conn = getConnection<DbConnection>();
    if (conn) {
      conn.reducers.clearChatHistory({ nodeId });
    }
  };

  return { addMessage, clearHistory, messages };
};
