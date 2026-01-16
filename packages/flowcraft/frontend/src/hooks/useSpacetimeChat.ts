import { create } from "@bufbuild/protobuf";
import { useCallback, useEffect, useMemo } from "react";
import { useSpacetimeDB, useTable } from "spacetimedb/react";

import { type ChatMessage } from "@/components/media/chat/types";
import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { wrapReducers } from "@/utils/pb-client";
import { type DbConnection, tables } from "@/generated/spacetime";

export const useSpacetimeChat = (nodeId: string, treeId?: string, headId?: string) => {
  const stdb = useSpacetimeDB();
  const getConnection = useCallback(() => stdb.getConnection<DbConnection>(), [stdb]);

  const [stMessages] = useTable(tables.chatMessages);

  useEffect(() => {
    const conn = getConnection();
    if (conn) {
      void conn.subscriptionBuilder().subscribe(["SELECT * FROM chat_messages"]);
    }
  }, [getConnection]);

  const { messages } = useMemo(() => {
    const filtered = stMessages

      .filter((msg: any) => (treeId ? msg.state.treeId === treeId : msg.state.treeId === nodeId))

      .map((msg: any) => {
        const m = msg.state;

        const modelId = m.metadata?.tag === "chatMetadata" ? m.metadata.value.modelId : "";

        return {
          createdAt: Number(m.timestamp),

          id: m.id,

          metadata: { modelId },

          parentId: m.parentId === "" ? undefined : m.parentId,

          parts: m.parts || [],

          role: m.role as "assistant" | "system" | "user",

          siblingIds: m.siblingIds || [],

          treeId: m.treeId,
        } as ChatMessage;
      });

    const branch: ChatMessage[] = [];
    let currentId = headId;
    const msgMap = new Map(filtered.map((m) => [m.id, m]));

    while (currentId) {
      const msg = msgMap.get(currentId);
      if (!msg) break;
      branch.unshift(msg);
      currentId = msg.parentId;
    }

    return {
      allMessagesInTree: filtered,
      messages: branch.length > 0 ? branch : filtered.sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0)),
    };
  }, [stMessages, nodeId, treeId, headId]);

  const addMessage = (msg: ChatMessage) => {
    const conn = getConnection();
    if (conn) {
      const client = wrapReducers(conn);
      client.pbreducers.addChatMessage({
        message: create(ChatSyncMessageSchema, {
          id: msg.id,
          modelId: (msg.metadata?.modelId as string) || "",
          parts: msg.parts || [],
          role: msg.role,
          timestamp: BigInt(msg.createdAt ?? Date.now()),
        }),
        nodeId: nodeId,
      });
    }
  };

  const clearHistory = () => {
    const conn = getConnection();
    if (conn) {
      const client = wrapReducers(conn);
      client.reducers.clearChatHistory({ nodeId });
    }
  };

  return { addMessage, clearHistory, messages };
};
