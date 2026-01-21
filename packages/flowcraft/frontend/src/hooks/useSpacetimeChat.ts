import { create } from "@bufbuild/protobuf";
import { useCallback, useMemo } from "react";

import { type ChatSyncMessage, ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { useChatStore } from "@/store/chatStore";
import { useFlowStore } from "@/store/flowStore";

/**
 * Hook to manage chat messages from the centralized store
 */
export const useSpacetimeChat = (filterId?: string, _treeId?: string, headId?: string) => {
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);
  const allMessages = useChatStore((s) => s.messages);

  const messages = useMemo(() => {
    if (!filterId) return allMessages;

    return allMessages.filter((m) => {
      // With washed PB messages, we can rely on standard property names
      return m.treeId === filterId;
    });
  }, [allMessages, filterId]);

  const addMessage = useCallback(
    (message: Partial<ChatSyncMessage>) => {
      if (spacetimeConn) {
        const parentId = headId || "";

        spacetimeConn.pbreducers.addChatMessage({
          message: create(ChatSyncMessageSchema, {
            id: message.id,
            modelId: message.modelId,
            parentId: parentId,
            parts: message.parts,
            role: message.role,
            timestamp: message.timestamp,
          }),
          nodeId: filterId || "",
        });
      }
    },
    [spacetimeConn, filterId, headId],
  );

  return {
    addMessage,
    messages,
  };
};
