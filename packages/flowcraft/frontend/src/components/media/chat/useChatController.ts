import { useCallback, useMemo, useState } from "react";

import { useSpacetimeChat } from "@/hooks/useSpacetimeChat";
import { useChatStore } from "@/store/chatStore";
import { useFlowStore } from "@/store/flowStore";
import { ChatStatus } from "@/types";

import { type ChatMessage, ChatRole } from "./types";

interface ChatState {
  lastRequest: null | {
    content: string;
    endpoint: string;
    modelId: string; // Corrected field name
    search: boolean;
  };
}

export function useChatController(conversationHeadId: string | undefined, nodeId: string, treeId?: string) {
  const { addMessage, messages: allMessages } = useSpacetimeChat(treeId || nodeId, undefined, conversationHeadId);
  const streamEntry = useChatStore((s) => s.streams[nodeId]);

  const [state, setState] = useState<ChatState>({
    lastRequest: null,
  });

  // Calculate active path from PB-converted messages
  const messages = useMemo(() => {
    let effectiveHeadId = conversationHeadId;

    if (!effectiveHeadId && allMessages.length > 0) {
      // Sort by timestamp DESC to get the latest message
      const sorted = [...allMessages].sort((a, b) => {
        const tA = BigInt(a.timestamp || 0n);
        const tB = BigInt(b.timestamp || 0n);
        return tA > tB ? -1 : tA < tB ? 1 : 0;
      });
      effectiveHeadId = sorted[0]?.id;
    }

    if (!effectiveHeadId || allMessages.length === 0) {
      return [];
    }

    const path: ChatMessage[] = [];
    let currentId: string | undefined = effectiveHeadId;
    const visited = new Set<string>();

    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const targetId: string = currentId;
      const msg = allMessages.find((m) => m.id === targetId) as ChatMessage | undefined;

      if (!msg) break;
      path.unshift(msg);
      currentId = msg.parentId;
    }

    return path;
  }, [allMessages, conversationHeadId]);

  const streamingMessage = useMemo(() => {
    if (!streamEntry || streamEntry.status === "idle") return null;

    return {
      createdAt: 0,
      id: "streaming-placeholder",
      parts: [
        {
          part: {
            case: "text",
            value: streamEntry.status === "thinking" && !streamEntry.content ? "..." : streamEntry.content,
          },
        },
      ],
      role: ChatRole.ASSISTANT,
    } as ChatMessage;
  }, [streamEntry]);

  // Derive consolidated status
  const status = useMemo<ChatStatus>(() => {
    if (streamEntry?.status === "streaming") {
      return ChatStatus.STREAMING;
    }
    if (streamEntry?.status === "thinking") {
      return ChatStatus.SUBMITTED;
    }
    if (streamEntry?.status === "error") {
      return ChatStatus.ERROR;
    }
    return ChatStatus.READY;
  }, [streamEntry]);

  const setLastRequest = useCallback(
    (request: ChatState["lastRequest"]) => {
      setState((prev) => ({ ...prev, lastRequest: request }));
    },

    [setState],
  );

  /**

       * Optimistically appends a user message.

       */

  const appendUserMessage = useCallback(
    (msg: ChatMessage) => {
      addMessage(msg);

      const node = useFlowStore.getState().allNodes.find((n) => n.id === nodeId);

      if (!node) return;

      const res = useFlowStore.getState().nodeDraft(node);

      if (res.ok) {
        const draft = res.value;

        if (draft.data.extension?.case === "chat") {
          draft.data.extension.value.conversationHeadId = msg.id;

          draft.data.extension.value.isHistoryCleared = false;
        }
      }
    },

    [addMessage, nodeId],
  );

  return {
    appendUserMessage,
    clearAll: () => {
      /* Implemented via SpacetimeDB signals */
    },
    handleStreamChunk: (_chunk: string) => {
      /* Handled via SpacetimeDB chat_streams table */
    },
    isLoading: status === ChatStatus.STREAMING || status === ChatStatus.SUBMITTED,
    lastRequest: state.lastRequest,
    messages,
    setLastRequest,
    sliceHistory: (_index: number) => {
      /* Local slice for optimistic UI if needed */
    },
    status,
    streamingMessage,
  };
}
