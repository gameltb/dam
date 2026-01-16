import { create } from "@bufbuild/protobuf";
import { useCallback, useMemo, useState } from "react";
import { useTable } from "spacetimedb/react";

import { ChatNodeStateSchema } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import { tables } from "@/generated/spacetime";
import { useSpacetimeChat } from "@/hooks/useSpacetimeChat";
import { useFlowStore } from "@/store/flowStore";
import { ChatStatus } from "@/types";

import { type ChatMessage } from "./types";

interface ChatState {
  lastRequest: null | {
    content: string;
    endpoint: string;
    model: string;
    search: boolean;
  };
}

export function useChatController(conversationHeadId: string | undefined, nodeId: string) {
  const { addMessage, messages: allMessages } = useSpacetimeChat(nodeId);
  const [stStreams] = useTable(tables.chatStreams);

  const [state, setState] = useState<ChatState>({
    lastRequest: null,
  });

  // Calculate active path from SpacetimeDB messages
  const messages = useMemo(() => {
    if (!conversationHeadId || allMessages.length === 0) return [];

    const path: ChatMessage[] = [];
    let currentId: string | undefined = conversationHeadId;
    const msgMap = new Map(allMessages.map((m) => [m.id, m]));

    while (currentId) {
      const msg = msgMap.get(currentId);
      if (!msg) break;
      path.unshift(msg);
      currentId = msg.parentId;
    }
    return path;
  }, [allMessages, conversationHeadId]);

  // Derive streaming message from Spacetime table
  const streamEntry = stStreams.find((s) => s.nodeId === nodeId);

  const streamingMessage = useMemo(() => {
    if (!streamEntry || streamEntry.status === "idle") return null;

    return {
      createdAt: 0, // Placeholder doesn't need real timestamp
      id: "streaming-placeholder",
      parts: [
        {
          part: {
            case: "text",
            value:
              streamEntry.status === "thinking" && !streamEntry.content
                ? "..." // Show indicator while thinking
                : streamEntry.content,
          },
        },
      ],
      role: "assistant",
    } as ChatMessage;
  }, [streamEntry]);

  // Derive consolidated status
  const status = useMemo<ChatStatus>(() => {
    if (stStreams.some((s) => s.nodeId === nodeId && s.status === "streaming")) {
      return ChatStatus.STREAMING;
    }
    if (stStreams.some((s) => s.nodeId === nodeId && s.status === "thinking")) {
      return ChatStatus.SUBMITTED;
    }
    return ChatStatus.READY;
  }, [stStreams, nodeId]);

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
      // Move head to the user message immediately
      useFlowStore.getState().updateNodeData(nodeId, {
        extension: {
          case: "chat",
          value: create(ChatNodeStateSchema, {
            conversationHeadId: msg.id,
            isHistoryCleared: false,
            treeId: msg.treeId ?? "",
          }),
        },
      });
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
