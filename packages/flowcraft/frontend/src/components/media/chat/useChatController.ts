import { useCallback, useEffect, useRef, useState } from "react";

import { type ChatMessagePart } from "@/generated/flowcraft/v1/actions/chat_actions_pb";

import { socketClient } from "@/utils/SocketClient";
import { type ChatMessage } from "./types";

interface ChatState {
  isLoading: boolean;
  lastRequest: null | {
    content: string;
    endpoint: string;
    model: string;
    search: boolean;
  };
  messages: ChatMessage[];
  streamingMessage: ChatMessage | null;
}

export function useChatController(conversationHeadId: string | undefined) {
  const [state, setState] = useState<ChatState>({
    isLoading: false,
    lastRequest: null,
    messages: [],
    streamingMessage: null,
  });

  // Ref to access latest state in effects/callbacks without triggering re-renders
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const setLastRequest = useCallback(
    (request: ChatState["lastRequest"]) => {
      setState((prev) => ({ ...prev, lastRequest: request }));
    },
    [setState],
  );

  /**
   * Fetches history from the backend.
   * Smartly merges if possible, or replaces if complete divergence.
   */
  const syncHistory = useCallback(async (headId: string) => {
    // If we already have this head as the last message, we are good.
    const currentMsgs = stateRef.current.messages;
    const lastMsg = currentMsgs[currentMsgs.length - 1];

    if (lastMsg?.id === headId) {
      return;
    }

    setState((prev) => ({ ...prev, isLoading: true }));
    try {
      const res = await socketClient.getChatHistory(headId);
      const mapped: ChatMessage[] = res.entries.map((m) => {
        let metadata: Record<string, unknown> = {};
        if (m.metadata.case === "chatMetadata") {
          metadata = {
            attachments: m.metadata.value.attachmentUrls,
            modelId: m.metadata.value.modelId,
          };
        } else if (m.metadata.case === "metadataStruct") {
          metadata = m.metadata.value as Record<string, unknown>;
        }

        return {
          createdAt: Number(m.timestamp),
          id: m.id,
          metadata,
          parentId: m.parentId || undefined,
          parts: m.parts,
          role: (["assistant", "system", "user"].includes(m.role)
            ? m.role
            : "user") as "assistant" | "system" | "user",
          siblingIds: m.siblingIds,
          treeId: m.treeId,
        };
      });

      setState((prev) => ({
        ...prev,
        isLoading: false,
        messages: mapped,
        streamingMessage: null, // Clear streaming artifact once synced
      }));
      console.log(
        `[useChatController] Synced ${String(mapped.length)} messages. Head: ${String(mapped[mapped.length - 1]?.id)}`,
      );
    } catch (e) {
      console.error("[useChatController] Failed to fetch:", e);
      setState((prev) => ({ ...prev, isLoading: false }));
    }
  }, []);

  /**
   * Handles incoming stream chunks.
   */
  const handleStreamChunk = useCallback((chunk: string) => {
    setState((prev) => {
      const currentParts = prev.streamingMessage?.parts ?? [];
      const newParts = [...currentParts];
      const firstItem = newParts[0];

      if (!firstItem || firstItem.part.case !== "text") {
        newParts.unshift({
          part: { case: "text", value: chunk },
        } as ChatMessagePart);
      } else {
        const firstPart = firstItem.part;
        newParts[0] = {
          ...firstItem,
          part: {
            ...firstPart,
            case: "text",
            value: firstPart.value + chunk,
          },
        } as ChatMessagePart;
      }

      return {
        ...prev,
        streamingMessage: {
          createdAt: Date.now(),
          id: "streaming-placeholder", // Temp ID
          parts: newParts,
          role: "assistant",
        },
      };
    });
  }, []);

  /**
   * Optimistically appends a user message.
   */
  const appendUserMessage = useCallback((msg: ChatMessage) => {
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, msg],
    }));
  }, []);

  /**
   * Clears history after a specific index (for regenerate/delete).
   */
  const sliceHistory = useCallback((endIndex: number) => {
    setState((prev) => ({
      ...prev,
      messages: prev.messages.slice(0, endIndex),
    }));
  }, []);

  /**
   * Complete reset.
   */
  const clearAll = useCallback(() => {
    setState({
      isLoading: false,
      lastRequest: null,
      messages: [],
      streamingMessage: null,
    });
  }, []);

  // -- Reaction to Head ID Changes --

  // We use a ref to track the last synced head to avoid loops
  const lastSyncedHeadRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (conversationHeadId === lastSyncedHeadRef.current) return;

    console.log(
      `[useChatController] Head changed: ${lastSyncedHeadRef.current ?? "none"} -> ${conversationHeadId ?? "none"}`,
    );

    // If undefined, it usually means empty or initializing.
    if (!conversationHeadId) {
      // Check if we have an optimistic user message at the end
      const lastMsg =
        stateRef.current.messages[stateRef.current.messages.length - 1];
      // If the last message is a user message, we likely just sent it and the backend
      // reset to root to start a new branch. We should NOT clear it.
      if (lastMsg?.role === "user") {
        lastSyncedHeadRef.current = conversationHeadId;
        return;
      }

      if (stateRef.current.messages.length > 0) {
        // Use a microtask to avoid synchronous setState during render/effect phase
        queueMicrotask(() => {
          setState((prev) => {
            if (prev.messages.length === 0) return prev;
            return {
              ...prev,
              isLoading: false,
              messages: [],
              streamingMessage: null,
            };
          });
        });
      }
      lastSyncedHeadRef.current = conversationHeadId;
      return;
    }

    // Check if we can perform a "Zero-Fetch" transition.
    // This happens when the stream finishes and the backend tells us the new ID.
    // If our streaming message matches the content, we just "commit" it.
    const { messages, streamingMessage } = stateRef.current;

    // 1. Check if the headId matches the last message we ALREADY have in history (e.g. initial load or simple navigation)
    const lastHistoryMsg = messages[messages.length - 1];
    if (lastHistoryMsg?.id === conversationHeadId) {
      lastSyncedHeadRef.current = conversationHeadId;
      return;
    }

    // 2. Check if we have a streaming message that is ready to be "committed"
    if (streamingMessage) {
      // We assume that if the head ID changes while we have a streaming message,
      // this new head ID *is* that message.
      // We optimistically "promote" the streaming message to a history message.

      const committedMsg: ChatMessage = {
        ...streamingMessage,
        id: conversationHeadId,
        // We infer parent from the last message in history
        parentId: messages[messages.length - 1]?.id,
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, committedMsg],
        streamingMessage: null, // Remove the floater
      }));

      lastSyncedHeadRef.current = conversationHeadId;
      return;
    }

    // 3. Fallback: We are out of sync (e.g. another user updated, or time travel). Fetch.
    void syncHistory(conversationHeadId);
    lastSyncedHeadRef.current = conversationHeadId;
  }, [conversationHeadId, syncHistory, clearAll]);

  return {
    appendUserMessage,
    clearAll,
    handleStreamChunk,
    isLoading: state.isLoading,
    lastRequest: state.lastRequest,
    messages: state.messages,
    setLastRequest,
    sliceHistory,
    streamingMessage: state.streamingMessage,
  };
}
