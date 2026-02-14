import { useCallback, useMemo } from "react";

import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { useNodeController } from "@/hooks/nodes/useNodeController";
import { ChatStatus } from "@/types";
import { ChatLenses } from "@/utils/lenses";

import { type ChatMessage } from "./types";

/**
 * useChatController (Redesigned)
 * Manages chat state and provides control methods for the chat UI.
 */
export function useChatController(_conversationHeadId: string, nodeId: string, _treeId: string) {
  // 1. Use lens to get real-time history
  const [messages] = useSyncedBinding(useMemo(() => ChatLenses.history(nodeId), [nodeId]));

  const nodeController = useNodeController(nodeId);

  const status = useMemo(() => {
    switch (nodeController.status) {
      case "busy":
        return ChatStatus.STREAMING;
      case "error":
        return ChatStatus.ERROR;
      default:
        return ChatStatus.READY;
    }
  }, [nodeController.status]);

  // Implement methods to match component expectations
  // Note: Most logic has been moved to atomic commit in useChatActions
  const appendUserMessage = useCallback((_msg: ChatMessage) => {
    console.debug("[useChatController] appendUserMessage called", _msg);
  }, []);

  const handleStreamChunk = useCallback((_chunk: string) => {
    console.debug("[useChatController] handleStreamChunk called", _chunk);
  }, []);

  const sliceHistory = useCallback((_index: number) => {
    console.debug("[useChatController] sliceHistory called", _index);
  }, []);

  return {
    appendUserMessage,
    handleStreamChunk,
    messages: (messages || []) as ChatMessage[],
    sliceHistory,
    status,
    streamingMessage: null as ChatMessage | null,
  };
}
