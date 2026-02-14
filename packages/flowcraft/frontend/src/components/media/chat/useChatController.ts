import { useCallback, useMemo } from "react";

import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { useNodeController } from "@/hooks/nodes/useNodeController";
import { ChatLenses } from "@/utils/lenses";

import { type ChatMessage } from "./types";

/**
 * useChatController (Redesigned)
 */
export function useChatController(_conversationHeadId: string, nodeId: string, _treeId: string) {
  // 1. Use lens to get real-time history
  const [messages] = useSyncedBinding(useMemo(() => ChatLenses.history(nodeId), [nodeId]));

  const nodeController = useNodeController(nodeId);

  // Implement missing methods to match component expectations
  const appendUserMessage = useCallback((_msg: ChatMessage) => {
    // Logic has been moved to atomic commit in useChatActions
  }, []);

  const handleStreamChunk = useCallback((_chunk: string) => {}, []);
  const sliceHistory = useCallback((_index: number) => {}, []);

  return {
    appendUserMessage,
    handleStreamChunk,
    messages: messages,
    sliceHistory,
    status: nodeController.status as any,
    streamingMessage: null as any,
  };
}
