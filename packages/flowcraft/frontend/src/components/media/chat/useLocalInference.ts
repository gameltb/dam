import { useCallback } from "react";
import { useFlowStore } from "@/store/flowStore";
import { useSettingsStore } from "@/store/ui/settingsStore";
import { type LocalLLMClientConfig } from "@/types";
import { ChatStatus, type ChatMessage } from "./types";

export function useLocalInference(nodeId: string) {
  const { localClients } = useSettingsStore();
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);

  const performLocalInference = useCallback(
    async (
      _client: LocalLLMClientConfig,
      _modelId: string,
      _getHistory: () => ChatMessage[],
      _setStatus: (s: ChatStatus) => void,
      _handleStreamChunk: (chunk: string) => void,
      _userMsgId: string,
      _userParts: any[],
    ) => {
      // TODO: Implement local LLM invocation logic via fetch or SDK
      console.warn("Local inference not yet fully implemented");
    },
    [nodeId, spacetimeConn],
  );

  return { localClients, performLocalInference };
}