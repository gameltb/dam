import { useCallback } from "react";

import { useFlowStore } from "@/store/flowStore";
import { useSettingsStore } from "@/store/ui/settingsStore";
import { type LocalLLMClientConfig } from "@/types";
import { log } from "@/utils/logger";

import { type ChatMessage, ChatStatus } from "./types";

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
      _userParts: unknown[],
    ) => {
      log.debug("useLocalInference", `Local inference requested for node ${nodeId}, but not yet implemented`);
    },
    [nodeId, spacetimeConn],
  );

  return { localClients, performLocalInference };
}
