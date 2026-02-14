import { type ChatMessage, ChatRole } from "@/components/media/chat/types";
import { NodeLenses } from "@/store/materializers/nodeMaterializer";
import { type ChatNodeData, isChatNode } from "@/types";

import { type SyncedLens } from "./lens-types";

export * from "@/store/materializers/nodeMaterializer";
export * from "@/store/materializers/viewportMaterializer";

/**
 * ChatLenses - Chat graph structure query lenses
 */
export const ChatLenses = {
  history: (chatNodeId: string): SyncedLens<ChatMessage[]> => ({
    category: "node",
    description: `Compute chat history for ${chatNodeId}`,
    get: (s) => {
      const chatNode = s.nodesById[chatNodeId];
      if (!chatNode || !isChatNode(chatNode)) return [];

      const data = chatNode.data as ChatNodeData;
      const headId = data.extension.value.conversationHeadId;
      if (!headId) return [];

      const history: ChatMessage[] = [];
      let currentId = headId;
      const visited = new Set<string>();

      while (currentId && !visited.has(currentId)) {
        visited.add(currentId);
        const node = s.nodesById[currentId];
        if (!node) break;

        const nodeData = node.data;

        history.unshift({
          content: NodeLenses.messageContent(node.id).get(s),
          id: node.id,
          parts: nodeData.metadata?.parts_json ? JSON.parse(nodeData.metadata.parts_json) : [],
          role: (nodeData.metadata?.role as ChatRole) || ChatRole.USER,
          timestamp: nodeData.metadata?.timestamp ? BigInt(nodeData.metadata.timestamp) : undefined,
        });

        const parentEdge = s.edges.find((e) => e.target === currentId);
        currentId = parentEdge ? parentEdge.source : "";
        if (currentId === chatNodeId) break;
      }
      return history;
    },
    set: () => {},
  }),
};

/**
 * UiLenses - Transient UI state lenses
 */
export const UiLenses = {
  node: <T>(_nodeId: string, key: string, defaultValue: T): SyncedLens<T> => ({
    category: "ui",
    description: `UI state update: ${key}`,
    get: (_s: any) => defaultValue, // Placeholder
    set: () => {},
  }),
};
