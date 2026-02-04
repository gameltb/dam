import { type ChatMessage, ChatRole } from "@/components/media/chat/types";
import { type ChatNodeData, isChatNode } from "@/types";
import { NodeLenses } from "@/store/materializers/nodeMaterializer";
import { type SyncedLens } from "./lens-types";

export * from "@/store/materializers/nodeMaterializer";
export * from "@/store/materializers/viewportMaterializer";

/**
 * ChatLenses - Chat graph structure query lenses
 */
export const ChatLenses = {
  history: (chatNodeId: string): SyncedLens<ChatMessage[]> => ({
    category: 'node',
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
          id: node.id,
          role: (nodeData.metadata?.role as ChatRole) || ChatRole.USER,
          content: NodeLenses.messageContent(node.id).get(s),
          timestamp: nodeData.metadata?.timestamp ? BigInt(nodeData.metadata.timestamp) : undefined,
          parts: nodeData.metadata?.parts_json ? JSON.parse(nodeData.metadata.parts_json) : [],
        });

        const parentEdge = s.edges.find(e => e.target === currentId);
        currentId = parentEdge ? parentEdge.source : "";
        if (currentId === chatNodeId) break;
      }
      return history;
    },
    set: () => {},
    description: `Compute chat history for ${chatNodeId}`
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