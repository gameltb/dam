import { create as createProto, toJsonString } from "@bufbuild/protobuf";
import { type Edge as RFEdge } from "@xyflow/react";
import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { commit } from "@/store/orchestrator";
import { AppNodeType, type AppNode } from "@/types";
import { useLayoutOperations } from "./useLayoutOperations";

/**
 * useImportExport
 */
export const useImportExport = () => {
  const { autoLayout } = useLayoutOperations();

  const importGraph = (data: string) => {
    try {
      const parsed = JSON.parse(data);
      commit(
        (draft) => {
          draft.nodesById = {};
          draft.edgesById = {};
          
          if (parsed.nodes) {
            parsed.nodes.forEach((n: any) => {
              if (n.id) draft.nodesById[n.id] = n;
            });
          }
          if (parsed.edges) {
            parsed.edges.forEach((e: any) => {
              if (e.id) draft.edgesById[e.id] = e;
            });
          }
        },
        { description: "Import graph from JSON", isHistoryOp: true }
      );
    } catch (err) {
      console.error("Failed to import graph:", err);
    }
  };

  /**
   * Core business: Import conversation JSON as a message tree subgraph.
   */
  const importConversations = (data: string) => {
    try {
      const sessions = JSON.parse(data);
      if (!Array.isArray(sessions)) return;

      commit((draft) => {
        sessions.forEach((session, sIdx) => {
          const { conv, messages } = session;
          const sessionRootId = conv.id || `session-${sIdx}`;
          
          // 1. Create session root node (Chat Node)
          const chatNode: AppNode = {
            id: sessionRootId,
            type: AppNodeType.DYNAMIC,
            position: { x: sIdx * 1000, y: 0 },
            width: 300,
            height: 200,
            scopeId: "root", // Imported sessions land in root by default
            data: {
              displayName: conv.name || "Imported Session",
              extension: {
                case: "chat",
                value: {
                  conversationHeadId: conv.currNode || "",
                  treeId: sessionRootId,
                  isHistoryCleared: false,
                }
              },
              availableModes: [1, 2, 3], // MODE_CHAT, etc.
              activeMode: 1, // MODE_CHAT
            } as any,
          };
          draft.nodesById[chatNode.id] = chatNode;

          // 2. Convert all messages to ChatMessageNode
          const msgNodes: Record<string, AppNode> = {};
          messages.forEach((msg: any, mIdx: number) => {
            const node: AppNode = {
              id: msg.id,
              type: AppNodeType.CHAT_MESSAGE,
              position: { x: sIdx * 1000 + (mIdx % 5) * 350, y: 300 + Math.floor(mIdx / 5) * 250 },
              width: 300,
              height: 150,
              scopeId: "root", // All messages share root scope for now
              data: {
                metadata: {
                  role: msg.role,
                  timestamp: msg.timestamp?.toString(),
                  parts_json: JSON.stringify([
                    JSON.parse(toJsonString(ChatMessagePartSchema, createProto(ChatMessagePartSchema, {
                      part: { case: "text", value: msg.content }
                    })))
                  ]),
                }
              } as any,
              parentId: sessionRootId, // Place into session subgraph
            };
            msgNodes[node.id] = node;
            draft.nodesById[node.id] = node;
          });

          // 3. Establish connections (message tree structure)
          messages.forEach((msg: any) => {
            if (msg.parent && msgNodes[msg.parent]) {
              const edgeId = `e-${msg.parent}-${msg.id}`;
              const edge: RFEdge = {
                id: edgeId,
                source: msg.parent,
                target: msg.id,
                type: "default",
              };
              draft.edgesById[edgeId] = edge;
            } else if (msg.type === "root" || !msg.parent) {
              // If it's a root message, connect to session node (optional, depending on UI convention)
              const edgeId = `e-\
${sessionRootId}-\
${msg.id}\
`;
              const edge: RFEdge = {
                id: edgeId,
                source: sessionRootId,
                target: msg.id,
                type: "default",
              };
              draft.edgesById[edgeId] = edge;
            }
          });
        });
      }, { description: "Imported message tree subgraphs", isHistoryOp: true });

      // Automatically trigger a layout after import
      setTimeout(() => autoLayout(), 100);
    } catch (err) {
      console.error("Failed to import conversations:", err);
    }
  };

  return { autoLayout, importConversations, importGraph };
};
