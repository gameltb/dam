import type { Edge as RFEdge } from "@xyflow/react";

import { create } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import type { AppNode, DynamicNodeData } from "@/types";

import { ActionExecutionRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { EdgeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphMutation, GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { appNodeToProto } from "@/utils/nodeProtoUtils";

/**
 * Hook to manage clipboard operations (copy, paste, duplicate).
 * Supports remapping IDs and maintaining relative positions.
 */
export function useClipboard() {
  const { applyMutations, edges, nodes, spacetimeConn } = useFlowStore();
  const { clipboard, setClipboard } = useUiStore();

  const copy = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedNodeIds = new Set(selectedNodes.map((n) => n.id));
    const selectedEdges = edges.filter((e) => selectedNodeIds.has(e.source) && selectedNodeIds.has(e.target));

    setClipboard({
      edges: selectedEdges,
      nodes: selectedNodes,
    });
  }, [nodes, edges, setClipboard]);

  const paste = useCallback(
    (offset = { x: 40, y: 40 }) => {
      if (!clipboard) return;

      const idMap: Record<string, string> = {};
      const chatNodesToDuplicate: {
        newId: string;
        oldHeadId: string;
        treeId: string;
      }[] = [];

      const newNodes: AppNode[] = clipboard.nodes.map((node) => {
        const newId = uuidv4();
        idMap[node.id] = newId;

        // If it's a chat node, prepare for branch duplication
        const ext = (node.data as DynamicNodeData).extension;
        if (ext?.case === "chat") {
          const chatData = ext.value;
          chatNodesToDuplicate.push({
            newId,
            oldHeadId: chatData.conversationHeadId || "",
            treeId: chatData.treeId || "",
          });
        }

        return {
          ...node,
          id: newId,
          position: {
            x: node.position.x + offset.x,
            y: node.position.y + offset.y,
          },
          selected: true,
        };
      });

      const newEdges: RFEdge[] = clipboard.edges.map((edge) => ({
        ...edge,
        id: uuidv4(),
        selected: true,
        source: idMap[edge.source] ?? edge.source,
        target: idMap[edge.target] ?? edge.target,
      }));

      // Deselect existing
      const deselectMutations: GraphMutation[] = nodes
        .filter((n) => n.selected)
        .map((n) =>
          create(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: { id: n.id },
            },
          }),
        );

      const addMutations: GraphMutation[] = [
        ...newNodes.map((n) =>
          create(GraphMutationSchema, {
            operation: {
              case: "addNode",
              value: { node: appNodeToProto(n) },
            },
          }),
        ),
        ...newEdges.map((e) =>
          create(GraphMutationSchema, {
            operation: {
              case: "addEdge",
              value: {
                edge: create(EdgeSchema, {
                  edgeId: e.id,
                  metadata: {},
                  sourceHandle: e.sourceHandle ?? "",
                  sourceNodeId: e.source,
                  targetHandle: e.targetHandle ?? "",
                  targetNodeId: e.target,
                }),
              },
            },
          }),
        ),
      ];

      applyMutations([...deselectMutations, ...addMutations], {
        description: "Paste subgraph",
      });

      // Trigger asynchronous branch duplication for Chat nodes
      chatNodesToDuplicate.forEach((item) => {
        if (item.oldHeadId && item.treeId) {
          // Trigger a specialized action that Worker handles to COW clone the branch
          if (spacetimeConn) {
            const request = create(ActionExecutionRequestSchema, {
              actionId: "chat.duplicateBranch",
              params: {
                case: "paramsStruct",
                value: {
                  fields: {
                    sourceHeadId: {
                      kind: { case: "stringValue", value: item.oldHeadId },
                    },
                    treeId: {
                      kind: { case: "stringValue", value: item.treeId },
                    },
                  },
                } as any,
              },
              sourceNodeId: item.newId,
            });

            spacetimeConn.pbreducers.executeAction({
              id: crypto.randomUUID(),
              request,
            });
          }
        }
      });
    },
    [clipboard, nodes, applyMutations, spacetimeConn],
  );

  const duplicate = useCallback(() => {
    copy();
    paste({ x: 50, y: 50 });
  }, [copy, paste]);

  return { canPaste: !!clipboard, copy, duplicate, paste };
}
