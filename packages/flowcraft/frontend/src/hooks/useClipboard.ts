import type { Edge as RFEdge } from "@xyflow/react";

import { create as createProto } from "@bufbuild/protobuf";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { useShallow } from "zustand/react/shallow";

import type { AppNode, DynamicNodeData } from "@/types";

import { ActionExecutionRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { AddEdgeRequestSchema, AddNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { appNodeToProto } from "@/utils/nodeProtoUtils";

/**
 * Hook to manage clipboard operations (copy, paste, duplicate).
 * Supports remapping IDs and maintaining relative positions.
 */
export function useClipboard() {
  const { applyMutations, edges, nodeDraft, nodes, spacetimeConn } = useFlowStore(
    useShallow((s) => ({
      applyMutations: s.applyMutations,
      edges: s.edges,
      nodeDraft: s.nodeDraft,
      nodes: s.nodes,
      spacetimeConn: s.spacetimeConn,
    })),
  );
  const { clipboard, setClipboard } = useUiStore(
    useShallow((s) => ({
      clipboard: s.clipboard,
      setClipboard: s.setClipboard,
    })),
  );

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

      // Deselect existing via ORM
      nodes
        .filter((n) => n.selected)
        .forEach((n) => {
          const res = nodeDraft(n);
          if (res.ok) {
            res.value.selected = false;
          }
        });

      const addMutations = [
        ...newNodes.map((n) => createProto(AddNodeRequestSchema, { node: appNodeToProto(n) })),
        ...newEdges.map((e) =>
          createProto(AddEdgeRequestSchema, {
            edge: {
              edgeId: e.id,
              metadata: {},
              sourceHandle: e.sourceHandle ?? "",
              sourceNodeId: e.source,
              targetHandle: e.targetHandle ?? "",
              targetNodeId: e.target,
            },
          }),
        ),
      ];

      applyMutations(addMutations, {
        description: "Paste subgraph",
      });

      chatNodesToDuplicate.forEach((item) => {
        if (item.oldHeadId && item.treeId) {
          if (spacetimeConn) {
            const request = createProto(ActionExecutionRequestSchema, {
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
    [clipboard, nodes, applyMutations, spacetimeConn, nodeDraft],
  );

  const duplicate = useCallback(() => {
    copy();
    paste({ x: 50, y: 50 });
  }, [copy, paste]);

  return { canPaste: !!clipboard, copy, duplicate, paste };
}
