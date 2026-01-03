import { useCallback } from "react";
import { useFlowStore } from "../store/flowStore";
import { useUiStore } from "../store/uiStore";
import { v4 as uuidv4 } from "uuid";
import type { AppNode } from "../types";
import type { Edge as RFEdge } from "@xyflow/react";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/flowcraft/v1/core/service_pb";
import { EdgeSchema } from "../generated/flowcraft/v1/core/node_pb";
import { create } from "@bufbuild/protobuf";
import { toProtoNode } from "../utils/protoAdapter";

/**
 * Hook to manage clipboard operations (copy, paste, duplicate).
 * Supports remapping IDs and maintaining relative positions.
 */
export function useClipboard() {
  const { nodes, edges, applyMutations } = useFlowStore();
  const { clipboard, setClipboard } = useUiStore();

  const copy = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedNodeIds = new Set(selectedNodes.map((n) => n.id));
    const selectedEdges = edges.filter(
      (e) => selectedNodeIds.has(e.source) && selectedNodeIds.has(e.target),
    );

    setClipboard({
      nodes: selectedNodes,
      edges: selectedEdges,
    });
  }, [nodes, edges, setClipboard]);

  const paste = useCallback(
    (offset = { x: 40, y: 40 }) => {
      if (!clipboard) return;

      const idMap: Record<string, string> = {};
      const newNodes: AppNode[] = clipboard.nodes.map((node) => {
        const newId = uuidv4();
        idMap[node.id] = newId;
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
        source: idMap[edge.source] ?? edge.source,
        target: idMap[edge.target] ?? edge.target,
        selected: true,
      }));

      // Deselect all existing nodes
      const deselectMutations: GraphMutation[] = nodes
        .filter((n) => n.selected)
        .map((n) =>
          create(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: {
                id: n.id,
                // selected is handled by store locally, but we might want to sync it
              },
            },
          }),
        );

      const addMutations: GraphMutation[] = [
        ...newNodes.map((n) =>
          create(GraphMutationSchema, {
            operation: {
              case: "addNode",
              value: { node: toProtoNode(n) },
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
                  sourceNodeId: e.source,
                  targetNodeId: e.target,
                  sourceHandle: e.sourceHandle ?? "",
                  targetHandle: e.targetHandle ?? "",
                  metadata: {},
                }),
              },
            },
          }),
        ),
      ];

      applyMutations([...deselectMutations, ...addMutations], {
        description: "Paste subgraph",
      });
    },
    [clipboard, nodes, applyMutations],
  );

  const duplicate = useCallback(() => {
    copy();
    paste({ x: 50, y: 50 });
  }, [copy, paste]);

  return { copy, paste, duplicate, canPaste: !!clipboard };
}
