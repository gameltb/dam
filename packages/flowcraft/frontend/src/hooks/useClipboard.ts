import { useCallback } from "react";
import { useUiStore } from "../store/uiStore";
import { v4 as uuidv4 } from "uuid";
import type { AppNode } from "../types";
import type { XYPosition, Edge as RFEdge } from "@xyflow/react";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/core/service_pb";
import { EdgeSchema, type Node as ProtoNode } from "../generated/core/node_pb";
import { create } from "@bufbuild/protobuf";
import { type MutationContext } from "../store/flowStore";

export const useClipboard = (
  nodes: AppNode[],
  edges: RFEdge[],
  applyMutations: (
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void,
) => {
  const copySelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    const selectedEdges = edges.filter((e) => {
      const isSourceSelected = selectedNodes.some((n) => n.id === e.source);
      const isTargetSelected = selectedNodes.some((n) => n.id === e.target);
      return isSourceSelected && isTargetSelected;
    });

    if (selectedNodes.length > 0) {
      useUiStore.getState().setClipboard({
        nodes: JSON.parse(JSON.stringify(selectedNodes)) as AppNode[],
        edges: JSON.parse(JSON.stringify(selectedEdges)) as RFEdge[],
      });
    }
  }, [nodes, edges]);

  const paste = useCallback(
    (targetPosition?: XYPosition) => {
      const clipboard = useUiStore.getState().clipboard;
      if (!clipboard) return;

      const idMap: Record<string, string> = {};
      const firstNodePos = clipboard.nodes[0]?.position ?? { x: 0, y: 0 };
      const offset = targetPosition
        ? {
            x: targetPosition.x - firstNodePos.x,
            y: targetPosition.y - firstNodePos.y,
          }
        : { x: 40, y: 40 };

      const newNodes = clipboard.nodes.map((node) => {
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

      const newEdges = clipboard.edges.map((edge) => ({
        ...edge,
        id: uuidv4(),
        source: idMap[edge.source] ?? edge.source,
        target: idMap[edge.target] ?? edge.target,
        selected: true,
      }));

      applyMutations(
        [
          create(GraphMutationSchema, {
            operation: {
              case: "addSubgraph",
              value: {
                nodes: newNodes as unknown as ProtoNode[],
                edges: newEdges.map((e) =>
                  create(EdgeSchema, {
                    id: e.id,
                    source: e.source,
                    target: e.target,
                    sourceHandle: e.sourceHandle ?? "",
                    targetHandle: e.targetHandle ?? "",
                  }),
                ),
              },
            },
          }),
        ],
        { taskId: uuidv4(), description: "Paste Subgraph" },
      );
    },
    [applyMutations],
  );

  const duplicateSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedEdges = edges.filter((e) => {
      const isSourceSelected = selectedNodes.some((n) => n.id === e.source);
      const isTargetSelected = selectedNodes.some((n) => n.id === e.target);
      return isSourceSelected && isTargetSelected;
    });

    const tempClipboard: { nodes: AppNode[]; edges: RFEdge[] } = {
      nodes: JSON.parse(JSON.stringify(selectedNodes)) as AppNode[],
      edges: JSON.parse(JSON.stringify(selectedEdges)) as RFEdge[],
    };

    const idMap: Record<string, string> = {};
    const newNodes = tempClipboard.nodes.map((n: AppNode) => {
      const newId = uuidv4();
      idMap[n.id] = newId;
      return {
        ...n,
        id: newId,
        position: { x: n.position.x + 40, y: n.position.y + 40 },
        selected: true,
      };
    });
    const newEdges = tempClipboard.edges.map((e: RFEdge) => ({
      ...e,
      id: uuidv4(),
      source: idMap[e.source] ?? e.source,
      target: idMap[e.target] ?? e.target,
      selected: true,
    }));

    applyMutations(
      [
        create(GraphMutationSchema, {
          operation: {
            case: "addSubgraph",
            value: {
              nodes: newNodes as unknown as ProtoNode[],
              edges: newEdges.map((e) =>
                create(EdgeSchema, {
                  id: e.id,
                  source: e.source,
                  target: e.target,
                  sourceHandle: e.sourceHandle ?? "",
                  targetHandle: e.targetHandle ?? "",
                }),
              ),
            },
          },
        }),
      ],
      { taskId: uuidv4(), description: "Duplicate Selected" },
    );
  }, [nodes, edges, applyMutations]);

  return { copySelected, paste, duplicateSelected };
};
