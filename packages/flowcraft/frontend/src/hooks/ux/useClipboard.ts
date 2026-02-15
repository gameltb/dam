import { useCallback } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { commit } from "@/store/orchestrator";
import { type AppEdge, type AppNode } from "@/types";
import { sanitizeNode } from "@/utils/nodeUtils";

/**
 * useClipboard
 */
export const useClipboard = () => {
  const { activeGraphId, clipboard, setClipboard } = useFlowStore(
    useShallow((s) => ({
      activeGraphId: s.activeGraphId,
      clipboard: s.clipboard,
      setClipboard: s.setClipboard,
    })),
  );

  const copy = useCallback(
    (nodes: AppNode[], edges: AppEdge[]) => {
      // Dehydrate nodes if they contain non-serializable data
      setClipboard({ edges, nodes: nodes.map((n) => ({ ...n, selected: false })) });
    },
    [setClipboard],
  );

  const paste = useCallback(
    (targetPos: { x: number; y: number }) => {
      if (!clipboard || clipboard.nodes.length === 0) return;

      const graphId = activeGraphId || "default";

      commit(
        (draft) => {
          const idMap: Record<string, string> = {};

          // Calculate center of clipboard group for relative positioning
          const minX = Math.min(...clipboard.nodes.map((n) => n.position?.x ?? 0));
          const minY = Math.min(...clipboard.nodes.map((n) => n.position?.y ?? 0));

          // 1. Deselect current nodes in draft
          Object.values(draft.nodesById).forEach((n) => {
            n.selected = false;
          });

          // 2. Clone Nodes
          clipboard.nodes.forEach((n) => {
            const newId = crypto.randomUUID();
            idMap[n.id] = newId;

            const originalPos = n.position || { x: 0, y: 0 };
            const relativeX = originalPos.x - minX;
            const relativeY = originalPos.y - minY;

            const newNode = sanitizeNode({
              ...n,
              graphId,
              id: newId,
              position: {
                x: targetPos.x + relativeX,
                y: targetPos.y + relativeY,
              },
              selected: true,
            });
            draft.nodesById[newId] = newNode;
          });

          // 3. Clone Edges
          clipboard.edges.forEach((e) => {
            const newId = crypto.randomUUID();
            const sourceId = idMap[e.source];
            const targetId = idMap[e.target];

            // Only paste edge if both source and target were part of the copy
            if (sourceId && targetId) {
              const newEdge: AppEdge = {
                ...e,
                graphId,
                id: newId,
                selected: true,
                source: sourceId,
                target: targetId,
              };
              draft.edgesById[newId] = newEdge;
            }
          });
        },
        { description: "Paste nodes and edges" },
      );
    },
    [activeGraphId, clipboard],
  );

  const duplicate = useCallback(
    (nodes: AppNode[], edges: AppEdge[]) => {
      const offset = 30;
      const graphId = activeGraphId || "default";
      if (nodes.length === 0) return;

      commit(
        (draft) => {
          const idMap: Record<string, string> = {};

          // 1. Deselect everything first
          Object.values(draft.nodesById).forEach((n) => {
            n.selected = false;
          });

          // 2. Duplicate nodes
          nodes.forEach((n) => {
            const newId = crypto.randomUUID();
            idMap[n.id] = newId;
            const pos = n.position || { x: 0, y: 0 };

            draft.nodesById[newId] = sanitizeNode({
              ...n,
              graphId,
              id: newId,
              position: { x: pos.x + offset, y: pos.y + offset },
              selected: true,
            });
          });

          // 3. Duplicate edges
          edges.forEach((e) => {
            const newId = crypto.randomUUID();
            const sourceId = idMap[e.source];
            const targetId = idMap[e.target];

            if (sourceId && targetId) {
              const newEdge: AppEdge = {
                ...e,
                graphId,
                id: newId,
                selected: true,
                source: sourceId,
                target: targetId,
              };
              draft.edgesById[newId] = newEdge;
            }
          });
        },
        { description: "Duplicate selection" },
      );
    },
    [activeGraphId],
  );

  return { copy, duplicate, paste };
};
