import { type Edge as RFEdge } from "@xyflow/react";
import { useCallback } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { commit } from "@/store/orchestrator";
import { type AppNode } from "@/types";

/**
 * useClipboard
 */
export const useClipboard = () => {
  const { clipboard, setClipboard } = useFlowStore(
    useShallow((s) => ({
      clipboard: s.clipboard,
      setClipboard: s.setClipboard,
    })),
  );

  const copy = useCallback(
    (nodes: AppNode[], edges: RFEdge[]) => {
      setClipboard({ edges, nodes });
    },
    [setClipboard],
  );

  const paste = useCallback(
    (position: { x: number; y: number }) => {
      if (!clipboard) return;

      commit(
        (draft) => {
          const idMap: Record<string, string> = {};

          clipboard.nodes.forEach((n) => {
            const newId = crypto.randomUUID();
            idMap[n.id] = newId;
            const newNode = {
              ...n,
              id: newId,
              position: {
                x: position.x + (n.position.x - (clipboard.nodes[0]?.position.x || 0)),
                y: position.y + (n.position.y - (clipboard.nodes[0]?.position.y || 0)),
              },
              selected: true,
            };
            draft.nodesById[newId] = newNode;
          });

          clipboard.edges.forEach((e) => {
            const newId = crypto.randomUUID();
            const newEdge = {
              ...e,
              id: newId,
              selected: true,
              source: idMap[e.source] || e.source,
              target: idMap[e.target] || e.target,
            };
            draft.edgesById[newId] = newEdge;
          });
        },
        { description: "Paste nodes and edges" },
      );
    },
    [clipboard],
  );

  const duplicate = useCallback((nodes: AppNode[], edges: RFEdge[]) => {
    const offset = 20;
    commit(
      (draft) => {
        const idMap: Record<string, string> = {};
        nodes.forEach((n) => {
          const newId = crypto.randomUUID();
          idMap[n.id] = newId;
          draft.nodesById[newId] = {
            ...n,
            id: newId,
            position: { x: n.position.x + offset, y: n.position.y + offset },
            selected: true,
          };
        });
        edges.forEach((e) => {
          const newId = crypto.randomUUID();
          draft.edgesById[newId] = {
            ...e,
            id: newId,
            selected: true,
            source: idMap[e.source] || e.source,
            target: idMap[e.target] || e.target,
          };
        });
      },
      { description: "Duplicate selection" },
    );
  }, []);

  return { copy, duplicate, paste };
};
