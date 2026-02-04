import { useCallback } from "react";

import { commit } from "@/store/orchestrator";
import { type AppNode } from "@/types";

export const useNodeOperations = () => {
  const addNode = useCallback((node: AppNode) => {
    commit(
      (draft) => {
        draft.nodesById[node.id] = node;
      },
      { description: `Add node ${node.id}` },
    );
  }, []);

  const deleteNode = useCallback((nodeId: string) => {
    commit(
      (draft) => {
        delete draft.nodesById[nodeId];
        // Cleanup connected edges
        Object.values(draft.edgesById).forEach((edge) => {
          if (edge.source === nodeId || edge.target === nodeId) {
            delete draft.edgesById[edge.id];
          }
        });
      },
      { description: `Delete node ${nodeId}` },
    );
  }, []);

  const deleteEdge = useCallback((edgeId: string) => {
    commit(
      (draft) => {
        delete draft.edgesById[edgeId];
      },
      { description: `Delete edge ${edgeId}` },
    );
  }, []);

  return { addNode, deleteEdge, deleteNode };
};
