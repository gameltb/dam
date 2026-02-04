import { useCallback } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";

/**
 * useNodeScope
 * Manages node hierarchy and scope navigation logic.
 */
export const useNodeScope = (nodeId: string) => {
  const { nodesById, spacetimeConn } = useFlowStore(
    useShallow((s) => ({
      nodesById: s.nodesById,
      spacetimeConn: s.spacetimeConn,
    })),
  );

  const reparentNode = useFlowStore((s) => s.reparentNode);

  const moveToScope = useCallback(
    (newParentId: null | string) => {
      reparentNode(nodeId, newParentId);
    },
    [nodeId, reparentNode],
  );

  const navigateToScope = useCallback(() => {
    // Logic: If the current node is a Group or Subgraph, enter its scope
    const parent = nodesById[nodeId];
    if (parent) {
      // Trigger UI layer navigation (handled by external UI Store)
      console.log(`[Scope] Navigating into ${nodeId}`);
    }
  }, [nodeId, nodesById]);

  return {
    moveToScope,
    navigateToScope,
    spacetimeConn,
  };
};
