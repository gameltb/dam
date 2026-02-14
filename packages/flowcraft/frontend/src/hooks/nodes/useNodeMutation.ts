import { useCallback } from "react";

import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { editNode } from "@/store/orchestrator";
import { type DynamicNodeData } from "@/types";
import { NodeLenses } from "@/utils/lenses";

/**
 * useNodeMutation
 */
export const useNodeMutation = (nodeId: string) => {
  const [, setLayout] = useSyncedBinding(NodeLenses.layout(nodeId));
  const [, setChatHead] = useSyncedBinding(NodeLenses.chatHead(nodeId));

  const updateData = useCallback(
    (recipe: (data: DynamicNodeData) => void) => {
      editNode(nodeId, (node) => {
        if (node.type === "dynamic" || node.type === "chatMessage") {
          recipe(node.data as DynamicNodeData);
        }
      });
    },
    [nodeId],
  );

  return {
    setChatHead,
    updateData,
    updateLayout: setLayout,
  };
};
