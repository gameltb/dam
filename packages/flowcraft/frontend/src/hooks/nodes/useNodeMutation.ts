import { useCallback } from "react";
import { NodeLenses } from "@/utils/lenses";
import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { editNode } from "@/store/orchestrator";
import { type DynamicNodeData } from "@/types";

/**
 * useNodeMutation
 */
export const useNodeMutation = (nodeId: string) => {
  const [, setLayout] = useSyncedBinding(NodeLenses.layout(nodeId));
  const [, setChatHead] = useSyncedBinding(NodeLenses.chatHead(nodeId));

  const updateData = useCallback((recipe: (data: DynamicNodeData) => void) => {
    editNode(nodeId, (node) => {
      if (node.type === "dynamic" || node.type === "chatMessage") {
        recipe(node.data as DynamicNodeData);
      }
    });
  }, [nodeId]);

  return {
    updateData,
    updateLayout: setLayout,
    setChatHead,
  };
};