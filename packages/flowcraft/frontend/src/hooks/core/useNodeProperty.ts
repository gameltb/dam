import { useMemo } from "react";

import { useNodeId } from "@/contexts/NodeContext";
import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { type DynamicNodeData } from "@/types";
import { NodeLenses } from "@/utils/lenses";

/**
 * useNodeProperty
 */
export function useNodeProperty<K extends keyof DynamicNodeData>(property: K) {
  const nodeId = useNodeId();
  const lens = useMemo(() => NodeLenses.prop(nodeId, property), [nodeId, property]);
  return useSyncedBinding(lens);
}
