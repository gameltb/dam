import { useFlowStore } from "@/store/flowStore";
import { type AppNode } from "@/types";

type NodeMapper = (node: AppNode) => AppNode;

export const applySyncUpdate = (nodeId: string, mapper: NodeMapper, skipIfDragging = true) => {
  useFlowStore.setState((s) => {
    const n = s.nodesById[nodeId];
    if (!n) return s;
    if (skipIfDragging && n.dragging) return s;
    return { nodesById: { ...s.nodesById, [nodeId]: mapper(n) } };
  });
};

export const applySyncInsert = (nodeId: string, node: AppNode) => {
  useFlowStore.setState((s) => ({
    nodesById: { ...s.nodesById, [nodeId]: node },
  }));
};

export const applySyncDelete = (nodeId: string) => {
  useFlowStore.setState((s) => {
    const { [nodeId]: _, ...rest } = s.nodesById;
    return { nodesById: rest };
  });
};
