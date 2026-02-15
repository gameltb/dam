import { useReactFlow } from "@xyflow/react";
import { useLayoutOperations } from "@/hooks/graph/useLayoutOperations";
import { useNodeOperations } from "@/hooks/nodes/useNodeOperations";
import { useClipboard } from "@/hooks/ux/useClipboard";
import { useFlowStore } from "@/store/flowStore";
import { commit } from "@/store/orchestrator";
import { type AppEdge } from "@/types";

export const useGraphOperations = () => {
  const { addNode, deleteEdge, deleteNode } = useNodeOperations();
  const { autoLayout } = useLayoutOperations();
  const { copy, duplicate, paste } = useClipboard();
  const { getEdges, getNodes } = useReactFlow();

  const handleCopy = () => {
    const selectedNodes = getNodes().filter((n) => n.selected);
    const selectedEdges = getEdges().filter((e) => e.selected);
    copy(selectedNodes as any, selectedEdges as AppEdge[]);
  };

  const handlePaste = () => {
    const pos = window.lastProcessedMousePos || { x: 500, y: 500 };
    paste(pos);
  };

  const handleDuplicate = () => {
    const selectedNodes = getNodes().filter((n) => n.selected);
    const selectedEdges = getEdges().filter((e) => e.selected);
    duplicate(selectedNodes as any, selectedEdges as AppEdge[]);
  };

  const handleDeleteSelected = () => {
    const selectedNodeIds = Object.values(useFlowStore.getState().nodesById)
      .filter((n) => n.selected)
      .map((n) => n.id);
    const selectedEdgeIds = Object.values(useFlowStore.getState().edgesById)
      .filter((e) => e.selected)
      .map((e) => e.id);

    commit(
      (draft) => {
        selectedNodeIds.forEach((id) => {
          delete draft.nodesById[id];
          Object.values(draft.edgesById).forEach((edge: any) => {
            if (edge.source === id || edge.target === id) {
              delete draft.edgesById[edge.id];
            }
          });
        });
        selectedEdgeIds.forEach((id) => {
          delete draft.edgesById[id];
        });
      },
      { description: `Delete selected entities` },
    );
  };

  const handleGroupSelected = () => {
    console.warn("groupSelected logic moved to LayoutOperations but not yet re-exposed");
  };

  return {
    addNode,
    autoLayout,
    copySelected: handleCopy,
    deleteEdge,
    deleteNode,
    deleteSelected: handleDeleteSelected,
    duplicateSelected: handleDuplicate,
    groupSelected: handleGroupSelected,
    paste: handlePaste,
  };
};
