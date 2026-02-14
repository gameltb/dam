import { useLayoutOperations } from "@/hooks/graph/useLayoutOperations";
import { useNodeOperations } from "@/hooks/nodes/useNodeOperations";
import { useClipboard } from "@/hooks/ux/useClipboard";
import { useFlowStore } from "@/store/flowStore";
import { commit } from "@/store/orchestrator";

export const useGraphOperations = () => {
  const { addNode, deleteEdge, deleteNode } = useNodeOperations();
  const { autoLayout } = useLayoutOperations();
  const { copy, duplicate, paste } = useClipboard();

  const handleCopy = () => {
    const nodes = Object.values(useFlowStore.getState().nodesById).filter((n) => n.selected);
    const edges = Object.values(useFlowStore.getState().edgesById).filter((e) => e.selected);
    copy(nodes, edges);
  };

  const handlePaste = () => {
    const pos = window.lastProcessedMousePos || { x: 500, y: 500 };
    paste(pos);
  };

  const handleDuplicate = () => {
    const nodes = Object.values(useFlowStore.getState().nodesById).filter((n) => n.selected);
    const edges = Object.values(useFlowStore.getState().edgesById).filter((e) => e.selected);
    duplicate(nodes, edges);
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
