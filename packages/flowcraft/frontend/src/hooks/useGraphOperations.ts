import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useClipboard } from "./useClipboard";
import { useLayoutOperations } from "./useLayoutOperations";
import { useNodeOperations } from "./useNodeOperations";

interface GraphOpsProps {
  clientVersion: number;
}

export const useGraphOperations = ({ clientVersion }: GraphOpsProps) => {
  const { applyMutations, edges, nodes } = useFlowStore(
    useShallow((state) => ({
      applyMutations: state.applyMutations,
      edges: state.edges,
      nodes: state.nodes,
    })),
  );

  const { addNode, deleteEdge, deleteNode } = useNodeOperations(applyMutations);
  const { copy, duplicate, paste } = useClipboard();
  const { autoLayout, groupSelected } = useLayoutOperations(
    nodes,
    edges,
    applyMutations,
  );

  return {
    addNode,
    autoLayout,
    clientVersion,
    copySelected: copy,
    deleteEdge,
    deleteNode,
    duplicateSelected: duplicate,
    groupSelected,
    paste,
  };
};
