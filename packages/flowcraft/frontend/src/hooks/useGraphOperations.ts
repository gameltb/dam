import { useFlowStore } from "../store/flowStore";
import { useShallow } from "zustand/react/shallow";
import { useClipboard } from "./useClipboard";
import { useLayoutOperations } from "./useLayoutOperations";
import { useNodeOperations } from "./useNodeOperations";

interface GraphOpsProps {
  clientVersion: number;
}

export const useGraphOperations = ({ clientVersion }: GraphOpsProps) => {
  const { nodes, edges, applyMutations } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      applyMutations: state.applyMutations,
    })),
  );

  const { addNode, deleteNode, deleteEdge } = useNodeOperations(applyMutations);
  const { copySelected, paste, duplicateSelected } = useClipboard(
    nodes,
    edges,
    applyMutations,
  );
  const { autoLayout, groupSelected } = useLayoutOperations(
    nodes,
    edges,
    applyMutations,
  );

  return {
    addNode,
    deleteNode,
    deleteEdge,
    copySelected,
    paste,
    duplicateSelected,
    autoLayout,
    groupSelected,
    clientVersion,
  };
};
