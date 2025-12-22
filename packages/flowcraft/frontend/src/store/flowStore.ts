import { create, type StoreApi, type UseBoundStore } from "zustand";
import { temporal } from "zundo";
import type {
  Connection,
  Edge,
  EdgeChange,
  NodeChange,
  OnNodesChange,
  OnEdgesChange,
} from "@xyflow/react";
import { addEdge, applyNodeChanges, applyEdgeChanges } from "@xyflow/react";
import { isTypedNodeData, type AppNode } from "../types";

export interface RFState {
  nodes: AppNode[];
  edges: Edge[];
  version: number;
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: (connection: Connection) => void;
  setNodes: (nodes: AppNode[]) => void;
  setEdges: (edges: Edge[]) => void;
  addNode: (node: AppNode) => void;
}

const useStore = create(
  temporal<RFState>((set, get) => ({
    nodes: [],
    edges: [],
    version: 0,
    onNodesChange: (changes: NodeChange[]) => {
      set((state) => ({
        nodes: applyNodeChanges(changes, state.nodes) as AppNode[],
        version: state.version + 1,
      }));
    },
    onEdgesChange: (changes: EdgeChange[]) => {
      set((state) => ({
        edges: applyEdgeChanges(changes, state.edges),
        version: state.version + 1,
      }));
    },
    onConnect: (connection: Connection) => {
      const { nodes, edges, version } = get();
      const sourceNode = nodes.find((node) => node.id === connection.source);
      const targetNode = nodes.find((node) => node.id === connection.target);

      if (
        sourceNode &&
        targetNode &&
        isTypedNodeData(sourceNode.data) &&
        isTypedNodeData(targetNode.data)
      ) {
        const sourceHandleType = sourceNode.data.outputType as string;
        const targetHandleType = targetNode.data.inputType as string;

        if (
          sourceHandleType === targetHandleType ||
          sourceHandleType === "any" ||
          targetHandleType === "any"
        ) {
          set({
            edges: addEdge(
              {
                ...connection,
                id: `e${connection.source}-${connection.target}`,
              },
              edges,
            ),
            version: version + 1,
          });
        } else {
          alert(
            `Type mismatch: Cannot connect ${sourceHandleType} to ${targetHandleType}`,
          );
        }
      } else {
        set({
          edges: addEdge(connection, edges),
          version: version + 1,
        });
      }
    },
    setNodes: (nodes: AppNode[]) => {
      set((state) => ({ nodes, version: state.version + 1 }));
    },
    setEdges: (edges: Edge[]) => {
      set((state) => ({ edges, version: state.version + 1 }));
    },
    addNode: (node: AppNode) => {
      set((state) => ({
        nodes: [...state.nodes, node],
        version: state.version + 1,
      }));
    },
  })),
);

export const useFlowStore = useStore;

interface TemporalState {
  pastStates: RFState[];
  futureStates: RFState[];
  undo: () => void;
  redo: () => void;
  clear: () => void;
  isTracking: boolean;
  pause: () => void;
  resume: () => void;
}

export const useTemporalStore = <T>(
  selector: (state: TemporalState) => T,
): T => {
  return (
    useStore as unknown as { temporal: UseBoundStore<StoreApi<TemporalState>> }
  ).temporal(selector);
};
