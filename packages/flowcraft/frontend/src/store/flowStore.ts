import { create } from "zustand";
import { temporal, type TemporalState } from "zundo";
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
import { useStoreWithEqualityFn } from "zustand/traditional";

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
  setVersion: (version: number) => void;
  setGraph: (
    graph: { nodes: AppNode[]; edges: Edge[] },
    version: number,
  ) => void;
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
    setVersion: (version: number) => {
      set({ version });
    },
    setGraph: (graph: { nodes: AppNode[]; edges: Edge[] }, version: number) => {
      set({ nodes: graph.nodes, edges: graph.edges, version });
    },
  })),
);

export const useFlowStore = useStore;

export function useTemporalStore<T>(
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  const store = useStore.temporal;
  if (!store) {
    throw new Error(
      "Temporal store not found. Make sure you have wrapped your store with temporal middleware",
    );
  }
  return useStoreWithEqualityFn(store, selector, equality);
}
