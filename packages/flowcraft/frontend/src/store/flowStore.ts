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
import {
  type AppNode,
  isDynamicNode,
  PortStyle,
  type DynamicNodeData,
} from "../types";
import { useStoreWithEqualityFn } from "zustand/traditional";
import { flowcraft } from "../generated/flowcraft";
import { validateConnection } from "../utils/portValidators";

export interface RFState {
  nodes: AppNode[];
  edges: Edge[];
  version: number;
  // Node Events
  lastNodeEvent: {
    type: string;
    payload: Record<string, unknown>;
    timestamp: number;
  } | null;
  dispatchNodeEvent: (type: string, payload: Record<string, unknown>) => void;

  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: (connection: Connection) => void;
  setNodes: (nodes: AppNode[]) => void;
  setEdges: (edges: Edge[]) => void;
  addNode: (node: AppNode) => void;
  updateNodeData: (nodeId: string, data: Record<string, unknown>) => void;
  setVersion: (version: number) => void;
  setGraph: (
    graph: { nodes: AppNode[]; edges: Edge[] },
    version: number,
  ) => void;

  // Mutation API
  applyMutations: (mutations: flowcraft.v1.IGraphMutation[]) => void;

  // Clipboard
  clipboard: { nodes: AppNode[]; edges: Edge[] } | null;
  setClipboard: (data: { nodes: AppNode[]; edges: Edge[] } | null) => void;

  // Active Connection Attempt
  connectionStartHandle: {
    nodeId: string;
    handleId: string;
    type: "source" | "target";
  } | null;
  setConnectionStartHandle: (
    handle: {
      nodeId: string;
      handleId: string;
      type: "source" | "target";
    } | null,
  ) => void;
}

const useStore = create(
  temporal<RFState>(
    (set, get) => ({
      nodes: [],
      edges: [],
      version: 0,
      lastNodeEvent: null,

      dispatchNodeEvent: (type, payload) => {
        set({ lastNodeEvent: { type, payload, timestamp: Date.now() } });
      },

      applyMutations: (mutations) => {
        set((state) => {
          let newNodes = [...state.nodes];
          let newEdges = [...state.edges];

          mutations.forEach((mut) => {
            if (mut.addNode?.node) {
              newNodes.push(mut.addNode.node as unknown as AppNode);
            } else if (mut.updateNode) {
              newNodes = newNodes.map((n) => {
                if (n.id === mut.updateNode!.id) {
                  const updated = { ...n } as AppNode;
                  if (mut.updateNode!.position) {
                    updated.position = {
                      x: mut.updateNode!.position.x ?? updated.position.x,
                      y: mut.updateNode!.position.y ?? updated.position.y,
                    };
                  }
                  if (mut.updateNode!.data) {
                    // Manual merge to keep internal types consistent
                    updated.data = {
                      ...updated.data,
                      ...(mut.updateNode!.data as unknown as DynamicNodeData),
                    };
                  }
                  return updated;
                }
                return n;
              });
            } else if (mut.removeNode) {
              newNodes = newNodes.filter((n) => n.id !== mut.removeNode!.id);
              newEdges = newEdges.filter(
                (e) =>
                  e.source !== mut.removeNode!.id &&
                  e.target !== mut.removeNode!.id,
              );
            } else if (mut.addEdge?.edge) {
              newEdges.push(mut.addEdge.edge as unknown as Edge);
            } else if (mut.removeEdge) {
              newEdges = newEdges.filter((e) => e.id !== mut.removeEdge!.id);
            } else if (mut.addSubgraph) {
              if (mut.addSubgraph.nodes)
                newNodes.push(
                  ...(mut.addSubgraph.nodes as unknown as AppNode[]),
                );
              if (mut.addSubgraph.edges)
                newEdges.push(...(mut.addSubgraph.edges as unknown as Edge[]));
            } else if (mut.clearGraph) {
              newNodes = [];
              newEdges = [];
            }
          });

          return {
            nodes: newNodes,
            edges: newEdges,
            version: state.version + 1,
          };
        });
      },

      onNodesChange: (changes: NodeChange[]) => {
        const isDragging = changes.some(
          (c) =>
            c.type === "position" && (c as { dragging?: boolean }).dragging,
        );
        set((state) => ({
          nodes: applyNodeChanges(changes, state.nodes) as AppNode[],
          version: isDragging ? state.version : state.version + 1,
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
          isDynamicNode(sourceNode) &&
          isDynamicNode(targetNode)
        ) {
          const sourcePort = (
            sourceNode.data as DynamicNodeData
          ).outputPorts?.find((p) => p.id === connection.sourceHandle);
          let targetPort = (
            targetNode.data as DynamicNodeData
          ).inputPorts?.find((p) => p.id === connection.targetHandle);
          if (!targetPort) {
            const widget = (targetNode.data as DynamicNodeData).widgets?.find(
              (w) => w.inputPortId === connection.targetHandle,
            );
            if (widget) {
              targetPort = (
                targetNode.data as DynamicNodeData
              ).inputPorts?.find((p) => p.id === widget.inputPortId);
            }
          }

          if (sourcePort && targetPort) {
            const result = validateConnection(
              {
                ...(sourcePort as unknown as flowcraft.v1.IPort),
                nodeId: connection.source!,
              },
              {
                ...(targetPort as unknown as flowcraft.v1.IPort),
                nodeId: connection.target!,
              },
              edges,
            );

            if (result.canConnect) {
              const isDash =
                sourcePort.style === PortStyle.PORT_STYLE_DASH ||
                targetPort.style === PortStyle.PORT_STYLE_DASH;

              set({
                edges: addEdge(
                  {
                    ...connection,
                    id: `e${connection.source}-${connection.sourceHandle}-${connection.target}-${connection.targetHandle}`,
                    type: isDash ? "system" : undefined,
                  },
                  edges,
                ),
                version: version + 1,
              });
            } else {
              console.warn(result.reason);
            }
          }
        } else {
          set({ edges: addEdge(connection, edges), version: version + 1 });
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
      updateNodeData: (nodeId: string, data: Record<string, unknown>) => {
        set((state) => ({
          nodes: state.nodes.map((n) =>
            n.id === nodeId
              ? ({ ...n, data: { ...n.data, ...data } } as AppNode)
              : n,
          ),
          version: state.version + 1,
        }));
      },
      setVersion: (version: number) => {
        set({ version });
      },
      setGraph: (
        graph: { nodes: AppNode[]; edges: Edge[] },
        version: number,
      ) => {
        set({ nodes: graph.nodes, edges: graph.edges, version });
      },

      clipboard: null,
      setClipboard: (clipboard) => set({ clipboard }),

      connectionStartHandle: null,
      setConnectionStartHandle: (connectionStartHandle) =>
        set({ connectionStartHandle }),
    }),
    {
      partialize: (state) => {
        const { nodes, edges, version } = state;
        return { nodes, edges, version } as RFState;
      },
      equality: (a, b) => (a as RFState).version === (b as RFState).version,
    },
  ),
);

export const useFlowStore = useStore;

export function useTemporalStore<T>(
  selector: (state: TemporalState<any>) => T, // eslint-disable-line @typescript-eslint/no-explicit-any
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
