import { create } from "zustand";
import { temporal } from "zundo";
import {
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import { type AppNode, MutationSource, type DynamicNodeData } from "../types";
import { flowcraft_proto } from "../generated/flowcraft_proto";
import { useTaskStore } from "./taskStore";
import * as Y from "yjs";
import { ydoc, yNodes, yEdges } from "./yjsInstance";
import { hydrateNodes, dehydrateNode } from "../utils/nodeUtils";

export interface MutationContext {
  taskId?: string;
  source?: MutationSource;
  description?: string;
}

export interface NodeHandlers {
  onChange: (id: string, data: Record<string, unknown>) => void;
  onWidgetClick?: (nodeId: string, widgetId: string) => void;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    mediaType: flowcraft_proto.v1.MediaType,
    x: number,
    y: number,
  ) => void;
}

export interface RFState {
  nodes: AppNode[];
  edges: Edge[];
  version: number;

  ydoc: Y.Doc;
  yNodes: Y.Map<unknown>;
  yEdges: Y.Map<unknown>;

  // Handlers for hydration
  nodeHandlers: NodeHandlers | null;
  registerNodeHandlers: (handlers: NodeHandlers) => void;

  lastNodeEvent: {
    type: string;
    payload: Record<string, unknown>;
    timestamp: number;
  } | null;
  dispatchNodeEvent: (type: string, payload: Record<string, unknown>) => void;

  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;

  setGraph: (
    graph: { nodes: AppNode[]; edges: Edge[] },
    version: number,
  ) => void;
  addNode: (node: AppNode) => void;
  updateNodeData: (id: string, data: Partial<DynamicNodeData>) => void;

  applyMutations: (
    mutations: flowcraft_proto.v1.IGraphMutation[],
    context?: MutationContext,
  ) => void;
  applyYjsUpdate: (update: Uint8Array) => void;
  syncFromYjs: () => void;
  resetStore: () => void;

  clipboard: { nodes: AppNode[]; edges: Edge[] } | null;
  setClipboard: (data: { nodes: AppNode[]; edges: Edge[] } | null) => void;
  connectionStartHandle: {
    nodeId: string;
    handleId: string;
    type: "source" | "target";
  } | null;
  setConnectionStartHandle: (
    h: { nodeId: string; handleId: string; type: "source" | "target" } | null,
  ) => void;
}

const useStore = create(
  temporal<RFState>(
    (set, get) => {
      // Setup observers for granular updates
      yNodes.observe((event) => {
        // If the change came from our own local sync or an undo/redo,
        // we don't need to update Zustand again (it's already done or will be done)
        if (
          event.transaction.origin === "zustand-sync" ||
          event.transaction.origin === "undo-redo"
        )
          return;

        get().syncFromYjs();
      });

      yEdges.observe((event) => {
        if (
          event.transaction.origin === "zustand-sync" ||
          event.transaction.origin === "undo-redo"
        )
          return;

        get().syncFromYjs();
      });

      return {
        nodes: [],
        edges: [],
        version: 0,
        ydoc: ydoc,
        yNodes: yNodes,
        yEdges: yEdges,
        nodeHandlers: null,
        lastNodeEvent: null,

        registerNodeHandlers: (handlers) => {
          set({ nodeHandlers: handlers });
          // Trigger a re-sync to apply handlers to existing nodes
          get().syncFromYjs();
        },

        dispatchNodeEvent: (type, payload) => {
          set({ lastNodeEvent: { type, payload, timestamp: Date.now() } });
        },

        syncFromYjs: () => {
          const rawNodes: AppNode[] = [];
          yNodes.forEach((v) => rawNodes.push(v as AppNode));

          // Apply hydration if handlers are available
          const handlers = get().nodeHandlers;
          const nodes = handlers ? hydrateNodes(rawNodes, handlers) : rawNodes;

          const edges: Edge[] = [];
          yEdges.forEach((v) => edges.push(v as Edge));
          set({ nodes, edges, version: get().version + 1 });
        },

        applyYjsUpdate: (update) => {
          Y.applyUpdate(ydoc, update, "remote");
          // remote updates will be caught by the observer and update Zustand
        },

        updateNodeData: (id, data) => {
          ydoc.transact(() => {
            const existing = yNodes.get(id) as AppNode | undefined;
            if (existing) {
              const updated = {
                ...existing,
                data: { ...existing.data, ...data },
              } as AppNode;
              yNodes.set(id, dehydrateNode(updated));
            }
          }, "zustand-sync");
          get().syncFromYjs();
        },

        applyMutations: (mutations, context) => {
          const taskId = context?.taskId ?? "manual-interaction";
          if (!useTaskStore.getState().tasks[taskId]) {
            useTaskStore.getState().registerTask({
              taskId,
              label: context?.description ?? "Action",
              source: context?.source ?? MutationSource.USER,
            });
          }
          useTaskStore.getState().logMutation({
            taskId,
            source: context?.source ?? MutationSource.USER,
            description: context?.description ?? "Applied",
            mutations,
          });

          ydoc.transact(() => {
            mutations.forEach((mut) => {
              if (mut.addNode?.node) {
                const node = mut.addNode.node as AppNode;
                if (node.id) {
                  // Incoming mutations usually don't have handlers yet, but dehydrate just in case
                  yNodes.set(node.id, dehydrateNode(node));
                }
              } else if (mut.updateNode) {
                const id = mut.updateNode.id;
                if (!id) return;
                const existing = yNodes.get(id) as AppNode | undefined;
                if (existing) {
                  const updated = { ...existing } as AppNode;
                  if (mut.updateNode.position) {
                    updated.position = {
                      x: mut.updateNode.position.x ?? updated.position.x,
                      y: mut.updateNode.position.y ?? updated.position.y,
                    };
                  }
                  if (mut.updateNode.width || mut.updateNode.height) {
                    updated.measured = {
                      width:
                        mut.updateNode.width ?? updated.measured?.width ?? 0,
                      height:
                        mut.updateNode.height ?? updated.measured?.height ?? 0,
                    };
                  }
                  if (mut.updateNode.data) {
                    updated.data = {
                      ...updated.data,
                      ...(mut.updateNode.data as Record<string, unknown>),
                    };
                  }
                  yNodes.set(id, dehydrateNode(updated));
                }
              } else if (mut.removeNode?.id) {
                yNodes.delete(mut.removeNode.id);
              } else if (mut.addEdge?.edge) {
                const edge = mut.addEdge.edge as Edge;
                yEdges.set(edge.id, edge);
              } else if (mut.removeEdge?.id) {
                yEdges.delete(mut.removeEdge.id);
              } else if (mut.addSubgraph) {
                mut.addSubgraph.nodes?.forEach((n) => {
                  const node = n as AppNode;
                  if (node.id) yNodes.set(node.id, dehydrateNode(node));
                });
                mut.addSubgraph.edges?.forEach((e) => {
                  const edge = e as Edge;
                  if (edge.id) yEdges.set(edge.id, edge);
                });
              } else if (mut.clearGraph) {
                yNodes.clear();
                yEdges.clear();
              }
            });
          }, "zustand-sync");
          // Since observer ignores "zustand-sync", we must sync Zustand manually
          get().syncFromYjs();
        },
        onNodesChange: (changes) => {
          const nextNodes = applyNodeChanges(changes, get().nodes) as AppNode[];
          ydoc.transact(() => {
            changes.forEach((c) => {
              if (
                c.type === "position" ||
                c.type === "dimensions" ||
                c.type === "select"
              ) {
                const n = nextNodes.find((node) => node.id === c.id);
                if (n) yNodes.set(n.id, dehydrateNode(n));
              } else if (c.type === "remove") yNodes.delete(c.id);
            });
          }, "zustand-sync");
          get().syncFromYjs();
        },
        onEdgesChange: (changes) => {
          const nextEdges = applyEdgeChanges(changes, get().edges);
          ydoc.transact(() => {
            changes.forEach((c) => {
              if (c.type === "remove") yEdges.delete(c.id);
              else if (c.type === "select") {
                const e = nextEdges.find((edge) => edge.id === c.id);
                if (e) yEdges.set(e.id, e);
              }
            });
          }, "zustand-sync");
          get().syncFromYjs();
        },
        onConnect: (connection) => {
          const newEdge = {
            ...connection,
            id: `e${String(Date.now())}`,
          } as Edge;
          ydoc.transact(() => {
            yEdges.set(newEdge.id, newEdge);
          }, "zustand-sync");
          get().syncFromYjs();
        },
        setGraph: (graph, version) => {
          ydoc.transact(() => {
            yNodes.clear();
            yEdges.clear();
            graph.nodes.forEach((n) => yNodes.set(n.id, dehydrateNode(n)));
            graph.edges.forEach((e) => yEdges.set(e.id, e));
          }, "zustand-sync");
          set({ version });
          get().syncFromYjs();
        },
        addNode: (node) => {
          ydoc.transact(() => {
            yNodes.set(node.id, dehydrateNode(node));
          }, "zustand-sync");
          get().syncFromYjs();
        },
        resetStore: () => {
          ydoc.transact(() => {
            yNodes.clear();
            yEdges.clear();
          }, "zustand-sync");
          set({ nodes: [], edges: [], version: 0 });
        },
        clipboard: null,
        setClipboard: (c) => {
          set({ clipboard: c });
        },
        connectionStartHandle: null,
        setConnectionStartHandle: (h) => {
          set({ connectionStartHandle: h });
        },
      };
    },
    {
      partialize: (state) => {
        const { nodes, edges, version } = state;
        return { nodes, edges, version } as RFState;
      },
      equality: (a, b) => a.version === b.version,
    },
  ),
);

// Subscribe to temporal store to sync back to Yjs on undo/redo
useStore.temporal.subscribe(() => {
  // If we are performing an undo or redo (state is from the past/future)
  // We need to sync the restored Zustand state back to Yjs
  const current = useStore.getState();
  ydoc.transact(() => {
    yNodes.clear();
    yEdges.clear();
    // We dehydrate nodes before putting them back into Yjs
    current.nodes.forEach((n) => yNodes.set(n.id, dehydrateNode(n)));
    current.edges.forEach((e) => yEdges.set(e.id, e));
  }, "undo-redo");
});

import { useStoreWithEqualityFn } from "zustand/traditional";
import { type TemporalState } from "zundo";

export const useFlowStore = useStore;

export function useTemporalStore<T>(
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  const store = useStore.temporal;
  if (!store) {
    throw new Error("Temporal store not found.");
  }
  return useStoreWithEqualityFn(store, selector, equality);
}
