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
import { type AppNode, MutationSource } from "../types";
import { flowcraft } from "../generated/flowcraft";
import { useTaskStore } from "./taskStore";
import * as Y from "yjs";

export interface MutationContext {
  taskId?: string;
  source?: MutationSource;
  description?: string;
}

export interface RFState {
  nodes: AppNode[];
  edges: Edge[];
  version: number;

  ydoc: Y.Doc;
  yNodes: Y.Map<unknown>;
  yEdges: Y.Map<unknown>;

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
  applyMutations: (
    mutations: flowcraft.v1.IGraphMutation[],
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

// Fixed references to ensure hooks never lose the context
const _ydoc = new Y.Doc();
const _yNodes = _ydoc.getMap("nodes");
const _yEdges = _ydoc.getMap("edges");

const useStore = create(
  temporal<RFState>(
    (set, get) => ({
      nodes: [],
      edges: [],
      version: 0,
      ydoc: _ydoc,
      yNodes: _yNodes,
      yEdges: _yEdges,
      lastNodeEvent: null,
      dispatchNodeEvent: (type, payload) => {
        set({ lastNodeEvent: { type, payload, timestamp: Date.now() } });
      },
      syncFromYjs: () => {
        const nodes: AppNode[] = [];
        _yNodes.forEach((v) => nodes.push(v as AppNode));
        const edges: Edge[] = [];
        _yEdges.forEach((v) => edges.push(v as Edge));
        set({ nodes, edges, version: get().version + 1 });
      },
      applyYjsUpdate: (update) => {
        Y.applyUpdate(_ydoc, update, "remote");
        get().syncFromYjs();
      },
      applyMutations: (mutations, context) => {
        const taskId = context?.taskId || "manual-interaction";
        if (!useTaskStore.getState().tasks[taskId]) {
          useTaskStore.getState().registerTask({
            taskId,
            label: context?.description || "Action",
            source: context?.source || MutationSource.USER,
          });
        }
        useTaskStore.getState().logMutation({
          taskId,
          source: context?.source || MutationSource.USER,
          description: context?.description || "Applied",
          mutations,
        });

        _ydoc.transact(() => {
          mutations.forEach((mut) => {
            if (mut.addNode?.node) {
              _yNodes.set(
                mut.addNode.node.id!,
                mut.addNode.node as flowcraft.v1.INode,
              );
            } else if (mut.updateNode) {
              const id = mut.updateNode.id!;
              const existing = _yNodes.get(id) as AppNode;
              if (existing) {
                const updated = { ...existing };
                if (mut.updateNode.position) {
                  updated.position = {
                    x: mut.updateNode.position.x ?? updated.position.x,
                    y: mut.updateNode.position.y ?? updated.position.y,
                  };
                }
                if (mut.updateNode.width || mut.updateNode.height) {
                  updated.measured = {
                    width: mut.updateNode.width ?? updated.measured?.width ?? 0,
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
                _yNodes.set(id, updated);
              }
            } else if (mut.removeNode) {
              _yNodes.delete(mut.removeNode.id!);
            } else if (mut.addEdge?.edge) {
              _yEdges.set(mut.addEdge.edge.id!, mut.addEdge.edge as Edge);
            } else if (mut.removeEdge) {
              _yEdges.delete(mut.removeEdge.id!);
            } else if (mut.addSubgraph) {
              mut.addSubgraph.nodes?.forEach((n) => {
                if (n.id) _yNodes.set(n.id, JSON.parse(JSON.stringify(n)));
              });
              mut.addSubgraph.edges?.forEach((e) => {
                if (e.id) _yEdges.set(e.id, JSON.parse(JSON.stringify(e)));
              });
            } else if (mut.clearGraph) {
              _yNodes.clear();
              _yEdges.clear();
            }
          });
        });
        get().syncFromYjs();
      },
      onNodesChange: (changes) => {
        const nextNodes = applyNodeChanges(changes, get().nodes) as AppNode[];
        _ydoc.transact(() => {
          changes.forEach((c) => {
            if (
              c.type === "position" ||
              c.type === "dimensions" ||
              c.type === "select"
            ) {
              const n = nextNodes.find((node) => node.id === c.id);
              if (n) _yNodes.set(n.id, n);
            } else if (c.type === "remove") _yNodes.delete(c.id);
          });
        });
        get().syncFromYjs();
      },
      onEdgesChange: (changes) => {
        const nextEdges = applyEdgeChanges(changes, get().edges);
        _ydoc.transact(() => {
          changes.forEach((c) => {
            if (c.type === "remove") _yEdges.delete(c.id);
            else if (c.type === "select") {
              const e = nextEdges.find((edge) => edge.id === c.id);
              if (e) _yEdges.set(e.id, e);
            }
          });
        });
        get().syncFromYjs();
      },
      onConnect: (connection) => {
        const newEdge = { ...connection, id: `e${Date.now()}` } as Edge;
        _yEdges.set(newEdge.id, newEdge);
        get().syncFromYjs();
      },
      setGraph: (graph, version) => {
        _ydoc.transact(() => {
          _yNodes.clear();
          _yEdges.clear();
          graph.nodes.forEach((n) => _yNodes.set(n.id, n));
          graph.edges.forEach((e) => _yEdges.set(e.id, e));
        });
        set({ version });
        get().syncFromYjs();
      },
      resetStore: () => {
        _ydoc.transact(() => {
          _yNodes.clear();
          _yEdges.clear();
        });
        set({ nodes: [], edges: [], version: 0 });
      },
      clipboard: null,
      setClipboard: (c) => set({ clipboard: c }),
      connectionStartHandle: null,
      setConnectionStartHandle: (h) => set({ connectionStartHandle: h }),
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

// Subscribe to temporal store to sync back to Yjs on undo/redo
useStore.temporal.subscribe(() => {
  // If we are performing an undo or redo (state is from the past/future)
  // We need to sync the restored Zustand state back to Yjs
  const current = useStore.getState();
  _ydoc.transact(() => {
    _yNodes.clear();
    _yEdges.clear();
    current.nodes.forEach((n) => _yNodes.set(n.id, n));
    current.edges.forEach((e) => _yEdges.set(e.id, e));
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
