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
import { type MediaType, type PortType } from "../generated/core/node_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
} from "../generated/core/service_pb";
import { type WidgetSignal } from "../generated/core/signals_pb";
import { create as createProto } from "@bufbuild/protobuf";
import * as Y from "yjs";
import { ydoc, yNodes, yEdges } from "./yjsInstance";
import { dehydrateNode } from "../utils/nodeUtils";
import { fromProtoNode } from "../utils/protoAdapter";
import { getValidator } from "../utils/portValidators";

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
    mediaType: MediaType,
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
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void;
  applyYjsUpdate: (update: Uint8Array) => void;
  syncFromYjs: () => void;
  resetStore: () => void;

  // --- Widget Interaction Signals ---
  sendWidgetSignal: (signal: WidgetSignal) => void;
  handleIncomingWidgetSignal: (signal: WidgetSignal) => void;
}

const widgetSignalListeners = new Map<string, (signal: WidgetSignal) => void>();

export const registerWidgetSignalListener = (
  nodeId: string,
  widgetId: string,
  callback: (signal: WidgetSignal) => void,
) => {
  const key = `${nodeId}-${widgetId}`;
  widgetSignalListeners.set(key, callback);
  return () => {
    widgetSignalListeners.delete(key);
  };
};

const useStore = create(
  temporal<RFState>(
    (set, get) => {
      // Setup observers for granular updates
      yNodes.observe((event) => {
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
        lastNodeEvent: null,

        dispatchNodeEvent: (type, payload) => {
          set({ lastNodeEvent: { type, payload, timestamp: Date.now() } });
        },

        syncFromYjs: () => {
          const rawNodes: AppNode[] = [];
          yNodes.forEach((v) => rawNodes.push(v as AppNode));
          const nodes: AppNode[] = [];
          const visited = new Set<string>();

          const visit = (node: AppNode) => {
            if (visited.has(node.id)) return;

            // If this node has a parent that we haven't visited yet, visit the parent first
            if (node.parentId && !visited.has(node.parentId)) {
              const parent = rawNodes.find((n) => n.id === node.parentId);
              if (parent) visit(parent);
            }

            visited.add(node.id);
            nodes.push(node);
          };

          rawNodes.forEach((n) => {
            visit(n);
          });

          const edges: Edge[] = [];
          yEdges.forEach((v) => edges.push(v as Edge));
          set({ nodes, edges, version: get().version + 1 });
        },

        applyYjsUpdate: (update) => {
          Y.applyUpdate(ydoc, update, "remote");
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
          get().dispatchNodeEvent("mutations-applied", { mutations, context });

          ydoc.transact(() => {
            mutations.forEach((mutInput) => {
              const mut = createProto(GraphMutationSchema, mutInput);

              const op = mut.operation;

              if (!op.case) return;

              switch (op.case) {
                case "addNode":
                  if (op.value.node) {
                    const node = fromProtoNode(op.value.node);

                    if (node.id) {
                      yNodes.set(node.id, dehydrateNode(node));
                    }
                  }

                  break;

                case "updateNode": {
                  const val = op.value;

                  const id = val.id;

                  if (!id) break;

                  const existing = yNodes.get(id) as AppNode | undefined;

                  if (existing) {
                    const updated = { ...existing } as AppNode;

                    if (val.position) {
                      updated.position = {
                        x: val.position.x || updated.position.x,

                        y: val.position.y || updated.position.y,
                      };
                    }

                    if (val.width !== 0 || val.height !== 0) {
                      const newWidth =
                        val.width ||
                        (updated.measured ? updated.measured.width : 0);

                      const newHeight =
                        val.height ||
                        (updated.measured ? updated.measured.height : 0);

                      updated.measured = {
                        width: newWidth,

                        height: newHeight,
                      };

                      updated.style = {
                        ...updated.style,

                        width: updated.measured.width,

                        height: updated.measured.height,
                      };
                    }

                    if (val.data) {
                      updated.data = {
                        ...updated.data,

                        ...(val.data as Record<string, unknown>),
                      };
                    }

                    const pId = val.parentId;

                    updated.parentId =
                      (pId === "" ? undefined : pId) ?? undefined;

                    // Usually when parentId is set, we want the node to be contained within parent

                    updated.extent = updated.parentId ? "parent" : undefined;

                    // Preserve extent if it was already set and not explicitly removed

                    if (updated.parentId && !updated.extent) {
                      updated.extent = "parent";
                    }

                    yNodes.set(id, dehydrateNode(updated));
                  }

                  break;
                }

                case "removeNode":
                  if (op.value.id) {
                    yNodes.delete(op.value.id);
                  }

                  break;

                case "addEdge":
                  if (op.value.edge) {
                    const edge = op.value.edge as Edge;

                    yEdges.set(edge.id, edge);
                  }

                  break;

                case "removeEdge":
                  if (op.value.id) {
                    yEdges.delete(op.value.id);
                  }

                  break;

                case "addSubgraph":
                  op.value.nodes.forEach((n) => {
                    const node = fromProtoNode(n);

                    if (node.id) yNodes.set(node.id, dehydrateNode(node));
                  });

                  op.value.edges.forEach((e) => {
                    const edge = e as Edge;

                    if (edge.id) yEdges.set(edge.id, edge);
                  });

                  break;

                case "clearGraph":
                  yNodes.clear();

                  yEdges.clear();

                  break;
              }
            });
          }, "zustand-sync");

          // Since observer ignores "zustand-sync", we must sync Zustand manually

          get().syncFromYjs();
        },
        onNodesChange: (changes) => {
          const nextNodes = applyNodeChanges(changes, get().nodes) as AppNode[];

          // Immediate Zustand update for smooth UI
          set({ nodes: nextNodes });

          // Yjs sync should be careful not to spam.
          // We only sync to Yjs if the change is NOT a purely local position update during dragging,
          // OR we can rely on React Flow's drag events to sync at the end.
          // For now, let's avoid syncing 'position' changes to Yjs if they are part of an active drag.
          // React Flow 12 sets dragging: true on nodes being dragged.

          ydoc.transact(() => {
            changes.forEach((c) => {
              if (c.type === "remove") {
                yNodes.delete(c.id);
              } else if (c.type === "dimensions" || c.type === "select") {
                // Dimensions and selection are relatively infrequent, sync immediately
                const n = nextNodes.find((node) => node.id === c.id);
                if (n) {
                  if (c.type === "dimensions" && c.dimensions) {
                    n.measured = c.dimensions;
                    n.style = {
                      ...n.style,
                      width: c.dimensions.width,
                      height: c.dimensions.height,
                    };
                  }
                  yNodes.set(n.id, dehydrateNode(n));
                }
              } else if (c.type === "position" && !c.dragging) {
                // Only sync position if dragging has finished or it's an external move (like layout)
                const n = nextNodes.find((node) => node.id === c.id);
                if (n) yNodes.set(n.id, dehydrateNode(n));
              }
            });
          }, "zustand-sync");

          // Note: syncFromYjs is skipped here because we already updated Zustand state locally
          // and we want to avoid the expensive topological sort on every frame.
        },
        onEdgesChange: (changes) => {
          const nextEdges = applyEdgeChanges(changes, get().edges);
          set({ edges: nextEdges });
          ydoc.transact(() => {
            changes.forEach((c) => {
              if (c.type === "remove") yEdges.delete(c.id);
              else if (c.type === "select") {
                const e = nextEdges.find((edge) => edge.id === c.id);
                if (e) yEdges.set(e.id, e);
              }
            });
          }, "zustand-sync");
        },
        onConnect: (connection) => {
          const { nodes, edges } = get();
          const targetNode = nodes.find((n) => n.id === connection.target);
          let maxInputs = 999; // Default to multiple

          if (targetNode?.type === "dynamic") {
            const data = targetNode.data;
            const port =
              data.inputPorts?.find((p) => p.id === connection.targetHandle) ??
              data.widgets?.find(
                (w) => w.inputPortId === connection.targetHandle,
              );

            if (port) {
              const validator = getValidator(
                "type" in port ? (port.type as PortType) : undefined,
              );
              maxInputs = validator.getMaxInputs();
            }
          }

          const newEdge = {
            ...connection,
            id: `e${String(Date.now())}`,
          } as Edge;

          ydoc.transact(() => {
            // If it's a single input port, remove existing connections to this handle
            if (maxInputs === 1) {
              edges.forEach((e) => {
                if (
                  e.target === connection.target &&
                  e.targetHandle === connection.targetHandle
                ) {
                  yEdges.delete(e.id);
                }
              });
            }
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
          lastPastLength = 0;
          lastFutureLength = 0;
          set({ nodes: [], edges: [], version: 0 });
        },

        sendWidgetSignal: (signal) => {
          void import("../utils/SocketClient").then(({ socketClient }) => {
            void socketClient.send({
              payload: { case: "widgetSignal", value: signal },
            });
          });
        },

        handleIncomingWidgetSignal: (signal) => {
          const key = `${signal.nodeId}-${signal.widgetId}`;
          widgetSignalListeners.get(key)?.(signal);
        },
      };
    },
    {
      partialize: (state) => {
        const { nodes, edges, version } = state;
        return { nodes, edges, version };
      },
      equality: (a, b) => a.version === b.version,
      handleSet: (handleSet) => (state) => {
        // Skip recording history if we're in the middle of a drag
        const isDragging = (state as RFState).nodes.some((n) => n.dragging);
        if (isDragging) return;
        handleSet(state);
      },
    },
  ),
);

// Subscribe to temporal store to sync back to Yjs on undo/redo
let isSyncingFromTemporal = false;
let lastPastLength = 0;
let lastFutureLength = 0;

useStore.temporal.subscribe((state) => {
  // Only sync back to Yjs if an actual undo or redo happened.
  // We can detect this by checking if the lengths changed and the present state is different.
  const isUndo = state.pastStates.length < lastPastLength;
  const isRedo = state.futureStates.length < lastFutureLength;

  lastPastLength = state.pastStates.length;
  lastFutureLength = state.futureStates.length;

  if (isUndo || isRedo) {
    if (isSyncingFromTemporal) return;
    isSyncingFromTemporal = true;
    try {
      const current = useStore.getState();
      ydoc.transact(() => {
        yNodes.clear();
        yEdges.clear();
        // We dehydrate nodes before putting them back into Yjs
        current.nodes.forEach((n) => yNodes.set(n.id, dehydrateNode(n)));
        current.edges.forEach((e) => yEdges.set(e.id, e));
      }, "undo-redo");
    } finally {
      isSyncingFromTemporal = false;
    }
  }
});

import { useStoreWithEqualityFn } from "zustand/traditional";
import { type TemporalState } from "zundo";

export const useFlowStore = useStore;

export function useTemporalStore<T>(
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  return useStoreWithEqualityFn(useStore.temporal, selector, equality);
}