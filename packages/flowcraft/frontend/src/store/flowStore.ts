import { create } from "zustand";
import { temporal, type TemporalState } from "zundo";
import {
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import { type AppNode, MutationSource, FlowEvent } from "../types";
import {
  type MediaType,
  type PortType,
} from "../generated/flowcraft/v1/core/node_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
} from "../generated/flowcraft/v1/core/service_pb";
import { type WidgetSignal } from "../generated/flowcraft/v1/core/signals_pb";
import * as Y from "yjs";
import { ydoc, yNodes, yEdges } from "./yjsInstance";
import { dehydrateNode, findPort } from "../utils/nodeUtils";
import { getValidator } from "../utils/portValidators";
import { handleGraphMutation } from "./mutationHandlers";
import { getWidgetSignalListener } from "./signalHandlers";
import { pipeline } from "./middleware/pipeline";
import { MutationDirection } from "./middleware/types";
import { MutationSource as ProtoSource } from "../generated/flowcraft/v1/core/base_pb";
import { PresentationSchema } from "../generated/flowcraft/v1/core/base_pb";
import { create as createProto } from "@bufbuild/protobuf";
import { toProtoNodeData, toProtoNode } from "../utils/protoAdapter";
import { type DynamicNodeData } from "../types";

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
  isLayoutDirty: boolean; // Flag to indicate if topological sort is needed

  ydoc: Y.Doc;
  yNodes: Y.Map<unknown>;
  yEdges: Y.Map<unknown>;

  lastNodeEvent: {
    type: FlowEvent;
    payload: Record<string, unknown>;
    timestamp: number;
  } | null;
  dispatchNodeEvent: (
    type: FlowEvent,
    payload: Record<string, unknown>,
  ) => void;

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
        isLayoutDirty: false,
        ydoc: ydoc,
        yNodes: yNodes,
        yEdges: yEdges,
        lastNodeEvent: null,

        dispatchNodeEvent: (type, payload) => {
          set({ lastNodeEvent: { type, payload, timestamp: Date.now() } });
        },

        syncFromYjs: () => {
          const state = get();
          const rawNodes: AppNode[] = [];
          yNodes.forEach((v) => rawNodes.push(v as AppNode));

          const edges: Edge[] = [];
          yEdges.forEach((v) => edges.push(v as Edge));

          // If layout is not dirty, we can just update nodes without sorting
          if (!state.isLayoutDirty) {
            set({ nodes: rawNodes, edges, version: state.version + 1 });
            return;
          }

          const nodes: AppNode[] = [];
          const visited = new Set<string>();

          const visit = (node: AppNode) => {
            if (visited.has(node.id)) return;

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

          set({
            nodes,
            edges,
            isLayoutDirty: false,
            version: state.version + 1,
          });
        },

        applyYjsUpdate: (update) => {
          // Updates from remote might change parent/child relationships
          set({ isLayoutDirty: true });
          Y.applyUpdate(ydoc, update, "remote");
          // syncFromYjs will be called by the observers
        },

        updateNodeData: (id, data) => {
          const node = get().nodes.find((n) => n.id === id);
          if (node) {
            const updatedData: DynamicNodeData = {
              ...(node.data as DynamicNodeData),
              ...data,
            };
            get().applyMutations([
              createProto(GraphMutationSchema, {
                operation: {
                  case: "updateNode",
                  value: {
                    id: id,
                    data: toProtoNodeData(updatedData),
                  },
                },
              }),
            ]);
          }
        },

        applyMutations: (mutations, context) => {
          const source = context?.source ?? MutationSource.SOURCE_USER;

          let direction = MutationDirection.OUTGOING;
          if (source === ProtoSource.SOURCE_SYNC) {
            direction = MutationDirection.INCOMING;
          }

          pipeline.execute(
            { mutations, context: context ?? {}, direction },
            (finalEvent) => {
              set({ isLayoutDirty: true });
              ydoc.transact(() => {
                finalEvent.mutations.forEach((mutInput) => {
                  handleGraphMutation(mutInput, yNodes, yEdges);
                });
              }, "zustand-sync");
              get().syncFromYjs();
            },
          );
        },
        onNodesChange: (changes) => {
          const nextNodes = applyNodeChanges(changes, get().nodes) as AppNode[];
          set({ nodes: nextNodes });

          const mutations: GraphMutation[] = [];

          changes.forEach((c) => {
            if (c.type === "remove") {
              mutations.push(
                createProto(GraphMutationSchema, {
                  operation: { case: "removeNode", value: { id: c.id } },
                }),
              );
            } else if (c.type === "position" && !c.dragging) {
              const n = nextNodes.find((node) => node.id === c.id);
              if (n) {
                const presentation = createProto(PresentationSchema, {
                  position: { x: n.position.x, y: n.position.y },
                  width: n.measured?.width ?? 0,
                  height: n.measured?.height ?? 0,
                  parentId: n.parentId ?? "",
                  isInitialized: true,
                });

                mutations.push(
                  createProto(GraphMutationSchema, {
                    operation: {
                      case: "updateNode",
                      value: {
                        id: n.id,
                        presentation,
                        data: toProtoNodeData(n.data as DynamicNodeData),
                      },
                    },
                  }),
                );
              }
            }
          });

          if (mutations.length > 0) {
            get().applyMutations(mutations);
          }

          // Local-only state updates (like selection) that don't need sync
          ydoc.transact(() => {
            changes.forEach((c) => {
              if (c.type === "select") {
                const n = nextNodes.find((node) => node.id === c.id);
                if (n) {
                  yNodes.set(n.id, dehydrateNode(n));
                }
              }
            });
          }, "zustand-sync");
        },
        onEdgesChange: (changes) => {
          const nextEdges = applyEdgeChanges(changes, get().edges);
          set({ edges: nextEdges });

          const removals = changes.filter((c) => c.type === "remove") as {
            id: string;
            type: "remove";
          }[];
          if (removals.length > 0) {
            get().applyMutations(
              removals.map((r) =>
                createProto(GraphMutationSchema, {
                  operation: { case: "removeEdge", value: { id: r.id } },
                }),
              ),
            );
            return;
          }

          ydoc.transact(() => {
            changes.forEach((c) => {
              if (c.type === "select") {
                const e = nextEdges.find((edge) => edge.id === c.id);
                if (e) yEdges.set(e.id, e);
              }
            });
          }, "zustand-sync");
        },
        onConnect: (connection) => {
          const { nodes, edges } = get();
          const targetNode = nodes.find((n) => n.id === connection.target);
          let maxInputs = 999;

          if (targetNode) {
            const port = findPort(targetNode, connection.targetHandle ?? "");
            if (port) {
              const validator = getValidator(
                port.type as unknown as PortType | undefined,
              );
              maxInputs = validator.getMaxInputs();
            }
          }

          const mutations: GraphMutation[] = [];
          if (maxInputs === 1) {
            edges.forEach((e) => {
              if (
                e.target === connection.target &&
                e.targetHandle === connection.targetHandle
              ) {
                mutations.push(
                  createProto(GraphMutationSchema, {
                    operation: { case: "removeEdge", value: { id: e.id } },
                  }),
                );
              }
            });
          }

          const edgeId = `e${String(Date.now())}`;
          mutations.push(
            createProto(GraphMutationSchema, {
              operation: {
                case: "addEdge",
                value: {
                  edge: {
                    edgeId,
                    sourceNodeId: connection.source,
                    targetNodeId: connection.target,
                    sourceHandle: connection.sourceHandle ?? "",
                    targetHandle: connection.targetHandle ?? "",
                    metadata: {},
                  },
                },
              },
            }),
          );

          get().applyMutations(mutations, { description: "Connect handles" });
        },
        setGraph: (graph, version) => {
          set({ isLayoutDirty: true });
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
          get().applyMutations([
            createProto(GraphMutationSchema, {
              operation: {
                case: "addNode",
                value: { node: toProtoNode(node) },
              },
            }),
          ]);
        },
        resetStore: () => {
          get().applyMutations([
            createProto(GraphMutationSchema, {
              operation: { case: "clearGraph", value: {} },
            }),
          ]);
          lastPastLength = 0;
          lastFutureLength = 0;
          set({ nodes: [], edges: [], version: 0, isLayoutDirty: false });
        },

        sendWidgetSignal: (signal) => {
          void import("../utils/SocketClient").then(({ socketClient }) => {
            void socketClient.send({
              payload: { case: "widgetSignal", value: signal },
            });
          });
        },

        handleIncomingWidgetSignal: (signal) => {
          getWidgetSignalListener(signal.nodeId, signal.widgetId)?.(signal);
        },
      };
    },
    {
      partialize: (state) => {
        const { nodes, edges, version } = state;
        return { nodes, edges, version } as unknown as RFState;
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
      useStore.setState({ isLayoutDirty: true });
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

export const useFlowStore = useStore;

export function useTemporalStore<T>(
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  return useStoreWithEqualityFn(
    useStore.temporal,
    (state) => selector(state),
    equality,
  );
}
