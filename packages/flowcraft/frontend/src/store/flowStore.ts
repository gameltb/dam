import { create as createProto } from "@bufbuild/protobuf";
import {
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";
import * as Y from "yjs";
import { temporal, type TemporalState } from "zundo";
import { create } from "zustand";

import { MutationSource as ProtoSource } from "../generated/flowcraft/v1/core/base_pb";
import { PresentationSchema } from "../generated/flowcraft/v1/core/base_pb";
import {
  type MediaType,
  type PortType,
} from "../generated/flowcraft/v1/core/node_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
} from "../generated/flowcraft/v1/core/service_pb";
import { type WidgetSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { type AppNode, FlowEvent, MutationSource } from "../types";
import { type DynamicNodeData } from "../types";
import { dehydrateNode, findPort } from "../utils/nodeUtils";
import { getValidator } from "../utils/portValidators";
import { toProtoNode, toProtoNodeData } from "../utils/protoAdapter";
import { pipeline } from "./middleware/pipeline";
import { MutationDirection } from "./middleware/types";
import { handleGraphMutation } from "./mutationHandlers";
import { getWidgetSignalListener } from "./signalHandlers";
import { ydoc, yEdges, yNodes } from "./yjsInstance";

export interface MutationContext {
  description?: string;
  source?: MutationSource;
  taskId?: string;
}

export interface NodeHandlers {
  onChange: (id: string, data: Record<string, unknown>) => void;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
    x: number,
    y: number,
  ) => void;
  onWidgetClick?: (nodeId: string, widgetId: string) => void;
}

export interface RFState {
  addNode: (node: AppNode) => void;
  applyMutations: (
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void;
  applyYjsUpdate: (update: Uint8Array) => void;
  dispatchNodeEvent: (
    type: FlowEvent,
    payload: Record<string, unknown>,
  ) => void;

  edges: Edge[];
  handleIncomingWidgetSignal: (signal: WidgetSignal) => void;
  isLayoutDirty: boolean; // Flag to indicate if topological sort is needed

  lastNodeEvent: null | {
    payload: Record<string, unknown>;
    timestamp: number;
    type: FlowEvent;
  };
  nodes: AppNode[];

  onConnect: (connection: Connection) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;

  resetStore: () => void;
  // --- Widget Interaction Signals ---
  sendWidgetSignal: (signal: WidgetSignal) => void;
  setGraph: (
    graph: { edges: Edge[]; nodes: AppNode[] },
    version: number,
  ) => void;

  syncFromYjs: () => void;
  updateNodeData: (id: string, data: Partial<DynamicNodeData>) => void;
  version: number;
  ydoc: Y.Doc;

  yEdges: Y.Map<unknown>;
  yNodes: Y.Map<unknown>;
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
        applyMutations: (mutations, context) => {
          const source = context?.source ?? MutationSource.SOURCE_USER;

          let direction = MutationDirection.OUTGOING;
          if (source === ProtoSource.SOURCE_SYNC) {
            direction = MutationDirection.INCOMING;
          }

          pipeline.execute(
            { context: context ?? {}, direction, mutations },
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
        applyYjsUpdate: (update) => {
          // Updates from remote might change parent/child relationships
          set({ isLayoutDirty: true });
          Y.applyUpdate(ydoc, update, "remote");
          // syncFromYjs will be called by the observers
        },
        dispatchNodeEvent: (type, payload) => {
          set({ lastNodeEvent: { payload, timestamp: Date.now(), type } });
        },
        edges: [],
        handleIncomingWidgetSignal: (signal) => {
          getWidgetSignalListener(signal.nodeId, signal.widgetId)?.(signal);
        },
        isLayoutDirty: false,
        lastNodeEvent: null,

        nodes: [],

        onConnect: (connection) => {
          const { edges, nodes } = get();
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
                    metadata: {},
                    sourceHandle: connection.sourceHandle ?? "",
                    sourceNodeId: connection.source,
                    targetHandle: connection.targetHandle ?? "",
                    targetNodeId: connection.target,
                  },
                },
              },
            }),
          );

          get().applyMutations(mutations, { description: "Connect handles" });
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
                  height: n.measured?.height ?? 0,
                  isInitialized: true,
                  parentId: n.parentId ?? "",
                  position: { x: n.position.x, y: n.position.y },
                  width: n.measured?.width ?? 0,
                });

                mutations.push(
                  createProto(GraphMutationSchema, {
                    operation: {
                      case: "updateNode",
                      value: {
                        data: toProtoNodeData(n.data as DynamicNodeData),
                        id: n.id,
                        presentation,
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

        resetStore: () => {
          get().applyMutations([
            createProto(GraphMutationSchema, {
              operation: { case: "clearGraph", value: {} },
            }),
          ]);
          lastPastLength = 0;
          lastFutureLength = 0;
          set({ edges: [], isLayoutDirty: false, nodes: [], version: 0 });
        },
        sendWidgetSignal: (signal) => {
          void import("../utils/SocketClient").then(({ socketClient }) => {
            void socketClient.send({
              payload: { case: "widgetSignal", value: signal },
            });
          });
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
        syncFromYjs: () => {
          const state = get();
          const rawNodes: AppNode[] = [];
          yNodes.forEach((v) => rawNodes.push(v as AppNode));

          const edges: Edge[] = [];
          yEdges.forEach((v) => edges.push(v as Edge));

          // If layout is not dirty, we can just update nodes without sorting
          if (!state.isLayoutDirty) {
            set({ edges, nodes: rawNodes, version: state.version + 1 });
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
            edges,
            isLayoutDirty: false,
            nodes,
            version: state.version + 1,
          });
        },
        updateNodeData: (id, data) => {
          const node = get().nodes.find((n) => n.id === id);
          if (node) {
            const updatedData = {
              ...node.data,
              ...data,
            } as DynamicNodeData;
            get().applyMutations([
              createProto(GraphMutationSchema, {
                operation: {
                  case: "updateNode",
                  value: {
                    data: toProtoNodeData(updatedData),
                    id: id,
                  },
                },
              }),
            ]);
          }
        },
        version: 0,
        ydoc: ydoc,

        yEdges: yEdges,

        yNodes: yNodes,
      };
    },
    {
      equality: (a, b) => a.version === b.version,
      handleSet: (handleSet) => (state) => {
        // Skip recording history if we're in the middle of a drag
        const isDragging = (state as RFState).nodes.some((n) => n.dragging);
        if (isDragging) return;
        handleSet(state);
      },
      partialize: (state) => {
        const { edges, nodes, version } = state;
        return { edges, nodes, version } as unknown as RFState;
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
