import { create as createProto } from "@bufbuild/protobuf";
import { applyEdgeChanges, applyNodeChanges } from "@xyflow/react";
import * as Y from "yjs";
import { temporal, type TemporalState } from "zundo";
import { create } from "zustand";

import { MutationSource as ProtoSource } from "@/generated/flowcraft/v1/core/base_pb";
import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type PortType } from "@/generated/flowcraft/v1/core/node_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type RFState } from "@/store/types";
import { type AppNode, type DynamicNodeData, MutationSource } from "@/types";
import { dehydrateNode, findPort } from "@/utils/nodeUtils";
import { getValidator } from "@/utils/portValidators";
import { toProtoNode, toProtoNodeData } from "@/utils/protoAdapter";

import { pipeline } from "./middleware/pipeline";
import { MutationDirection } from "./middleware/types";
import { handleGraphMutation } from "./mutationHandlers";
import { getWidgetSignalListener } from "./signalHandlers";
import { syncFromYjs } from "./utils";
import { ydoc, yEdges, yNodes } from "./yjsInstance";

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
          set({ edges: [], isLayoutDirty: false, nodes: [], version: 0 });
        },
        sendNodeSignal: (signal) => {
          void import("@/utils/SocketClient").then(({ socketClient }) => {
            void socketClient.send({
              payload: { case: "nodeSignal", value: signal },
            });
          });
        },
        sendWidgetSignal: (signal) => {
          void import("@/utils/SocketClient").then(({ socketClient }) => {
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
          const newState = syncFromYjs(get());
          set(newState);
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

import {
  setupTemporalSync,
  useTemporalStore as useTemporalStoreInternal,
} from "./temporalSync";

export const useFlowStore = useStore;

setupTemporalSync(useStore);

export function useTemporalStore<T>(
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  return useTemporalStoreInternal(useStore, selector, equality);
}
