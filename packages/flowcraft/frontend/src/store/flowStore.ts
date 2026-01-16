import { type JsonObject, create as createProto } from "@bufbuild/protobuf";
import { applyEdgeChanges, applyNodeChanges } from "@xyflow/react";
import { temporal, type TemporalState } from "zundo";
import { create } from "zustand";

import { MutationSource as ProtoSource } from "@/generated/flowcraft/v1/core/base_pb";
import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type PortType } from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphMutation, GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { type RFState } from "@/store/types";
import { type AppNode, type DynamicNodeData, MutationSource } from "@/types";
import { appNodeDataToProto, appNodeToProto } from "@/utils/nodeProtoUtils";
import { dehydrateNode, findPort } from "@/utils/nodeUtils";
import { getValidator } from "@/utils/portValidators";

import { pipeline } from "./middleware/pipeline";
import { MutationDirection } from "./middleware/types";
import { handleGraphMutation } from "./mutationHandlers";
import { getWidgetSignalListener } from "./signalHandlers";
import { dispatchToSpacetime } from "./spacetimeDispatcher";

const useStore = create(
  temporal<RFState>(
    (set, get) => {
      return {
        addNode: (node) => {
          get().applyMutations([
            createProto(GraphMutationSchema, {
              operation: {
                case: "addNode",
                value: { node: appNodeToProto(node) },
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

          pipeline.execute({ context: context ?? {}, direction, mutations }, (finalEvent) => {
            const { edges: currentEdges, nodes: currentNodes, spacetimeConn } = get();

            let nextNodes = currentNodes;
            let nextEdges = currentEdges;

            finalEvent.mutations.forEach((mutInput) => {
              const result = handleGraphMutation(mutInput, nextNodes, nextEdges);
              nextNodes = result.nodes;
              nextEdges = result.edges;

              // Sync to SpacetimeDB
              if (direction === MutationDirection.OUTGOING && spacetimeConn) {
                // Set implicit task context if a taskId is provided in the mutation
                if (mutInput.originTaskId) {
                  spacetimeConn.reducers.assignCurrentTask({
                    taskId: mutInput.originTaskId,
                  });
                }
                dispatchToSpacetime(spacetimeConn, mutInput);
              }
            });

            set({
              edges: nextEdges,
              isLayoutDirty: true,
              nodes: nextNodes,
            });
          });
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
              const validator = getValidator(port.type as unknown as PortType | undefined);
              maxInputs = validator.getMaxInputs();
            }
          }

          const mutations: GraphMutation[] = [];
          if (maxInputs === 1) {
            edges.forEach((e) => {
              if (e.target === connection.target && e.targetHandle === connection.targetHandle) {
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
          }
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
            } else if ((c.type === "position" && !c.dragging) || c.type === "dimensions") {
              const n = nextNodes.find((node) => node.id === c.id);
              if (n) {
                const presentation = createProto(PresentationSchema, {
                  height: n.measured?.height ?? Number(n.style?.height ?? 0),
                  isInitialized: true,
                  parentId: n.parentId ?? "",
                  position: { x: n.position.x, y: n.position.y },
                  width: n.measured?.width ?? Number(n.style?.width ?? 0),
                });

                const dynData = n.data as DynamicNodeData;
                mutations.push(
                  createProto(GraphMutationSchema, {
                    operation: {
                      case: "updateNode",
                      value: {
                        data: appNodeDataToProto(dynData),
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
        },

        resetStore: () => {
          get().applyMutations([
            createProto(GraphMutationSchema, {
              operation: { case: "clearGraph", value: {} },
            }),
          ]);
          set({ edges: [], isLayoutDirty: false, nodes: [] });
        },
        sendNodeSignal: (signal) => {
          const { spacetimeConn } = get();
          if (spacetimeConn) {
            spacetimeConn.pbreducers.sendNodeSignal({
              signal,
            });
          }
        },
        sendWidgetSignal: (signal) => {
          const { spacetimeConn } = get();
          if (spacetimeConn) {
            // Mapping widget signal to node signal with structured parameters.
            // Using google.protobuf.Struct to hold the payload and widgetId.
            spacetimeConn.pbreducers.sendNodeSignal({
              signal: createProto(NodeSignalSchema, {
                nodeId: signal.nodeId,
                payload: {
                  case: "parameters",
                  value: {
                    fields: {
                      payload: {
                        kind: {
                          case: "structValue",
                          value: signal.payload as JsonObject,
                        },
                      },
                      widgetId: {
                        kind: { case: "stringValue", value: signal.widgetId },
                      },
                    },
                  },
                },
              }),
            });
          }
        },
        setGraph: (graph) => {
          set({
            edges: graph.edges,
            isLayoutDirty: true,
            nodes: graph.nodes.map(dehydrateNode),
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
                    data: appNodeDataToProto(updatedData),
                    id: id,
                  },
                },
              }),
            ]);
          }
        },
      };
    },
    {
      equality: () => true, // version is gone, we don't rely on it for equality here anymore
      handleSet: (handleSet) => (state) => {
        // Skip recording history if we're in the middle of a drag
        const isDragging = (state as RFState).nodes.some((n) => n.dragging);
        if (isDragging) return;
        handleSet(state);
      },
      partialize: (state) => {
        const { edges, nodes } = state;
        return { edges, nodes } as unknown as RFState;
      },
    },
  ),
);

import { useTemporalStore as useTemporalStoreInternal } from "./temporalSync";

export const useFlowStore = useStore;

export function useTemporalStore<T>(
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  return useTemporalStoreInternal(useStore, selector, equality);
}
