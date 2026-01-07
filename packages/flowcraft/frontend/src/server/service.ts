import type { ServiceImpl } from "@connectrpc/connect";

import {
  create,
  fromBinary,
  type JsonObject,
  toBinary,
} from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { MutationSource } from "../generated/flowcraft/v1/core/base_pb";
import { type TaskUpdate } from "../generated/flowcraft/v1/core/node_pb";
import {
  type FlowMessage,
  FlowMessageSchema,
  FlowService,
  GraphMutationSchema,
  type GraphSnapshot,
  GraphSnapshotSchema,
  type MutationList,
  MutationListSchema,
  type NodeEvent,
  NodeEventSchema,
  PathUpdate_UpdateType,
} from "../generated/flowcraft/v1/core/service_pb";
import { type WidgetSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { isDynamicNode } from "../types";
import { toProtoEdge, toProtoNode } from "../utils/protoAdapter";
import {
  eventBus,
  getChatHistory,
  getMutations,
  incrementVersion,
  logMutation,
  serverGraph,
  serverVersion,
} from "./db";
import { executeMutation } from "./mutationExecutor";
import { NodeRegistry } from "./registry";

export const FlowServiceImpl: ServiceImpl<typeof FlowService> = {
  applyMutations(req) {
    const { mutations, source } = req;
    mutations.forEach((mut) => {
      try {
        logMutation(
          mut.operation.case ?? "unknown",
          toBinary(GraphMutationSchema, mut),
          source,
        );
      } catch (e) {
        console.error("[Server] Log failed:", e);
      }
      executeMutation(mut, serverGraph);
    });

    incrementVersion();
    eventBus.emit(
      "mutations",
      create(MutationListSchema, {
        mutations,
        sequenceNumber: BigInt(serverVersion),
        source: source || MutationSource.SOURCE_USER,
      }),
    );
    return {};
  },

  cancelTask() {
    return {};
  },

  discoverActions(req) {
    const { nodeId } = req;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    let actions = NodeRegistry.getGlobalActions();
    if (node && isDynamicNode(node) && node.data.typeId) {
      actions = [
        ...actions,
        ...NodeRegistry.getActionsForNode(node.data.typeId),
      ];
    }
    return { actions };
  },

  discoverTemplates() {
    return { templates: NodeRegistry.getTemplates() };
  },

  executeAction(req) {
    const { actionId, contextNodeIds, params, sourceNodeId } = req;
    const handler = NodeRegistry.getActionHandler(actionId);
    const taskId = uuidv4();
    const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);

    const emitMutation = (m: MutationList) => {
      m.mutations.forEach((mut) => {
        executeMutation(mut, serverGraph);
      });
      incrementVersion();
      eventBus.emit("mutations", m);
    };

    const emitNodeEvent = (e: unknown) => {
      eventBus.emit("nodeEvent", e);
    };

    const emitWidgetStream = (
      widgetId: string,
      chunk: string,
      isDone = false,
    ) => {
      emitNodeEvent(
        create(NodeEventSchema, {
          nodeId: sourceNodeId,
          payload: {
            case: "widgetStream",
            value: {
              chunkData: chunk,
              isDone,
              widgetId,
            },
          },
        }),
      );
    };

    if (handler) {
      void handler({
        actionId,
        contextNodeIds,
        emitMutation,
        emitNodeEvent,
        emitTaskUpdate: (t: TaskUpdate) => eventBus.emit("taskUpdate", t),
        node,
        params,
        selectedNodeIds: [],
        sourceNodeId,
        taskId,
      });
    } else if (node && isDynamicNode(node)) {
      const executor = NodeRegistry.getExecutor(node.data.typeId ?? "");
      if (executor) {
        void executor({
          actionId,
          emitMutation,
          emitNodeEvent,
          emitTaskUpdate: (t: TaskUpdate) => eventBus.emit("taskUpdate", t),
          emitWidgetStream,
          node,
          params,
          taskId,
        });
      }
    }
    return {};
  },

  getChatHistory(req) {
    const history = getChatHistory(req.headId);
    return {
      entries: history.map((m) => ({
        content: m.content,
        id: m.id,
        metadata: {
          case: "metadataStruct",
          value: m.metadata as unknown as JsonObject,
        },
        role: m.role,
        timestamp: BigInt(m.timestamp),
      })),
    };
  },

  getHistory(req) {
    const { fromSeq, toSeq } = req;
    const dbMutations = getMutations(
      Number(fromSeq),
      toSeq ? Number(toSeq) : undefined,
    );
    return {
      entries: dbMutations.map((m) => ({
        description: m.description ?? "",
        mutation: fromBinary(GraphMutationSchema, m.payload),
        seq: BigInt(m.seq),
        source: m.source,
        timestamp: BigInt(m.timestamp),
        userId: m.user_id ?? "",
      })),
    };
  },

  rollback(req) {
    const targetSeq = Number(req.targetSeq);
    serverGraph.nodes = [];
    serverGraph.edges = [];
    getMutations(0, targetSeq).forEach((h) => {
      executeMutation(fromBinary(GraphMutationSchema, h.payload), serverGraph);
    });
    incrementVersion();
    const snapshot = create(GraphSnapshotSchema, {
      edges: serverGraph.edges.map(toProtoEdge),
      nodes: serverGraph.nodes.map(toProtoNode),
      version: BigInt(serverVersion),
    });
    eventBus.emit("snapshot", snapshot);
    return {};
  },

  sendWidgetSignal(req) {
    eventBus.emit("widgetSignal", req);
    return {};
  },
  updateNode(req) {
    const { data, nodeId, presentation } = req;
    const mut = create(GraphMutationSchema, {
      operation: {
        case: "updateNode",
        value: { data, id: nodeId, presentation },
      },
    });
    executeMutation(mut, serverGraph);
    incrementVersion();
    eventBus.emit(
      "mutations",
      create(MutationListSchema, {
        mutations: [mut],
        sequenceNumber: BigInt(serverVersion),
        source: MutationSource.SOURCE_USER,
      }),
    );
    return {};
  },

  updateViewport() {
    return {};
  },

  updateWidget(req) {
    const { nodeId, value, widgetId } = req;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && isDynamicNode(node) && node.data.widgets) {
      const widget = node.data.widgets.find((w) => w.id === widgetId);
      if (widget && value) {
        widget.value = value;
        incrementVersion();
        eventBus.emit(
          "mutations",
          create(MutationListSchema, {
            mutations: [
              create(GraphMutationSchema, {
                operation: {
                  case: "pathUpdate",
                  value: {
                    path: `data.widgetsValues.${widgetId}`,
                    targetId: nodeId,
                    type: PathUpdate_UpdateType.REPLACE,
                    value: value,
                  },
                },
              }),
            ],
            sequenceNumber: BigInt(serverVersion),
            source: MutationSource.SOURCE_USER,
          }),
        );
      }
    }
    return {};
  },

  async *watchGraph(_req, ctx) {
    console.log("[Server] Client connected");

    try {
      const snapshot = create(FlowMessageSchema, {
        messageId: uuidv4(),
        payload: {
          case: "snapshot",
          value: create(GraphSnapshotSchema, {
            edges: serverGraph.edges.map(toProtoEdge),
            nodes: serverGraph.nodes.map(toProtoNode),
            version: BigInt(serverVersion),
          }),
        },
        timestamp: BigInt(Date.now()),
      });
      yield snapshot;
    } catch (err) {
      console.error("[Server] Error generating snapshot:", err);
    }

    const queue: FlowMessage[] = [];
    let waitingPromise: null | { promise: Promise<void>; resolve: () => void } =
      null;

    const pushToQueue = <K extends NonNullable<FlowMessage["payload"]>["case"]>(
      type: K,
      payload: Extract<
        NonNullable<FlowMessage["payload"]>,
        { case: K }
      >["value"],
    ) => {
      queue.push(
        create(FlowMessageSchema, {
          messageId: uuidv4(),
          payload: { case: type, value: payload } as NonNullable<
            FlowMessage["payload"]
          >,
          timestamp: BigInt(Date.now()),
        }),
      );
      if (waitingPromise) {
        waitingPromise.resolve();
        waitingPromise = null;
      }
    };

    const onMutations = (m: MutationList) => {
      pushToQueue("mutations", m);
    };
    const onTaskUpdate = (t: TaskUpdate) => {
      pushToQueue("taskUpdate", t);
    };
    const onNodeEvent = (e: NodeEvent) => {
      pushToQueue("nodeEvent", e);
    };
    const onWidgetSignal = (s: WidgetSignal) => {
      pushToQueue("widgetSignal", s);
    };
    const onSnapshot = (s: GraphSnapshot) => {
      pushToQueue("snapshot", s);
    };

    eventBus.on("mutations", onMutations);
    eventBus.on("taskUpdate", onTaskUpdate);
    eventBus.on("nodeEvent", onNodeEvent);
    eventBus.on("widgetSignal", onWidgetSignal);
    eventBus.on("snapshot", onSnapshot);

    try {
      while (!ctx.signal.aborted) {
        while (queue.length > 0) {
          const msg = queue.shift();
          if (msg) yield msg;
        }
        if (!waitingPromise) {
          let resolve: () => void = () => {
            /* empty */
          };
          const promise = new Promise<void>((r) => {
            resolve = r;
          });
          waitingPromise = { promise, resolve };
        }
        await Promise.race([
          waitingPromise.promise,
          new Promise((r) => setTimeout(r, 10000)),
        ]);
      }
    } finally {
      eventBus.off("mutations", onMutations);
      eventBus.off("taskUpdate", onTaskUpdate);
      eventBus.off("nodeEvent", onNodeEvent);
      eventBus.off("widgetSignal", onWidgetSignal);
      eventBus.off("snapshot", onSnapshot);
    }
  },
};
