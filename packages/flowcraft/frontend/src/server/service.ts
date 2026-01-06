import { create, toBinary, fromBinary } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";
import type { ServiceImpl } from "@connectrpc/connect";
import {
  FlowService,
  FlowMessageSchema,
  GraphSnapshotSchema,
  MutationListSchema,
  GraphMutationSchema,
  type FlowMessage,
  type MutationList,
  type StreamChunk,
  PathUpdate_UpdateType,
} from "../generated/flowcraft/v1/core/service_pb";
import { type TaskUpdate } from "../generated/flowcraft/v1/core/node_pb";
import { MutationSource } from "../generated/flowcraft/v1/core/base_pb";
import {
  getChatHistory,
  serverGraph,
  serverVersion,
  incrementVersion,
  eventBus,
  logMutation,
  getMutations,
} from "./db";
import { NodeRegistry } from "./registry";
import { toProtoNode, toProtoEdge } from "../utils/protoAdapter";
import { isDynamicNode } from "../types";
import { executeMutation } from "./mutationExecutor";

export const FlowServiceImpl: ServiceImpl<typeof FlowService> = {
  async *watchGraph(_req, ctx) {
    console.log("[Server] Client connected");

    try {
      const snapshot = create(FlowMessageSchema, {
        messageId: uuidv4(),
        timestamp: BigInt(Date.now()),
        payload: {
          case: "snapshot",
          value: create(GraphSnapshotSchema, {
            nodes: serverGraph.nodes.map(toProtoNode),
            edges: serverGraph.edges.map(toProtoEdge),
            version: BigInt(serverVersion),
          }),
        },
      });
      yield snapshot;
    } catch (err) {
      console.error("[Server] Error generating snapshot:", err);
    }

    const queue: FlowMessage[] = [];
    let waitingPromise: { resolve: () => void; promise: Promise<void> } | null =
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
          timestamp: BigInt(Date.now()),
          payload: { case: type, value: payload } as any,
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
    const onStreamChunk = (c: StreamChunk) => {
      pushToQueue("streamChunk", c);
    };
    const onWidgetSignal = (s: any) => {
      pushToQueue("widgetSignal", s);
    };
    const onSnapshot = (s: any) => {
      pushToQueue("snapshot", s);
    };

    eventBus.on("mutations", onMutations);
    eventBus.on("taskUpdate", onTaskUpdate);
    eventBus.on("streamChunk", onStreamChunk);
    eventBus.on("widgetSignal", onWidgetSignal);
    eventBus.on("snapshot", onSnapshot);

    try {
      while (!ctx.signal.aborted) {
        while (queue.length > 0) yield queue.shift()!;
        if (!waitingPromise) {
          let resolve: () => void = () => {};
          const promise = new Promise<void>((r) => {
            resolve = r;
          });
          waitingPromise = { resolve, promise };
        }
        await Promise.race([
          waitingPromise.promise,
          new Promise((r) => setTimeout(r, 10000)),
        ]);
      }
    } finally {
      eventBus.off("mutations", onMutations);
      eventBus.off("taskUpdate", onTaskUpdate);
      eventBus.off("streamChunk", onStreamChunk);
      eventBus.off("widgetSignal", onWidgetSignal);
      eventBus.off("snapshot", onSnapshot);
    }
  },

  async applyMutations(req) {
    const { mutations, source } = req;
    mutations.forEach((mut) => {
      try {
        logMutation(
          mut.operation.case || "unknown",
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

  async discoverTemplates() {
    return { templates: NodeRegistry.getTemplates() };
  },

  async discoverActions(req) {
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

  async executeAction(req) {
    const { actionId, sourceNodeId, contextNodeIds, paramsJson } = req;
    const handler = NodeRegistry.getActionHandler(actionId);
    const params = paramsJson ? JSON.parse(paramsJson) : {};
    const taskId = uuidv4();
    const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);

    const emitMutation = (m: MutationList) => {
      m.mutations.forEach((mut) => {
        executeMutation(mut, serverGraph);
      });
      incrementVersion();
      eventBus.emit("mutations", m);
    };

    if (handler) {
      void handler({
        actionId,
        sourceNodeId,
        node,
        contextNodeIds,
        selectedNodeIds: [],
        params,
        taskId,
        emitTaskUpdate: (t: TaskUpdate) => eventBus.emit("taskUpdate", t),
        emitMutation,
        emitStreamChunk: (chunk: StreamChunk) =>
          eventBus.emit("streamChunk", chunk),
      });
    } else if (node && isDynamicNode(node)) {
      const executor = NodeRegistry.getExecutor(node.data.typeId || "");
      if (executor) {
        void executor({
          actionId,
          node,
          params,
          taskId,
          emitTaskUpdate: (t: TaskUpdate) => eventBus.emit("taskUpdate", t),
          emitMutation,
          emitStreamChunk: (chunk: StreamChunk) =>
            eventBus.emit("streamChunk", chunk),
        });
      }
    }
    return {};
  },

  async updateWidget(req) {
    const { nodeId, widgetId, valueJson } = req;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && isDynamicNode(node) && node.data.widgets) {
      const widget = node.data.widgets.find((w) => w.id === widgetId);
      if (widget) {
        widget.value = JSON.parse(valueJson);
        incrementVersion();
        eventBus.emit(
          "mutations",
          create(MutationListSchema, {
            mutations: [
              create(GraphMutationSchema, {
                operation: {
                  case: "pathUpdate",
                  value: {
                    targetId: nodeId,
                    path: `data.widgetsValues.${widgetId}`,
                    valueJson,
                    type: PathUpdate_UpdateType.REPLACE,
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

  async updateNode(req) {
    const { nodeId, data, presentation } = req;
    const mut = create(GraphMutationSchema, {
      operation: {
        case: "updateNode",
        value: { id: nodeId, data, presentation },
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

  async sendWidgetSignal(req) {
    eventBus.emit("widgetSignal", req);
    return {};
  },

  async updateViewport() {
    return {};
  },
  async cancelTask() {
    return {};
  },

  async getHistory(req) {
    const { fromSeq, toSeq } = req;
    const dbMutations = getMutations(
      Number(fromSeq),
      toSeq ? Number(toSeq) : undefined,
    );
    return {
      entries: dbMutations.map((m) => ({
        seq: BigInt(m.seq),
        mutation: fromBinary(GraphMutationSchema, m.payload),
        timestamp: BigInt(m.timestamp),
        source: m.source,
        description: m.description || "",
        userId: m.user_id || "",
      })),
    };
  },

  async rollback(req) {
    const targetSeq = Number(req.targetSeq);
    serverGraph.nodes = [];
    serverGraph.edges = [];
    getMutations(0, targetSeq).forEach((h) => {
      executeMutation(fromBinary(GraphMutationSchema, h.payload), serverGraph);
    });
    incrementVersion();
    const snapshot = create(GraphSnapshotSchema, {
      nodes: serverGraph.nodes.map(toProtoNode),
      edges: serverGraph.edges.map(toProtoEdge),
      version: BigInt(serverVersion),
    });
    eventBus.emit("snapshot", snapshot);
    return {};
  },

  async getChatHistory(req) {
    const history = getChatHistory(req.headId);
    return {
      entries: history.map((m: any) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        metadataJson: JSON.stringify(m.metadata),
        timestamp: BigInt(m.timestamp),
      })),
    };
  },
};
