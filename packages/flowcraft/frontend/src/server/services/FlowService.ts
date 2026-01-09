import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { type ServiceImpl } from "@connectrpc/connect";

import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import {
  FlowService,
  GraphMutationSchema,
  GraphSnapshotSchema,
  MutationListSchema,
  PathUpdate_UpdateType,
} from "@/generated/flowcraft/v1/core/service_pb";
import { isDynamicNode } from "@/types";
import { toProtoEdge, toProtoNode } from "@/utils/protoAdapter";

import { watchGraph } from "./GraphWatcher";
import { inferenceService } from "./InferenceService";
import { executeMutation } from "./MutationExecutor";
import { runAction, runNodeSignal } from "./NodeExecutor";
import { NodeRegistry } from "./NodeRegistry";
import {
  eventBus,
  getChatHistory,
  getMutations,
  incrementVersion,
  logMutation,
  serverGraph,
  serverVersion,
} from "./PersistenceService";

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

  clearChatHistory(req) {
    void import("./PersistenceService").then(({ clearChatHistory }) => {
      clearChatHistory(req.nodeId);
    });
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

  discoverInferenceConfig() {
    const config = inferenceService.getConfig();
    return {
      defaultEndpointId: config.defaultEndpointId,
      defaultModel: config.defaultModel,
      endpoints: config.endpoints.map((e) => ({
        id: e.id,
        models: e.models,
        name: e.name,
      })),
    };
  },

  discoverTemplates() {
    return { templates: NodeRegistry.getTemplates() };
  },

  executeAction(req) {
    const { actionId, contextNodeIds, params, sourceNodeId } = req;
    void runAction(actionId, sourceNodeId, params, contextNodeIds);
    return {};
  },

  getChatHistory(req) {
    const history = getChatHistory(req.headId);
    return {
      entries: history.map((m) => {
        const metadata = m.metadata as {
          attachments?: string[];
          modelId?: string;
        };
        return {
          id: m.id,
          metadata: {
            case: "chatMetadata",
            value: {
              attachmentUrls: metadata.attachments ?? [],
              modelId: metadata.modelId ?? "",
            },
          },
          parentId: m.parentId ?? "",
          parts: m.parts,
          role: m.role,
          siblingIds: m.siblingIds,
          timestamp: BigInt(m.timestamp),
          treeId: m.treeId,
        };
      }),
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

  sendNodeSignal(req) {
    console.log(
      `[FlowService] Received NodeSignal for node: ${req.nodeId}, payload case: ${req.payload.case ?? "unknown"}`,
    );
    eventBus.emit("nodeSignal", req);
    void runNodeSignal(req.nodeId, req.payload);
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

  watchGraph,
};
