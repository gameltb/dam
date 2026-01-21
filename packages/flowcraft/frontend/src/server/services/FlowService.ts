import { toBinary } from "@bufbuild/protobuf";
import { type ServiceImpl } from "@connectrpc/connect";

import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import { FlowService, GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";

import { getChatHistory } from "./ChatService";
import { watchGraph } from "./GraphWatcher";
import { inferenceService } from "./InferenceService";
import { executeMutation } from "./MutationExecutor";
import { runAction, runNodeSignal } from "./NodeExecutor";
import { NodeRegistry } from "./NodeRegistry";
import { eventBus, getMutations, incrementVersion, logMutation, serverGraph } from "./PersistenceService";

export const FlowServiceImpl: ServiceImpl<typeof FlowService> = {
  applyMutations(req) {
    req.mutations.forEach((mut) => {
      executeMutation(mut, serverGraph);
      incrementVersion();
      logMutation(mut.operation.case ?? "unknown", toBinary(GraphMutationSchema, mut), req.source);
    });

    eventBus.emit("mutations", req);
    return {};
  },

  async cancelTask(_req) {
    return {};
  },

  async clearChatHistory(_req) {
    return {};
  },

  async discoverActions(req) {
    const actions = NodeRegistry.getActionsForNode(req.nodeId);
    return { actions };
  },

  async discoverInferenceConfig() {
    return inferenceService.getConfig();
  },

  async discoverTemplates() {
    const templates = NodeRegistry.getTemplates();
    return { templates };
  },

  async executeAction(req) {
    if (req.params.case) {
      runAction(req.actionId, req.sourceNodeId, req.params.value as any);
    }
    return {};
  },

  async getChatHistory(req) {
    const entries = await getChatHistory(req.headId);
    return { entries };
  },

  async getHistory(req) {
    const rawEntries = getMutations(Number(req.fromSeq), Number(req.toSeq));
    const entries = rawEntries.map((e) => ({
      description: e.description || "",
      mutation: undefined, // Requires translation from binary payload
      seq: BigInt(e.seq),
      source: e.source as MutationSource,
      timestamp: BigInt(e.timestamp),
      userId: e.user_id || "",
    }));
    return { entries };
  },

  async rollback(_req) {
    return {};
  },

  async sendNodeSignal(req) {
    if (req.payload.case) {
      runNodeSignal(req.nodeId, req.payload);
    }
    return {};
  },

  async sendWidgetSignal(_req) {
    return {};
  },

  updateViewport() {
    return {};
  },

  watchGraph,
};
