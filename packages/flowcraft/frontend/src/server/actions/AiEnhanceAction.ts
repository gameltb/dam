import { create } from "@bufbuild/protobuf";

import { ActionExecutionStrategy, ActionTemplateSchema } from "@/generated/flowcraft/v1/core/action_pb";

import { type ActionHandlerContext, NodeRegistry } from "../services/NodeRegistry";

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "flowcraft.action.media.enhance",
    label: "Enhance Media",
    strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
  }),
  async (ctx: ActionHandlerContext) => {
    const { taskId } = ctx;
    console.log(`[AiEnhanceAction] Executing action for task ${taskId}`);
    await Promise.resolve();
  },
);
