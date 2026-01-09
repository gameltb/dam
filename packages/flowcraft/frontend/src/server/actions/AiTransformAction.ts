import { create } from "@bufbuild/protobuf";

import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
} from "@/generated/flowcraft/v1/core/action_pb";

import {
  type ActionHandlerContext,
  NodeRegistry,
} from "../services/NodeRegistry";

NodeRegistry.registerGlobalAction(
  create(ActionTemplateSchema, {
    id: "flowcraft.action.node.transform",
    label: "Transform Node",
    strategy: ActionExecutionStrategy.EXECUTION_IMMEDIATE,
  }),
  async (ctx: ActionHandlerContext) => {
    const { taskId } = ctx;
    console.log(`[AiTransformAction] Executing action for task ${taskId}`);
    await Promise.resolve();
  },
);
