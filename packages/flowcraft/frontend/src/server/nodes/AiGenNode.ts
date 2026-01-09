import { create } from "@bufbuild/protobuf";

import {
  NodeDataSchema,
  NodeTemplateSchema,
  RenderMode,
} from "@/generated/flowcraft/v1/core/node_pb";

import {
  type NodeExecutionContext,
  NodeRegistry,
} from "../services/NodeRegistry";

NodeRegistry.register({
  execute: async (ctx: NodeExecutionContext) => {
    const { emitWidgetStream, taskId } = ctx;
    console.log(`[AiGenNode] Executing task ${taskId}`);

    // Mock generation for now
    emitWidgetStream("prompt", "Generating...", false);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    emitWidgetStream("prompt", "Done!", true);
  },
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_WIDGETS,
      availableModes: [RenderMode.MODE_WIDGETS],
      displayName: "AI Generator",
    }),
    displayName: "AI Image Generator",
    menuPath: ["AI"],
    templateId: "flowcraft.node.ai.generator",
  }),
});
