import { create } from "@bufbuild/protobuf";

import { NodeDataSchema, NodeTemplateSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";

import { type NodeExecutionContext, NodeRegistry } from "../services/NodeRegistry";

NodeRegistry.register({
  execute: async (ctx: NodeExecutionContext) => {
    const { emitWidgetStream, taskId } = ctx;
    console.log(`[StreamNode] Executing task ${taskId}`);

    for (let i = 0; i < 10; i++) {
      emitWidgetStream("output", `Chunk ${i.toString()} `, false);
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    emitWidgetStream("output", "Done.", true);
  },
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_WIDGETS,
      availableModes: [RenderMode.MODE_WIDGETS],
      displayName: "Streamer",
    }),
    displayName: "Stream Demo Node",
    menuPath: ["Debug"],
    templateId: "flowcraft.node.debug.stream",
  }),
});
