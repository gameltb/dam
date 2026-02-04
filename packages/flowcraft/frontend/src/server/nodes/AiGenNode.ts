import { create } from "@bufbuild/protobuf";

import { NodeDataSchema, NodeTemplateSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { AiGenNodeStateSchema } from "@/generated/flowcraft/v1/nodes/ai_gen_node_pb";

import { NodeRegistry } from "../services/NodeRegistry";

NodeRegistry.register({
  execute: async () => {
    // Execution logic for AI Generator node
  },
  schema: AiGenNodeStateSchema,
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      displayName: "AI Generator",
      extension: {
        case: "aiGen",
        value: {
          currentStatus: "Idle",
          modelId: "default",
          progress: 0,
        },
      },
    }),
    displayName: "AI Image Generator",
    menuPath: ["AI"],
  }),
});
