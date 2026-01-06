import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  NodeDataSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import { NodeRegistry } from "../registry";

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-stream-node",
    displayName: "AI Stream Result",
    menuPath: ["AI"],
    defaultState: create(NodeDataSchema, {
      displayName: "AI Result",
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_MEDIA,
      widgetsValuesJson: JSON.stringify({
        agent_name: "Assistant",
        logs: "Waiting for content...",
      }),
    }),
  }),
});
