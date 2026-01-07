import { create, type JsonObject } from "@bufbuild/protobuf";

import {
  NodeDataSchema,
  NodeTemplateSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import { NodeRegistry } from "../registry";

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "AI Result",
      widgetsValues: {
        agent_name: "Assistant",
        logs: "Waiting for content...",
      } as unknown as JsonObject,
    }),
    displayName: "Real-time Stream Viewer",
    menuPath: ["AI"],
    templateId: "flowcraft.node.utility.stream_viewer",
  }),
});
