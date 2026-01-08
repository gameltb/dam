import { create } from "@bufbuild/protobuf";

import {
  NodeDataSchema,
  NodeTemplateSchema,
  RenderMode,
} from "@/generated/flowcraft/v1/core/node_pb";
import { NodeRegistry } from "../services/NodeRegistry";

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Document",
    }),
    displayName: "Document Viewer",
    menuPath: ["Media"],
    templateId: "flowcraft.node.media.document",
  }),
});

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Visual",
    }),
    displayName: "Image Viewer",
    menuPath: ["Media"],
    templateId: "flowcraft.node.media.visual",
  }),
});
