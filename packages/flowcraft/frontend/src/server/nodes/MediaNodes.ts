import { create } from "@bufbuild/protobuf";

import { NodeDataSchema, NodeTemplateSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import {
  AcousticNodeStateSchema,
  DocumentNodeStateSchema,
  VisualNodeStateSchema,
} from "@/generated/flowcraft/v1/nodes/media_node_pb";

import { NodeRegistry } from "../services/NodeRegistry";

// 1. Document (Markdown)
NodeRegistry.register({
  schema: DocumentNodeStateSchema,
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      displayName: "Document",
      extension: {
        case: "document",
        value: { content: "# New Document\nStart typing..." },
      },
    }),
    displayName: "Markdown Document",
    menuPath: ["Media"],
  }),
});

// 2. Visual (Image/Video)
NodeRegistry.register({
  schema: VisualNodeStateSchema,
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      displayName: "Media",
      extension: {
        case: "visual",
        value: { mimeType: "image/png", url: "" },
      },
    }),
    displayName: "Visual Media",
    menuPath: ["Media"],
  }),
});

// 3. Acoustic (Audio)
NodeRegistry.register({
  schema: AcousticNodeStateSchema,
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      displayName: "Audio",
      extension: {
        case: "acoustic",
        value: { url: "" },
      },
    }),
    displayName: "Acoustic Node",
    menuPath: ["Media"],
  }),
});
