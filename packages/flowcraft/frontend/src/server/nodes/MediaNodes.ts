import { create, type JsonObject } from "@bufbuild/protobuf";

import { PortMainType } from "../../generated/flowcraft/v1/core/base_pb";
import {
  NodeDataSchema,
  NodeTemplateSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import { NodeRegistry } from "../registry";

// 1. 图片节点
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Visual Media",
      extension: {
        case: "visual",
        value: { altText: "New Image", mimeType: "image/jpeg", url: "" },
      },
      inputPorts: [
        { id: "in", label: "Source", type: { mainType: PortMainType.IMAGE } },
      ],
      outputPorts: [
        { id: "out", label: "Result", type: { mainType: PortMainType.IMAGE } },
      ],
      widgetsValues: {
        content: "",
        mimeType: "image/jpeg",
        url: "",
      } as unknown as JsonObject,
    }),
    displayName: "Visual Media",
    menuPath: ["Media"],
    templateId: "flowcraft.node.media.visual",
  }),
});

// 2. 视频节点 (暂用通用媒体)
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Motion Media",
      inputPorts: [
        { id: "in", label: "Source", type: { mainType: PortMainType.VIDEO } },
      ],
      outputPorts: [
        { id: "out", label: "Result", type: { mainType: PortMainType.VIDEO } },
      ],
      widgetsValues: {
        content: "",
        mimeType: "video/mp4",
        url: "",
      } as unknown as JsonObject,
    }),
    displayName: "Motion Media",
    menuPath: ["Media"],
    templateId: "flowcraft.node.media.motion",
  }),
});

// 3. 音频节点
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Acoustic Media",
      extension: {
        case: "acoustic",
        value: { duration: 0, url: "" },
      },
      inputPorts: [
        { id: "in", label: "Source", type: { mainType: PortMainType.AUDIO } },
      ],
      outputPorts: [
        { id: "out", label: "Result", type: { mainType: PortMainType.AUDIO } },
      ],
      widgetsValues: {
        content: "",
        mimeType: "audio/mpeg",
        url: "",
      } as unknown as JsonObject,
    }),
    displayName: "Acoustic Media",
    menuPath: ["Media"],
    templateId: "flowcraft.node.media.acoustic",
  }),
});

// 4. Markdown 节点
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Notes",
      extension: {
        case: "document",
        value: {
          content: "# New Markdown\nType here...",
          mimeType: "text/markdown",
        },
      },
      widgetsValues: {
        content: "# New Markdown\nType here...",
        mimeType: "text/markdown",
      } as unknown as JsonObject,
    }),
    displayName: "Structured Document",
    menuPath: ["Media"],
    templateId: "flowcraft.node.media.document",
  }),
});
