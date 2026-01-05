import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  NodeDataSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import { PortMainType } from "../../generated/flowcraft/v1/core/base_pb";
import { NodeRegistry } from "../registry";

// 通用媒体配置工厂
const createMediaTemplate = (
  id: string,
  name: string,
  mainType: PortMainType,
) => {
  return {
    template: create(NodeTemplateSchema, {
      templateId: id,
      displayName: name,
      menuPath: ["Media"],
      defaultState: create(NodeDataSchema, {
        displayName: name,
        availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
        activeMode: RenderMode.MODE_MEDIA,
        inputPorts: [
          { id: "in", label: "Source", type: { mainType: mainType as any } },
        ],
        outputPorts: [
          { id: "out", label: "Result", type: { mainType: mainType as any } },
        ],
        widgetsValuesJson: JSON.stringify({
          url: "",
          mimeType: "",
          content: "",
        }),
      }),
    }),
  };
};

// 1. 图片节点
NodeRegistry.register(
  createMediaTemplate("tpl-media-image", "Image", PortMainType.IMAGE),
);

// 2. 视频节点
NodeRegistry.register(
  createMediaTemplate("tpl-media-video", "Video", PortMainType.VIDEO),
);

// 3. 音频节点
NodeRegistry.register(
  createMediaTemplate("tpl-media-audio", "Audio", PortMainType.AUDIO),
);

// 4. Markdown 节点
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-media-md",
    displayName: "Markdown",
    menuPath: ["Media"],
    defaultState: create(NodeDataSchema, {
      displayName: "Notes",
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_MEDIA,
      widgetsValuesJson: JSON.stringify({
        content: "# New Markdown\nType here...",
        mimeType: "text/markdown",
      }),
    }),
  }),
});
