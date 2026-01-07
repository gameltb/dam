import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { PortMainType } from "../generated/flowcraft/v1/core/base_pb";
import {
  MediaType,
  NodeTemplateSchema,
  PortStyle,
  RenderMode,
  WidgetType,
} from "../generated/flowcraft/v1/core/node_pb";
import {
  type AppNode,
  AppNodeType,
  type Edge,
  type NodeTemplate,
} from "../types";
import { fromProtoNodeData, toProtoNodeData } from "../utils/protoAdapter";

export const createNode = (
  id: string,
  label: string,
  x: number,
  y: number,
  typeId?: string,
  width = 300,
  height = 200,
): AppNode => {
  const data = fromProtoNodeData(
    toProtoNodeData({
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [],
      label,
      modes: [RenderMode.MODE_WIDGETS],
      outputPorts: [],
      typeId: typeId ?? "test-node",
      widgets: [],
    }),
  );

  return {
    data,
    id,
    measured: { height, width },
    position: { x, y },
    style: { height, width },
    type: AppNodeType.DYNAMIC,
  } as AppNode;
};

export const nodeTemplates: NodeTemplate[] = [
  create(NodeTemplateSchema, {
    defaultState: toProtoNodeData({
      activeMode: RenderMode.MODE_WIDGETS,
      label: "Widget Showcase",
      modes: [RenderMode.MODE_WIDGETS],
      widgets: [
        { id: "w1", label: "Text", type: WidgetType.WIDGET_TEXT, value: "" },
      ],
    }),
    displayName: "Widget Showcase",
    menuPath: ["Test"],
    templateId: "widgets-all",
  }),
];

export const generateGallery = () => {
  const nodes: AppNode[] = [];
  const edges: Edge[] = [];

  const startX = 50;
  const startY = 50;
  const colGap = 400;
  const rowGap = 600;

  // --- COLUMN 1: ALL WIDGET TYPES ---
  let currY = startY;
  const widgetTypes = [
    { label: "Text Field", type: WidgetType.WIDGET_TEXT, value: "Hello World" },
    {
      label: "Select",
      options: [
        { label: "Option 1", value: "opt1" },
        { label: "Option 2", value: "opt2" },
      ],
      type: WidgetType.WIDGET_SELECT,
      value: "opt1",
    },
    { label: "Checkbox", type: WidgetType.WIDGET_CHECKBOX, value: true },
    {
      config: { max: 100, min: 0 },
      label: "Slider",
      type: WidgetType.WIDGET_SLIDER,
      value: 50,
    },
    { label: "Button Action", type: WidgetType.WIDGET_BUTTON, value: "click" },
  ];

  const widgetNode = createNode(
    uuidv4(),
    "Widget Showcase",
    startX,
    currY,
    "widgets-all",
    320,
    450,
  );
  widgetNode.data.widgets = widgetTypes.map((w, i) => ({
    id: `w-${String(i)}`,
    ...w,
  }));
  nodes.push(widgetNode);

  // --- COLUMN 2: PORT STYLES ---
  currY = startY;
  const portStyleNode = createNode(
    uuidv4(),
    "Port Visual Styles",
    startX + colGap,
    currY,
    "port-styles",
    320,
    350,
  );
  portStyleNode.data.inputPorts = [
    {
      id: "in-1",
      label: "Circle (Default)",
      style: PortStyle.CIRCLE,
    },
    { id: "in-2", label: "Square", style: PortStyle.SQUARE },
    { id: "in-3", label: "Diamond", style: PortStyle.DIAMOND },
    { id: "in-4", label: "Dashed", style: PortStyle.DASH },
  ];
  nodes.push(portStyleNode);

  // --- COLUMN 3: PORT SEMANTICS (TYPES) ---
  currY = startY;
  const portTypeNode = createNode(
    uuidv4(),
    "Port Types (Semantics)",
    startX + colGap * 2,
    currY,
    "port-types",
    320,
    400,
  );
  portTypeNode.data.inputPorts = [
    {
      id: "it-1",
      label: "String Input",
      style: PortStyle.CIRCLE,
      type: { isGeneric: false, itemType: "", mainType: PortMainType.STRING },
    },
    {
      id: "it-2",
      label: "Number Input",
      style: PortStyle.CIRCLE,
      type: { isGeneric: false, itemType: "", mainType: PortMainType.NUMBER },
    },
    {
      id: "it-3",
      label: "Image List",
      style: PortStyle.CIRCLE,
      type: {
        isGeneric: false,
        itemType: "image",
        mainType: PortMainType.LIST,
      },
    },
    {
      id: "it-4",
      label: "Any Type",
      style: PortStyle.CIRCLE,
      type: { isGeneric: false, itemType: "", mainType: PortMainType.ANY },
    },
    {
      id: "it-5",
      label: "Generic T",
      style: PortStyle.CIRCLE,
      type: { isGeneric: true, itemType: "", mainType: PortMainType.ANY },
    },
  ];
  portTypeNode.data.outputPorts = [
    {
      id: "ot-1",
      label: "",
      style: PortStyle.CIRCLE,
      type: { isGeneric: false, itemType: "", mainType: PortMainType.ANY },
    },
    {
      id: "ot-2",
      label: "Signal Out",
      style: PortStyle.DASH,
      type: { isGeneric: false, itemType: "", mainType: PortMainType.SYSTEM },
    },
  ];
  nodes.push(portTypeNode);

  // --- ROW 2: MEDIA MODES ---
  currY = startY + rowGap;
  const mediaTypes = [
    {
      height: 200,
      id: uuidv4(),
      label: "Image Renderer",
      media: {
        galleryUrls: [
          "https://picsum.photos/id/238/400/300",
          "https://picsum.photos/id/239/400/300",
        ],
        type: MediaType.MEDIA_IMAGE,
        url: "https://picsum.photos/id/237/400/300",
      },
      templateId: "media-img",
      width: 300,
    },
    {
      height: 200,
      id: uuidv4(),
      label: "Video Renderer",
      media: {
        type: MediaType.MEDIA_VIDEO,
        url: "https://www.w3schools.com/html/mov_bbb.mp4",
      },
      templateId: "media-video",
      width: 300,
    },
    {
      height: 110,
      id: uuidv4(),
      label: "Audio Renderer",
      media: {
        type: MediaType.MEDIA_AUDIO,
        url: "https://www.w3schools.com/html/horse.mp3",
      },
      templateId: "media-audio",
      width: 240,
    },
    {
      height: 200,
      id: uuidv4(),
      label: "Markdown Renderer",
      media: {
        content:
          "# Markdown Title\n\n- List Item 1\n- List Item 2\n\n**Bold Text** and `code`.",
        type: MediaType.MEDIA_MARKDOWN,
      },
      templateId: "media-md",
      width: 300,
    },
  ];

  mediaTypes.forEach((m, i) => {
    const node = createNode(
      m.id,
      m.label,
      startX + i * colGap,
      currY,
      m.templateId,
      m.width,
      m.height,
    );
    node.data.modes = [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS];
    node.data.activeMode = RenderMode.MODE_MEDIA;
    node.data.media = m.media;

    // Define output port based on media type
    let portMainType = PortMainType.ANY;
    let portColor = "#a0aec0";

    switch (m.media.type) {
      case MediaType.MEDIA_AUDIO:
        portMainType = PortMainType.AUDIO;
        portColor = "#4299e1";
        break;
      case MediaType.MEDIA_IMAGE:
        portMainType = PortMainType.IMAGE;
        portColor = "#48bb78";
        break;
      case MediaType.MEDIA_MARKDOWN:
        portMainType = PortMainType.STRING;
        portColor = "#646cff";
        break;
      case MediaType.MEDIA_VIDEO:
        portMainType = PortMainType.VIDEO;
        portColor = "#ed64a6";
        break;
    }

    node.data.outputPorts = [
      {
        color: portColor,
        id: "out-1",
        label: "", // Empty label to hide text
        style: PortStyle.CIRCLE,
        type: { isGeneric: false, itemType: "", mainType: portMainType },
      },
    ];

    // Add some widgets too so we can test switching
    node.data.widgets = [
      {
        id: "sw-1",
        label: "Description",
        type: WidgetType.WIDGET_TEXT,
        value: `Setting for ${m.label}`,
      },
    ];
    nodes.push(node);
  });

  // --- ROW 3: COMPLEX COMBINATIONS ---
  currY = startY + rowGap * 2;

  // Node with implicit port binding (port tied to a widget)
  const implicitNode = createNode(
    uuidv4(),
    "Implicit Binding",
    startX,
    currY,
    "complex-implicit",
    320,
    300,
  );
  implicitNode.data.widgets = [
    {
      id: "linked-widget",
      inputPortId: "widget-port-1",
      label: "Manual / Remote",
      type: WidgetType.WIDGET_TEXT,
      value: "Override me via port",
    },
  ];
  implicitNode.data.inputPorts = [];
  nodes.push(implicitNode);

  // Large Processing Node (Custom Type)
  const procNode: AppNode = {
    data: {
      label: "AI Image Generation",
      progress: 45,
      status: 1, // PROCESSING
      taskId: "task-123",
    },
    id: uuidv4(),
    measured: { height: 120, width: 300 },
    position: { x: startX + colGap, y: currY },
    style: { height: 120, width: 300 },
    type: AppNodeType.PROCESSING,
  } as AppNode;
  nodes.push(procNode);

  // Group Node
  const groupId = uuidv4();
  const groupNode: AppNode = {
    data: { label: "Logical Group" },
    id: groupId,
    measured: { height: 300, width: 400 },
    position: { x: startX + colGap * 2, y: currY },
    style: { height: 300, width: 400 },
    type: AppNodeType.GROUP,
  } as AppNode;
  nodes.push(groupNode);

  // Child inside group
  const childNode = createNode(
    uuidv4(),
    "Child Node",
    startX + colGap * 2 + 50,
    currY + 50,
    "child-node",
    200,
    150,
  );
  childNode.parentId = groupId;
  childNode.extent = "parent";
  nodes.push(childNode);

  // --- ROW 4: TYPE TESTING ---
  currY = startY + rowGap * 3;

  // Image Batcher
  const batcherNode = createNode("batcher-1", "Image Batcher", startX, currY);
  batcherNode.data.inputPorts = [
    {
      color: "#48bb78",
      id: "in",
      label: "Img",
      type: { mainType: PortMainType.IMAGE },
    },
  ];
  batcherNode.data.outputPorts = [
    {
      color: "#48bb78",
      id: "out",
      label: "List",
      type: { itemType: "image", mainType: PortMainType.LIST },
    },
  ];
  nodes.push(batcherNode);

  // List Joiner (Generic Input)
  const joinerNode = createNode(
    "joiner-1",
    "Generic Joiner",
    startX + colGap,
    currY,
  );
  joinerNode.data.inputPorts = [
    {
      color: "#ecc94b",
      id: "list",
      label: "List",
      type: { isGeneric: true, mainType: PortMainType.LIST },
    },
  ];
  joinerNode.data.outputPorts = [
    {
      color: "#646cff",
      id: "out",
      label: "Str",
      type: { mainType: PortMainType.STRING },
    },
  ];
  nodes.push(joinerNode);

  // Pass-Through (Full Generic)
  const passNode = createNode(
    "pass-1",
    "Generic Pass",
    startX + colGap * 2,
    currY,
  );
  passNode.data.inputPorts = [
    {
      color: "#a0aec0",
      id: "in",
      label: "Any",
      type: { isGeneric: true, mainType: PortMainType.ANY },
    },
  ];
  passNode.data.outputPorts = [
    {
      color: "#a0aec0",
      id: "out",
      label: "Result",
      type: { isGeneric: true, mainType: PortMainType.ANY },
    },
  ];
  nodes.push(passNode);

  return { edges, nodes };
};
