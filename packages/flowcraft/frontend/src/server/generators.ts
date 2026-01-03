/* eslint-disable @typescript-eslint/no-explicit-any */
import { v4 as uuidv4 } from "uuid";
import { create } from "@bufbuild/protobuf";
import {
  WidgetType,
  RenderMode,
  MediaType,
  PortStyle,
  NodeTemplateSchema,
} from "../generated/flowcraft/v1/node_pb";
import { PortMainType } from "../generated/flowcraft/v1/base_pb";
import type { AppNode, Edge, NodeTemplate } from "../types";
import { toProtoNodeData, fromProtoNodeData } from "../utils/protoAdapter";

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
      typeId: typeId ?? "test-node",
      label,
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [],
      inputPorts: [],
      outputPorts: [],
    }),
  );

  return {
    id,
    type: "dynamic",
    position: { x, y },
    style: { width, height },
    measured: { width, height },
    data,
  } as AppNode;
};

export const nodeTemplates: NodeTemplate[] = [
  create(NodeTemplateSchema, {
    templateId: "widgets-all",
    displayName: "Widget Showcase",
    menuPath: ["Test"],
    defaultState: toProtoNodeData({
      label: "Widget Showcase",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [
        { id: "w1", type: WidgetType.WIDGET_TEXT, label: "Text", value: "" },
      ],
    }),
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
    { type: WidgetType.WIDGET_TEXT, label: "Text Field", value: "Hello World" },
    {
      type: WidgetType.WIDGET_SELECT,
      label: "Select",
      value: "opt1",
      options: [
        { label: "Option 1", value: "opt1" },
        { label: "Option 2", value: "opt2" },
      ],
    },
    { type: WidgetType.WIDGET_CHECKBOX, label: "Checkbox", value: true },
    {
      type: WidgetType.WIDGET_SLIDER,
      label: "Slider",
      value: 50,
      config: { min: 0, max: 100 },
    },
    { type: WidgetType.WIDGET_BUTTON, label: "Button Action", value: "click" },
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
  ] as any;
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
      type: { mainType: PortMainType.STRING },
    },
    {
      id: "it-2",
      label: "Number Input",
      type: { mainType: PortMainType.NUMBER },
    },
    {
      id: "it-3",
      label: "Image List",
      type: { mainType: PortMainType.LIST, itemType: "image" },
    },
    {
      id: "it-4",
      label: "Any Type",
      type: { mainType: PortMainType.ANY },
    },
    {
      id: "it-5",
      label: "Generic T",
      type: { mainType: PortMainType.ANY, isGeneric: true },
    },
  ] as any;
  portTypeNode.data.outputPorts = [
    {
      id: "ot-1",
      label: "",
      type: { mainType: PortMainType.ANY },
    },
    {
      id: "ot-2",
      label: "Signal Out",
      type: { mainType: PortMainType.SYSTEM },
      style: PortStyle.DASH,
    },
  ] as any;
  nodes.push(portTypeNode);

  // --- ROW 2: MEDIA MODES ---
  currY = startY + rowGap;
  const mediaTypes = [
    {
      id: uuidv4(),
      templateId: "media-img",
      label: "Image Renderer",
      width: 300,
      height: 200,
      media: {
        type: MediaType.MEDIA_IMAGE,
        url: "https://picsum.photos/id/237/400/300",
        galleryUrls: [
          "https://picsum.photos/id/238/400/300",
          "https://picsum.photos/id/239/400/300",
        ],
      },
    },
    {
      id: uuidv4(),
      templateId: "media-video",
      label: "Video Renderer",
      width: 300,
      height: 200,
      media: {
        type: MediaType.MEDIA_VIDEO,
        url: "https://www.w3schools.com/html/mov_bbb.mp4",
      },
    },
    {
      id: uuidv4(),
      templateId: "media-audio",
      label: "Audio Renderer",
      width: 240,
      height: 110,
      media: {
        type: MediaType.MEDIA_AUDIO,
        url: "https://www.w3schools.com/html/horse.mp3",
      },
    },
    {
      id: uuidv4(),
      templateId: "media-md",
      label: "Markdown Renderer",
      width: 300,
      height: 200,
      media: {
        type: MediaType.MEDIA_MARKDOWN,
        content:
          "# Markdown Title\n\n- List Item 1\n- List Item 2\n\n**Bold Text** and `code`.",
      },
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
      case MediaType.MEDIA_IMAGE:
        portMainType = PortMainType.IMAGE;
        portColor = "#48bb78";
        break;
      case MediaType.MEDIA_VIDEO:
        portMainType = PortMainType.VIDEO;
        portColor = "#ed64a6";
        break;
      case MediaType.MEDIA_AUDIO:
        portMainType = PortMainType.AUDIO;
        portColor = "#4299e1";
        break;
      case MediaType.MEDIA_MARKDOWN:
        portMainType = PortMainType.STRING;
        portColor = "#646cff";
        break;
    }

    node.data.outputPorts = [
      {
        id: "out-1",
        label: "", // Empty label to hide text
        type: { mainType: portMainType },
        color: portColor,
      },
    ] as any;

    // Add some widgets too so we can test switching
    node.data.widgets = [
      {
        id: "sw-1",
        type: WidgetType.WIDGET_TEXT,
        label: "Description",
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
      type: WidgetType.WIDGET_TEXT,
      label: "Manual / Remote",
      value: "Override me via port",
      inputPortId: "widget-port-1",
    },
  ];
  implicitNode.data.inputPorts = [] as any;
  nodes.push(implicitNode);

  // Large Processing Node (Custom Type)
  const procNode: AppNode = {
    id: uuidv4(),
    type: "processing",
    position: { x: startX + colGap, y: currY },
    style: { width: 300, height: 120 },
    measured: { width: 300, height: 120 },
    data: {
      taskId: "task-123",
      label: "AI Image Generation",
      progress: 45,
      status: 1, // PROCESSING
    },
  } as AppNode;
  nodes.push(procNode);

  // Group Node
  const groupId = uuidv4();
  const groupNode: AppNode = {
    id: groupId,
    type: "groupNode",
    position: { x: startX + colGap * 2, y: currY },
    style: { width: 400, height: 300 },
    measured: { width: 400, height: 300 },
    data: { label: "Logical Group" },
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
      id: "in",
      label: "Img",
      type: { mainType: PortMainType.IMAGE },
      color: "#48bb78",
    },
  ];
  batcherNode.data.outputPorts = [
    {
      id: "out",
      label: "List",
      type: { mainType: PortMainType.LIST, itemType: "image" },
      color: "#48bb78",
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
      id: "list",
      label: "List",
      type: { mainType: PortMainType.LIST, isGeneric: true },
      color: "#ecc94b",
    },
  ];
  joinerNode.data.outputPorts = [
    {
      id: "out",
      label: "Str",
      type: { mainType: PortMainType.STRING },
      color: "#646cff",
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
      id: "in",
      label: "Any",
      type: { mainType: PortMainType.ANY, isGeneric: true },
      color: "#a0aec0",
    },
  ];
  passNode.data.outputPorts = [
    {
      id: "out",
      label: "Result",
      type: { mainType: PortMainType.ANY, isGeneric: true },
      color: "#a0aec0",
    },
  ];
  nodes.push(passNode);

  return { nodes, edges };
};
