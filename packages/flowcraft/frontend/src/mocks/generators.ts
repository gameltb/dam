import type { AppNode, Edge } from "../types";
import { WidgetType, RenderMode, MediaType, PortStyle } from "../types";

export const createNode = (
  id: string,
  label: string,
  x: number,
  y: number,
): AppNode =>
  ({
    id,
    type: "dynamic",
    position: { x, y },
    style: { width: 350, height: 300 },
    data: {
      label,
      onChange: () => {},
      inputPorts: [],
      outputPorts: [],
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [],
    },
  }) as AppNode;

export const generateGallery = () => {
  const nodes: AppNode[] = [];
  const edges: Edge[] = [];

  let x = 50;
  let y = 50;
  const colGap = 450;
  const rowGap = 500;

  // 1. All Widgets
  const wShowcase = createNode(
    "gallery-widgets",
    "1. All Widgets Matrix",
    x,
    y,
  );
  wShowcase.style = { width: 350, height: 550 };
  wShowcase.data.widgets = [
    {
      id: "t1",
      type: WidgetType.WIDGET_TEXT,
      label: "Text Field",
      value: "Protocol V2",
    },
    {
      id: "s1",
      type: WidgetType.WIDGET_SELECT,
      label: "Dropdown",
      value: "v1",
      options: [
        { label: "Option A", value: "v1" },
        { label: "Option B", value: "v2" },
      ],
    },
    {
      id: "c1",
      type: WidgetType.WIDGET_CHECKBOX,
      label: "Toggle Switch",
      value: true,
    },
    {
      id: "sl1",
      type: WidgetType.WIDGET_SLIDER,
      label: "Range Slider",
      value: 42,
      config: { min: 0, max: 100 },
    },
    {
      id: "b1",
      type: WidgetType.WIDGET_BUTTON,
      label: "Task Trigger",
      value: "task:demo",
    },
    {
      id: "b2",
      type: WidgetType.WIDGET_BUTTON,
      label: "Stream Trigger",
      value: "stream-to:t1",
    },
  ];
  nodes.push(wShowcase);

  // 2. Ports
  x += colGap;
  const portNode = createNode(
    "gallery-ports",
    "2. Port Semantic & Styles",
    x,
    y,
  );
  portNode.style = { width: 350, height: 450 };
  portNode.data.inputPorts = [
    {
      id: "in-std",
      label: "Standard (Any)",
      type: { mainType: "any" },
      style: PortStyle.PORT_STYLE_CIRCLE,
      color: "#646cff",
    },
    {
      id: "in-list",
      label: "List<String>",
      type: { mainType: "list", itemType: "string" },
      style: PortStyle.PORT_STYLE_SQUARE,
      color: "#ed64a6",
    },
    {
      id: "in-set",
      label: "Set<ID>",
      type: { mainType: "set", itemType: "id" },
      style: PortStyle.PORT_STYLE_CIRCLE,
      color: "#f6ad55",
    },
    {
      id: "in-gen",
      label: "Generic T",
      type: { mainType: "any", isGeneric: true },
      style: PortStyle.PORT_STYLE_DIAMOND,
      color: "#a0aec0",
    },
  ];
  portNode.data.outputPorts = [
    {
      id: "out-std",
      label: "Result Out",
      type: { mainType: "any" },
      style: PortStyle.PORT_STYLE_CIRCLE,
      color: "#646cff",
    },
    {
      id: "out-sys",
      label: "System Flow",
      type: { mainType: "system" },
      style: PortStyle.PORT_STYLE_DASH,
      color: "#cbd5e0",
    },
  ];
  nodes.push(portNode);

  // 3. Implicit
  x += colGap;
  const implicitNode = createNode(
    "gallery-implicit",
    "3. Implicit Port Binding",
    x,
    y,
  );
  implicitNode.data.widgets = [
    {
      id: "linked",
      type: WidgetType.WIDGET_TEXT,
      label: "External Input",
      value: "Lock when connected",
      inputPortId: "implicit-in",
    },
  ];
  implicitNode.data.inputPorts = [
    {
      id: "implicit-in",
      label: "Remote Data",
      type: { mainType: "string" },
      style: PortStyle.PORT_STYLE_CIRCLE,
      color: "#646cff",
    },
  ];
  nodes.push(implicitNode);

  // 4. Media
  x = 50;
  y += rowGap;
  const imgNode = createNode("gallery-img", "4. Image & Gallery", x, y);
  imgNode.data.modes = [RenderMode.MODE_MEDIA];
  imgNode.data.activeMode = RenderMode.MODE_MEDIA;
  imgNode.data.media = {
    type: MediaType.MEDIA_IMAGE,
    url: "https://picsum.photos/id/1011/400/300",
    aspectRatio: 1.33,
    galleryUrls: [
      "https://picsum.photos/id/1012/400/300",
      "https://picsum.photos/id/1013/400/300",
    ],
  };
  nodes.push(imgNode);

  return { nodes, edges };
};
