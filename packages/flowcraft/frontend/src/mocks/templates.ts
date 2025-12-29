import type { NodeTemplate } from "../types";
import {
  WidgetType,
  RenderMode,
  ActionExecutionStrategy,
  PortStyle,
} from "../types";

export const nodeTemplates: NodeTemplate[] = [
  {
    id: "tpl-text",
    label: "Text Node",
    path: ["Input", "Basic"],
    defaultData: {
      label: "Text Input",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [
        { id: "w1", type: WidgetType.WIDGET_TEXT, label: "Content", value: "" },
      ],
    },
  },
  {
    id: "tpl-slider",
    label: "Slider Input",
    path: ["Input", "Advanced"],
    defaultData: {
      label: "Range",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [
        {
          id: "sl1",
          type: WidgetType.WIDGET_SLIDER,
          label: "Value",
          value: 50,
          config: { min: 0, max: 100 },
        },
      ],
    },
  },
  {
    id: "tpl-logic-and",
    label: "AND Gate",
    path: ["Logic", "Boolean"],
    defaultData: {
      label: "AND",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "a",
          label: "A",
          type: { mainType: "bool" },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
        {
          id: "b",
          label: "B",
          type: { mainType: "bool" },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Out",
          type: { mainType: "bool" },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
    },
  },
  {
    id: "tpl-logic-not",
    label: "NOT Gate",
    path: ["Logic", "Boolean"],
    defaultData: {
      label: "NOT",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "in",
          label: "In",
          type: { mainType: "bool" },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Out",
          type: { mainType: "bool" },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
    },
  },
  {
    id: "tpl-ai-gen",
    label: "Image Generator",
    path: ["AI", "Generation"],
    defaultData: {
      label: "Stable Diffusion",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [
        {
          id: "p1",
          type: WidgetType.WIDGET_TEXT,
          label: "Prompt",
          value: "A futuristic city",
        },
        {
          id: "b1",
          type: WidgetType.WIDGET_BUTTON,
          label: "Generate",
          value: "task:gen",
        },
      ],
    },
  },
  // --- MEDIA PROCESSING ---
  {
    id: "tpl-img-proc",
    label: "Image Filter",
    path: ["Media", "Image"],
    defaultData: {
      label: "Grayscale",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "in",
          label: "Image",
          type: { mainType: "image" },
          color: "#48bb78",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Filtered",
          type: { mainType: "image" },
          color: "#48bb78",
        },
      ],
    },
  },
  {
    id: "tpl-vid-proc",
    label: "Video Trimmer",
    path: ["Media", "Video"],
    defaultData: {
      label: "Trimmer",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "in",
          label: "Video",
          type: { mainType: "video" },
          color: "#ed64a6",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Trimmed",
          type: { mainType: "video" },
          color: "#ed64a6",
        },
      ],
    },
  },
  // --- COLLECTIONS & CONTAINERS ---
  {
    id: "tpl-list-merge",
    label: "List Joiner",
    path: ["Logic", "Collections"],
    defaultData: {
      label: "Join List",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "list",
          label: "Input List",
          type: { mainType: "list", isGeneric: true },
          color: "#ecc94b",
        },
      ],
      outputPorts: [
        {
          id: "str",
          label: "String",
          type: { mainType: "string" },
          color: "#646cff",
        },
      ],
      widgets: [
        {
          id: "sep",
          type: WidgetType.WIDGET_TEXT,
          label: "Separator",
          value: ", ",
        },
      ],
    },
  },
  {
    id: "tpl-batch-image",
    label: "Image Batcher",
    path: ["Media", "Image"],
    defaultData: {
      label: "Batch",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "img",
          label: "Image",
          type: { mainType: "image" },
          color: "#48bb78",
        },
      ],
      outputPorts: [
        {
          id: "batch",
          label: "Image List",
          type: { mainType: "list", itemType: "image" },
          color: "#48bb78",
          style: PortStyle.SQUARE,
        },
      ],
    },
  },
  // --- GENERIC TYPES ---
  {
    id: "tpl-any-pass",
    label: "Generic Pass",
    path: ["Utility", "Debug"],
    defaultData: {
      label: "Pass-Through",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "any",
          label: "Any",
          type: { mainType: "any", isGeneric: true },
          color: "#a0aec0",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Result",
          type: { mainType: "any", isGeneric: true },
          color: "#a0aec0",
        },
      ],
    },
  },
];

export const actionTemplates = [
  {
    id: "spawn-sys",
    label: "Spawn Linked Node",
    path: ["Development"],
    strategy: ActionExecutionStrategy.EXECUTION_IMMEDIATE,
  },
];
