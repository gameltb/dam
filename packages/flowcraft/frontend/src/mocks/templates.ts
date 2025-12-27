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
          style: PortStyle.PORT_STYLE_SQUARE,
          color: "#f6ad55",
        },
        {
          id: "b",
          label: "B",
          type: { mainType: "bool" },
          style: PortStyle.PORT_STYLE_SQUARE,
          color: "#f6ad55",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Out",
          type: { mainType: "bool" },
          style: PortStyle.PORT_STYLE_SQUARE,
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
          style: PortStyle.PORT_STYLE_SQUARE,
          color: "#f6ad55",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Out",
          type: { mainType: "bool" },
          style: PortStyle.PORT_STYLE_SQUARE,
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
];

export const actionTemplates = [
  {
    id: "spawn-sys",
    label: "Spawn Linked Node",
    path: ["Development"],
    strategy: ActionExecutionStrategy.EXECUTION_IMMEDIATE,
  },
];
