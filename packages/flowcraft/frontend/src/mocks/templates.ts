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
    path: ["Basic"],
    defaultData: {
      label: "New Node",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [
        { id: "w1", type: WidgetType.WIDGET_TEXT, label: "Content", value: "" },
      ],
    },
  },
  {
    id: "tpl-logic",
    label: "AND Gate",
    path: ["Logic"],
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
];

export const actionTemplates = [
  {
    id: "spawn-sys",
    label: "Spawn Linked Node",
    path: ["Development"],
    strategy: ActionExecutionStrategy.EXECUTION_IMMEDIATE,
  },
];
