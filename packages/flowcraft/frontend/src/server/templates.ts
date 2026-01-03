import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  WidgetType,
  RenderMode,
  TaskStatus,
  PortStyle,
} from "../generated/flowcraft/v1/node_pb";
import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
} from "../generated/flowcraft/v1/action_pb";
import {
  MutationSource,
  PortMainType,
} from "../generated/flowcraft/v1/base_pb";
import { toProtoNodeData, toProtoNode } from "../utils/protoAdapter";
import { NodeRegistry } from "./registry";
import { TaskUpdateSchema } from "../generated/flowcraft/v1/node_pb";
import { MutationListSchema } from "../generated/flowcraft/v1/service_pb";
import { isDynamicNode } from "../types";
import { incrementVersion } from "./db";

// --- 注册 Basic Nodes ---
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-text",
    displayName: "Text Node",
    menuPath: ["Input", "Basic"],
    defaultState: toProtoNodeData({
      label: "Text Input",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgets: [
        { id: "w1", type: WidgetType.WIDGET_TEXT, label: "Content", value: "" },
      ],
    }),
  }),
});

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-slider",
    displayName: "Slider Input",
    menuPath: ["Input", "Advanced"],
    defaultState: toProtoNodeData({
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
    }),
  }),
});

// --- 注册 Logic Nodes ---
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-logic-and",
    displayName: "AND Gate",
    menuPath: ["Logic", "Boolean"],
    defaultState: toProtoNodeData({
      label: "AND",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "a",
          label: "A",
          type: { mainType: PortMainType.BOOLEAN },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
        {
          id: "b",
          label: "B",
          type: { mainType: PortMainType.BOOLEAN },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Out",
          type: { mainType: PortMainType.BOOLEAN },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
    } as any),
  }),
});

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-logic-not",
    displayName: "NOT Gate",
    menuPath: ["Logic", "Boolean"],
    defaultState: toProtoNodeData({
      label: "NOT",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "in",
          label: "In",
          type: { mainType: PortMainType.BOOLEAN },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Out",
          type: { mainType: PortMainType.BOOLEAN },
          style: PortStyle.SQUARE,
          color: "#f6ad55",
        },
      ],
    } as any),
  }),
});

// --- 注册 Media Nodes ---
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-img-proc",
    displayName: "Image Filter",
    menuPath: ["Media", "Image"],
    defaultState: toProtoNodeData({
      label: "Grayscale",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "in",
          label: "Image",
          type: { mainType: PortMainType.IMAGE },
          color: "#48bb78",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Filtered",
          type: { mainType: PortMainType.IMAGE },
          color: "#48bb78",
        },
      ],
    } as any),
  }),
});

// --- 注册 AI Nodes ---
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-ai-gen",
    displayName: "Image Generator",
    menuPath: ["AI", "Generation"],
    defaultState: toProtoNodeData({
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
    }),
  }),
  actions: [
    create(ActionTemplateSchema, {
      id: "gen",
      label: "Trigger Generation",
      strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
    }),
  ],
  execute: async (ctx) => {
    ctx.emitTaskUpdate(
      create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_PROCESSING,
        progress: 0,
        message: "Initializing AI model...",
      }),
    );

    await new Promise((r) => setTimeout(r, 1000));

    // 更新节点标签作为结果
    if (isDynamicNode(ctx.node)) {
      ctx.node.data.label = "AI Generated Result";
      const protoNode = toProtoNode(ctx.node);

      incrementVersion();
      ctx.emitMutation(
        create(MutationListSchema, {
          mutations: [
            {
              operation: {
                case: "updateNode",
                value: {
                  id: ctx.node.id,
                  data: protoNode.state,
                  presentation: protoNode.presentation,
                },
              },
            },
          ],
          source: MutationSource.SOURCE_REMOTE_TASK,
        }),
      );
    }

    ctx.emitTaskUpdate(
      create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_COMPLETED,
        progress: 100,
        message: "Image generated successfully!",
      }),
    );
  },
});

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-stream-node",
    displayName: "AI Agent (Streaming)",
    menuPath: ["AI", "Agent"],
    widgetsSchemaJson: JSON.stringify({
      type: "object",
      properties: {
        agent_name: {
          type: "string",
          title: "Agent Name",
          default: "Assistant",
        },
        logs: {
          type: "string",
          title: "Execution Logs",
          uiWidget: "streamingText",
        },
        run: {
          type: "boolean",
          title: "Run Command",
          uiWidget: "signalButton",
        },
      },
    }),
    defaultState: toProtoNodeData({
      label: "AI Agent",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgetsValues: {
        agent_name: "FlowAgent v1",
        logs: "",
      },
    }),
  }),
  execute: async (ctx) => {
    ctx.emitTaskUpdate(
      create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_PROCESSING,
        progress: 10,
        message: "Agent thinking...",
      }),
    );

    const text =
      "Connecting to LLM... Done. Thinking... I can help you build this workflow. Adding a result node now.";
    const words = text.split(" ");

    for (const word of words) {
      ctx.emitStreamChunk({
        nodeId: ctx.node.id,
        widgetId: "logs",
        chunkData: word + " ",
        isDone: false,
      });
      await new Promise((r) => setTimeout(r, 150));
    }

    ctx.emitTaskUpdate(
      create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_COMPLETED,
        progress: 100,
        message: "Agent task finished.",
      }),
    );
  },
});

// --- 注册 Utility Nodes ---
NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-any-pass",
    displayName: "Generic Pass",
    menuPath: ["Utility", "Debug"],
    defaultState: toProtoNodeData({
      label: "Pass-Through",
      modes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      inputPorts: [
        {
          id: "any",
          label: "Any",
          type: { mainType: PortMainType.ANY, isGeneric: true },
          color: "#a0aec0",
        },
      ],
      outputPorts: [
        {
          id: "out",
          label: "Result",
          type: { mainType: PortMainType.ANY, isGeneric: true },
          color: "#a0aec0",
        },
      ],
    } as any),
  }),
});
