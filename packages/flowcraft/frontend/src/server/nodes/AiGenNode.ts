import { create } from "@bufbuild/protobuf";

import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
} from "../../generated/flowcraft/v1/core/action_pb";
import { MutationSource } from "../../generated/flowcraft/v1/core/base_pb";
import {
  NodeTemplateSchema,
  RenderMode,
  TaskStatus,
  TaskUpdateSchema,
  WidgetType,
} from "../../generated/flowcraft/v1/core/node_pb";
import { MutationListSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { isDynamicNode } from "../../types";
import { toProtoNode, toProtoNodeData } from "../../utils/protoAdapter";
import { incrementVersion } from "../db";
import { type NodeExecutionContext, NodeRegistry } from "../registry";

async function executeAiGen(ctx: NodeExecutionContext) {
  const { actionId, emitMutation, emitTaskUpdate, node, taskId } = ctx;

  if (actionId !== "flowcraft.action.node.generator.run") return;

  emitTaskUpdate(
    create(TaskUpdateSchema, {
      message: "Initializing AI model...",
      progress: 0,
      status: TaskStatus.TASK_PROCESSING,
      taskId: taskId,
    }),
  );

  await new Promise((r) => setTimeout(r, 1000));

  // 更新节点标签作为结果
  if (isDynamicNode(node)) {
    node.data.label = "AI Generated Result";
    const protoNode = toProtoNode(node);

    incrementVersion();
    emitMutation(
      create(MutationListSchema, {
        mutations: [
          {
            operation: {
              case: "updateNode",
              value: {
                data: protoNode.state,
                id: node.id,
                presentation: protoNode.presentation,
              },
            },
          },
        ],
        source: MutationSource.SOURCE_REMOTE_TASK,
      }),
    );
  }

  emitTaskUpdate(
    create(TaskUpdateSchema, {
      message: "Image generated successfully!",
      progress: 100,
      status: TaskStatus.TASK_COMPLETED,
      taskId: taskId,
    }),
  );
}

NodeRegistry.register({
  actions: [
    create(ActionTemplateSchema, {
      id: "flowcraft.action.node.generator.run",
      label: "Run Content Generation",
      strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
    }),
  ],
  execute: executeAiGen,
  template: create(NodeTemplateSchema, {
    defaultState: toProtoNodeData({
      activeMode: RenderMode.MODE_WIDGETS,
      label: "Stable Diffusion",
      modes: [RenderMode.MODE_WIDGETS],
      widgets: [
        {
          id: "p1",
          label: "Prompt",
          type: WidgetType.WIDGET_TEXT,
          value: "A futuristic city",
        },
        {
          id: "b1",
          label: "Generate",
          type: WidgetType.WIDGET_BUTTON,
          value: "task:flowcraft.action.node.generator.run",
        },
      ],
    }),
    displayName: "AI Content Generator",
    menuPath: ["AI", "Generation"],
    templateId: "flowcraft.node.ai.generator",
  }),
});
