import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  WidgetType,
  RenderMode,
  TaskStatus,
  TaskUpdateSchema,
} from "../../generated/flowcraft/v1/core/node_pb";
import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
} from "../../generated/flowcraft/v1/core/action_pb";
import { MutationSource } from "../../generated/flowcraft/v1/core/base_pb";
import { MutationListSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { toProtoNodeData, toProtoNode } from "../../utils/protoAdapter";
import { NodeRegistry, type NodeExecutionContext } from "../registry";
import { isDynamicNode } from "../../types";
import { incrementVersion } from "../db";

async function executeAiGen(ctx: NodeExecutionContext) {
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
}

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
  execute: executeAiGen,
});
