import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  NodeDataSchema,
  RenderMode,
  TaskUpdateSchema,
  TaskStatus,
} from "../../generated/flowcraft/v1/core/node_pb";
import { StreamChunkSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { NodeRegistry } from "../registry";
import { OpenAI } from "openai";

// 直接使用现有的 OpenAI 配置（从 .env 读取）
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "your-key-here",
  baseURL: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
});

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-ai-chat",
    displayName: "AI Chat",
    menuPath: ["AI"],
    widgetsSchemaJson: JSON.stringify({
      type: "object",
      properties: {
        conversation: { type: "object", title: "Conversation", uiWidget: "chatInterface" }
      }
    }),
    defaultState: create(NodeDataSchema, {
      displayName: "AI Assistant",
      availableModes: [RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_WIDGETS,
      widgetsValuesJson: JSON.stringify({
        conversation: {
          messages: [{ id: "1", role: "assistant", content: "Hello! How can I help you today?" }],
          activeLeafId: "1"
        }
      })
    }),
  }),
  // 核心：使用现有的 gRPC 执行通道
  execute: async (ctx) => {
    const { node, params, emitStreamChunk, emitTaskUpdate } = ctx;
    const history = params.messages || [];

    try {
      emitTaskUpdate(create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_PROCESSING,
        message: "Thinking..."
      }));

      const stream = await openai.chat.completions.create({
        model: process.env.OPENAI_MODEL || "gpt-3.5-turbo",
        messages: history.map((m: any) => ({ role: m.role, content: m.content })),
        stream: true,
      });

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        if (content) {
          // 通过 Protobuf 协议发送流式数据
          emitStreamChunk(create(StreamChunkSchema, {
            nodeId: node.id,
            widgetId: "conversation", // 对应 widgetId
            chunkData: content,
            isDone: false,
          } as any));
        }
      }

      emitTaskUpdate(create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_COMPLETED,
        message: "Done"
      }));
    } catch (err: any) {
      emitTaskUpdate(create(TaskUpdateSchema, {
        taskId: ctx.taskId,
        status: TaskStatus.TASK_FAILED,
        message: err.message
      }));
    }
  }
});